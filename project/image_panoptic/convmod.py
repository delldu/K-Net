import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import pdb


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    conv --
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    norm --
        nn.GroupNorm(norm_num_groups, num_channels, eps=1e-05, affine=True)
    activation --
        nn.ReLU(inplace=True)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm_num_groups=-1,
    ):
        super(ConvModule, self).__init__()
        if norm_num_groups > 0:
            bias = False  # if norm exists, bias is unnecessary.

        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_num_groups > 0:
            # norm layer is after conv layer
            self.gn = nn.GroupNorm(norm_num_groups, out_channels)
            self.activate = nn.ReLU(inplace=True)
        else:
            self.gn = nn.Identity()
            self.activate = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.activate(x)
        return x


class FPN(nn.Module):
    r"""Feature Pyramid Network for R50."""

    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=4, start_level=0):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.end_level = self.num_ins
        self.start_level = start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.end_level):
            l_conv = ConvModule(in_channels[i], out_channels, kernel_size=1)
            fpn_conv = ConvModule(out_channels, out_channels, kernel_size=3, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(inputs) == len(self.in_channels)

        # build laterals
        # self.lateral_convs -- ModuleList(
        # (0): ConvModule( ... )
        # (3): ConvModule( ... ))

        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)  # -- 4
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode="nearest")

        # build outputs
        outs: List[torch.Tensor] = []
        for i, block in enumerate(self.fpn_convs):
            outs.append(block(laterals[i]))

        return outs


class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.
    """

    def __init__(self,
                 num_feats=128,
                 temperature=10000,
                 normalize=True,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.):
        super(SinePositionalEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset
        

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


class SemanticFPNWrapper(nn.Module):
    """Implementation of Semantic FPN used in Panoptic FPN.

    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 out_channels=256,
                 start_level=0,
                 end_level=3,
                 cat_coors_level=3):
        super(SemanticFPNWrapper, self).__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.cat_coors_level = cat_coors_level
        self.positional_encoding = SinePositionalEncoding()

        self.convs_all_levels = nn.ModuleList()

        # convs_all_levels -- 0, 1, 2 (upsample), 3 (upsample)
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    stride=2,
                    norm_num_groups=32)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
            else:
                for j in range(i):
                    chn = self.in_channels if j == 0 else self.feat_channels
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        norm_num_groups=32)
                    convs_per_level.add_module('conv' + str(j), one_conv)

                    if j < i - 1:
                        one_upsample = nn.Upsample(
                            scale_factor=2, mode='bilinear', align_corners=False)
                        convs_per_level.add_module('upsample' + str(j),
                                                   one_upsample)
                self.convs_all_levels.append(convs_per_level)

        self.conv_pred = ConvModule(
            self.feat_channels,
            self.out_channels,
            1,
            padding=0,
            norm_num_groups=32)

        self.aux_convs = nn.ModuleList()
        self.aux_convs.append(
            ConvModule(
                self.feat_channels,
                self.out_channels,
                1,
                padding=0,
                norm_num_groups=32))

    def forward(self, inputs):
        mlvl_feats = []
        for i in range(self.start_level, self.end_level + 1):
            input_p = inputs[i]
            if i == self.cat_coors_level: # 3
                ignore_mask = input_p.new_zeros(
                    (input_p.shape[0], input_p.shape[-2],
                     input_p.shape[-1]),
                    dtype=torch.bool)
                positional_encoding = self.positional_encoding(ignore_mask)
                input_p = input_p + positional_encoding

            mlvl_feats.append(self.convs_all_levels[i](input_p))

        feature_add_all_level = sum(mlvl_feats)

        out = self.conv_pred(feature_add_all_level)
        outs = [out]
        for conv in self.aux_convs:
            outs.append(conv(feature_add_all_level))
        return outs

class KernelUpdator(nn.Module):
    '''
    nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
    '''
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 out_channels=256,
                 input_feat_shape=3):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        if isinstance(input_feat_shape, int):
            input_feat_shape = [input_feat_shape] * 2
        self.input_feat_shape = input_feat_shape
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)

        self.norm_in = nn.LayerNorm((self.feat_channels,))
        self.norm_out = nn.LayerNorm((self.feat_channels,))
        self.input_norm_in = nn.LayerNorm((self.feat_channels,))
        self.input_norm_out = nn.LayerNorm((self.feat_channels,))
        self.activation = nn.ReLU(inplace=True)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = nn.LayerNorm((self.out_channels,))


    def forward(self, update_feature, input_feature):
        update_feature = update_feature.reshape(-1, self.in_channels)
        num_proposals = update_feature.size(0)
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.num_params_in].view(
            -1, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels)

        input_feats = self.input_layer(
            input_feature.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in.unsqueeze(-2)

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        input_gate = input_gate.sigmoid()
        update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features



class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(proj_drop)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.
    """
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 **kwargs):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class KernelUpdateHead(nn.Module):
    def __init__(self,
                 num_classes=133,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 feedforward_channels=2048,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 mask_thr=0.5,
                 conv_kernel_size=1,
                 hard_mask_thr=0.5,
                 feat_gather_stride=1,
                 ):
        super(KernelUpdateHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.conv_kernel_size = conv_kernel_size

        self.attention = MultiheadAttention(in_channels * conv_kernel_size**2, num_heads)

        # LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.attention_norm = nn.LayerNorm((in_channels * conv_kernel_size**2,))

        self.kernel_update_conv = KernelUpdator()

        self.feat_transform = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=feat_gather_stride,
            padding=int(feat_gather_stride // 2))

        self.ffn = FFN(in_channels, feedforward_channels, num_ffn_fcs, dropout=dropout)
        self.ffn_norm = nn.LayerNorm((in_channels,))

        self.cls_fcs = nn.ModuleList()
        self.cls_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
        self.cls_fcs.append(nn.LayerNorm((in_channels,)))
        self.cls_fcs.append(nn.ReLU(inplace=True))

        self.fc_cls = nn.Linear(in_channels, self.num_classes)

        self.mask_fcs = nn.ModuleList()
        self.mask_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
        self.mask_fcs.append(nn.LayerNorm((in_channels,)))
        self.mask_fcs.append(nn.ReLU(inplace=True))

        self.fc_mask = nn.Linear(in_channels, out_channels)

    def forward(self,
                x,
                proposal_feat,
                mask_preds,
                prev_cls_score=None,
                mask_shape=None,
                img_metas=None):

        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)
        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.float()

        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape(N, num_proposals,
                                              self.in_channels,
                                              -1).permute(0, 1, 3, 2)
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)

        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1).permute(1, 0, 2)
        obj_feat = self.attention_norm(self.attention(obj_feat))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.permute(1, 0, 2)

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1, self.in_channels)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        mask_x = x
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # fold_x = F.unfold(
        #     mask_x,
        #     self.conv_kernel_size,
        #     padding=int(self.conv_kernel_size // 2))
        # mask_feat = mask_feat.reshape(N, num_proposals, -1)
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(mask_x[i:i + 1], mask_feat[i], padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds, mask_shape, align_corners=False, mode='bilinear')

        return cls_score, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size, self.conv_kernel_size)

if __name__ == "__main__":
    # model = FPN()
    # model.eval()
    # print(model)

    # model = torch.jit.script(model)

    # inputs = [
    #     torch.randn(1, 256, 200, 200),
    #     torch.randn(1, 512, 100, 100),
    #     torch.randn(1, 1024, 50, 50),
    #     torch.randn(1, 2048, 25, 25),
    # ]
    # with torch.no_grad():
    #     output = model(inputs)

    # for i in range(len(output)):
    #     print(output[i].size())



    # model = SinePositionalEncoding()
    # model.eval()
    # print(model)


    # model = SemanticFPNWrapper()
    # print(model)

    # model = KernelUpdator()
    # print(model)

    model = KernelUpdateHead()
    print(model)

    pdb.set_trace()
