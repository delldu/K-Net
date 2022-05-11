import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         build_transformer_layer)
# from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.utils import get_root_logger

import pdb

@HEADS.register_module()
class KernelUpdateHead(nn.Module):
    # xxxx1111 -- ?
    def __init__(self,
                 num_classes=133,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_mask_fcs=1,
                 feedforward_channels=2048,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 mask_thr=0.5,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 conv_kernel_size=1,
                 feat_transform_cfg={'conv_cfg': {'type': 'Conv2d'}, 'act_cfg': None},
                 hard_mask_thr=0.5,
                 kernel_init=False,
                 with_ffn=True,
                 mask_out_stride=4,
                 relative_coors=False,
                 relative_coors_off=False,
                 feat_gather_stride=1,
                 mask_transform_stride=1,
                 mask_upsample_stride=2,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 kernel_updator_cfg={'type': 'KernelUpdator', 'in_channels': 256, 'feat_channels': 256, 'out_channels': 256, 'input_feat_shape': 3, 'act_cfg': {'type': 'ReLU', 'inplace': True}, 'norm_cfg': {'type': 'LN'}},
                 loss_rank={'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.1},
                 loss_mask={'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0},
                 loss_dice={'type': 'DiceLoss', 'loss_weight': 4.0},
                 loss_cls={'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}):
        super(KernelUpdateHead, self).__init__()

        self.num_classes = num_classes
        # xxxx8888
        self.loss_cls = build_loss(loss_cls)

        # xxxx8888
        self.loss_mask = build_loss(loss_mask)

        # xxxx8888
        self.loss_dice = build_loss(loss_dice)
        if loss_rank is not None: # True
            self.loss_rank = build_loss(loss_rank) # xxxx8888
        else:
            self.loss_rank = loss_rank

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.mask_upsample_stride = mask_upsample_stride

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg

        # xxxx8888
        self.attention = MultiheadAttention(in_channels * conv_kernel_size**2,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size**2)[1]

        self.kernel_update_conv = build_transformer_layer(kernel_updator_cfg)

        # feat_transform_cfg -- {'conv_cfg': {'type': 'Conv2d'}, 'act_cfg': None}
        if feat_transform_cfg is not None: # True
            kernel_size = feat_transform_cfg.pop('kernel_size', 1) # 1
            # xxxx8888 ConvModule
            self.feat_transform = ConvModule(
                in_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg)
            #self.feat_transform -- ConvModule(
            #     (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            # )
        else:
            self.feat_transform = None

        if self.with_ffn: # True
            # xxxx8888
            self.ffn = FFN(
                in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs): #  num_cls_fcs -- 1
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(act_cfg))

        if self.loss_cls.use_sigmoid: # True
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs): # num_mask_fcs -- 1
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.fc_mask = nn.Linear(in_channels, out_channels)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        if self.kernel_init:
            logger = get_root_logger()
            logger.info(
                'mask kernel in mask head is normal initialized by std 0.01')
            nn.init.normal_(self.fc_mask.weight, mean=0, std=0.01)

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
        if self.with_ffn:
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

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
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
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                mask_shape,
                align_corners=False,
                mode='bilinear')

        return cls_score, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size)


    def rescale_masks(self, masks_per_img, img_meta):
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks
