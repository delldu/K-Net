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

        self.blocks = nn.ModuleList()

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
        self.blocks.append(self.conv)

        if norm_num_groups > 0:
            # norm layer is after conv layer
            self.gn = nn.GroupNorm(norm_num_groups, out_channels)
            self.activate = nn.ReLU(inplace=True)

            self.blocks.append(self.gn)
            self.blocks.append(self.activate)

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
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


if __name__ == "__main__":
    model = FPN()
    model.eval()
    print(model)

    model = torch.jit.script(model)

    inputs = [
        torch.randn(1, 256, 200, 200),
        torch.randn(1, 512, 100, 100),
        torch.randn(1, 1024, 50, 50),
        torch.randn(1, 2048, 25, 25),
    ]
    with torch.no_grad():
        output = model(inputs)

    for i in range(len(output)):
        print(output[i].size())

    pdb.set_trace()
