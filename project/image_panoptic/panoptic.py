"""Create model."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 22日 星期三 23:13:05 CST
# ***
# ************************************************************************************/
#

import math
import os
import pdb  # For debug
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    model = FPN()
    print(model)

    # model = model.cuda()
    model.eval()

    input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(input)

    print(output)
    # print(output.size())

    # (Pdb) len(output) -- 4
    # (Pdb) output[0].size()
    # torch.Size([1, 256, 56, 56])
    # (Pdb) output[1].size()
    # torch.Size([1, 512, 28, 28])
    # (Pdb) output[2].size()
    # torch.Size([1, 1024, 14, 14])
    # (Pdb) output[3].size()
    # torch.Size([1, 2048, 7, 7])

    pdb.set_trace()
