# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from model.backbone.resnet import ResNetBase

from utils import config


_config = config.Config()

M = _config.STRUCTURE.m


class AliveUNetBase(ResNetBase):
    BLOCK = None  # default
    PLANES = None  # default

    BLOCK = BasicBlock
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (32, 64, 96, 128, 160, 192, 224, 224, 192, 160, 128, 96, 64, 32)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.conv5p16s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block5= self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])

        self.conv6p32s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn6 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])

        self.conv7p64s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn7 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[8] + self.PLANES[6] * self.BLOCK.expansion
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[8],
                                       self.LAYERS[8])
        self.convtr8 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[8], kernel_size=2, stride=2, dimension=D)
        self.bntr8 = ME.MinkowskiBatchNorm(self.PLANES[8])

        self.inplanes = self.PLANES[9] + self.PLANES[5] * self.BLOCK.expansion
        self.block9 = self._make_layer(self.BLOCK, self.PLANES[9],
                                       self.LAYERS[9])
        self.convtr9 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[9], kernel_size=2, stride=2, dimension=D)
        self.bntr9 = ME.MinkowskiBatchNorm(self.PLANES[9])

        self.inplanes = self.PLANES[10] + self.PLANES[4] * self.BLOCK.expansion
        self.block10 = self._make_layer(self.BLOCK, self.PLANES[10],
                                       self.LAYERS[10])
        self.convtr10 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[10], kernel_size=2, stride=2, dimension=D)
        self.bntr10 = ME.MinkowskiBatchNorm(self.PLANES[10])

        self.inplanes = self.PLANES[11] + self.PLANES[3] * self.BLOCK.expansion
        self.block11 = self._make_layer(self.BLOCK, self.PLANES[11],
                                       self.LAYERS[11])
        self.convtr11 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[11], kernel_size=2, stride=2, dimension=D)
        self.bntr11 = ME.MinkowskiBatchNorm(self.PLANES[11])

        self.inplanes = self.PLANES[12] + self.PLANES[2] * self.BLOCK.expansion
        self.block12 = self._make_layer(self.BLOCK, self.PLANES[12],
                                       self.LAYERS[12])
        self.convtr12 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[12], kernel_size=2, stride=2, dimension=D)
        self.bntr12 = ME.MinkowskiBatchNorm(self.PLANES[12])

        self.inplanes = self.PLANES[13] + self.PLANES[1] * self.BLOCK.expansion
        self.block13 = self._make_layer(self.BLOCK, self.PLANES[13],
                                       self.LAYERS[13])
        self.convtr13 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[13], kernel_size=2, stride=2, dimension=D)
        self.bntr13 = ME.MinkowskiBatchNorm(self.PLANES[13])

        self.inplanes = self.PLANES[13] + self.INIT_DIM
        self.block14 = self._make_layer(self.BLOCK, self.PLANES[13],
                                       self.LAYERS[13])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[13] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_b4p16 = self.block4(out)

        out = self.conv5p16s2(out_b4p16)
        out = self.bn5(out)
        out = self.relu(out)
        out_b5p32 = self.block5(out)

        out = self.conv6p32s2(out_b5p32)
        out = self.bn6(out)
        out = self.relu(out)
        out_b6p64 = self.block6(out)

        out = self.conv7p64s2(out_b6p64)
        out = self.bn7(out)
        out = self.relu(out)
        out = self.block7(out)

        out = self.convtr7(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_b6p64)
        out = self.block8(out)

        out = self.convtr8(out)
        out = self.bntr8(out)
        out = self.relu(out)

        out = ME.cat(out, out_b5p32)
        out = self.block9(out)

        out = self.convtr9(out)
        out = self.bntr9(out)
        out = self.relu(out)

        out = ME.cat(out, out_b4p16)
        out = self.block10(out)

        out = self.convtr10(out)
        out = self.bntr10(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block11(out)

        out = self.convtr11(out)
        out = self.bntr11(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block12(out)

        out = self.convtr12(out)
        out = self.bntr12(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block13(out)

        out = self.convtr13(out)
        out = self.bntr13(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block14(out)

        return self.final(out)


class AliveUnet(AliveUNetBase):
    BLOCK = BasicBlock
    PLANES = tuple(i * M for i in (list(range(1, 8)) + list(range(7, 0, -1))))
    LAYERS = tuple(_config.STRUCTURE.block_reps for _ in range(len(PLANES)))

# class MinkUNet18(MinkUNetBase):
#     BLOCK = BasicBlock
#     LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
