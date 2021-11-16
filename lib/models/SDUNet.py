# coding:utf-8
import torch
import torch.nn as nn
# from tools import *
from .dense_layers import *
import pdb


##################Denseblock Unet#########################
class SDUNet(nn.Module):
    def __init__(self, in_channels, down_blocks, up_blocks, bottleneck_layers, growth_rate, out_chans_first_conv,
                 n_classes):
        super(SDUNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        # down_blocks=(4, 4, 4, 4)
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            if i == 0 or i == 2:
                self.transUpBlocks.append(
                    TransitionUp_DULR(prev_block_channels, prev_block_channels, skip_connection_channel_counts[i]))
            else:
                self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.firstconv(x)
        # pdb.set_trace()

        skip_connections = []
        # down sampling process
        for i in range(len(self.down_blocks)):  # down_blocks is the number of levels
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)  # down sampling feature map

        out = self.bottleneck(out)
        # up sampling and skip connection process
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)  # up sampling feature map
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        # out = self.softmax(out)
        return out


def defineSDUNet(n_classes):
    return SDUNet(in_channels=3, down_blocks=(4, 4, 4, 4),
                  up_blocks=(4, 4, 4, 4),
                  bottleneck_layers=4,
                  growth_rate=12,
                  out_chans_first_conv=48, n_classes=n_classes)


if __name__ == "__main__":

    model = defineSDUNet(n_classes=1)
    input = torch.rand(1, 3, 512, 512)
    output = model(input)