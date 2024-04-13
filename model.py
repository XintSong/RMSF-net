import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(
            mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class RegressionHead(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, final_ch):
        super(RegressionHead, self).__init__()
        self.regress1 = nn.Sequential(nn.Conv3d(in_ch, mid_ch, kernel_size=1, padding=0, bias=True),
                                      nn.BatchNorm3d(mid_ch), nn.ReLU(inplace=True), nn.Dropout(0.2))

        self.regress2 = nn.Sequential(nn.Conv3d(mid_ch, out_ch, kernel_size=1, padding=0, bias=True),
                                      nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

        self.final = nn.Conv3d(mid_ch+out_ch, final_ch,
                               kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.regress1(x)
        x2 = self.regress2(x1)
        x = self.final(torch.cat([x1, x2], 1))
        return x


def normoaliza_box(box_list):

    max_list = torch.max(box_list.reshape(
        box_list.shape[0], box_list.shape[1], -1), dim=2)[0]
    box_list = torch.clip(box_list, min=0)
    box_list = torch.div(
        box_list, max_list.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    return box_list


class RMSF_net_model(nn.Module):

    def __init__(self, in_channels=1, out_channels_first_conv=32, n_classes=1, norm=True):
        super().__init__()

        n1 = out_channels_first_conv
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        # self.Up = nn.Upsample(scale_factor=2, align_corners=True)
        self.norm = norm

        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])

        self.conv0_1 = conv_block_nested(
            filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(
            filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(
            filters[2] + filters[3], filters[2], filters[2])

        self.conv0_2 = conv_block_nested(
            filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(
            filters[1]*2 + filters[2], filters[1], filters[1])

        self.conv0_3 = conv_block_nested(
            filters[0]*3 + filters[1], filters[0], filters[0])

        self.regression_head = RegressionHead(
            filters[0], filters[0], filters[0], n_classes)

    def forward(self, x, sse=None, input_size=40, out_size=10):

        if self.norm:
            x = normoaliza_box(x)

        if sse is not None:

            if sse.shape[-1] != input_size:
                csize = (input_size-sse.shape[-1])//2
                x = x[:, :, csize:-csize, csize:-csize, csize:-csize]
                if self.norm:
                    x = normoaliza_box(x)
            x = torch.cat([x, sse], 1)

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))  # down 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))  # up 1

        x2_0 = self.conv2_0(self.pool(x1_0))  # down 2
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))  # up 1
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))  # up 2

        x3_0 = self.conv3_0(self.pool(x2_0))  # down 3
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))  # up 1
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))  # up 2
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))  # up 3

        output = self.regression_head(x0_3)

        if out_size != input_size:
            csize = (input_size-out_size)//2
            output = output[:, :, csize:-csize, csize:-csize, csize:-csize]

        return output
