import torch
import torch.nn as nn
from torch.nn import functional as F

# from thop import clever_format
# from thop import profile

class FoldConv_aspp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=0):
        super(FoldConv_aspp, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.BatchNorm2d(out_channel),
             nn.ReLU(inplace=True))
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim,kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d( down_dim), nn.ReLU(inplace=True)
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(fold_C, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU(inplace=True)  #如果batch=1 ，进行batchnorm会有问题
        # )

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, fold_C, kernel_size=1), nn.BatchNorm2d(fold_C), nn.ReLU(inplace=True)
        )

        # self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)

        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        in_feature = self.down_conv(in_feature)
        in_feature = self.unfold(in_feature)
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size, W // self.win_size)
        in_feature1 = self.conv1(in_feature)
        in_feature2 = self.conv2(in_feature)
        in_feature3 = self.conv3(in_feature)
        in_feature4 = self.conv4(in_feature)
        # in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(in_feature, 1)), size=in_feature.size()[2:], mode='bilinear')
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)


        in_feature = F.fold(input=in_feature, output_size=[H, W], kernel_size=2, dilation=1, padding=0, stride=2)
        in_feature = self.up_conv(in_feature)
        return in_feature


class ASPP(nn.Module): # deeplab

    def __init__(self, dim,in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.ReLU(inplace=True))
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
         )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim),nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4,conv5), 1))

class ASPP_crop(nn.Module): # deeplab

    def __init__(self, dim,in_dim):
        super(ASPP_crop, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.ReLU(inplace=True))
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
         )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU(inplace=True)
        # )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim),nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        # conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4), 1))

class FoldConv_denseaspp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=0):
        super(FoldConv_denseaspp, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.BatchNorm2d(out_channel),
             nn.ReLU(inplace=True))
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim,kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_C+down_dim, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.BatchNorm2d(down_dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_C+down_dim*2, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_C+down_dim*3, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d( down_dim), nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fold_C+down_dim*4, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.ReLU(inplace=True)  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, fold_C, kernel_size=1), nn.BatchNorm2d(fold_C), nn.ReLU(inplace=True)
        )

        # self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)

        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        in_feature = self.down_conv(in_feature)
        in_feature = self.unfold(in_feature)
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size, W // self.win_size)
        in_feature1 = self.conv1(in_feature)
        feature = torch.cat((in_feature1, in_feature), dim=1)
        in_feature2 = self.conv2(feature)
        feature = torch.cat((in_feature2, feature), dim=1)
        in_feature3 = self.conv3(feature)
        feature = torch.cat((in_feature3, feature), dim=1)
        in_feature4 = self.conv4(feature)
        feature = torch.cat((in_feature4, feature), dim=1)
        in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(feature, 1)), size=in_feature.size()[2:], mode='bilinear')
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4,in_feature5), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)
        in_feature = F.fold(input=in_feature, output_size=H, kernel_size=2, dilation=1, padding=0, stride=2)
        in_feature = self.up_conv(in_feature)

        return in_feature


class FoldConv_aspp_gn(nn.Module):
    def __init__(self, in_channel, out_channel, out_size,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=0,gn_groups=8):
        super(FoldConv_aspp_gn, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.GroupNorm(num_groups=gn_groups, num_channels=out_channel),nn.ReLU(inplace=True))
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim,kernel_size=1),nn.GroupNorm(num_groups=gn_groups, num_channels=down_dim),nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.GroupNorm(num_groups=gn_groups, num_channels=down_dim), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=4, padding=4),nn.GroupNorm(num_groups=gn_groups, num_channels=down_dim),nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=6, padding=6), nn.GroupNorm(num_groups=gn_groups, num_channels=down_dim),nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=1),nn.GroupNorm(num_groups=gn_groups, num_channels=down_dim),nn.ReLU(inplace=True) #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, fold_C, kernel_size=1), nn.GroupNorm(num_groups=gn_groups, num_channels=fold_C),nn.ReLU(inplace=True)
        )

        self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)
        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        in_feature = self.down_conv(in_feature)
        in_feature = self.unfold(in_feature)
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size, W // self.win_size)
        in_feature1 = self.conv1(in_feature)
        in_feature2 = self.conv2(in_feature)
        in_feature3 = self.conv3(in_feature)
        in_feature4 = self.conv4(in_feature)
        in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(in_feature, 1)), size=in_feature.size()[2:], mode='bilinear')
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4,in_feature5), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)
        in_feature = self.fold(in_feature)
        in_feature = self.up_conv(in_feature)
        return in_feature

if __name__ == "__main__":
    model = FoldConv_aspp(in_channel=2048,
                      out_channel=64,
                      out_size=384 // 16,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      win_size=2,
                      win_padding=0,
        )
    # depth = torch.randn(1, 3, 256, 256)
    input = torch.randn(4, 2048, 12, 12)
    # flops, params = profile(model,inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops,params)