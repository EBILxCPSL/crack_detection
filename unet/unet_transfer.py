from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from copy import deepcopy
# from torchvision.models.vgg import VGG16_Weights

input_size = (448, 448)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        # self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

# Efficient's activation is LeakyReLU
class ConvRelu_efficient(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class skipConnection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skipLayer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.skipLayer(x)

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            # Decoder
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
    
class DecoderBlockV2_efficient(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlockV2_efficient, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            ConvRelu_efficient(in_channels, out_channels),
            ConvRelu_efficient(out_channels, out_channels),
        )

    def forward(self, x1, x2):
        # print(x2.shape)
        # print(x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode='reflect')

        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.block(x)

# VGG16
class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        # self.encoder = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0], # c
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
            # x_out = F.sigmoid(x_out)

        return x_out

# ResNet
class UNetResNet(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))

# Efficient
class EfficientUNet(nn.Module):
    def __init__(self):
        super(EfficientUNet, self).__init__()
        # from github 'NVIDIA/DeepLearningExamples:torchhub' import 'nvidia_efficientnet_b4'
        self.efficientModel = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        self.efficient_layer = []  # will get 8 layers
        self.get_efficient_layer()
        self.inputs_0_0 = self.efficient_layer[0]  # out channels 48
        self.down1_1_0 = self.efficient_layer[1]  # out channels 24
        self.down2_2_0 = self.efficient_layer[2]  # out channels 32
        self.down3_3_0 = self.efficient_layer[3]  # out channels 56
        self.down4_4_0 = self.efficient_layer[4]  # out channels 112
        self.down5_5_0 = self.efficient_layer[5]  # out channels 160
        self.down6_6_0 = self.efficient_layer[6]  # out channels 272

        self.down7_7_0 = self.efficient_layer[7]  # out channels 448

        # 448 concate 272
        self.up1_6_1 = DecoderBlockV2_efficient(448+272, 272)
        self.up2_5_2 = DecoderBlockV2_efficient(272+160, 160)
        self.up3_4_3 = DecoderBlockV2_efficient(160+112, 112)
        self.up4_3_4 = DecoderBlockV2_efficient(112+56, 56)
        self.up5_2_5 = DecoderBlockV2_efficient(56+32, 32)
        self.up6_1_6 = DecoderBlockV2_efficient(32+24, 24)
        self.up7_0_7 = DecoderBlockV2_efficient(24+48, 48)
        
        self.outputs = nn.Conv2d(48, 1, kernel_size=1)

    def get_efficient_layer(self):
        for index, module in enumerate(self.efficientModel.named_children()):
            # conv3x3 layer
            if index == 0:
                self.efficient_layer.append(deepcopy(module[1]))
            # all MBConv layer
            if index == 1:
                for _, mm in enumerate(module[1].named_children()):
                    self.efficient_layer.append(deepcopy(mm[1]))

    def forward(self, x):
        x1 = self.inputs_0_0(x)
        x2 = self.down1_1_0(x1)
        x3 = self.down2_2_0(x2)
        x4 = self.down3_3_0(x3)
        x5 = self.down4_4_0(x4)
        x6 = self.down5_5_0(x5)
        x7 = self.down6_6_0(x6)
        x8 = self.down7_7_0(x7)

        x = self.up1_6_1(x8, x7)
        x = self.up2_5_2(x, x6)
        x = self.up3_4_3(x, x5)
        x = self.up4_3_4(x, x4)
        x = self.up5_2_5(x, x3)
        x = self.up6_1_6(x, x2)
        x = self.up7_0_7(x, x1)
        logits = self.outputs(x)
        return logits
