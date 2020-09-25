import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ResNetUNet(nn.Module):
    def __init__(self, input_channel, output_channel_num):
        super(ResNetUNet,self).__init__()

        base_model = models.resnet18(pretrained=True)
        base_model.conv1 = torch.nn.Conv2d(input_channel,64,kernel_size=(7, 7),stride=(2, 2),padding=(3, 3),bias=False)

        self.base_layers = list(base_model.children())
        self.layer0 = nn.Sequential(
            *self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(
            *self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(input_channel, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, output_channel_num, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    


#====================================
# Generators
#====================================

class Pix2PixUNetGenerator( nn.Module ):
    """
    UNet 構造での生成器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_out_channels = 3,
        n_fmaps = 64,
        dropout = 0.5   # 生成器 G に入力する入力ノイズ z は、直接 dropout を施すという意味でのノイズとして実現する。
    ):
        super( Pix2PixUNetGenerator, self ).__init__()

        def conv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True ),

                nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.Dropout( dropout )
            )
            return model

        def dconv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.ConvTranspose2d( in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1 ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU( 0.2, inplace=True ),
                nn.Dropout( dropout )
            )
            return model

        # Encoder（ダウンサンプリング）
        self.conv1 = conv_block( n_in_channels, n_fmaps )
        self.pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv2 = conv_block( n_fmaps*1, n_fmaps*2 )
        self.pool2 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv3 = conv_block( n_fmaps*2, n_fmaps*4 )
        self.pool3 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv4 = conv_block( n_fmaps*4, n_fmaps*8 )
        self.pool4 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )

        #
        self.bridge=conv_block( n_fmaps*8, n_fmaps*16 )

        # Decoder（アップサンプリング）
        self.dconv1 = dconv_block( n_fmaps*16, n_fmaps*8 )
        self.up1 = conv_block( n_fmaps*16, n_fmaps*8 )
        self.dconv2 = dconv_block( n_fmaps*8, n_fmaps*4 )
        self.up2 = conv_block( n_fmaps*8, n_fmaps*4 )
        self.dconv3 = dconv_block( n_fmaps*4, n_fmaps*2 )
        self.up3 = conv_block( n_fmaps*4, n_fmaps*2 )
        self.dconv4 = dconv_block( n_fmaps*2, n_fmaps*1 )
        self.up4 = conv_block( n_fmaps*2, n_fmaps*1 )

        # 出力層
        self.out_layer = nn.Sequential(
		    nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),
		    nn.Tanh(),
		)
        return

    def forward( self, input ):
        # Encoder（ダウンサンプリング）
        conv1 = self.conv1( input )
        pool1 = self.pool1( conv1 )
        conv2 = self.conv2( pool1 )
        pool2 = self.pool2( conv2 )
        conv3 = self.conv3( pool2 )
        pool3 = self.pool3( conv3 )
        conv4 = self.conv4( pool3 )
        pool4 = self.pool4( conv4 )

        #
        bridge = self.bridge( pool4 )

        # Decoder（アップサンプリング）& skip connection
        dconv1 = self.dconv1(bridge)

        concat1 = torch.cat( [dconv1,conv4], dim=1 )
        up1 = self.up1(concat1)

        dconv2 = self.dconv2(up1)
        concat2 = torch.cat( [dconv2,conv3], dim=1 )

        up2 = self.up2(concat2)
        dconv3 = self.dconv3(up2)
        concat3 = torch.cat( [dconv3,conv2], dim=1 )

        up3 = self.up3(concat3)
        dconv4 = self.dconv4(up3)
        concat4 = torch.cat( [dconv4,conv1], dim=1 )

        up4 = self.up4(concat4)

        # 出力層
        output = self.out_layer( up4 )

        return output

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

#====================================
# Discriminators
#====================================
class Pix2PixPatchGANDiscriminator( nn.Module ):
    """
    PatchGAN の識別器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_fmaps = 32
    ):
        super( Pix2PixPatchGANDiscriminator, self ).__init__()

        # 識別器のネットワークでは、Patch GAN を採用するが、
        # patchを切り出したり、ストライドするような処理は、直接的には行わない
        # その代りに、これを畳み込みで表現する。
        # つまり、CNNを畳み込んで得られる特徴マップのある1pixelは、入力画像のある領域(Receptive field)の影響を受けた値になるが、
        # 裏を返せば、ある1pixelに影響を与えられるのは、入力画像のある領域だけ。
        # そのため、「最終出力をあるサイズをもった特徴マップにして、各pixelにて真偽判定をする」ことと 、「入力画像をpatchにして、各patchの出力で真偽判定をする」ということが等価になるためである。
        def discriminator_block1( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        def discriminator_block2( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.InstanceNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        self.layer1 = discriminator_block1( n_in_channels * 2, n_fmaps )
        self.layer2 = discriminator_block2( n_fmaps, n_fmaps*2 )
        self.layer3 = discriminator_block2( n_fmaps*2, n_fmaps*4 )
        self.layer4 = discriminator_block2( n_fmaps*4, n_fmaps*8 )

        self.output_layer = nn.Sequential(
            nn.ZeroPad2d( (1, 0, 1, 0) ),
            nn.Conv2d( n_fmaps*8, 1, 4, padding=1, bias=False ),
            nn.Sigmoid()
        )


    def forward(self, x, y ):
        output = torch.cat( [x, y], dim=1 )
        output = self.layer1( output )
        output = self.layer2( output )
        output = self.layer3( output )
        output = self.layer4( output )
        output = self.output_layer( output )
        output = output.view(-1)
        return output

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    