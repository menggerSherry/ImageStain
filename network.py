import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import Parameter
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.vgg import VGG
# from tensorboardX import SummaryWriter

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, 3, 1, 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))



class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super(VGGNet, self).__init__(make_layers(cfg))
        self.ranges = ranges

        if pretrained:
            # exec('self.load_state_dict(models.%s(pretrained=True).state_dict())'%model)
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth',
                                                  progress=True)
            self.load_state_dict(state_dict)
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False
        if remove_fc:
            del self.classifier
        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())
        self.new_classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4)
        )

    def forward(self, x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.new_classifier(x)
        return x

def transform_block(in_d,ngf,dropout=0.0):
    model=[

        nn.Conv2d(in_d, ngf, 1, 1),
        nn.BatchNorm2d(ngf),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        model.append(nn.Dropout(dropout))
    return nn.Sequential(*model)

class SeparableConv2d(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        '''
        '''
        super(XceptionBlock, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.InstanceNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(nn.ReflectionPad2d(1))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, bias=False))
            rep.append(nn.InstanceNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(nn.ReflectionPad2d(1))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, bias=False))
            rep.append(nn.InstanceNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(nn.ReflectionPad2d(1))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,  bias=False))
            rep.append(nn.InstanceNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class SPGenerator(nn.Module):
    def __init__(self,backbone_name,num_channels,ngf,pretrained=True):
        super(SPGenerator, self).__init__()
        # self.alpha=torch.tensor(1.,requires_grad=True)
        self.alpha=Parameter(torch.Tensor(1))
        self.backbone_list=nn.ModuleList()
        for i in range(4):
            self.backbone_list.append(
                IntermediateLayerGetter(resnet.__dict__[backbone_name](
                    pretrained=pretrained,
                    replace_stride_with_dilation=[True,False,True],
                ),return_layers={'layer2':'mid_layer','layer4':'out_layer'})
            )
        self.mid_block=nn.ModuleList()
        self.parallel_block=nn.ModuleList()
        self.up_list=nn.ModuleList()
        self.conv_list=nn.ModuleList()
        # self.interm_block=nn.ModuleList()
        self.mid_block.append(XceptionBlock(ngf , ngf, 2, start_with_relu=False))
        for i in range(1,4):
            self.mid_block.append(XceptionBlock(ngf,ngf,2,start_with_relu=False))
        for i in range(4):
            self.parallel_block.append(transform_block(2048,ngf,dropout=0.5))
        for i in range(4):
            self.up_list.append(nn.Upsample(scale_factor=2))
        for i in range(3):
            self.conv_list.append(transform_block(ngf*3,ngf))

        self.upsample_block=nn.Sequential(
            nn.Upsample(scale_factor=2),
            # nn.Conv2d(2*ngf,ngf,3,1,1),
            nn.ReflectionPad2d(1),
            SeparableConv2d(2*ngf,ngf,3,1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            SeparableConv2d(ngf,ngf,3,1),
            # nn.Conv2d(ngf,ngf,3,1,1),
            nn.BatchNorm2d(ngf),
        )
        self.skip_connection=nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(2*ngf,ngf,1,1),
        )
        self.to_rgb=nn.Sequential(
            nn.Conv2d(ngf,num_channels,1,1),
            nn.Tanh(),

        )
        # for i in range(4):
        #     self.interm_block.append(transform_block(1024,ngf))

        self._init_weight()
    def forward(self,input_list):
        # batch 512 8 8
        # get the backbone
        backbone_out=self.backbone_list[0](input_list[0])
        mid_layer=self.up_list[0](self.parallel_block[0](backbone_out['out_layer']))
        mid_layer=self.mid_block[0](mid_layer)
        inter_layer=backbone_out['mid_layer']
        for i in range(1,4):

            backbone_out=self.backbone_list[i](input_list[i])
            # print(backbone_out['out_layer'].size())
            # print(mid_layer.size())
            # print(inter_layer.size())

            mid_input=self.conv_list[i-1](torch.cat((mid_layer,inter_layer,self.parallel_block[i](backbone_out['out_layer'])),dim=1))
            # mid_input=torch.cat((mid_input,torch.randn((mid_input.shape[0],1,mid_input.shape[2],mid_input.shape[3]),device=mid_input.device)),dim=1)
            mid_input=self.up_list[i](mid_input)
            mid_layer=self.mid_block[i](mid_input)
            inter_layer=backbone_out['mid_layer']
        paramid_output=torch.cat((inter_layer,mid_layer),dim=1)
        upsample_output=self.upsample_block(paramid_output)
        skip_output=self.skip_connection(paramid_output)
        out=self.to_rgb(upsample_output+skip_output)

        return out


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.normal_(m.weight,1)
                nn.init.normal_(m.bias,0)
    def set_alpha(self,alpha):
        self.alpha=alpha
        print('soft growing')



class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.BatchNorm2d
        else:
            use_bias = norm_layer == nn.BatchNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Sigmoid()
        ]


        self.net = nn.Sequential(*self.net)

    def forward(self, input,label):
        """Standard forward."""
        return self.net(torch.cat((input,label),dim=1))

def dis_block_model(ind,ndf,norm_layer):
    model=[
        nn.Conv2d(ind,ndf,3,2,1,bias=False),
        ]
    if norm_layer:
        model.append(norm_layer(ndf))

    model.append(nn.LeakyReLU(0.2,inplace=True))

    return nn.Sequential(*model)

def block_model(ind,ndf,norm_layer):
    model=[
        nn.Conv2d(ind,ndf,3,1,1,bias=False)
    ]
    if norm_layer is not None:
        model.append(norm_layer(ndf))
    model.append(nn.LeakyReLU(0.2,inplace=True))
    return nn.Sequential(*model)

class PatchDiscriminator(nn.Module):
    def __init__(self,input_nc,ndf=128,norm_layer=nn.BatchNorm2d):
        super(PatchDiscriminator, self).__init__()
        self.dis_block1=dis_block_model(input_nc,ndf,norm_layer)
        self.dis_block2=dis_block_model(ndf,2*ndf,norm_layer)
        self.dis_block3=dis_block_model(2*ndf,2*ndf,norm_layer)
        self.block1=block_model(2*ndf,2*ndf,norm_layer)
        self.block2=block_model(2*ndf,1,norm_layer)
    def forward(self,x,label):
        x=self.dis_block1(torch.cat((x,label),dim=1))
        x=self.dis_block2(x)
        x=self.dis_block3(x)
        x=self.block1(x)
        x=self.block2(x)
        return x

def dis_down(in_features,out_features,normalize=True):
    model=[nn.Conv2d(in_features,out_features,4,2,1)]
    if normalize:
        model.append(nn.BatchNorm2d(out_features))
    model.append(nn.LeakyReLU(0.2,inplace=True))

    return model
def dis_feature_up(in_features,out_features):
    model=[
        nn.Conv2d(in_features,out_features,1,1),
        # nn.BatchNorm2d(out_features),
        nn.LeakyReLU(0.2,inplace=True),
    ]

    return model

ranges=((0,2),(2,4),(4,6),(6,8),(8,10),(10,11))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.ngpu=ngpu
        self.ranges=ranges
        self.model=nn.Sequential(
            *dis_feature_up(6,64),#1024,512,256
            *dis_feature_up(64,128),#512,256,128
            *dis_feature_up(128,256),#256,128,64
            *dis_feature_up(256,256),#128,64,32
            *dis_feature_up(256, 256),
            # *dis_feature_up(512,512),
            # *dis_feature_up(512,512),
            nn.Conv2d(256,1,1,1),

        )

    def forward(self, x):
        out=[]
        for i in range(len(self.ranges)):
            for layer in range(self.ranges[i][0],self.ranges[i][1]):
                x=self.model[layer](x)
                out.append(x)
        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self,num_D=2):
        super(MultiScaleDiscriminator, self).__init__()
        self.num=num_D
        self.downsample=nn.AvgPool2d(3,stride=2,padding=[1,1])
        # self.netD1=Discriminator(1)
        # self.netD2=Discriminator(1)
        for i in range(self.num):
            setattr(self,'Scale_%d'%i,Discriminator())



    def forward(self,x):
        output=[]
        for i in range(self.num):
            output.append(getattr(self,'Scale_%d'%i)(x))
            x=self.downsample(x)
        # out1=self.netD1(x)
        # output.append(out1)
        # input=self.downsample(x)
        # out2=self.netD2(input)
        # output.append(out2)

        return output

class FeatureDiscriminator(nn.Module):
    def __init__(self,input_nc,ndf=128,norm_layer=nn.BatchNorm2d):
        super(FeatureDiscriminator, self).__init__()
        self.discriminator=IntermediateLayerGetter(
            PatchDiscriminator(input_nc,ndf,norm_layer),
            return_layers={'dis_block1':'feature1','dis_block2':'feature2','dis_block3':'feature3',
                           'block1':'feature4','block2':'feature5'}
        )

    def forward(self,x,label):
        out=self.discriminator(torch.cat((x,label),dim=1))
        return out




# class FeatureExtract(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super(FeatureExtract, self).__init__()


class XceptionBlockDown(nn.Module):
    def __init__(self,in_channels,out_channels,dropout=0.0):
        super(XceptionBlockDown, self).__init__()
        model=[]
        model.append(XceptionBlock(in_channels , out_channels, 2, start_with_relu=False))
        model.append(nn.AvgPool2d(4,2,1))
        if dropout:
            model.append(nn.Dropout(dropout))
        self.model=nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)


class GeneratorUP(nn.Module):

    def __init__(self,in_channels,out_channels,dropout=0.0):
        super(GeneratorUP, self).__init__()
        model=[]
        model.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        model.append(nn.Conv2d(in_channels,in_channels,3,1,1,groups=in_channels))
        model.append(nn.Conv2d(in_channels,out_channels,1,1,0,1,1))
        model.append(nn.InstanceNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        if dropout:
            model.append(nn.Dropout(dropout))
        self.model=nn.Sequential(*model)

    def forward(self,x,skip,trim_skip):

        x = self.model(x)
        return torch.cat((x,skip,trim_skip),dim=1)


class DownSkip(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownSkip, self).__init__()
        self.up=nn.AvgPool2d(3,2,1)
        self.skip=nn.Conv2d(in_channels,out_channels,1,1,0,bias=False)

    def forward(self,x):
        x=self.up(x)
        return self.skip(x)

class GeneratorSUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorSUNet, self).__init__()
        # self.ngpu = ngpu
        model1 = [
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 7,groups=in_channels),
            nn.Conv2d(in_channels,32,1,1,0,1,1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        self.start_layers = nn.Sequential(*model1)
        self.down1 = XceptionBlockDown(32, 64)  # (batch,64,128,128)
        self.down2 = XceptionBlockDown(64, 128)  # (batch,128,64,64)
        self.down3 = XceptionBlockDown(128, 256)  # (batch,256,32,32)
        self.down4 = XceptionBlockDown(256, 512, dropout=0.5)  # (batch,512,16,16)
        self.down5 = XceptionBlockDown(512, 512, dropout=0.5)  # (batch,512,8,8)
        self.down6 = XceptionBlockDown(512, 512, dropout=0.5)  # (batch,512,4,4)
        self.down7 = XceptionBlockDown(512, 512, dropout=0.5)  # (batch,512,2,2)
        self.down8 = XceptionBlockDown(512, 512, dropout=0.5)  # (batch,512,1,1)

        self.up1 = GeneratorUP(512, 512, dropout=0.5)  # (batch,1024,2,2)
        self.up2 = GeneratorUP(1024, 512, dropout=0.5)  # (batch,1024,4,4)
        self.up3 = GeneratorUP(1536, 512, dropout=0.5)  # (batch,1024,8,8)
        self.up4 = GeneratorUP(1536, 512, dropout=0.5)  # (batch,1024,16,16)
        self.up5 = GeneratorUP(1536, 256)  # (batch,512,32,32)
        self.up6 = GeneratorUP(768, 128)  # (batch,256,64,64)
        self.up7 = GeneratorUP(384, 64)  # (batch,128,128,128)
        self.up8 = NetUp(192, 32)

        self.skip1 = DownSkip(512,512)
        self.skip2 = DownSkip(512,512)
        self.skip3 = DownSkip(512,512)
        # self.skip4 = UpSkip(512,512)
        self.skip4 = DownSkip(512,256)
        self.skip5 = DownSkip(256,128)
        self.skip6 = DownSkip(128,64)
        self.skip7 = DownSkip(64,32)

        self.finalup = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True),  

            nn.Conv2d(128, 32, 3, 1, 1),  
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )
        model2 = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(96, 96,3,1,groups=96),
            nn.Conv2d(96, 3,1,1,0),

            # nn.Conv2d(96, in_channels, 7),
            nn.Tanh(),
        ]
        self.end_layer = nn.Sequential(*model2)

    def forward(self, x):
        # down
        start = self.start_layers(x)
        d1 = self.down1(start)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        #skip
        s1 = self.skip1(start)
        s2 = self.skip2(d1)
        s3 = self.skip3(d2)
        s4 = self.skip4(d3)
        # print(d3.size())
        s5 = self.skip5(d4)
        s6 = self.skip6(d5)
        s7 = self.skip7(d6)

        # up
        # 2
        u1 = self.up1(d8, d7, s7)
        # 2
        # print(u1.size())
        # print(d6.size())
        # print(s1.size())
        u2 = self.up2(u1, d6, s6)
        u3 = self.up3(u2, d5, s5)
        u4 = self.up4(u3, d4, s4)
        u5 = self.up5(u4, d3, s3)
        u6 = self.up6(u5, d2, s2)
        u7 = self.up7(u6, d1, s1)
        u8 = self.up8(u7, start)
        return self.end_layer(u8)


class GeneratorUNet(nn.Module):
    def __init__(self, ngpu, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.ngpu = ngpu
        model1 = [
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, 32, 7),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        self.start_layers = nn.Sequential(*model1)
        self.down1 = NetDown(32, 64, normalize=False)  # (batch,64,128,128)
        self.down2 = NetDown(64, 128)  # (batch,128,64,64)
        self.down3 = NetDown(128, 256)  # (batch,256,32,32)
        self.down4 = NetDown(256, 512, dropout=0.5)  # (batch,512,16,16)
        self.down5 = NetDown(512, 512, dropout=0.5)  # (batch,512,8,8)
        self.down6 = NetDown(512, 512, dropout=0.5)  # (batch,512,4,4)
        self.down7 = NetDown(512, 512, dropout=0.5)  # (batch,512,2,2)
        self.down8 = NetDown(512, 512, normalize=False, dropout=0.5)  # (batch,512,1,1)

        self.up1 = NetUp(512, 512, dropout=0.5)  # (batch,1024,2,2)
        self.up2 = NetUp(1024, 512, dropout=0.5)  # (batch,1024,4,4)
        self.up3 = NetUp(1024, 512, dropout=0.5)  # (batch,1024,8,8)
        self.up4 = NetUp(1024, 512, dropout=0.5)  # (batch,1024,16,16)
        self.up5 = NetUp(1024, 256)  # (batch,512,32,32)
        self.up6 = NetUp(512, 128)  # (batch,256,64,64)
        self.up7 = NetUp(256, 64)  # (batch,128,128,128)

        self.finalup = nn.Sequential(
            nn.Upsample(scale_factor=2),  

            nn.Conv2d(128, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )
        model2 = [
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(32, in_channels, 7),
            nn.Tanh(),
        ]
        self.end_layer = nn.Sequential(*model2)

    def forward(self, x):
        # down
        start = self.start_layers(x)
        d1 = self.down1(start)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # up
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.finalup(u7)
        return self.end_layer(u8)

class NetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(NetUp, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SeparableConv2d(in_size,out_size,3,1,1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class NetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(NetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)




class UNet2(nn.Module):
    def __init__(self,channels):
        super(UNet2, self).__init__()
        # self.channels=channels
        self.down1=XceptionBlockDown(channels,channels)
        self.down2=XceptionBlockDown(channels,channels)

        self.up1=GeneratorUP(channels,channels)
        self.up2=NetUp(channels*3,channels)

        self.skip=DownSkip(channels,channels)
        self.model=nn.Sequential(
            nn.ReflectionPad2d(1),
            SeparableConv2d(channels*2,channels,3,1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        d1=self.down1(x)
        d2=self.down2(d1)

        s=self.skip(x)

        u1=self.up1(d2,d1,s)
        u2 = self.up2(u1,x)

        return self.model(u2)


class UNet4(nn.Module):
    def __init__(self,channels):
        super(UNet4, self).__init__()
        self.down1 = XceptionBlockDown(channels, channels)
        self.down2 = XceptionBlockDown(channels, channels)
        self.down3 = XceptionBlockDown(channels, channels)
        self.down4 = XceptionBlockDown(channels, channels,dropout=0.1)

        self.up1 = GeneratorUP(channels, channels,dropout=0.1)
        self.up2 = GeneratorUP(channels * 3, channels)
        self.up3 = GeneratorUP(channels * 3, channels)
        self.up4 = NetUp(channels * 3, channels)

        self.skip1 = DownSkip(channels, channels)
        self.skip2 = DownSkip(channels, channels)
        self.skip3 = DownSkip(channels, channels)

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            SeparableConv2d(channels * 2, channels, 3, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        s1 = self.skip1(x)
        s2 = self.skip2(d1)
        s3 = self.skip3(d2)

        u1 = self.up1(d4,d3,s3)
        u2 = self.up2(u1,d2,s2)
        u3 = self.up3(u2,d1,s1)
        u4 = self.up4(u3,x)
        return self.model(u4)

class UNetS(nn.Module):
    def __init__(self,channels):
        super(UNetS, self).__init__()
        self.unet=UNet6(channels)
        model1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 128, 7),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, 512, 7),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        ]
        self.start_layers = nn.Sequential(*model1)
        model2 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(512, 128, 7),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, 3, 7),
            nn.Tanh()
        ]
        self.out_layers = nn.Sequential(*model2)
    def forward(self,x):
        inx=self.start_layers(x)
        inx=self.unet(inx)
        return self.out_layers(inx)


class UNet6(nn.Module):
    def __init__(self,channels):
        super(UNet6, self).__init__()
        self.down1 = XceptionBlockDown(channels, channels)
        self.down2 = XceptionBlockDown(channels, channels)
        self.down3 = XceptionBlockDown(channels, channels)
        self.down4 = XceptionBlockDown(channels, channels)
        self.down5 = XceptionBlockDown(channels, channels,dropout=0.4)
        self.down6 = XceptionBlockDown(channels, channels,dropout=0.4)

        self.up1 = GeneratorUP(channels, channels,dropout=0.4)
        self.up2 = GeneratorUP(channels * 3, channels,dropout=0.4)
        self.up3 = GeneratorUP(channels * 3, channels)
        self.up4 = GeneratorUP(channels * 3, channels)
        self.up5 = GeneratorUP(channels * 3, channels)
        self.up6 = NetUp(channels * 3, channels)

        self.skip1 = DownSkip(channels, channels)
        self.skip2 = DownSkip(channels, channels)
        self.skip3 = DownSkip(channels, channels)
        self.skip4 = DownSkip(channels, channels)
        self.skip5 = DownSkip(channels, channels)

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            SeparableConv2d(channels * 2, channels, 3, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        s1 = self.skip1(x)
        s2 = self.skip2(d1)
        s3 = self.skip3(d2)
        s4 = self.skip4(d3)
        s5 = self.skip5(d4)

        u1 = self.up1(d6,d5,s5)
        u2 = self.up2(u1,d4,s4)
        u3 = self.up3(u2,d3,s3)
        u4 = self.up4(u3,d2,s2)
        u5 = self.up5(u4, d1, s1)
        u6 = self.up6(u5, x)
        return self.model(u6)


class FromRGB(nn.Module):
    def __init__(self,channel,ngf,out_channel):
        super(FromRGB, self).__init__()
        self.model1=nn.Sequential(
            nn.ReflectionPad2d(1),
            SeparableConv2d(channel,ngf,3,1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            SeparableConv2d(ngf,out_channel,3,1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.model2=nn.Sequential(
            nn.ReflectionPad2d(1),
            SeparableConv2d(channel,out_channel,3,1),
            nn.ReLU(inplace=True),

        )

        self.model3=nn.Sequential(
            nn.Conv2d(channel,out_channel,1,1,0),
            nn.InstanceNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.project=nn.Sequential(
            SeparableConv2d(3*out_channel,out_channel,1,bias=False),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self,x):
        out1=self.model1(x)
        out2=self.model2(x)
        out3=self.model3(x)
        final=self.project(torch.cat((out1,out2,out3),dim=1))
        return final


class BackBoneUP(nn.Module):
    def __init__(self,in_channel,out_channel,dropout=0.0):
        super(BackBoneUP, self).__init__()
        model=[
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SeparableConv2d(in_channel,out_channel,3,1,1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True),
            #
        ]
        if dropout:
            model.append(nn.Dropout(dropout))
        self.model=nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)


class TRGenerator(nn.Module):

    def __init__(self,ngf,channel):
        super(TRGenerator, self).__init__()
        unet_model=[]
        unet_model.append(UNet2(channel))
        unet_model.append(UNet2(channel))
        unet_model.append(UNet4(channel))
        unet_model.append(UNet4(channel))
        unet_model.append(UNet6(channel))
        unet_model.append(UNet6(channel))
        from_rgb_model=[]
        for i in range(6):
            from_rgb_model.append(FromRGB(3,ngf,channel))
        up_model=[]
        up_model.append(BackBoneUP(channel,channel,dropout=0.4))
        for i in range(4):
            up_model.append(BackBoneUP(2*channel,channel))
        self.unets=nn.ModuleList(unet_model)
        self.rgb_model=nn.ModuleList(from_rgb_model)
        self.ups=nn.ModuleList(up_model)
        # self.skip=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),SeparableConv2d(channel*2,channel*2,3,1,1))
        self.projection=\
            nn.Sequential(
            XceptionBlock(2*channel , ngf, 2, start_with_relu=False),
            # XceptionBlock(channel, ngf ,2 ,start_with_relu=False),
            nn.ReflectionPad2d(1),
            SeparableConv2d(ngf,3,3,1),
            # Conv********************
            nn.Tanh()
        )
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1])
        self._init_weight()

    def forward(self,inx):
        unet_output=[]
        # get the UNet output
        for inp in range(6):
            # x=inx[i]
            stride = 2 ** (5 - inp)

            x = inx[...,::stride,::stride]
            unet_output.append(self.unets[inp](self.rgb_model[inp](x)))
        u1=torch.cat((self.ups[0](unet_output[0]),unet_output[1]),dim=1)
        u2 = torch.cat((self.ups[1](u1),unet_output[2]),dim=1)
        u3 = torch.cat((self.ups[2](u2),unet_output[3]),dim=1)
        u4=torch.cat((self.ups[3](u3),unet_output[4]),dim=1)
        u5 = torch.cat((self.ups[4](u4), unet_output[5]), dim=1)
        # print(u5.size())
        # input = torch.cat((u5,unet_output[6]),dim=1)
        out=self.projection(u5)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.normal_(m.weight,1)
                nn.init.normal_(m.bias,0)


class ToRGB(nn.Module):
    def __init__(self,n_feature,ngf,channel):
        super(ToRGB, self).__init__()
        self.model1=nn.Sequential(
            nn.ReflectionPad2d(1),
            SeparableConv2d(n_feature,channel,3,1),
            nn.Tanh(),
        )
        # self.model2=nn.Sequential(
        #     XceptionBlock(n_feature,ngf,2,start_with_relu=False),
        #     SeparableConv2d(ngf,channel,3,1,1),
        #     nn.Tanh(),
        # )
        self.model3=nn.Sequential(
            XceptionBlock(n_feature,ngf,2,start_with_relu=False),
            SeparableConv2d(ngf,ngf//2,3,1,1),
            nn.InstanceNorm2d(ngf//2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            SeparableConv2d(ngf//2,channel,3,1),
            nn.Tanh()
        )
        self.project=nn.Sequential(

        )

    def forward(self,x):
        out1=self.model1(x)
        # out2=self.model2(x)
        out3=self.model3(x)
        return out1+out3


class TrapTRGenerator(TRGenerator):
    def __init__(self,ngf,channel,trap_idx):
        super(TrapTRGenerator, self).__init__(ngf,channel)
        self.trap_idx=trap_idx

    def forward(self,inx):
        unet_output = []
        for inp in range(6):
            # x=inx[i]
            stride = 2 ** (5 - inp)
            x = inx[...,::stride,::stride]
            if inp != self.trap_idx:
                x=torch.zeros_like(x)
          #  print('11')
            unet_output.append(self.unets[inp](self.rgb_model[inp](x)))
        u1 = torch.cat((self.ups[0](unet_output[0]), unet_output[1]), dim=1)
        u2 = torch.cat((self.ups[1](u1), unet_output[2]), dim=1)
        u3 = torch.cat((self.ups[2](u2), unet_output[3]), dim=1)
        u4 = torch.cat((self.ups[3](u3), unet_output[4]), dim=1)
        u5 = torch.cat((self.ups[4](u4), unet_output[5]), dim=1)
        # print(u5.size())
        # input = torch.cat((u5,unet_output[6]),dim=1)
        out = self.projection(u5)
        return out














