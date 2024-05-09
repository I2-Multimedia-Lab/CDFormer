import torch
import torch.nn as nn
import model.common as common


class Encoder_lr(nn.Module):
    def __init__(self,feats = 64, scale=4):
        super(Encoder_lr, self).__init__()
        self.scale=scale
        self.E = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )
    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1
    

class Encoder_gt(nn.Module):
    def __init__(self,feats = 64,scale=4):
        super(Encoder_gt, self).__init__()
        if scale == 2:
            in_dim = 15
        elif scale == 3:
            in_dim = 30
        elif scale == 4:
            in_dim = 51
        else :
            print('Upscale error!!!!')

        self.D = nn.Sequential(
            nn.Conv2d(in_dim, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )


        self.C = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=7, stride=7, padding=0),
            nn.LeakyReLU(0.1, True),            
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),           
            nn.Conv2d(feats, feats*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(0.1, True),            
            nn.Conv2d(feats*2, feats*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(feats * 4 * 2, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
    def forward(self, x, gt):
        gt0 = self.pixel_unshuffle(gt) 
        x = torch.cat([x, gt0], dim=1)
        x1_ave = self.D(x).squeeze(-1).squeeze(-1)
        x2_ave = self.C(gt).squeeze(-1).squeeze(-1)
        fea = self.mlp(torch.cat([x1_ave, x2_ave], dim=1))
        return fea

class denoise(nn.Module):
    def __init__(self,feats = 64, timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        self.mlp=nn.Sequential(
            nn.Linear(feats*4*2+1, feats*4),
            nn.LeakyReLU(0.1, True),           
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        fea = self.mlp(torch.cat([x,t,c],dim=1))
        return fea 