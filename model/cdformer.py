import torch
import torch.nn as nn
from model.diffusion import diffusion
from model.transformer import CDFormer_SR
from model.encoder import Encoder_lr, Encoder_gt, denoise

def make_model(args):
    return CDFormer(args)

class CDFormer(nn.Module):
    def __init__(self,  args):
        super(CDFormer, self).__init__()
        timesteps = 4 # default is 4
        self.SR = CDFormer_SR(upscale=int(args.scale[0])).cuda()
        self.encoder = Encoder_gt(feats=64, scale=int(args.scale[0])).cuda()
        self.condition = Encoder_lr(feats=64, scale=int(args.scale[0])).cuda()
        self.denoise = denoise(feats=64, timesteps=4).cuda()
        self.netG = diffusion.DDPM(denoise=self.denoise, 
        condition=self.condition ,feats=64, timesteps = timesteps
        ).cuda()
        self.img_range = 1.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1).cuda()
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
    def forward(self, x):
        if self.training:
            #  normalization (optinal). It makes little difference to the results.
            img = x[0]/255.
            gt = x[1]/255.
            img = (img - self.mean) * self.img_range
            gt = (gt - self.mean) * self.img_range
            diffusion = x[2]
            if diffusion == False:      
                # encode CDP
                cdp = self.encoder(img,gt)
                cdp_diff = cdp
                sr = self.SR(img, cdp)
            else:
                # freezen CDP
                self.freeze_module(self.encoder) 
                # encode CDP
                cdp = self.encoder(img,gt)
                # generate CDP
                cdp_diff = self.netG(img, cdp)
                sr = self.SR(img, cdp_diff)
            # Anti-normalization
            sr = sr / self.img_range + self.mean
            sr = sr*255
            return cdp_diff, cdp, sr
        else:
            #  normalization (optinal). It makes little difference to the results.
            img = x[0]/255.
            gt = x[1]/255.
            img = (img - self.mean) * self.img_range
            gt = (gt - self.mean) * self.img_range
            diffusion = x[2]
            if diffusion == False:      
                # encode CDP
                cdp = self.encoder(img,gt)
                sr = self.SR(img, cdp)
            else:
                # generate CDP
                cdp_diff = self.netG(img)    
                sr = self.SR(img, cdp_diff) 
            # Anti-normalization
            sr = sr / self.img_range + self.mean   
            sr = sr*255
            return sr
