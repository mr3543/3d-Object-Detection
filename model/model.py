import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from .. import config
from config import cfg

class PPFeatureNet(nn.Module):
    def __init__(self,in_channels,out_channels,batch_norm,last_layer):
        super(PPFeatureNet,self).__init__()
        self.last_layer
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn(x)
        if self.last_layer:
            x = torch.max(x,dim=2)[0]
        return x

class PPScatter(nn.Module):
    def __init__(self):
        super(PPScatter,self).__init__()
    
    def forward(self,x,inds):
        out_sh = x.size()
        out = torch.zeros(out_sh[0],out_sh[1],cfg.FM_HEIGHT,cfg.FM_WIDTH)
        non_empty = np.where(inds[:,:,0] == 1)
        x_inds = inds[non_empty][:,1]
        y_inds = inds[non_empty][:,2]
        batch = non_empty[0]
        pillars = non_empty[1]
        out[batch,:,y_inds,x_inds] = x[batch,:,pillars]
        return out

class PPDownBlock(nn.Module):
    def __init__(self,num_layers,in_channels,out_channels):
        super(PPDownBlock,self).__init__()
        block = []
        block.append(nn.Conv2d(in_channels,out_channels,
                               kernel_size=3,stride=2,padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorn2d(out_channels))
        for i in range(num_layers - 1):
            block.append(nn.Conv2d(out_channels,out_channels,
                                   kernel_size = 3,stride=1,padding=1))
            block.append(nn.ReLU())
            block.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*block)
    def forward(self,x):
        out = self.block(x)
        return out

class PPUpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride,
                 padding,output_padding):
        
        super(PPUpBlock,self).__init__()
        self.conv2d_t = nn.ConvTranspose2d(in_channels,out_channels,
                                           kernel_size=3,stride=stride,
                                           padding=padding,
                                           output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv2d_t(x)
        x = F.relu(x)
        x = self.bn(x)
        return x

class PPBackbone(nn.Module):
    def __init__(self,in_channels):
        super(PPBackbone,self).__init__()
        # add down block one
        self.down1 = PPDownBlock(4,in_channels,in_channels)
        self.up1 = PPUpBlock(in_channels,2*in_channels,1,1,0)
        self.down2 = PPDownBlock(6,in_channels,2*in_channels)
        self.up1 = PPUpBlock(in_channels*2,2*in_channels,2,1,1)
        self.down3 = PPDownBlock(6,2*in_channels,4*in_channels)
        self.up3 = PPUpBlock(in_channels*4,in_channels*2,4,1,3)

    def forward(self,x):
        x = self.down1(x)
        out1 = self.up1(x)
        x = self.down2(x)
        out2 = self.up2(x)
        x = self.down3(x)
        out3 = self.up3(x)
        return torch.cat((out1,out2,out3),dim=1)


class PPDetectionHead(nn.Module):
    def __init__(self,in_channels,cls_out_channels,reg_out_channels):
        super(PPDetectionHead,self).__init__()
        self.cls = nn.Conv2d(in_channels,cls_out_channels,
                             kernel_size=1,stride=1)
        self.reg = nn.Conv2d(in_channels,reg_out_channels,
                             kernel_size=1,stride=1)
    def forward(self,x):
        cls_scores = self.cls(x)
        reg_scores = self.reg(x)

        return (cls_scores,reg_scores)


