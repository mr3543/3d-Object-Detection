import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import sys

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PPFeatureNet(nn.Module):
    """
    This layer takes in the pillar pseudo-image and runs a 1x1 convolution
    on the input.
    
    The input is a tensor of size [B,D,P,N], where B is the mini-batch size,
    D is the size of a single lidar point feature (x,y,z,r,xc,yc,zc,xp,yp). P 
    is the number of pillars, and N is the number of points per pillar.
    
    This tensor is fed through a 1x1 convolution to produce a tensor of size
    [B,C,P,N]. We then take the max of the last dimension to output a tensor of
    size [B,C,P]
    """
    def __init__(self,in_channels,out_channels):
        super(PPFeatureNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn(x)
        x = torch.max(x,dim=3)[0]
        return x

class PPScatter(nn.Module):
    """
    This layer takes a tensor of size [B,C,P] and transforms it to 
    a tensor of [B,C,H,W] by scattering back the pillars to their 
    original H,W location. We pass `inds` to the `forward` method
    which contain the H,W locations of the non-empty pillars.
    """
    def __init__(self,device):
        super(PPScatter,self).__init__()
        self.device = device 
        
    def forward(self,x,inds):
        out_sh = x.size()
        out = torch.zeros(out_sh[0],out_sh[1],cfg.DATA.CANVAS_HEIGHT,cfg.DATA.CANVAS_WIDTH).to(self.device)
        non_empty = torch.nonzero(inds[:,:,0])
        batch = non_empty[:,0]
        pillar = non_empty[:,1]
        x_inds = inds[batch,pillar][:,1]
        y_inds = inds[batch,pillar][:,2]
        out[batch,:,y_inds,x_inds] = x[batch,:,pillar]
        return out

class PPDownBlock(nn.Module):
    """
    Standard down convolutional block. First layer takes 
    the input and reduces its H,W by 2 with a stride 2 
    convolution. The next layers are stride 1 and do
    not change the feature map size. 
    """
    def __init__(self,num_layers,in_channels,out_channels):
        super(PPDownBlock,self).__init__()
        block = []
        block.append(nn.Conv2d(in_channels,out_channels,
                               kernel_size=3,stride=2,padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))
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
    """
    Standard up convolutional block. First layer takes in input
    and increases in H,W size by 2 with a fractionally strided 
    convolution layer. The next layers are stride 1 and do not change
    the feature map size. 
    """
    def __init__(self,in_channels,out_channels,stride,
                 padding,output_padding):
        super(PPUpBlock,self).__init__()
        
        self.conv2d_t = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,
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
    """
    Model backbone, consists of down blocks
    and up blocks.
    """
    
    def __init__(self,in_channels):
        super(PPBackbone,self).__init__()
        # add down block one
        self.down1 = PPDownBlock(4,in_channels,in_channels)
        self.up1 = PPUpBlock(in_channels,2*in_channels,1,1,0)
        self.down2 = PPDownBlock(6,in_channels,2*in_channels)
        self.up2 = PPUpBlock(in_channels*2,2*in_channels,2,1,1)
        self.down3 = PPDownBlock(6,2*in_channels,4*in_channels)
        self.up3 = PPUpBlock(in_channels*4,in_channels*2,4,1,1)

    def forward(self,x):
        x = self.down1(x)
        out1 = self.up1(x)
        x = self.down2(x)
        out2 = self.up2(x)
        x = self.down3(x)
        out3 = self.up3(x)
        out = torch.cat((out1,out2,out3),dim=1)
        return out


class PPDetectionHead(nn.Module):
    """
    Detection head. The classification and regression layers are
    implemented as stride 1, kernel 1 convolutional layers. The output
    channels are the model's unnormalized predictions for the classification 
    and regression targests. This is similar to the SSD architecture. 
    """
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

class PPModel(nn.Module):
    """
    End to end model architecture. Currently not used due to the need
    for LSUV initialization. 
    """
    def __init__(self,feature_net_in_channels,feature_net_out_channels,
                 class_layer_channels,reg_layer_channels,device):
        super(PPModel,self).__init__()
        self.feature_net = PPFeatureNet(feature_net_in_channels,feature_net_out_channels)
        self.scatter = PPScatter(device)
        self.backbone = PPBackbone(feature_net_out_channels)
        self.det_head = PPDetectionHead(6*feature_net_out_channels,class_layer_channels,
                                        reg_layer_channels)
    def forward(self,x,inds):
        x = self.feature_net(x)
        x = self.scatter(x,inds)
        x = self.backbone(x)
        cls_tensor,reg_tensor = self.det_head(x)
        return (cls_tensor,reg_tensor)