import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
import pdb


class PPLoss(nn.Module):
    """
    """
    def __init__(self,b_ort,b_reg,b_cls,gamma,device):
        super(PPLoss,self).__init__()
        self.b_ort,self.b_reg,self.b_cls,self.gamma,self.device = b_ort,b_reg,b_cls,gamma,device


    def forward(self,cls_tensor,reg_tensor,cls_targets,reg_targets):
        #cls_tensor: [batch,cls_channels,FM_H,FM_W]
        #reg_tensor: [batch,reg_channels,FM_H,FM_W]
        #cls_channels = anchor_dims * (num_classes+1)
        #reg_channels = anchor_dims * reg_dims
       
        #pdb.set_trace() 
        cls_tensor  = cls_tensor.permute(0,2,3,1)
        cls_size    = cls_tensor.size()
        cls_tensor  = cls_tensor.reshape(cls_size[0],-1)
        cls_targets = cls_targets.reshape(cls_size[0],-1)
        p           = torch.sigmoid(cls_tensor)
        pt          = torch.where(cls_targets == 1,p,1-p)
        at          = torch.where(cls_targets == 1,torch.ones(pt.size(),device=self.device)*1000,torch.ones(pt.size(),device=self.device))
        w           = (at*(1-pt)**self.gamma).detach()
        
        cls_loss    = F.binary_cross_entropy_with_logits(cls_tensor,cls_targets,weight=w)        
        
        reg_tensor  = reg_tensor.permute(0,2,3,1)
        reg_tensor[...,6] = torch.tanh(reg_tensor[...,6])
        reg_size    = reg_tensor.size()
        reg_tensor  = reg_tensor.reshape(reg_size[0],-1,cfg.DATA.REG_DIMS)
        pos_anchors = torch.where(reg_targets[...,0] == 1)
        reg_scores  = reg_tensor[pos_anchors][...,:7]
        loss_targs  = reg_targets[pos_anchors][...,1:8]
        reg_loss    = F.smooth_l1_loss(reg_scores,loss_targs,reduction='mean')
        
        ort_scores  = reg_tensor[pos_anchors][...,7:]
        ort_targets = reg_targets[pos_anchors][...,8].long()
        ort_loss    = F.cross_entropy(ort_scores,ort_targets)

        return self.b_cls*cls_loss + self.b_ort*ort_loss + self.b_reg*reg_loss









