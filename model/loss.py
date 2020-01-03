import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
import pdb


class PPLoss(nn.Module):
    """
        Loss module. We use a multi-task loss with a classification component
        for the object detection and a regression component for the bouding
        box regression
    """

    def __init__(self,b_ort,b_reg,b_cls,gamma,device):
        super(PPLoss,self).__init__()
        # parameters are set to weight the loss function so that
        # classification and regression losses are approximately on the same scale

        self.b_ort,self.b_reg,self.b_cls,self.gamma,self.device = b_ort,b_reg,b_cls,gamma,device
        self.cls_weights = torch.from_numpy(cfg.NET.CLASS_WEIGHTS).to(self.device)

    def forward(self,cls_tensor,reg_tensor,cls_targets,reg_targets):
        #cls_tensor: [batch,cls_channels,FM_H,FM_W]
        #reg_tensor: [batch,reg_channels,FM_H,FM_W]
        #cls_channels = anchor_dims * (num_classes+1)
        #reg_channels = anchor_dims * reg_dims
       
        
        cls_tensor  = cls_tensor.permute(0,2,3,1)
        cls_size    = cls_tensor.size()
        cls_tensor  = cls_tensor.reshape(cls_size[0],-1)
        cls_targets = cls_targets.reshape(cls_size[0],-1)
        # compute the p vector for focal loss
        p           = torch.sigmoid(cls_tensor)
        ct          = cls_targets.reshape(cls_size[0],cls_size[1],
                                          cls_size[2],cfg.DATA.NUM_ANCHORS,
                                          cfg.DATA.NUM_CLASSES)*self.cls_weights
        ct          = ct.reshape(cls_size[0],-1)
        pt          = torch.where(cls_targets == 1,p,1-p)
        #at          = torch.where(cls_targets == 1,torch.ones(pt.size(),device=self.device)*25,torch.ones(pt.size(),device=self.device))
        at          = torch.where(cls_targets == 1,ct,torch.ones(pt.size(),device=self.device))
        # compute focal loss weights
        w           = (at*(1-pt)**self.gamma).detach()
        cls_loss    = F.binary_cross_entropy_with_logits(cls_tensor,cls_targets,weight=w)

        reg_tensor  = reg_tensor.permute(0,2,3,1)
        # the dt element needs to put through a tanh since
        # the element represents sin(\theta_gt - \theta_a)
        reg_tensor[...,6] = torch.tanh(reg_tensor[...,6])
        reg_size    = reg_tensor.size()
        reg_tensor  = reg_tensor.reshape(reg_size[0],-1,cfg.DATA.REG_DIMS)
        pos_anchors = torch.where(reg_targets[...,0] == 1)
        reg_scores  = reg_tensor[pos_anchors][...,:7]
        loss_targs  = reg_targets[pos_anchors][...,1:8]
        reg_loss    = F.smooth_l1_loss(reg_scores,loss_targs,reduction='mean')
        
        
        # we are excluding the orientation loss since the only 
        # use is object detection
        ort_scores  = reg_tensor[pos_anchors][...,7:]
        ort_targets = reg_targets[pos_anchors][...,8].long()
        ort_loss    = F.cross_entropy(ort_scores,ort_targets)
        
        total_loss = self.b_cls*cls_loss + self.b_reg*reg_loss + self.b_ort*ort_loss
        return cls_loss,reg_loss,ort_loss,total_loss






