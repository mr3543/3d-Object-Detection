class PPLoss(nn.Module):
    def __init__(self,b_ort,b_reg,b_cls,gamma):
        super(PPLoss,self).__init__()
        self.b_ort,self.b_reg,self.b_cls,self.gamma = b_ort,b_reg,b_cls,gamma


    def forward(self,cls_tensor,reg_tensor,targets):
        #cls_tensor: [batch,cls_channels,FM_H,FM_W]
        #reg_tensor: [batch,reg_channels,FM_H,FM_W]
        #target: [batch,FM_H,FM_W,anchor_dims,target_dims]
        #cls_channels = anchor_dims * num_classes
        #reg_channels = anchor_dims * reg_dims

        c_sh = cls_tensor.size()
        r_sh = reg_tensor.size()
        
        cls_tensor = cls_tensor.cpu()
        reg_tensor = reg_tensor.cpu()
        targets = target.cpu()

        cls_tensor = cls_tensor.view(c_sh[0],cfg.DATA.NUM_CLASSES + 1,
                                     cfg.DATA.NUM_ANCHORS,c_sh[2],c_sh[3])
        reg_tensor = reg_tensor.view(r_sh[0],cfg.DATA.REG_DIMS,cfg.DATA.NUM_ANCHORS,
                                     r_sh[2],r_sh[3])
        cls_tensor = cls_tensor.permute(0,3,4,2,1)
        reg_tensor = reg_tensor.permute(0,3,4,2,1)

        cls_tensor = F.softmax(cls_tensor,dim=4)
        anchor_classes = targets[...,-1].numpy() # cpu 
        cls_tensor = cls_tensor.view(-1,cls_tensor.size()[-1])
        anchor_scores = cls_tensor[np.arange(cls_tensor.size()[0]),anchor_classes]
        cls_loss = torch.sum(-(1 - anchor_scores)**self.gamma*torch.log(anchor_scores))

        pos_anchors = np.where(targets[...,0] == 1)
        reg_targets = targets[pos_anchors][:,1:8].float()
        reg_tensor = reg_tensor.contiguous().view(reg_tensor.size()[0],-1,reg_tensor.size()[-1])
        reg_scores = reg_tensor[pos_anchors][:,:7]
        reg_loss = F.smooth_l1_loss(reg_scores,reg_targets,reduction='sum')

        ort_scores = F.softmax(reg_tensor[pos_anchors][:,7:9],dim=1)
        ort_targets = targets[pos_anchors][:,8]
        ort_targets = np.stack((ort_targets,1- ort_targets),axis=1)
        ort_targets = torch.from_numpy(ort_targets).float()
        ort_loss = F.binary_cross_entropy(ort_scores,ort_targets)

        num_pos = len(pos_anchors[0])
        return 1/num_pos*(self.b_cls*cls_loss + self.b_ort*ort_loss + self.b_reg*reg_loss)