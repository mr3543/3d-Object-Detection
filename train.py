import pickle
import torch.utils.data
from LSUV_pytorch.LSUV import LSUVinit
from config import cfg
from model.model import PPBackbone,PPDetectionHead,PPFeatureNet,PPScatter
from model.loss import PPLoss
from data.dataset import PPDataset
from tqdm import tqdm
import sys
import pdb
import os.path as osp

def get_batch(dataset,batch_size):
    pil_list = []
    ind_list = []
    tar_list = []
    for i in range(batch_size):
        pil,ind,tar = dataset[i]
        pil_list.append(pil)
        ind_list.append(ind)
        tar_list.append(tar)
    pil_out = torch.stack(pil_list)
    ind_out = torch.stack(ind_list)
    tar_out = torch.stack(tar_list)
    return (pil_out,ind_out,tar_out)

fn_in = cfg.NET.FEATURE_NET_IN
fn_out = cfg.NET.FEATURE_NET_OUT
cls_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.NUM_CLASSES
reg_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.REG_DIMS

data_dict = pickle.load(open('data_dict.pkl','rb'))
lidar_filepaths = pickle.load(open('lidar_filepaths.pkl','rb'))
anchor_boxes = pickle.load(open('anchor_boxes.pkl','rb'))
anchor_corners = pickle.load(open('anchor_corners.pkl','rb'))
anchor_centers = pickle.load(open('anchor_centers.pkl','rb'))
data_mean = pickle.load(open('pillar_means.pkl','rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pp_dataset = PPDataset(lidar_filepaths,data_dict,anchor_boxes,
                       anchor_corners,anchor_centers,data_mean)

batch_size = cfg.NET.BATCH_SIZE
epochs = cfg.NET.EPOCHS
num_workers = cfg.NET.NUM_WORKERS
dataloader = torch.utils.data.DataLoader(pp_dataset,batch_size,
                                         shuffle=False,num_workers=num_workers)

pp_featurenet = PPFeatureNet(fn_in,fn_out)
pp_scatter = PPScatter(device)
pp_backbone = PPBackbone(fn_out)
pp_det_head = PPDetectionHead(6*fn_out,cls_channels,reg_channels)
pp_loss = PPLoss(cfg.NET.B_ORT,cfg.NET.B_REG,cfg.NET.B_CLS,cfg.NET.GAMMA)

pp_featurenet = pp_featurenet.to(device)
pp_scatter = pp_scatter.to(device)
pp_backbone = pp_backbone.to(device)
pp_det_head = pp_det_head.to(device)
pp_loss = pp_loss.to(device)

(p,i,_,__) = next(iter(dataloader))
p = p.to(device)
i = i.to(device)

#(p,i,t) = pp_dataset[0]
#p = p[None,...].to(device)
#i = i[None,...].to(device)

pp_featurenet = LSUVinit(pp_featurenet,p,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
feature_out = pp_featurenet(p)
scatter_out = pp_scatter(feature_out,i)
pp_backbone = LSUVinit(pp_backbone,scatter_out,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
backbone_out = pp_backbone(scatter_out)
pp_det_head = LSUVinit(pp_det_head,backbone_out,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
params = list(pp_featurenet.parameters()) + list(pp_scatter.parameters()) + list(pp_backbone.parameters()) + \
         list(pp_det_head.parameters()) + list(pp_loss.parameters())

pp_loss = pp_loss.to(device)
lr = cfg.NET.LEARNING_RATE
wd = cfg.NET.WEIGHT_DECAY
optim = torch.optim.Adam(params,lr=lr,weight_decay=wd)

print('STARTING TRAINING')


for epoch in range(epochs):
    print('EPOCH: ',epoch)
    epoch_losses = []
    progress_bar = tqdm(dataloader)
    #i = 0
    #while True:
    for i,(pillar,inds,c_target,r_target) in enumerate(progress_bar):
        print('training on batch: ',i)
        #(pillar,inds,target) = get_batch(pp_dataset,batch_size)
        #if i == 0:
        #    print(pillar.size())
        #    print(inds.size())
        #    print(target.size())
        pillar = pillar.to(device)
        inds = inds.to(device)
        c_target = c_target.to(device)
        r_target = r_target.to(device)
        feature_out = pp_featurenet(pillar)
        scatter_out = pp_scatter(feature_out,inds)
        backbone_out = pp_backbone(scatter_out)
        cls_tensor,reg_tensor = pp_det_head(backbone_out)
        batch_loss = pp_loss(cls_tensor,reg_tensor,c_target,r_target)
        print('loss: ',batch_loss)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        #i += 1
        """
        if i % 5 == 0:
            with torch.no_grad():
                for param in params:
                    if param.requires_grad and param.name:
                        print('name: {}, mean: {}, std: {}'.format(param.name,torch.mean(param.data),torch.std(param.data)))
       """ 
        if i != 0 and i % 25 == 0:
            print('saving model ckpt')
            cpd = cfg.DATA.CKPT_DIR
            feature_ckpt  = osp.join(cpd,'pp_checkpoint_featurenet{}_{}.pth'.format(epoch,i))
            backbone_ckpt = osp.join(cpd,'pp_checkpoint_backbone{}_{}.pth'.format(epoch,i))
            dethead_ckpt  = osp.join(cpd,'pp_checkpoint_dethead{}_{}.pth'.format(epoch,i))
            torch.save(pp_featurenet.state_dict(),feature_ckpt)
            torch.save(pp_backbone.state_dict(),backbone_ckpt)
            torch.save(pp_det_head.state_dict(),dethead_ckpt)

    epoch_losses.append(batch_loss.detach().cpu().numpy())
