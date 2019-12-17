import pickle
import torch.utils.data
from LSUV_pytorch.LSUV import LSUVinit
from config import cfg
from model.model import PPBackbone,PPDetectionHead,PPFeatureNet,PPScatter,PPModel
from model.loss import PPLoss
from data.dataset import PPDataset
from tqdm import tqdm
from apex import amp
import sys
import pdb
import os.path as osp
from evaluate import evaluate


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


dd_fp  = osp.join(cfg.DATA.LIDAR_TRAIN_DIR,'data_dict.pkl')
li_fp  = osp.join(cfg.DATA.LIDAR_TRAIN_DIR,'lidar_filepaths.pkl')
box_fp = osp.join(cfg.DATA.ANCHOR_DIR,'anchor_boxes.pkl')
crn_fp = osp.join(cfg.DATA.ANCHOR_DIR,'anchor_corners.pkl')
cen_fp = osp.join(cfg.DATA.ANCHOR_DIR,'anchor_centers.pkl')

data_dict       = pickle.load(open(dd_fp,'rb'))
lidar_filepaths = pickle.load(open(li_fp,'rb'))
anchor_boxes    = pickle.load(open(box_fp,'rb'))
anchor_corners  = pickle.load(open(crn_fp,'rb'))
anchor_centers  = pickle.load(open(cen_fp,'rb'))

data_mean  = pickle.load(open('pillar_means.pkl','rb'))
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device    = 'cpu'
pp_dataset = PPDataset(lidar_filepaths,data_dict,anchor_boxes,
                       anchor_corners,anchor_centers,data_mean=data_mean,training=True)

batch_size  = cfg.NET.BATCH_SIZE
epochs      = cfg.NET.EPOCHS
num_workers = cfg.NET.NUM_WORKERS
dataloader  = torch.utils.data.DataLoader(pp_dataset,batch_size,
                                         shuffle=False,num_workers=num_workers)
 
pp_model = PPModel(fn_in,fn_out,cls_channels,reg_channels,device)
pp_loss  = PPLoss(cfg.NET.B_ORT,cfg.NET.B_REG,cfg.NET.B_CLS,cfg.NET.GAMMA,device)

model_fp = osp.join(cfg.DATA.CKPT_DIR,'pp_checkpoint0_500.pth')
pp_model.load_state_dict(torch.load(model_fp))
pp_model = pp_model.to(device)
pp_loss  = pp_loss.to(device)


#LSUV INIT

(p,i,_,__) = next(iter(dataloader))
p = p.to(device)
i = i.to(device)

pp_model.feature_net = LSUVinit(pp_model.feature_net,p,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
feature_out = pp_model.feature_net(p)
scatter_out = pp_model.scatter(feature_out,i)
pp_model.backbone = LSUVinit(pp_model.backbone,scatter_out,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
backbone_out = pp_model.backbone(scatter_out)
pp_model.det_head = LSUVinit(pp_model.det_head,backbone_out,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)

lr = cfg.NET.LEARNING_RATE
wd = cfg.NET.WEIGHT_DECAY


params = list(pp_model.parameters())
optim  = torch.optim.Adam(params,lr=lr,weight_decay=wd)
optim_fp = osp.join(cfg.DATA.CKPT_DIR,'optim_checkpoint0_500.pth')
optim.load_state_dict(torch.load(optim_fp))

#pp_model,optim = amp.initialize(pp_model,optim,opt_level="O1")

"""
for layer in pp_model.modules():
    if isinstance(layer,torch.nn.BatchNorm2d):
        print('BN')
        layer.float()
"""


print('STARTING TRAINING')


for epoch in range(epochs):
    print('EPOCH: ',epoch)
    epoch_losses = []
    progress_bar = tqdm(dataloader)
    for i,(pillar,inds,c_target,r_target) in enumerate(progress_bar):
        pillar = pillar.to(device)
        inds = inds.to(device)
        c_target = c_target.to(device)
        r_target = r_target.to(device)
        cls_tensor,reg_tensor = pp_model(pillar,inds)
        batch_loss = pp_loss(cls_tensor,reg_tensor,c_target,r_target) 
        optim.zero_grad()
        #with amp.scale_loss(batch_loss,optim) as scaled_loss:
        #    scaled_loss.backward()
        batch_loss.backward()
        optim.step()
        
        if i % 25 == 0:
           # with torch.no_grad():
           #     evaluate(pp_model,anchor_boxes,data_mean,device)
            print('loss: ',batch_loss)
    
        if i != 0 and i % 100 == 0:
            print('saving model ckpt')
            cpd = cfg.DATA.CKPT_DIR
            model_ckpt = osp.join(cpd,'pp_checkpoint{}_{}.pth'.format(epoch,i))
            print('saving model to: ',model_ckpt)
            torch.save(pp_model.state_dict(),model_ckpt)
            optim_ckpt = osp.join(cpd,'optim_checkpoint{}_{}.pth'.format(epoch,i))
            torch.save(optim.state_dict(),optim_ckpt)
        if i != 0 and i % 500 == 0:
            with torch.no_grad():
                evaluate(pp_model,anchor_boxes,data_mean,device)
    epoch_losses.append(batch_loss.detach().cpu().numpy())
