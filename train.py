import pickle
import torch.utils.data
from LSUV_pytorch.LSUV import LSUVinit
from config import cfg
from model.model import PPBackbone,PPDetectionHead,PPFeatureNet,PPScatter
from model.loss import PPLoss
from data.dataset import PPDataset
from tqdm import tqdm

fn_in = cfg.NET.FEATURE_NET_IN
fn_out = cfg.NET.FEATURE_NET_OUT
cls_channels = len(cfg.DATA.ANCHOR_DIMS)*(cfg.DATA.NUM_CLASSES + 1)
reg_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.REG_DIMS

data_dict = pickle.load(open('data_dict.pkl','rb'))
lidar_filepaths = pickle.load(open('lidar_filepaths.pkl','rb'))
anchor_boxes = pickle.load(open('anchor_boxes.pkl','rb'))
anchor_corners = pickle.load(open('anchor_corners.pkl','rb'))
anchor_centers = pickle.load(open('anchor_centers.pkl','rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pp_dataset = PPDataset(lidar_filepaths,data_dict,anchor_boxes,
                       anchor_corners,anchor_centers)

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

(p,i,t) = pp_dataset[0]
p = p[None,...].to(device)
i = i[None,...].to(device)
t = t[None,...].to(device)

pp_featurenet = LSUVinit(pp_featurenet,p,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
feature_out = pp_featurenet(p)
scatter_out = pp_scatter(feature_out,i)
pp_backbone = LSUVinit(pp_backbone,scatter_out,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
backbone_out = pp_backbone(scatter_out)
pp_det_head = LSUVinit(pp_det_head,backbone_out,needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = False)
params = list(pp_featurenet.parameters()) + list(pp_scatter.parameters()) + list(pp_backbone.parameters()) + \
         list(pp_det_head.parameters()) + list(pp_loss.parameters())

pp_loss = pp_loss.to(device)
optim = torch.optim.Adam(params,lr=2e-5)

for epoch in range(epochs):
    print('EPOCH: ',epoch)
    epoch_losses = []
    progress_bar = tqdm(dataloader)

    for i,(pillar,inds,target) in enumerate(progress_bar):
        print('training on batch: ',i)
        pillar = pillar.to(device)
        inds = inds.to(device)
        target = target.to(device)
        feature_out = pp_featurenet(pillar)
        scatter_out = pp_scatter(feature_out,inds)
        backbone_out = pp_backbone(scatter_out)
        cls_tensor,reg_tensor = pp_det_head(backbone_out)
        batch_loss = pp_loss(cls_tensor,reg_tensor,target)
        print('loss: ',batch_loss)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        """
        if i % 5 == 0:
            with torch.no_grad():
                for param in params:
                    if param.requires_grad and param.name:
                        print('name: {}, mean: {}, std: {}'.format(param.name,torch.mean(param.data),torch.std(param.data)))
        
        if i != 0 and i % 100 == 0:
            ckpt_filename = 'pp_checkpoint_{}_{}.pth'.format(epoch,i)
            ckpt_filepath = osp.join(cfg.DATA.CKPT_DIR,ckpt_filename)
            torch.save(pp_model.state_dict(),ckpt_filepath)
        """

    epoch_losses.append(batch_loss.detach().cpu().numpy())