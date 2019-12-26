import torch
import pickle
import os.path as osp
from config import cfg
from data.dataset import PPDataset
from tqdm import tqdm

ddfp = osp.join(cfg.DATA.LIDAR_TRAIN_DIR,'data_dict.pkl')
tkfp = osp.join(cfg.DATA.TOKEN_TRAIN_DIR,'token_list.pkl')

token_list = pickle.load(open(tkfp,'rb'))
data_dict  = pickle.load(open(ddfp,'rb'))

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pp_dataset = PPDataset(token_list,data_dict,None,
                       None,None,data_mean=None,training=False)

batch_size  = 10
num_workers = 2
dataloader  = torch.utils.data.DataLoader(pp_dataset,batch_size=batch_size,
                                         shuffle=False,num_workers=num_workers)

means = torch.zeros(cfg.DATA.MAX_POINTS_PER_PILLAR*cfg.DATA.MAX_PILLARS*9,device=device)
for i,(p,_) in enumerate(tqdm(dataloader,total = len(pp_dataset)//batch_size)):
    p = p.to(device)
    p = p.reshape(p.size()[0],-1)
    m = torch.mean(p,dim=0)
    means = means*(i/(i+1)) + m*(1/(i+1))

means = means.cpu()
pickle.dump(means,open('pillar_means.pkl','wb'))


