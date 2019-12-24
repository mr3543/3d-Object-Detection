import torch
import pickle
import os.path as osp
from config import cfg
from data.dataset import PPDataset

ddfp = osp.join(cfg.DATA.LIDAR_TRAIN_DIR,'data_dict.pkl')
tkdp = osp.join(cfg.DATA.TOKEN_TRAIN_DIR,'token_list.pkl')

token_list = pickle.load(open(tkfp,'rb'))
data_dict = pickle.load(open(ddfp,'rb'))

pp_dataset = PPDataset(token_list,data_dict,None,
                       None,None,data_mean=None,training=False)

means = torch.zeros(cfg.DATA.MAX_POINTS_PER_PILLAR*cfg.DATA.MAX_PILLARS*9)
for i in tqdm(range(len(pp_dataset))):
    (p,_) = pp_dataset[i]
    p = p.reshape(-1)
    means = means*(i/(i+1)) + p*(1/(i+1))

pickle.dump(means,open('pillar_means.pkl','wb'))


