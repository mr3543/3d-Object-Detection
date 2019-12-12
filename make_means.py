from data.dataset import PPDataset
import pickle
from tqdm import tqdm
import torch

data_dict = pickle.load(open('data_dict.pkl','rb'))
lidar_filepaths = pickle.load(open('lidar_filepaths.pkl','rb'))
anchor_boxes = pickle.load(open('anchor_boxes.pkl','rb'))
anchor_corners = pickle.load(open('anchor_corners.pkl','rb'))
anchor_centers = pickle.load(open('anchor_centers.pkl','rb'))
pp_dataset = PPDataset(lidar_filepaths,data_dict,anchor_boxes,
                       anchor_corners,anchor_centers,data_mean=None,training=False)

means = torch.zeros(10800000)
for i in tqdm(range(len(pp_dataset))):
    (p,_,__,___) = pp_dataset[i]
    p = p.reshape(-1)
    means = means*(i/(i+1)) + p*(1/(i+1))

pickle.dump(means,open('pillar_means.pkl','wb'))


