import pandas as pd
from datetime import datetime
import os
import gc
import os.path as osp
import pickle
import torch
import torch.nn as nn
import model.model as model
import model.loss as loss
from tqdm import tqdm
from data.data_prep import make_anchor_boxes
from data.dataset import PPDataset
from config import cfg
from lyft_dataset_sdk.lyftdataset import LyftDataset
from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from data_prep import move_boxes_to_canvas_space

def make_token_list(df,l5d):
    first_samples = df.first_sample_token.values
    token_list = []
    for token in first_samples:
        while token:
            token_list.append(token)
            sample = l5d.get('sample',token)
            token = sample['next']
    return token_list

    
data_path = cfg.DATA.DATA_PATH
json_path = cfg.DATA.TRAIN_JSON_PATH
l5d = LyftDataset(data_path=data_path,json_path=json_path,
                  verbose=True)

records = [(l5d.get('sample', record['first_sample_token'])['timestamp'], record) for record in l5d.scene]

entries = []

for start_time, record in sorted(records):
    start_time = l5d.get('sample', record['first_sample_token'])['timestamp'] / 1000000

    token = record['token']
    name = record['name']
    date = datetime.utcfromtimestamp(start_time)
    host = "-".join(record['name'].split("-")[:2])
    first_sample_token = record["first_sample_token"]

    entries.append((host, name, date, token, first_sample_token))

df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
validation_hosts = ["host-a007", "host-a008", "host-a009"]

val_df = df[df["host"].isin(validation_hosts)]
vi = val_df.index
train_df = df[~df.index.isin(vi)]

token_list = make_token_list(train_df,l5d)
lidar_filepaths = []
data_dict = {}

for token in tqdm_notebook(token_list,total=len(token_list)):
    sample = l5d.get('sample',token)
    sample_lidar_token = sample['data']['LIDAR_TOP']
    try:
        lidar_filepath = l5d.get_sample_data_path(sample_lidar_token)
        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
    except Exception as e:
        print('Failed to load LIDAR cloud for {}: {}:'.format(token,e))
        continue
    lidar_data = l5d.get('sample_data',sample_lidar_token)

    ego_pose = l5d.get('ego_pose',lidar_data['ego_pose_token'])
    calibrated_sensor = l5d.get('calibrated_sensor',lidar_data['calibrated_sensor_token'])
    car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                        Quaternion(calibrated_sensor['rotation']),
                                        inverse = False)
    boxes = l5d.get_boxes(sample_lidar_token)
    boxes = move_boxes_to_canvas_space(boxes,ego_pose)
    lidar_filepath = l5d.get_sample_data_path(sample_lidar_token)
    lidar_filepaths.append(lidar_filepath)
    box_filepath = osp.join(cfg.DATA.BOX_DIR,token + '_boxes.pkl')
    pickle.dump(boxes,open(box_filepath,'wb'))
    data_dict[lidar_filepath] = {'boxes':box_filepath,'trans_matrix':car_from_sensor}

del l5d,df,train_df,val_df,vi,token_list,records,entries
gc.collect()

anchor_boxes,anchor_corners,anchor_centers = make_anchor_boxes()

fn_in = cfg.NET.FEATURE_NET_IN
fn_out = cfg.NET.FEATURE_NET_OUT
cls_channels = len(cfg.DATA.ANCHOR_DIMS)*(cfg.DATA.NUM_CLASSES + 1)
reg_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.REG_DIMS

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
pp_model = PPModel(fn_in,fn_out,cls_channels,reg_channels,device)
pp_loss = PPLoss(cfg.NET.B_ORT,cfg.NET.B_REG,cfg.NET.B_CLS,cfg.NET.GAMMA)
pp_dataset = PPDataset(lidar_filepaths,data_dict,anchor_boxes,
                       anchor_corners,anchor_centers)

pp_model = pp_model.to(device)
pp_loss = pp_loss.to(device)
optim = torch.optim.Adam(pp_model.parameters(),lr=2e-5)
batch_size = cfg.NET.BATCH_SIZE
epochs = cfg.NET.EPOCHS
#num_workers = cfg.NET.NUM_WORKERS
num_workers=0
dataloader = torch.utils.data.DataLoader(pp_dataset,batch_size,
                                         shuffle=False,num_workers=num_workers)
for epoch in range(epochs):
    print('EPOCH: ',epoch)
    epoch_losses = []
    progress_bar = tqdm_notebook(dataloader)

    for i,(pillar,inds,target) in enumerate(progress_bar):
        print('training on sample: ',i)
        print('nan pillars: ',check_nan(pillar))
        print('nan inds: ',check_nan(inds))
        print('nan target: ',check_nan(target))
        print('pillars mean: ',torch.mean(pillar))
        print('pillars std: ',torch.std(pillar))
        pillar = pillar.to(device)
        inds = inds.to(device)
        target = target.to(device)
        cls_tensor,reg_tensor = pp_model(pillar,inds)
        batch_loss = pp_loss(cls_tensor,reg_tensor,target)
        print(batch_loss)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        if i % 5 == 0:
            with torch.no_grad():
                for name,params in pp_model.named_parameters():
                    if params.requires_grad:
                        print('name: {}, mean: {}, std: {}'.format(name,torch.mean(params.data),torch.std(params.data)))
        if i != 0 and i % 100 == 0:
            ckpt_filename = 'pp_checkpoint_{}_{}.pth'.format(epoch,i)
            ckpt_filepath = osp.join(cfg.DATA.CKPT_DIR,ckpt_filename)
            torch.save(pp_model.state_dict(),ckpt_filepath)


    epoch_losses.append(batch_loss.detach().cpu().numpy())