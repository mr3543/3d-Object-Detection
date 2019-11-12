import pandas as pd
from datetime import datetime
import os
import os.path as osp
import torch
import torch.nn as nn
import model.model as model
import model.loss as loss
from tqdm import tqdm
from data.data_prep import make_anchor_boxes
from data.dataset import PPDataset
from config import cfg
from lyft_dataset_sdk.lyftdataset import LyftDataset

def make_token_list(df,l5d):
    first_samples = df.first_sample_token.values
    token_list = []
    for token in first_samples:
        while token:
            token_list.append(token)
    return token_list


if __name__ == '__main__':
    
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
    anchor_boxes,anchor_corners,anchor_centers = make_anchor_boxes()
    pp_dataset = PPDataset(token_list,l5d,anchor_boxes,
                           anchor_corners,anchor_centers)

    fn_in = cfg.NET.FEATURE_NET_IN
    fn_out = cfg.NET.FEATURE_NET_OUT
    cls_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.NUM_CLASSES
    reg_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.REG_DIMS
    
    pp_model = model.PPModel(fn_in,fn_out,cls_channels,reg_channels)
    pp_loss = loss.PPLoss(cfg.NET.B_ORT,cfg.NET.B_REG,cfg.NET.B_CLS,cfg.NET.GAMMA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pp_model = pp_model.to(device)
    optim = torch.optim.Adam(pp_model.parameters(),lr=2e-4)
    batch_size = cfg.NET.BATCH_SIZE
    epochs = cfg.NET.EPOCHS
    num_workers = cfg.NET.NUM_WORKERS
    dataloader = torch.utils.data.Dataloader(pp_dataset,batch_size,shuffle=False,
                                             num_workers=num_workers)

    for epoch in epochs:
        print('EPOCH: ',epoch)
        epoch_losses = []
        progress_bar = tqdm(dataloader)

        for i,(pillars,inds,target) in enumerate(progress_bar):
            pillars = pillars.to(device)
            cls_tensor,reg_tensor = pp_model(pillars,inds)
            batch_loss = pp_loss(cls_tensor,reg_tensor,target)
            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            if i != 0 and 1 % 0 == 1000:
                ckpt_filename = 'pp_checkpoint_{}_{}.pth'.format(epoch,i)
                ckpt_filepath = osp.join(cfg.DATA.CKPT_DIR,ckpt_filename)
                torch.save(pp_model.state_dict(),ckpt_filepath)
                

            epoch_losses.append(batch_loss.detatch().cpu().numpy())


