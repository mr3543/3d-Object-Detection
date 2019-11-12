import pandas as pd
import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.model as model
import model.loss as loss
from tqdm import tqdm
from pyquaternion import Quaternion
from data.data_prep import make_anchor_boxes
from data.dataset import PPDataset
from config import cfg
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
from datetime import datetime
from torch.ops.torchvision import nms


def make_token_list(df,l5d):
    first_samples = df.first_sample_token.values
    token_list = []
    for token in first_samples:
        while token:
            token_list.append(token)
    return token_list

def write_submission(boxes):
    sub = {}
    for i in range(len(boxes)):
        yaw = 2*np.arccos(boxes[i].rotation[0])
        pred = str(boxes[i].score) + ' ' + \
               str(boxes[i].center_x) + ' ' + \
               str(boxes[i].center_y) + ' ' + \
               str(boxes[i].center_z) + ' ' + \
               str(boxes[i].width) + ' ' + \
               str(boxes[i].length) + ' ' + \
               str(boxes[i].height) + ' ' + \
               str(yaw) + ' ' + \
               str(boxes[i].name) + ' '

        if boxes[i].sample_token in sub.keys():
            sub[boxes[i].sample_token] += pred
        else:
            sub[boxes[i].sample_token] = pred

    sub = pd.DataFrame(list(sub.items()))
    sub.columns = ['Id','PredictionString']
    sub.to_csv('lyft3d_pred.csv',index=False)

def adjustment_box(offsets,a_box,token,class_ind,score):
    a_box_diag = np.sqrt(a_box.size[0]**2 + a_box.size[1]**2)
    box_x = a_box.center[0] + offsets[0] * a_box_diag
    box_y = a_box.center[1] + offsets[1] * a_box_diag
    box_z = a_box.center[2] + offsets[2] * a_box_diag

    box_w = np.exp(offsets[3])*a_box.wlh[0]
    box_l = np.exp(offsets[4])*a_box.wlh[1]
    box_h = np.exp(offsets[5])*a_box.wlh[2]

    box_name = cfg.DATA.IND_TO_NAME[class_ind]

    box_yaw = np.arcsin(offsets[6]) + a_box.orientation.yaw_pitch_roll[0]
    if offsets[7] > offsets[8]:
        box_ort = 1
    else:
        box_ort = -1
    
    box_yaw *= box_ort
    box = Box(sample_token=token,
                center=[box_x,box_y,box_z],
                size = [box_w,box_l,box_h],
                orientation=Quaternion([box_yaw,0,0]),
                label=class_ind,
                score=score,
                token=token)
    return box

def extract_boxes(cls_tensor,reg_tensor,tokens,anchor_boxes,l5d):
    
    box_list = []
    
    pred_classes = np.argmax(cls_tensor,axis=-1)
    pred_classes = pred_classes.reshape(pred_classes.shape[0],-1)
    pos_boxes = np.where(pred_classes != 0)
    reg_tensor = reg_tensor.reshape(reg_tensor.shape[0],-1,reg_tensor.shape[-1])
    cls_tensor = cls_tensor.reshape(cls_tensor.shape[0],-1,cls_tensor.shape[-1])

    batches = pos_boxes[0]
    anchors = pos_boxes[1]
    for i in range(len(batches)):
        token = tokens[batches[i]]
        offsets = reg_tensor[anchors[i]]
        a_box = anchor_boxes[anchors[i]]
        class_ind = int(pred_classes[batches[i],anchors[i]])
        score = cls_tensor[batches[i],anchors[i],class_ind]    
        box_list.append(adjustment_box(offsets,a_box,token,class_ind,score))

    return box_list

def move_boxes_to_world_space(pred_boxes,l5d):
    
    for box in pred_boxes:
        
        token = box.token
        sample = l5d.get('sample',token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = l5d.get("sample_data", sample_lidar_token)
        ego_pose = l5d.get("ego_pose", lidar_data["ego_pose_token"])
        translation = np.array(ego_pose['translation'])
        rotation = Quaternion(ego_pose['rotation'])
        box.translate(translation)
        box.rotate(rotation)


def make_predictions(model,dataloader,anchor_boxes,l5d):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    progress_bar = tqdm(dataloader)
    pred_boxes = []
    for (pillars,inds,tokens) in progress_bar:
        pillars = pillars.to(device)
        cls_tensor,reg_tensor = model(pillars,inds)
        c_sh = cls_tensor.size()
        r_sh = reg_tensor.size()

        cls_tensor = cls_tensor.view(c_sh[0],cfg.DATA.NUM_CLASSES,
                                     cfg.DATA.NUM_ANCHORS,c_sh[2],c_sh[3])
        reg_tensor = reg_tensor.view(r_sh[0],cfg.DATA.REG_DIMS,cfg.DATA.NUM_ANCHORS,
                                     r_sh[2],r_sh[3])
        cls_tensor = cls_tensor.permute(0,3,4,2,1)
        reg_tensor = reg_tensor.permute(0,3,4,2,1)
        cls_tensor = F.softmax(cls_tensor,dim=4)
        reg_tensor[...,7:9] = F.softmax(reg_tensor[...,7:9])
        cls_tensor = cls_tensor.detach().cpu().numpy()
        reg_tensor = reg_tensor.detach().cpu().numpy()
        batch_boxes = extract_boxes(cls_tensor,reg_tensor,tokens,anchor_boxes,l5d)
        pred_boxes.extend(batch_boxes)

    move_boxes_to_world_space(pred_boxes,l5d)

    write_submission(pred_boxes)


if __name__ == '__main__':

    data_path = cfg.DATA.DATA_PATH
    json_path = cfg.DATA.TEST_JSON_PATH
    model_to_load = cfg.NET.MODEL_TO_LOAD

    l5d = LyftDataset(data_path = data_path, json_path = json_path,
                      verbose = True)
    
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

    token_list = make_token_list(df,l5d)
    anchor_boxes,anchor_corners,anchor_centers = make_anchor_boxes()
    pp_dataset = PPDataset(token_list,l5d,anchor_boxes,
                           anchor_corners,anchor_centers,training=False)
    
    batch_size = cfg.NET.BATCH_SIZE
    num_workers = cfg.NET.NUM_WORKERS
    pp_dataloader = torch.utils.data.Dataloader(pp_dataset,batch_size,shuffle=False,
                                             num_workers=num_workers)
    fn_in = cfg.NET.FEATURE_NET_IN
    fn_out = cfg.NET.FEATURE_NET_OUT
    cls_channels = len(cfg.DATA.ANCHOR_DIMS)*(cfg.DATA.NUM_CLASSES+1)
    reg_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.REG_DIMS

    pp_model = model.PPModel(fn_in,fn_out,cls_channels,reg_channels)
    pp_model.load_state_dict(torch.load(model_to_load))

    make_predictions(pp_model,pp_dataloader,anchor_boxes,l5d)




