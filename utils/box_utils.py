import numpy as np
import pandas as pd
import torch
import time
import os
import sys
from tqdm import tqdm
from config import cfg
import data.pillars as pillars
from datetime import datetime
from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from lyft_dataset_sdk.lyftdataset import LyftDataset

def create_pillars_py(lidar_pts,max_pts_per_pillar,
                      max_pillars,x_step,y_step,
                      x_min,y_min,z_min,
                      x_max,y_max,z_max):

    L = pd.DataFrame(lidar_pts,columns=['x','y','z','r'])
    invalid = (L['x'] < x_min) | (L['x'] > x_max) | (L['y'] < y_min) | (L['y'] > y_max) | \
              (L['z'] < z_min) | (L['z'] > z_max)
    L = L[~invalid]
    L['canv_x'] = np.floor((L['x'] - x_min)/x_step)
    L['canv_y'] = np.floor((L['y'] - y_min)/y_step)
    
    L['xp'],L['yp'] = L['x'] - L['canv_x'],L['y'] - L['canv_y']
    gb = L.groupby(['canv_x','canv_y'])
    means = gb.transform('mean')
    L['xc'],L['yc'],L['zc'] = L['x'] - means['x'], L['y'] - means['y'], L['z'] - means['z']
    features = ['x','y','z','r','xp','yp','xc','yc','zc']
    pillars = np.zeros((max_pillars,max_pts_per_pillar,len(features)))
    p_xy = np.zeros((max_pillars,3))

    for i,g in enumerate(gb.groups):
        grp_inds = gb.groups[g]
        to_sample = min(max_pts_per_pillar,len(grp_inds))
        inds = np.random.choice(grp_inds,to_sample)
        pillars[np.array([i]*to_sample),np.arange(to_sample),:] = L.loc[inds,features].values
        p_xy[i,0],p_xy[i,1],p_xy[i,2] = 1,g[0],g[1]
        
    return pillars,p_xy


def make_target_torch(anchor_box,gt_box):

    ax,ay,az = anchor_box.center
    gx,gy,gz = gt_box.center
    aw,al,ah = anchor_box.wlh
    gw,gl,gh = gt_box.wlh
    ad = np.sqrt(aw**2 + al**2)
    at = anchor_box.orientation.yaw_pitch_roll[0]
    gt = gt_box.orientation.yaw_pitch_roll[0]

    dx = (gx - ax)/ad
    dy = (gy - ay)/ad
    dz = (gz - az)/ah

    dw = np.log(gw/aw)
    dl = np.log(gl/al)
    dh = np.log(gh/ah)

    dt = np.sin(gt - at)

    if gt > 0:
        ort = 1
    else:
        ort = 0
    
    return torch.Tensor([1,dx,dy,dz,dw,dl,dh,dt,ort])



def make_target(anchor_box,gt_box):

    ax,ay,az = anchor_box.center
    gx,gy,gz = gt_box.center
    aw,al,ah = anchor_box.wlh
    gw,gl,gh = gt_box.wlh
    ad = np.sqrt(aw**2 + al**2)
    at = anchor_box.orientation.yaw_pitch_roll[0]
    gt = gt_box.orientation.yaw_pitch_roll[0]

    dx = (gx - ax)/ad
    dy = (gy - ay)/ad
    dz = (gz - az)/ah

    dw = np.log(gw/aw)
    dl = np.log(gl/al)
    dh = np.log(gh/ah)

    dt = np.sin(gt - at)

    if gt > 0:
        ort = 1
    else:
        ort = 0

    return [1,dx,dy,dz,dw,dl,dh,dt,ort]

def make_anchor_boxes():
    
    fm_height   = cfg.DATA.FM_HEIGHT
    fm_width    = cfg.DATA.FM_WIDTH
    fm_scale    = cfg.DATA.FM_SCALE
    anchor_dims = cfg.DATA.ANCHOR_DIMS
    anchor_yaws = cfg.DATA.ANCHOR_YAWS
    anchor_zs   = cfg.DATA.ANCHOR_ZS

    corners_list = []
    boxes_list   = []
    centers_list = []
    xy_list      = []
    for y in tqdm(range(0,fm_height)):
        for x in range(0,fm_width):
            for d in range(0,len(anchor_dims)):
                x_center = (x + 0.5)/fm_scale
                y_center = (y + 0.5)/fm_scale
                z_center = anchor_zs[d]
                width  = anchor_dims[d][0]
                length = anchor_dims[d][1]
                height = anchor_dims[d][2]
                yaw  = anchor_yaws[d]
                quat = Quaternion(axis=[0,0,1],degrees = yaw)
                box  = Box(center=[x_center,y_center,z_center],size=[width,length,height],
                          orientation=quat)
                boxes_list.append(box)
                bc = box.bottom_corners().transpose([1,0])
                corners_list.append(bc[:,:2])
                centers_list.append([x_center,y_center,z_center])
                if yaw > 0:
                    xy_list.append(np.concatenate((bc[1,:2],bc[3,:2])))
                else:
                    xy_list.append(np.concatenate((bc[2,:2],bc[0,:2]))) 
                
    return boxes_list,np.array(corners_list),np.array(centers_list),np.array(xy_list)

def create_target_torch(anchor_corners,
                  gt_corners,
                  anchor_centers,
                  gt_centers,
                  anchor_box_list,
                  gt_box_list,
                  device):
   
    pos_thresh = cfg.DATA.IOU_POS_THRESH
    
    ious = np.zeros((len(anchor_box_list),len(gt_box_list)))
    pillars.make_ious(anchor_corners,gt_corners,
                   anchor_centers,gt_centers,ious)
   
    ious = torch.from_numpy(ious).to(device) 
    cls_targets = torch.zeros((len(anchor_box_list),cfg.DATA.NUM_CLASSES)).to(device)
    reg_targets = torch.zeros((len(anchor_box_list),cfg.DATA.REG_DIMS)).to(device)
    
    gt_box_classes = torch.from_numpy(np.array([cfg.DATA.NAME_TO_IND[box.name] for box in gt_box_list])).long()
    
    max_ious,arg_max_ious = torch.max(ious,dim=1)
    pos_anchors = torch.where(max_ious > pos_thresh)[0]
    pos_boxes = arg_max_ious[pos_anchors]
    
    ious = ious.permute([1,0])
    top_anchor_for_box = torch.max(ious,dim=1)[1]
    
    cls_targets[pos_anchors,gt_box_classes[pos_boxes]] = 1
    cls_targets[top_anchor_for_box,:] = 0
    cls_targets[top_anchor_for_box,gt_box_classes] = 1 

    for i,anch in enumerate(pos_anchors):
        reg_targets[anch,:] = make_target_torch(anchor_box_list[anch],
                                          gt_box_list[pos_boxes[i]]).to(device)

    for i,anch in enumerate(top_anchor_for_box):
        reg_targets[anch,:] = make_target_torch(anchor_box_list[anch],
                                          gt_box_list[i]).to(device)

    return cls_targets,reg_targets


def create_target(anchor_corners,
                  gt_corners,
                  anchor_centers,
                  gt_centers,
                  anchor_box_list,
                  gt_box_list):
   
    pos_thresh = cfg.DATA.IOU_POS_THRESH
    neg_thresh = cfg.DATA.IOU_NEG_THRESH
    
    ious = np.zeros((len(anchor_box_list),len(gt_box_list)))
    pillars.make_ious(anchor_corners,gt_corners,
                   anchor_centers,gt_centers,ious)
    
    cls_targets = np.zeros((len(anchor_box_list),cfg.DATA.NUM_CLASSES))   
    reg_targets = np.zeros((len(anchor_box_list),cfg.DATA.REG_DIMS))
    
    gt_box_classes = np.array([cfg.DATA.NAME_TO_IND[box.name] for box in gt_box_list],dtype=np.int32)
    
    max_ious = np.max(ious,axis=1)
    arg_max_ious = np.argmax(ious,axis=1)
    pos_anchors = np.where(max_ious > pos_thresh)[0]
    pos_boxes = arg_max_ious[pos_anchors]
    
    ious = ious.transpose([1,0])
    top_anchor_for_box = np.argmax(ious,axis=1)
    
    cls_targets[pos_anchors,gt_box_classes[pos_boxes]] = 1
    cls_targets[top_anchor_for_box,:] = 0
    cls_targets[top_anchor_for_box,gt_box_classes] = 1 

    for i,anch in enumerate(pos_anchors):
        reg_targets[anch,:] = make_target(anchor_box_list[anch],
                                          gt_box_list[pos_boxes[i]])

    for i,anch in enumerate(top_anchor_for_box):
        reg_targets[anch,:] = make_target(anchor_box_list[anch],
                                          gt_box_list[i])

    return cls_targets,reg_targets



def move_boxes_to_canvas_space(boxes,ego_pose):

   # print('STARING MOVE BOXES')
    box_list = []
    x_min = cfg.DATA.X_MIN
    x_max = cfg.DATA.X_MAX
    y_min = cfg.DATA.Y_MIN
    y_max = cfg.DATA.Y_MAX
    z_min = cfg.DATA.Z_MIN
    z_max = cfg.DATA.Z_MAX
    x_step = cfg.DATA.X_STEP
    y_step = cfg.DATA.Y_STEP
    canvas_height = cfg.DATA.CANVAS_HEIGHT

    box_translation = -np.array(ego_pose['translation'])
    box_rotation = Quaternion(ego_pose['rotation']).inverse

    for box in boxes:
        #transform to car space
        box.translate(box_translation)
        box.rotate(box_rotation)

        box_x,box_y,box_z = box.center
        if (box_x < x_min) or (box_x > x_max) or \
           (box_y < y_min) or (box_y > y_max) or \
           (box_z < z_min) or (box_z > z_max): continue
    
        #print('ORIG BOX LOC: ',box.center)
        #print('ORIG BOX SIZE: ',box.wlh)
        # transform to canvas space
        box_x = (box_x - cfg.DATA.X_MIN)/x_step
        box_y = (box_y - cfg.DATA.Y_MIN)/y_step

        box_y = (canvas_height - 1) - box_y

        box_w,box_l,box_h = box.wlh
        box_w /= y_step
        box_l /= x_step

        box.wlh = np.array([box_w,box_l,box_h])
        box.center = np.array([box_x,box_y,box_z])
        
        #print('CANVAS BOX LOC: ',box.center)
        #print('CANVAS BOX SIZE: ',box.wlh)
        box_list.append(box)
    
   # print('RETURNING {} BOXES '.format(len(box_list)))
    return box_list


