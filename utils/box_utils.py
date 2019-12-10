import numpy as np
import pandas as pd
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

    class_ind = cfg.DATA.NAME_TO_IND[gt_box.name]
    if gt > 0:
        ort = 1
    else:
        ort = -1

    return [1,dx,dy,dz,dw,dl,dh,dt,ort,class_ind]

def make_anchor_boxes():
    
    fm_height = cfg.DATA.FM_HEIGHT
    fm_width = cfg.DATA.FM_WIDTH
    fm_scale = cfg.DATA.FM_SCALE
    anchor_dims = cfg.DATA.ANCHOR_DIMS
    anchor_yaws = cfg.DATA.ANCHOR_YAWS
    anchor_zs = cfg.DATA.ANCHOR_ZS

    corners_list = []
    boxes_list = []
    centers_list = []
    for y in tqdm(range(0,fm_height)):
        for x in range(0,fm_width):
            for d in range(0,len(anchor_dims)):
                x_center = (x + 0.5)/fm_scale
                y_center = (y + 0.5)/fm_scale
                z_center = anchor_zs[d]
                width = anchor_dims[d][0]
                length = anchor_dims[d][1]
                height = anchor_dims[d][2]
                yaw = anchor_yaws[d]
                quat = Quaternion(axis=[0,0,1],degrees = yaw)
                box = Box(center=[x_center,y_center,z_center],size=[width,length,height],
                          orientation=quat)
                boxes_list.append(box)
                corners_list.append(box.bottom_corners()[:2,:].transpose([1,0]))
                centers_list.append([x_center,y_center,z_center])
    return boxes_list,np.array(corners_list),np.array(centers_list)

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
    t_max_ious = np.max(ious.transpose([1,0]),axis=1)
    t_argmax_ious = np.argmax(ious.transpose([1,0]),axis=1)
    targets = np.zeros((len(anchor_boxes),10))   
 
    max_ious = np.max(ious,axis=1)
    arg_max_ious = np.argmax(ious,axis=1)
    pos_anchors = np.where(max_ious > pos_thresh)
    neg_anchors = np.where(max_ious < neg_thresh)
    pos_boxes = arg_max_ious[pos_anchors]
    
    ious = ious.transpose([1,0])
    top_anchor_for_box = np.argmax(ious,axis=1)

    targets[neg_anchors[0],0] = -1

    for i,anch in enumerate(pos_anchors[0]):
        anchor = anchor_box_list[anch]
        matching_gt_ind = pos_boxes[i]
        matching_box = gt_box_list[matching_gt_ind]
        targets[anch,:] = make_target(anchor,matching_box)

    for i,anch in enumerate(top_anchor_for_box):
        targets[anch,:] = make_target(anchor_box_list[anch],gt_box_list[i])
    
    #print(np.mean(targets))
    return targets


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