import numpy as np
import pandas as pd
import pdb
import torch
import pickle
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

def boxes_to_image_space(boxes):
    """
    takes a list of boxes in canvas space and returns the 
    centers and corners in image space. This is done by
    flipping the canvas across the midpoint on the y-axis
    """

    centers = np.stack([box.center.copy() for box in boxes])
    corners = np.stack([box.bottom_corners().transpose([1,0])[:,:2] for box in boxes])

    centers[...,1] = (cfg.DATA.CANVAS_HEIGHT - 1) - centers[...,1]
    corners[...,1] = (cfg.DATA.CANVAS_HEIGHT - 1) - corners[...,1]

    return centers,corners


def create_pillars_py(lidar_pts,max_pts_per_pillar,
                      max_pillars,x_step,y_step,
                      x_min,y_min,z_min,
                      x_max,y_max,z_max):


    """
    function to create pillars in python. This was only used
    to compare speed vs python C extension.
    """

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

def make_target(anchor_box,gt_box,anch):

    """
    makes a regression target for an anchor box and gt box
    """
    ax,ay,az = anchor_box.center
    gx,gy,gz = gt_box.center
    aw,al,ah = anchor_box.wlh
    gw,gl,gh = gt_box.wlh
    ad = np.sqrt(aw**2 + al**2)
    at = anchor_box.orientation.yaw_pitch_roll[0]
    gt = gt_box.orientation.yaw_pitch_roll[0]

    gy = (cfg.DATA.CANVAS_HEIGHT - 1) - gy
    dx = (gx - ax)/ad
    dy = (gy - ay)/ad
    dz = (gz - az)/ah

    dw = np.log(gw/aw)
    dl = np.log(gl/al)
    dh = np.log(gh/ah)

    dt = np.sin(gt - at)

    if (gt >= np.pi/2 and gt <= np.pi) or (gt >= (3*np.pi)/2 and gt <= 2*np.pi):
        ort = 1
    else:
        ort = 0

    return [1,dx,dy,dz,dw,dl,dh,dt,ort]

def make_anchor_boxes():
    
    """
    constructs the anchor boxes for the model, returns a list of
    Box objects, an array of xy coordinates of the corners of shape
    [N,4,2], an array of the centers of shape [N,3] and an array of
    the boxes in (x1,y1,x2,y2) format where (x1,y1) are the top left
    corner and (x2,y2) the bottom right
    """ 

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

    # loop through the height,width and dimensions of the boxes
    for y in tqdm(range(0,fm_height)):
        for x in range(0,fm_width):
            for d in range(0,len(anchor_dims)):
                x_center = (x + 0.5)/fm_scale
                y_center = (y + 0.5)/fm_scale
                # the wlh, z coord and yaw of the anchors are set in config
                z_center = anchor_zs[d]
                width  = anchor_dims[d][0]
                length = anchor_dims[d][1]
                height = anchor_dims[d][2]
                yaw  = anchor_yaws[d]
                quat = Quaternion(axis=[0,0,1],degrees = yaw)
                # create new box object
                box  = Box(center=[x_center,y_center,z_center],size=[width,length,height],
                          orientation=quat)
                boxes_list.append(box)
                # take only the xy coords of the corners
                bc = box.bottom_corners().transpose([1,0])
                corners_list.append(bc[:,:2])
                centers_list.append([x_center,y_center,z_center])
                if yaw > 0:
                    xy_list.append(np.concatenate((bc[1,:2],bc[3,:2])))
                else:
                    xy_list.append(np.concatenate((bc[2,:2],bc[0,:2]))) 
                
    return boxes_list,np.array(corners_list),np.array(centers_list),np.array(xy_list)


def create_target(anchor_corners,
                  gt_corners,
                  anchor_centers,
                  gt_centers,
                  anchor_box_list,
                  gt_box_list):
   
    """
        creates the classification and regression targets for a
        single training sample.

        anchor boxes with a threshold iou against a ground truth box
        are matched together. For positive anchor boxes a binary target
        is created for classification loss, and the bounding box regression
        targets are created.
    """
    pos_thresh = cfg.DATA.IOU_POS_THRESH

    # compute ious
    ious = np.zeros((len(anchor_box_list),len(gt_box_list)))
    pillars.make_ious(anchor_corners,gt_corners,
                   anchor_centers,gt_centers,ious)
    
    cls_targets = np.zeros((len(anchor_box_list),cfg.DATA.NUM_CLASSES))   
    reg_targets = np.zeros((len(anchor_box_list),cfg.DATA.REG_DIMS))
    
    # get the classes for each ground truth box
    gt_box_classes = np.array([cfg.DATA.NAME_TO_IND[box.name] for box in gt_box_list],dtype=np.int32)
    
    # positive anchors are defined as anchor boxes where the max
    # iou with a ground truth box is above the threshold
    max_ious = np.max(ious,axis=1)
    arg_max_ious = np.argmax(ious,axis=1)
    pos_anchors = np.where(max_ious > pos_thresh)[0]
    pos_boxes = arg_max_ious[pos_anchors]
    
    # we also compute the highest iou anchor box for each ground truth box
    ious = ious.transpose([1,0])
    top_anchor_for_box = np.argmax(ious,axis=1)
    
    # it is possible that there are no positive anchors for a gt box, filter
    # these out
    filter_inds = np.nonzero(top_anchor_for_box)
    top_anchor_for_box = top_anchor_for_box[filter_inds]
    
    # make binary labels, first the positive boxes are labeled accrording
    # to the class of the highest matching ground truth box. These labels
    # are then potentially overwritten by labeling the anchor box with 
    # highest iou of each ground truth box
    cls_targets[pos_anchors,gt_box_classes[pos_boxes]] = 1
    cls_targets[top_anchor_for_box,:] = 0
    cls_targets[top_anchor_for_box,gt_box_classes[filter_inds]] = 1 
    #cls_targets[top_anchor_for_box,gt_box_classes] = 1

    # regression targets are made in the the same manner, first by positive
    # anchors, then by highest iou anchor box for each ground truth box
    #anch_dict = {}
    for i,anch in enumerate(pos_anchors):
        reg_targets[anch,:] = make_target(anchor_box_list[anch],
                                          gt_box_list[pos_boxes[i]],anch)
        #anch_dict[anch] = pos_boxes[i]

    matched_boxes = [gt_box_list[i] for i in filter_inds[0]]
    #matched_boxes = gt_box_list
    for i,anch in enumerate(top_anchor_for_box):
        reg_targets[anch,:] = make_target(anchor_box_list[anch],
                                          matched_boxes[i],anch)
        #anch_dict[anch] = i
    
    #pickle.dump(anch_dict,open('anch_dict.pkl','wb'))
    return cls_targets,reg_targets



def move_boxes_to_canvas_space(boxes,ego_pose):

    """
        takes a list of ground truth boxes in global space
        and moves them to canvas space. We first move the 
        boxes to the ego car coordinate system, then we 
        move the boxes to the voxelized canvas space
    """
    box_list = []
    x_min = cfg.DATA.X_MIN
    x_max = cfg.DATA.X_MAX
    y_min = cfg.DATA.Y_MIN
    y_max = cfg.DATA.Y_MAX
    z_min = cfg.DATA.Z_MIN
    z_max = cfg.DATA.Z_MAX
    x_step = cfg.DATA.X_STEP
    y_step = cfg.DATA.Y_STEP

    box_translation = -np.array(ego_pose['translation'])
    box_rotation = Quaternion(ego_pose['rotation']).inverse

    for box in boxes:
        #transform to car space
        box.translate(box_translation)
        box.rotate(box_rotation)

        # ignore boxes that are outside of the bounds
        # of the lidar point cloud
        box_x,box_y,box_z = box.center
        if (box_x < x_min) or (box_x > x_max) or \
           (box_y < y_min) or (box_y > y_max) or \
           (box_z < z_min) or (box_z > z_max): continue
    
        # compute new xy coordinates in canvas space
        canv_x = (box_x - cfg.DATA.X_MIN)/x_step
        canv_y = (box_y - cfg.DATA.Y_MIN)/y_step
        canv_z = box_z

        # adjust wlh
        box_w,box_l,box_h = box.wlh
        canv_w = box_w/y_step
        canv_l = box_l/x_step
        canv_h = box_h

        box_wlh    = np.array([canv_w,canv_l,canv_h])
        box_center = np.array([canv_x,canv_y,canv_z])
        
        box_list.append(Box(box_center,
                            box_wlh,
                            label=box.label,
                            orientation=Quaternion(list(box.orientation)),
                            name=box.name,
                            token=box.token))

    return box_list
