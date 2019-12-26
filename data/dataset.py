import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import data.pillars as pillars
import pickle
import pathlib
from functools import reduce
import gc
import pdb
from config import cfg
from utils.box_utils import move_boxes_to_canvas_space,create_target,create_pillars_py,boxes_to_image_space
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

"""

class PPDataset(torch.utils.data.Dataset):
    def __init__(self,lidars,data_dict,anchor_boxes,
                 anchor_corners,anchor_centers,data_mean=None,training=True):
        super(PPDataset,self).__init__()
        
        self.data_dict = data_dict
        self.lidars = lidars
        self.training = training
        self.anchor_boxes = anchor_boxes
        self.anchor_corners = anchor_corners
        self.anchor_centers = anchor_centers
        self.data_mean = data_mean

    def __len__(self):
        return len(self.lidars)

    def __getitem__(self,ind):
        lidar_fp = pathlib.Path(self.lidars[ind])
        lidar_pointcloud = LidarPointCloud.from_file(lidar_fp)
        car_from_sensor = self.data_dict[str(lidar_fp)]['trans_matrix']
        lidar_pointcloud.transform(car_from_sensor)
        if self.training:
            box_fp = self.data_dict[str(lidar_fp)]['boxes']
            boxes = pickle.load(open(box_fp,'rb'))
            #gt_corners = np.array([box.bottom_corners()[:2,:].transpose([1,0]) for box in boxes])
            #gt_centers = np.array([box.center.copy() for box in boxes])
            gt_centers,gt_corners = boxes_to_image_space(boxes)
 
        lidar_points = lidar_pointcloud.points
        pillar = np.zeros((cfg.DATA.MAX_PILLARS,cfg.DATA.MAX_POINTS_PER_PILLAR,9))
        indices = np.zeros((cfg.DATA.MAX_PILLARS,3))
        
        pillars.create_pillars(lidar_points.transpose([1,0]),pillar,
                            indices,cfg.DATA.MAX_POINTS_PER_PILLAR,
                            cfg.DATA.MAX_PILLARS,cfg.DATA.X_STEP,cfg.DATA.Y_STEP,
                            cfg.DATA.X_MIN,cfg.DATA.Y_MIN,cfg.DATA.Z_MIN,
                            cfg.DATA.X_MAX,cfg.DATA.Y_MAX,cfg.DATA.Z_MAX,
                            cfg.DATA.CANVAS_HEIGHT)
        #pillar,indices = create_pillars_py(lidar_points.transpose([1,0]),cfg.DATA.MAX_POINTS_PER_PILLAR,
        #                                    cfg.DATA.MAX_PILLARS,cfg.DATA.X_STEP,cfg.DATA.Y_STEP,
        #                                    cfg.DATA.X_MIN,cfg.DATA.Y_MIN,cfg.DATA.Z_MIN,
        #                                    cfg.DATA.X_MAX,cfg.DATA.Y_MAX,cfg.DATA.Z_MAX)

        pillar = pillar.transpose([2,0,1])
        pillar_size = pillar.shape
        pillar = torch.from_numpy(pillar).float()
        if self.data_mean is not None:
            pillar = pillar.reshape(-1) - self.data_mean
            pillar = pillar.reshape(pillar_size)
        indices = torch.from_numpy(indices).long()
        if self.training:
            c_target,r_target = create_target(self.anchor_corners,gt_corners,
                                    self.anchor_centers,gt_centers,
                                    self.anchor_boxes,boxes)
            c_target = torch.from_numpy(c_target).float()
            r_target = torch.from_numpy(r_target).float()
            return (pillar,indices,c_target,r_target)
        
        return (pillar,indices)
"""

class PPDataset(torch.utils.data.Dataset):
    def __init__(self,tokens,data_dict,anchor_boxes,
                 anchor_corners,anchor_centers,data_mean=None,training=True):
        super(PPDataset,self).__init__()
        
        self.data_dict = data_dict
        self.tokens = tokens
        self.training = training
        self.anchor_boxes = anchor_boxes
        self.anchor_corners = anchor_corners
        self.anchor_centers = anchor_centers
        self.data_mean = data_mean
        self.min_dist = 0.001
        self.num_sweeps = 10

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self,ind):

        token = self.tokens[ind]
        ref_sensor = self.data_dict[token]['cal_sensor']
        ref_pose   = self.data_dict[token]['ego_pose']
        ref_car_from_global = transform_matrix(ref_pose['translation'],Quaternion(ref_pose['rotation']),
                                               inverse=True)
        agg_pc = np.zeros((4,0))
        curr_token = token
        for _ in range(self.num_sweeps):
            
            if not self.data_dict[curr_token]['lidar_fp']:
                curr_token = self.data_dict[curr_token]['prev_token']
                continue 
            lidar_fp = pathlib.Path(self.data_dict[curr_token]['lidar_fp'])
            curr_pc  = LidarPointCloud.from_file(lidar_fp)
            curr_pose = self.data_dict[curr_token]['ego_pose']
            curr_sensor = self.data_dict[curr_token]['cal_sensor']
            global_from_curr_car = transform_matrix(curr_pose['translation'],Quaternion(curr_pose['rotation']),
                                                     inverse=False)
            curr_car_from_curr_sensor = transform_matrix(curr_sensor['translation'],
                                                        Quaternion(curr_sensor['rotation']),
                                                        inverse=False)
            transmat = reduce(np.dot,[ref_car_from_global,global_from_curr_car,curr_car_from_curr_sensor])
            curr_pc.transform(transmat)
            curr_pc.remove_close(self.min_dist)
            agg_pc = np.hstack((agg_pc,curr_pc.points))
            curr_token = self.data_dict[curr_token]['prev_token']
            if not curr_token:
                break

        if self.training:
            box_fp = self.data_dict[token]['boxes']
            boxes = pickle.load(open(box_fp,'rb'))
            gt_centers,gt_corners = boxes_to_image_space(boxes)
 
        lidar_points = agg_pc.transpose([1,0])
        pillar = np.zeros((cfg.DATA.MAX_PILLARS,cfg.DATA.MAX_POINTS_PER_PILLAR,9))
        indices = np.zeros((cfg.DATA.MAX_PILLARS,3))
        
        pillars.create_pillars(lidar_points,pillar,
                            indices,cfg.DATA.MAX_POINTS_PER_PILLAR,
                            cfg.DATA.MAX_PILLARS,cfg.DATA.X_STEP,cfg.DATA.Y_STEP,
                            cfg.DATA.X_MIN,cfg.DATA.Y_MIN,cfg.DATA.Z_MIN,
                            cfg.DATA.X_MAX,cfg.DATA.Y_MAX,cfg.DATA.Z_MAX,
                            cfg.DATA.CANVAS_HEIGHT)

        pillar = pillar.transpose([2,0,1])
        pillar_size = pillar.shape
        pillar = torch.from_numpy(pillar).float()
        if self.data_mean is not None:
            pillar = pillar.reshape(-1) - self.data_mean
            pillar = pillar.reshape(pillar_size)
        indices = torch.from_numpy(indices).long()
        if self.training:
            c_target,r_target = create_target(self.anchor_corners,gt_corners,
                                    self.anchor_centers,gt_centers,
                                    self.anchor_boxes,boxes)
            c_target = torch.from_numpy(c_target).float()
            r_target = torch.from_numpy(r_target).float()
            gc.collect()
            return (pillar,indices,c_target,r_target)
        
        return (pillar,indices)


