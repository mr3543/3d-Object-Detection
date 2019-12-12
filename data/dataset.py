import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import data.pillars as pillars
import pickle
from config import cfg
from utils.box_utils import move_boxes_to_canvas_space,create_target
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

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
        lidar_fp = self.lidars[ind]
        lidar_pointcloud = LidarPointCloud.from_file(lidar_fp)
        car_from_sensor = self.data_dict[lidar_fp]['trans_matrix']
        lidar_pointcloud.transform(car_from_sensor)

        if self.training:
            box_fp = self.data_dict[lidar_fp]['boxes']
            boxes = pickle.load(open(box_fp,'rb'))
            gt_corners = np.array([box.bottom_corners()[:2,:].transpose([1,0]) for box in boxes])
            gt_centers = np.array([box.center for box in boxes])
        
        lidar_points = lidar_pointcloud.points
        pillar = np.zeros((cfg.DATA.MAX_PILLARS,cfg.DATA.MAX_POINTS_PER_PILLAR,9))
        indices = np.zeros((cfg.DATA.MAX_PILLARS,3))
        
        pillars.create_pillars(lidar_points.transpose([1,0]),pillar,
                            indices,cfg.DATA.MAX_POINTS_PER_PILLAR,
                            cfg.DATA.MAX_PILLARS,cfg.DATA.X_STEP,cfg.DATA.Y_STEP,
                            cfg.DATA.X_MIN,cfg.DATA.Y_MIN,cfg.DATA.Z_MIN,
                            cfg.DATA.X_MAX,cfg.DATA.Y_MAX,cfg.DATA.Z_MAX,
                            cfg.DATA.CANVAS_HEIGHT)
        pillar = pillar.transpose([2,0,1])
        pillar_size = pillar.shape
        pillar = torch.from_numpy(pillar).float()
        if self.data_mean:
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
        
        return (pillar,indices,None,None)
