import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pickle
import pillars
from .. import config
from config import cfg
from data_prep import move_boxes_to_canvas_space,create_target
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

class PPDataset(torch.utils.data.dataset):
    def __init__(self,token_list,l5d,anchor_boxes,
                 anchor_corners,anchor_centers,training=True):
        self.token_list = token_list
        self.training = training
        self.l5d = l5d
        self.anchor_boxes = anchor_boxes
        self.anchor_corners = anchor_corners
        self.anchor_centers = anchor_centers

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self,ind):

        sample_token = self.token_list[ind]
        sample = self.l5d.get('sample',sample_token)
        sample_lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.l5d.get('sample_data',sample_lidar_token)

        ego_pose = self.l5d.get('ego_pose',lidar_data['ego_pose_token'])
        calibrated_sensor = self.l5d.get('calibrated_sensor',lidar_data['calibrated_sensor_token'])
        car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                           Quaternion(calibrated_sensor['rotation']),
                                           inverse = False)
        
        if self.training:
            boxes = self.l5d.get_boxes(sample_lidar_token)
            boxes = move_boxes_to_canvas_space(boxes,ego_pose)
            gt_corners = np.array([box.bottom_corners[:2,:].transpose([1,0]) for box in boxes])
            gt_centers = np.array([box.center for box in boxes])
        
        try:
            lidar_filepath = self.l5d.get_sample_data_path(sample_lidar_token)
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print('Failed to load LIDAR cloud for {}: {}:'.format(sample_token,e))
            return None
        
        lidar_points = lidar_pointcloud.points
        pillar = np.zeros((cfg.DATA.MAX_PILLARS,cfg.DATA.MAX_POINTS_PER_PILLAR,9))
        indices = np.zeros((cfg.DATA.MAX_PILLARS,3))

        pillars.create_pillars(lidar_points.transpose([1,0]),pillar,
                            indices,cfg.DATA.MAX_POINTS_PER_PILLAR,
                            cfg.DATA.MAX_PILLARS,cfg.DATA.X_STEP,cfg.DATA.Y_STEP,
                            cfg.DATA.X_MIN,cfg.DATA.Y_MIN,cfg.DATA.Z_MIN,
                            cfg.DATA.X_MAX,cfg.DATA.Y_MAX,cfg.DATA.Z_MAX,
                            cfg.DATA.CANVAS_HEIGHT)

        if self.training:
            target = create_target(self.anchor_cornres,gt_corners,
                                   self.anchor_centers,gt_centers,
                                   self.anchor_boxes,boxes)
            return (pillars,indices,target)
        
        return (pillars,indices,None)