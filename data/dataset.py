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
        self.num_sweeps = 3

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self,ind):
        
        """
        This function takes an index into the list of tokens and creates the pillar tensor, 
        scatter tensor and training targets for the model. 

        self.num_sweeps lidar point clouds are aggerated and transformed to the current 
        reference coordinates. The resulting point cloud is then used to create the pillar
        and scatter tensors (pillars, indices).
        """

        token = self.tokens[ind]
        # get the reference pose for the sample at token and create transformation matrix from the global
        # coordinate system to the reference car's coordinate system
        ref_pose   = self.data_dict[token]['ego_pose']
        ref_car_from_global = transform_matrix(ref_pose['translation'],Quaternion(ref_pose['rotation']),
                                               inverse=True)
        agg_pc = np.zeros((4,0))
        curr_token = token

        # aggregate num_sweeps previous lidars and move them to the reference car's coordinate system 
        for _ in range(self.num_sweeps):
            
            if not self.data_dict[curr_token]['lidar_fp']:
                curr_token = self.data_dict[curr_token]['prev_token']
                continue 
            # get the current point cloud for curr_token
            lidar_fp = pathlib.Path(self.data_dict[curr_token]['lidar_fp'])
            curr_pc  = LidarPointCloud.from_file(lidar_fp)
            curr_pose = self.data_dict[curr_token]['ego_pose']
            curr_sensor = self.data_dict[curr_token]['cal_sensor']

            # create transformation matrices from current sensor to current car and from
            # current car to global
            global_from_curr_car = transform_matrix(curr_pose['translation'],Quaternion(curr_pose['rotation']),
                                                     inverse=False)
            curr_car_from_curr_sensor = transform_matrix(curr_sensor['translation'],
                                                        Quaternion(curr_sensor['rotation']),
                                                        inverse=False)
            # combine transformation matrices into a single matrix that moves the point cloud 
            # from current sensor -> current car -> global -> reference car
            transmat = reduce(np.dot,[ref_car_from_global,global_from_curr_car,curr_car_from_curr_sensor])
            curr_pc.transform(transmat)
            curr_pc.remove_close(self.min_dist)
            # aggreate the resulting pointcloud 
            agg_pc = np.hstack((agg_pc,curr_pc.points))
            curr_token = self.data_dict[curr_token]['prev_token']
            if not curr_token:
                break

        # create pillar tensor and scatter tensor
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
            # subtract the data mean from the pillar tensor
            pillar = pillar.reshape(-1) - self.data_mean
            pillar = pillar.reshape(pillar_size)
        indices = torch.from_numpy(indices).long()
        if self.training:
            # get the ground truth boxes of the token 
            box_fp = self.data_dict[token]['boxes']
            boxes = pickle.load(open(box_fp,'rb'))
            gt_centers,gt_corners = boxes_to_image_space(boxes)

            # create training labels
            c_target,r_target = create_target(self.anchor_corners,gt_corners,
                                    self.anchor_centers,gt_centers,
                                    self.anchor_boxes,boxes)
            c_target = torch.from_numpy(c_target).float()
            r_target = torch.from_numpy(r_target).float()
            gc.collect()
            return (pillar,indices,c_target,r_target)
        
        return (pillar,indices)


