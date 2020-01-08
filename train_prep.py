import pandas as pd
from datetime import datetime
import os.path as osp
import pickle
from tqdm import tqdm
from utils.box_utils import make_anchor_boxes
from config import cfg
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from utils.box_utils import move_boxes_to_canvas_space

"""
    this script prepares the data for training and validation. For each sample we get the 
    ground boxes and the ego pose and calibrated sensor objects needed to transform that sample's
    lidar points to the correct reference frame. This reduces the amount of disk reads we need to 
    do during training. It also greatly reduces the memory footprint of the training process. The 
    LyftDataset object used to retreive the data from disk takes up a large amount of RAM. If we want
    to use multiple dataloader threads to read the training data we cannot store the LyftDataset object
    in memory since it will be copied across each thread. Pre-fetching the necessary data speeds up the
    training and reduces memory.

    We create a `data_dict` object which is a dictionary using the sample tokens as keys. The values are 
    the filepath to the sample's lidar data and the ego and calibrated sensor objects needed to transform
    the lidar data.

    data_dict[token] = {'lidar_fp': lidar_filepath, 'ego_pose': ego_pose_object,
                        'cal_sensor': calibrated_sensor_object, 'boxes': list_of_gt_boxes,
                        'prev_token': previous_sample_token} 

    We also create and serialize all the anchor box data need in the model
"""


data_path = cfg.DATA.DATA_PATH
json_path = cfg.DATA.TRAIN_JSON_PATH
l5d = LyftDataset(data_path=data_path,json_path=json_path,verbose=True)

# loop through each scene in the dataset
entries = []
for scene in l5d.scene:
    token = scene['first_sample_token']
    name = scene['name']
    host = "-".join(name.split("-")[:2])
    entries.append((host,token))

# split into training and validation 
df = pd.DataFrame(entries,columns=['host_name','first_sample_token'])

val_hosts = ["host-a007", "host-a008", "host-a009"] 
val_df    = df[df['host_name'].isin(val_hosts)]
val_ind   = val_df.index
train_df  = df[~df.index.isin(val_ind)]

dfs = [train_df,val_df]
box_dirs = [cfg.DATA.BOX_TRAIN_DIR,cfg.DATA.BOX_VAL_DIR]
lidar_dirs = [cfg.DATA.LIDAR_TRAIN_DIR,cfg.DATA.LIDAR_VAL_DIR]
token_dirs = [cfg.DATA.TOKEN_TRAIN_DIR,cfg.DATA.TOKEN_VAL_DIR]


# loop through the training and validation data frames
for df,box_dir,lidar_dir,token_dir in zip(dfs,box_dirs,lidar_dirs,token_dirs):
    data_dict  = {}
    token_list = []
    # loop through each sample in a scene
    for token in tqdm(df.first_sample_token):
        while token:
            sample = l5d.get('sample',token)
            lidar_token = sample['data']['LIDAR_TOP']
            try:
                # there may be currupted lidar files - mark these tokens 
                # by setting the file path to the lidar file as None
                lidar_filepath   = l5d.get_sample_data_path(lidar_token)
                lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            except Exception as e:
                print('Failed to load LIDAR cloud for {}: {}:'.format(token,e))
                prev_token = sample['prev']
                data_dict[token] = {'lidar_fp':None,
                                    'ego_pose':None,
                                    'cal_sensor':None,
                                    'boxes':None,
                                    'prev_token':prev_token}
                token = sample['next']
                continue
            # get the lidar, ego and sensor objects for the token
            lidar_data      = l5d.get('sample_data',lidar_token)
            ego_pose        = l5d.get('ego_pose',lidar_data['ego_pose_token'])
            cal_sensor      = l5d.get('calibrated_sensor',lidar_data['calibrated_sensor_token'])
            car_from_sensor = transform_matrix(cal_sensor['translation'],
                                               Quaternion(cal_sensor['rotation']),
                                               inverse=False)
            lidar_pointcloud.transform(car_from_sensor)
            lidar_points = lidar_pointcloud.points[:3,:]
            boxes      = l5d.get_boxes(lidar_token)
            # collect the ground truth boxes
            canv_boxes = move_boxes_to_canvas_space(boxes,ego_pose,lidar_points)
            boxes_fp   = osp.join(box_dir,token + '_boxes.pkl')
            pickle.dump(canv_boxes,open(boxes_fp,'wb'))
            prev_token = sample['prev']
            data_dict[token] = {'lidar_fp':str(lidar_filepath),
                                'ego_pose':ego_pose,
                                'cal_sensor':cal_sensor,
                                'boxes':boxes_fp,
                                'prev_token':prev_token}
            token_list.append(token)
            token = sample['next']
    
    ddfp = osp.join(lidar_dir,'data_dict.pkl')
    tkfp = osp.join(token_dir,'token_list.pkl')
    pickle.dump(data_dict,open(ddfp,'wb'))
    pickle.dump(token_list,open(tkfp,'wb'))

# create anchor boxes
anchor_boxes,anchor_corners,anchor_centers,anchor_xy = make_anchor_boxes()
a_dir = cfg.DATA.ANCHOR_DIR
pickle.dump(anchor_boxes,open(osp.join(a_dir,'anchor_boxes.pkl'),'wb'))
pickle.dump(anchor_corners,open(osp.join(a_dir,'anchor_corners.pkl'),'wb'))
pickle.dump(anchor_centers,open(osp.join(a_dir,'anchor_centers.pkl'),'wb'))
pickle.dump(anchor_xy,open(osp.join(a_dir,'anchor_xy.pkl'),'wb'))


