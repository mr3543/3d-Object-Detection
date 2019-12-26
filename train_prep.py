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
def make_token_list(df,l5d):
    first_samples = df.first_sample_token.values
    token_list = []
    for token in first_samples:
        while token:
            token_list.append(token)
            sample = l5d.get('sample',token)
            token = sample['next']
    return token_list


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

token_lists = [make_token_list(train_df,l5d),make_token_list(val_df,l5d)]
box_dirs    = [cfg.DATA.BOX_TRAIN_DIR,cfg.DATA.BOX_VAL_DIR]
lidar_dirs  = [cfg.DATA.LIDAR_TRAIN_DIR,cfg.DATA.LIDAR_VAL_DIR]

for token_list,box_dir,lidar_dir in zip(token_lists,box_dirs,lidar_dirs):
    
    lidar_filepaths = []
    data_dict = {}
    for token in tqdm(token_list,total=len(token_list)):
        sample = l5d.get('sample',token)
        sample_lidar_token = sample['data']['LIDAR_TOP']
        try:
            lidar_filepath = l5d.get_sample_data_path(sample_lidar_token)
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
        except Exception as e:
            print('Failed to load LIDAR cloud for {}: {}:'.format(token,e))
            continue
        lidar_data = l5d.get('sample_data',sample_lidar_token)

        ego_pose = l5d.get('ego_pose',lidar_data['ego_pose_token'])
        calibrated_sensor = l5d.get('calibrated_sensor',lidar_data['calibrated_sensor_token'])
        car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                            Quaternion(calibrated_sensor['rotation']),
                                            inverse = False)
        boxes = l5d.get_boxes(sample_lidar_token)
        move_boxes_to_canvas_space(boxes,ego_pose)
        lidar_filepath = str(l5d.get_sample_data_path(sample_lidar_token))
        lidar_filepaths.append(lidar_filepath)
        box_filepath = osp.join(box_dir,token + '_boxes.pkl')
        pickle.dump(boxes,open(box_filepath,'wb'))
        data_dict[lidar_filepath] = {'boxes':box_filepath,'trans_matrix':car_from_sensor}
    
    dd_fp  = osp.join(lidar_dir,'data_dict.pkl')
    ld_fp  = osp.join(lidar_dir,'lidar_filepaths.pkl') 
    pickle.dump(data_dict,open(dd_fp,'wb'))
    pickle.dump(lidar_filepaths,open(ld_fp,'wb'))

anchor_boxes,anchor_corners,anchor_centers,anchor_xy = make_anchor_boxes()
a_dir = cfg.DATA.ANCHOR_DIR
pickle.dump(anchor_boxes,open(osp.join(a_dir,'anchor_boxes.pkl'),'wb'))
pickle.dump(anchor_corners,open(osp.join(a_dir,'anchor_corners.pkl'),'wb'))
pickle.dump(anchor_centers,open(osp.join(a_dir,'anchor_centers.pkl'),'wb'))
pickle.dump(anchor_xy,open(osp.join(a_dir,'anchor_xy.pkl'),'wb'))

pickle.dump(token_lists[0],open(osp.join(cfg.DATA.TOKEN_TRAIN_DIR,'training_tokens.pkl'),'wb'))
pickle.dump(token_lists[1],open(osp.join(cfg.DATA.TOKEN_VAL_DIR,'val_tokens.pkl'),'wb'))
"""

data_path = cfg.DATA.DATA_PATH
json_path = cfg.DATA.TRAIN_JSON_PATH
l5d = LyftDataset(data_path=data_path,json_path=json_path,verbose=True)

entries = []
for scene in l5d.scene:
    token = scene['first_sample_token']
    name = scene['name']
    host = "-".join(name.split("-")[:2])
    entries.append((host,token))

df = pd.DataFrame(entries,columns=['host_name','first_sample_token'])

val_hosts = ["host-a007", "host-a008", "host-a009"] 
val_df    = df[df['host_name'].isin(val_hosts)]
val_ind   = val_df.index
train_df  = df[~df.index.isin(val_ind)]

dfs = [train_df,val_df]
box_dirs = [cfg.DATA.BOX_TRAIN_DIR,cfg.DATA.BOX_VAL_DIR]
lidar_dirs = [cfg.DATA.LIDAR_TRAIN_DIR,cfg.DATA.LIDAR_VAL_DIR]
token_dirs = [cfg.DATA.TOKEN_TRAIN_DIR,cfg.DATA.TOKEN_VAL_DIR]

for df,box_dir,lidar_dir,token_dir in zip(dfs,box_dirs,lidar_dirs,token_dirs):
    data_dict  = {}
    token_list = []
    box_list   = []
    for token in tqdm(df.first_sample_token):
        while token:
            sample = l5d.get('sample',token)
            lidar_token = sample['data']['LIDAR_TOP']
            try:
                lidar_filepath = l5d.get_sample_data_path(lidar_token)
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
            lidar_data = l5d.get('sample_data',lidar_token)
            ego_pose   = l5d.get('ego_pose',lidar_data['ego_pose_token'])
            cal_sensor = l5d.get('calibrated_sensor',lidar_data['calibrated_sensor_token'])
            boxes      = l5d.get_boxes(lidar_token)
            move_boxes_to_canvas_space(boxes,ego_pose)
            box_list.append(boxes)
            boxes_fp   = osp.join(box_dir,token + '_boxes.pkl')
            pickle.dump(boxes,open(boxes_fp,'wb'))
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
    
anchor_boxes,anchor_corners,anchor_centers,anchor_xy = make_anchor_boxes()
a_dir = cfg.DATA.ANCHOR_DIR
pickle.dump(anchor_boxes,open(osp.join(a_dir,'anchor_boxes.pkl'),'wb'))
pickle.dump(anchor_corners,open(osp.join(a_dir,'anchor_corners.pkl'),'wb'))
pickle.dump(anchor_centers,open(osp.join(a_dir,'anchor_centers.pkl'),'wb'))
pickle.dump(anchor_xy,open(osp.join(a_dir,'anchor_xy.pkl'),'wb'))


