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

token_list = make_token_list(train_df,l5d)
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
    boxes = move_boxes_to_canvas_space(boxes,ego_pose)
    lidar_filepath = l5d.get_sample_data_path(sample_lidar_token)
    lidar_filepaths.append(lidar_filepath)
    box_filepath = osp.join(cfg.DATA.BOX_DIR,token + '_boxes.pkl')
    pickle.dump(boxes,open(box_filepath,'wb'))
    data_dict[lidar_filepath] = {'boxes':box_filepath,'trans_matrix':car_from_sensor}

pickle.dump(data_dict,open('data_dict.pkl','wb'))
pickle.dump(lidar_filepaths,open('lidar_filepaths.pkl','wb'))

anchor_boxes,anchor_corners,anchor_centers = make_anchor_boxes()
pickle.dump(anchor_boxes,open('anchor_boxes.pkl','wb'))
pickle.dump(anchor_corners,open('anchor_corners.pkl','wb'))
pickle.dump(anchor_centers,open('anchor_centers.pkl','wb'))
