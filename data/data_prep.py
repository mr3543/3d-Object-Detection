import numpy as np
import pandas as pd
import time
import os
import os.path as osp
import pickle as pkl
import sys
import gc
import pillars as plrs
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from scipy.sparse import csr_matrix

from shapely.geometry import Polygon
from .. import config 
from config import cfg
from datetime import datetime
from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from lyft_dataset_sdk.lyftdataset import LyftDataset

ARTIFACTS_FOLDER = './artifacts'
X_MIN = -75
Y_MIN = -75
Z_MIN = -3
X_MAX = 75
Y_MAX = 75
Z_MAX = 3
X_STEP = .3
Y_STEP = .3
STEP = .3
FM_SCALE = .5
FM_HEIGHT = np.int32(((Y_MAX - Y_MIN)/Y_STEP)*FM_SCALE)
FM_WIDTH = np.int32(((X_MAX - X_MIN)/X_STEP)*FM_SCALE)
CANVAS_HEIGHT = np.int32((Y_MAX - Y_MIN)/Y_STEP) + 2

animal = np.array([.5,1,.5])/STEP
bicycle = np.array([.75,2,1.5])/STEP
bus = np.array([3,12.5,3.5])/STEP
car = np.array([2,5,1.75])/STEP
emergency_vehicle = np.array([2.5,6.5,2.5])/STEP
motorcycle = np.array([1,2.5,1.5])/STEP
other_vehicle = np.array([2.75,8.5,3.5])/STEP
pedestrian = np.array([.75,.75,1.75])/STEP
truck = np.array([3,10,3.5])/STEP
ANCHOR_DIMS = [animal,animal,bicycle,bicycle,bus,bus,\
               car,car,emergency_vehicle,emergency_vehicle, \
               motorcycle,motorcycle,other_vehicle,other_vehicle,\
               pedestrian,pedestrian,truck,truck]
ANCHOR_YAWS = [0,90]*9
ANCHOR_ZS = [0]*2 +[.75]*2 + [1.5]*2 +[.75]*2 + [1.15]*2 + [.5]*2 + [1.15]*2 +[1]*2 +[1.5]*2
MAX_POINTS_PER_PILLAR = 100
MAX_PILLARS = 12000
DATA_PATH = '/home/michaelregan/data'
TRAIN_JSON_PATH = '/home/michaelregan/data/train_data'
IOU_POS_THRESH = .6
IOU_NEG_THRESH = .45
NUM_WORKERS = 2
TRAIN_DATA_FOLDER = './artifacts/training_data'
VAL_DATA_FOLDER = './artifacts/validation_data'
NAME_TO_IND = {'animal':0,'bicycle':1,'bus':2,'car':3,'emergency_vehicle':4,'motorcycle':5,'other_vehicle':6,'pedestrian':7,'truck':8}


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

    class_ind = NAME_TO_IND[gt_box.name]
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
    for y in range(0,fm_height):
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
    plrs.make_ious(anchor_corners,gt_corners,
                   anchor_centers,gt_centers,ious)
    t_max_ious = np.max(ious.transpose([1,0]),axis=1)
    t_argmax_ious = np.argmax(ious.transpose([1,0]),axis=1)
    targets = np.zeros((len(anchor_boxes),10))   
 
    """
    for i in range(len(t_max_ious)):
        print('--------------------')
        print('BOX CENTER: ',gt_box_list[i].center)
        print('BOX SIZE: ',gt_box_list[i].wlh)
        print('BOX YAW: ',gt_box_list[i].orientation.yaw_pitch_roll[0])
        print('BOX TYPE: ',gt_box_list[i].name)
        print('MAX IOU: ',t_max_ious[i])
        print('AB CENTER: ',anchor_box_list[t_argmax_ious[i]].center)
        print('AB SIZE: ',anchor_box_list[t_argmax_ious[i]].wlh)
        print('AB YAW: ',anchor_box_list[t_argmax_ious[i]].orientation.yaw_pitch_roll[0])
        print('---------------------')
    """ 
    max_ious = np.max(ious,axis=1)
    arg_max_ious = np.argmax(ious,axis=1)
    pos_anchors = np.where(max_ious > pos_thresh)
    neg_anchors = np.where(max_ious < neg_thresh)
    pos_boxes = arg_max_ious[pos_anchors]
    #num_pos = len(pos_anchors[0])
    #num_neg = len(neg_anchors[0])
    #num_neut = len(anchor_box_list) - num_pos - num_neg
    #print('NUM POS ANCHORS: ',num_pos)
    #print('NUM NEG ANCHORS: ',num_neg)
    #print('NUM NEUT ANCHORS: ',num_neut)
    
    ious = ious.transpose([1,0])
    top_anchor_for_box = np.argmax(ious,axis=1)

    for anch in neg_anchors[0]:
        targets[anch,:] = np.array([-1] + [0]*9)

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
        box_x = (box_x - X_MIN)/x_step
        box_y = (box_y - Y_MIN)/y_step

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


def prep_training_val_data(sample_token,output_folder,l5d,anchor_boxes,
                           anchor_corners,anchor_centers):

    #num_sweeps = cfg.DATA.NUM_SWEEPS
    tokens_processed = 0
    start = time.time()
    while sample_token:
        
        if tokens_processed > 0 and tokens_processed % 100 == 0:
            end = time.time()
            print('PROCESSED 100 TOKENS IN: ',(end - start))
            start = time.time()

        sample = l5d.get('sample',sample_token)
        sample_lidar_token = sample['data']['LIDAR_TOP']

        lidar_data = l5d.get('sample_data',sample_lidar_token)

        ego_pose = l5d.get('ego_pose',lidar_data['ego_pose_token'])
        calibrated_sensor = l5d.get('calibrated_sensor',lidar_data['calibrated_sensor_token'])

        car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                           Quaternion(calibrated_sensor['rotation']),
                                           inverse = False)
        
        # move boxes to canvas space
        boxes = l5d.get_boxes(sample_lidar_token)
        boxes = move_boxes_to_canvas_space(boxes,ego_pose)
        try:
            lidar_filepath = l5d.get_sample_data_path(sample_lidar_token)
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print('Failed to load LIDAR cloud for {}: {}:'.format(sample_token,e))
            sample_token = sample['next']
            continue

        lidar_points = lidar_pointcloud.points
        #TODO: augment data here

        pillar = np.zeros((cfg.DATA.MAX_PILLARS,cfg.DATA.MAX_POINTS_PER_PILLAR,9))
        indices = np.zeros((cfg.DATA.MAX_PILLARS,3))
        
        gt_corners = np.array([box.bottom_corners()[:2,:].transpose([1,0]) for box in boxes])
        gt_centers = np.array([box.center for box in boxes])
         
        plrs.create_pillars(lidar_points.transpose([1,0]),pillar,
                            indices,cfg.DATA.MAX_POINTS_PER_PILLAR,
                            cfg.DATa.MAX_PILLARS,cfg.DATa.X_STEP,cfg.DATA.Y_STEP,
                            cfg.DATA.X_MIN,cfg.DATA.Y_MIN,cfg.DATA.Z_MIN,
                            cfg.DATA.X_MAX,cfg.DATA.Y_MAX,
                            cfg.DATA.Z_MAX,cfg.DATA.CANVAS_HEIGHT)
         
        target = create_target(anchor_corners,gt_corners,
                               anchor_centers,gt_centers,
                               anchor_boxes,boxes)
        
        pillar_file = osp.join(output_folder,'{}_pillar.pkl'.format(sample_token))
        target_file = osp.join(output_folder,'{}_target.pkl'.format(sample_token))
        indices_file = osp.join(output_folder,'{}_indices.pkl'.format(sample_token))

        pillar = pillar.reshape(pillar.shape[0],-1)
        indices = indices.reshape(indices.shape[0],-1)

        pillar_sparse = csr_matrix(pillar)
        indices_sparse = csr_matrix(indices)
        target_sparse = csr_matrix(target)        

        pkl.dump(pillar_sparse,open(pillar_file,'wb'))
        pkl.dump(indices_sparse,open(indices_file,'wb'))
        pkl.dump(target_sparse,open(target_file,'wb'))

        gc.collect()
        tokens_processed += 1
        sample_token = sample['next']


if __name__ == '__main__':

    data_path = cfg.DATA.DATA_PATH
    json_path = cfg.DATA.TRAIN_JSON_PATH

    l5d = LyftDataset(data_path=data_path,json_path = json_path,verbose = True)

    os.makedirs(cfg.DATA.ARTIFACTS_FOLDER,exist_ok=True)

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

    train_data_folder = cfg.DATA.TRAIN_DATA_FOLDER
    val_data_folder = cfg.DATA.VAL_DATA_FOLDER
    num_workers = cfg.DATA.NUM_WORKERS

    anchor_boxes,anchor_corners,anchor_centers = make_anchor_boxes()
    
    for df,data_folder in [(train_df,train_data_folder),(val_df,val_data_folder)]:
        print('Preparing data in {} using {} workers'.format(data_folder,num_workers))
        os.makedirs(data_folder,exist_ok=True)
        first_samples = df.first_sample_token.values
        
        process_func = partial(prep_training_val_data,output_folder=data_folder,
                                                      l5d=l5d,anchor_boxes=anchor_boxes,
                                                      anchor_corners=anchor_corners,
                                                      anchor_centers=anchor_centers)
        pool = Pool(num_workers)
        for _ in tqdm(pool.imap_unordered(process_func,first_samples),total = len(first_samples)):
            pass
        pool.close()
        del pool  