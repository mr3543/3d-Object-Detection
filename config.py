import os.path as osp
from easydict import EasyDict as edict
import numpy as np
import os
import torch

cfg = edict()
cfg.DATA = edict()
cfg.NET = edict()

#machine = 'local'
#machine = 'cloud'
machine = 'vastai'

# set the directories according to the machine type

if machine == 'vastai':
    pp_dir = '/root/PointPillars/'
    d_dir  = '/root/data/'
    cfg.DATA.ROOT_DIR        = d_dir
    cfg.DATA.CKPT_DIR        = pp_dir + 'ckpts'
    cfg.DATA.DATA_PATH       = d_dir
    cfg.DATA.TRAIN_JSON_PATH = d_dir  + 'train_data'
    cfg.DATA.BOX_TRAIN_DIR   = pp_dir + 'boxes/training'
    cfg.DATA.BOX_VAL_DIR     = pp_dir + 'boxes/validation'
    cfg.DATA.ANCHOR_DIR      = pp_dir + 'anchors'
    cfg.DATA.LIDAR_TRAIN_DIR = pp_dir + 'lidars/training'
    cfg.DATA.LIDAR_VAL_DIR   = pp_dir + 'lidars/validation'
    cfg.DATA.TOKEN_TRAIN_DIR = pp_dir + 'tokens/training'
    cfg.DATA.TOKEN_VAL_DIR   = pp_dir + 'tokens/validation'

# data location paramers
if machine == 'local':
    cfg.DATA.ROOT_DIR        = '/home/mmr/lyft_dataset'
    cfg.DATA.CKPT_DIR        = '/home/mmr/PointPillars/ckpts'
    cfg.DATA.DATA_PATH       = '/home/mmr/lyft_dataset'
    cfg.DATA.TRAIN_JSON_PATH = '/home/mmr/lyft_dataset/train_data'
    cfg.DATA.BOX_TRAIN_DIR   = '/home/mmr/PointPillars/boxes/training'
    cfg.DATA.BOX_VAL_DIR     = '/home/mmr/PointPillars/boxes/validation'
    cfg.DATA.ANCHOR_DIR      = '/home/mmr/PointPillars/anchors'
    cfg.DATA.LIDAR_TRAIN_DIR = '/home/mmr/PointPillars/lidars/training'
    cfg.DATA.LIDAR_VAL_DIR   = '/home/mmr/PointPillars/lidars/validation'
    cfg.DATA.TOKEN_TRAIN_DIR = '/home/mmr/PointPillars/tokens/training'
    cfg.DATA.TOKEN_VAL_DIR   = '/home/mmr/PointPillars/tokens/validation'


if machine == 'cloud':
    cfg.DATA.ROOT_DIR        = '/home/michaelregan/data/'
    cfg.DATA.CKPT_DIR        = '/home/michaelregan/PointPillars/ckpts'
    cfg.DATA.DATA_PATH       = '/home/michaelregan/data/'
    cfg.DATA.TRAIN_JSON_PATH = '/home/michaelregan/data/train_data'
    cfg.DATA.BOX_TRAIN_DIR   = '/home/michaelregan/PointPillars/boxes/training'
    cfg.DATA.BOX_VAL_DIR     = '/home/michaelregan/PointPillars/boxes/validation'
    cfg.DATA.ANCHOR_DIR      = '/home/michaelregan/PointPillars/anchors'
    cfg.DATA.LIDAR_TRAIN_DIR = '/home/michaelregan/PointPillars/lidars/training'
    cfg.DATA.LIDAR_VAL_DIR   = '/home/michaelregan/PointPillars/lidars/validation'
    cfg.DATA.TOKEN_TRAIN_DIR = '/home/michaelregan/PointPillars/tokens/training'
    cfg.DATA.TOKEN_VAL_DIR   = '/home/michaelregan/PointPillars/tokens/validation'

# pillar parameters 
cfg.DATA.X_MIN     = -60
cfg.DATA.Y_MIN     = -60
cfg.DATA.Z_MIN     = -10
cfg.DATA.X_MAX     = 60
cfg.DATA.Y_MAX     = 60
cfg.DATA.Z_MAX     = 10
cfg.DATA.X_STEP    = .2
cfg.DATA.Y_STEP    = .2
cfg.DATA.STEP      = .2
cfg.DATA.FM_SCALE  = .5

# compute the height and width of the canvas and feature map
cfg.DATA.FM_HEIGHT     = np.int32(((cfg.DATA.Y_MAX - cfg.DATA.Y_MIN)/cfg.DATA.Y_STEP)*cfg.DATA.FM_SCALE)
cfg.DATA.FM_WIDTH      = np.int32(((cfg.DATA.X_MAX - cfg.DATA.X_MIN)/cfg.DATA.X_STEP)*cfg.DATA.FM_SCALE)
cfg.DATA.CANVAS_HEIGHT = np.int32((cfg.DATA.Y_MAX - cfg.DATA.Y_MIN)/cfg.DATA.Y_STEP)
cfg.DATA.CANVAS_WIDTH  = np.int32((cfg.DATA.X_MAX - cfg.DATA.X_MIN)/cfg.DATA.X_STEP)

# approximate average box sizes of each category 
animal            = np.array([.5,1,.5])/cfg.DATA.STEP
bicycle           = np.array([.75,2,1.5])/cfg.DATA.STEP
bus               = np.array([3,12.5,3.5])/cfg.DATA.STEP
car               = np.array([2,5,1.75])/cfg.DATA.STEP
emergency_vehicle = np.array([2.5,6.5,2.5])/cfg.DATA.STEP
motorcycle        = np.array([1,2.5,1.5])/cfg.DATA.STEP
other_vehicle     = np.array([2.75,8.5,3.5])/cfg.DATA.STEP
pedestrian        = np.array([.75,.75,1.75])/cfg.DATA.STEP
truck             = np.array([3,10,3.5])/cfg.DATA.STEP

# group the classes into categories according to size
small = np.stack((animal,bicycle,pedestrian,motorcycle))
small = np.mean(small,axis=0)
med   = car
large = np.stack((bus,emergency_vehicle,truck,other_vehicle))
large = np.mean(large,axis=0)

# classes and class names
cfg.DATA.NUM_CLASSES = 9
cfg.DATA.CLASS_NAMES = ['animal','bicycle','bus','car','emergency_vehicle',
                        'motorcycle','other_vehicle','pedestrian','truck']

# uncomment according to whether you want to use 2 anchor boxes per category
# or group the classes into sizes
"""
cfg.DATA.ANCHOR_DIMS = [animal,animal,bicycle,bicycle,bus,bus,\
               car,car,emergency_vehicle,emergency_vehicle, \
               motorcycle,motorcycle,other_vehicle,other_vehicle,\
               pedestrian,pedestrian,truck,truck]
"""
cfg.DATA.ANCHOR_DIMS = [small,small,med,med,large,large]


#cfg.DATA.ANCHOR_YAWS = [0,90]*cfg.DATA.NUM_CLASSES
cfg.DATA.ANCHOR_YAWS = [0,90]*3
#cfg.DATA.ANCHOR_ZS   = [0]*2 +[.75]*2 + [1.5]*2 +[.75]*2 + [1.15]*2 + [.5]*2 + [1.15]*2 +[1]*2 +[1.5]*2
cfg.DATA.ANCHOR_ZS   = [.5]*2 + [.75]*2 + [1.0]*2
cfg.DATA.NUM_ANCHORS = len(cfg.DATA.ANCHOR_DIMS)

# pillar tensor & iou settings
cfg.DATA.MAX_POINTS_PER_PILLAR = 100
cfg.DATA.MAX_PILLARS    = 12000
cfg.DATA.REG_DIMS       = 9
cfg.DATA.IOU_POS_THRESH = .6
cfg.DATA.IOU_NEG_THRESH = .45

# training set construction parameters
cfg.DATA.TRAIN_DATA_FOLDER = osp.join(cfg.DATA.ROOT_DIR,'data/training_data')
cfg.DATA.VAL_DATA_FOLDER   = osp.join(cfg.DATA.ROOT_DIR,'data/validation_data')
cfg.DATA.NAME_TO_IND       = {'animal':0,'bicycle':1,'bus':2,'car':3,'emergency_vehicle':4,'motorcycle':5,'other_vehicle':6,'pedestrian':7,'truck':8}
cfg.DATA.IND_TO_NAME       = {'0':'animal','1':'bicycle','2':'bus','3':'car','4':'emergency_vehicle','5':'motorcycle','6':'other_vehicle','7':'pedestrian','8':'truck'}


# model parameters
cfg.NET.FEATURE_NET_IN  = 9
cfg.NET.FEATURE_NET_OUT = 64
cfg.NET.BATCH_SIZE      = 2
cfg.NET.EPOCHS          = 20
cfg.NET.LEARNING_RATE   = 1e-4
cfg.NET.WEIGHT_DECAY    = 1e-4
cfg.NET.NUM_WORKERS     = 4
cfg.NET.LR_SCHED        = np.concatenate([np.linspace(5e-5,1e-3,8),np.linspace(1e-3,5e-5,8),np.array([1e-5]*4)])


# loss parameters
cfg.NET.B_ORT = .2
cfg.NET.B_REG = 1
cfg.NET.B_CLS = 250
cfg.NET.GAMMA = 2
cfg.NET.POS_LABEL_WEIGHT = 500
cfg.NET.CLASS_WEIGHTS = np.array([2177,28,79,1,4059,616,20,20,54])
#cfg.NET.CLASS_WEIGHTS = np.array([1,1,1,1,1,1,1,1,1])

# validation
cfg.NET.VAL_MODEL = ''
cfg.DATA.VAL_POS_THRESH = .5
cfg.DATA.VAL_NMS_THRESH = .1
cfg.DATA.VAL_THRESH_LIST = np.arange(.5,1.0,0.05)

