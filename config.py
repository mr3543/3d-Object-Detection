import os.path as osp
from easydict import EasyDict as edict
import numpy as np
import os

cfg = edict()
cfg.DATA = edict()
cfg.NET = edict()

#machine = 'local'
#machine = 'kaggle'
machine = 'cloud'

# data location paramers
if machine == 'local':
    cfg.DATA.ROOT_DIR = '/home/mmr/lyft_dataset'
    cfg.DATA.CKPT_DIR = '/home/mmr/PointPillars/ckpts'
    cfg.DATA.DATA_PATH = '/home/mmr/lyft_dataset'
    cfg.DATA.TRAIN_JSON_PATH = '/home/mmr/lyft_dataset/train_data'
    cfg.DATA.BOX_DIR = '/home/mmr/PointPillars/boxes'

if machine == 'kaggle':
    cfg.DATA.ROOT_DIR = '/kaggle/input/3d-object-detection-for-autonomous-vehicles'
    cfg.DATA.CKPT_DIR = '/kaggle/working/ckpts'
    cfg.DATA.DATA_PATH = '.'
    cfg.DATA.TRAIN_JSON_PATH = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data'
    cfg.DATA.BOX_DIR = '/kaggle/working/PointPillars/boxes/'

if machine == 'cloud':
    cfg.DATA.ROOT_DIR = '/home/michaelregan/data/'
    cfg.DATA.CKPT_DIR = '/home/PointPillars/ckpts'
    cfg.DATA.DATA_PATH = '/home/michaelregan/data/'
    cfg.DATA.TRAIN_JSON_PATH = '/home/michaelregan/data/train_data'
    cfg.DATA.BOX_DIR = '/home/michaelregan/data/boxes/'
    

# pillar parameters 
cfg.DATA.X_MIN = -75
cfg.DATA.Y_MIN = -75
cfg.DATA.Z_MIN = -3
cfg.DATA.X_MAX = 75
cfg.DATA.Y_MAX = 75
cfg.DATA.Z_MAX = 3
cfg.DATA.X_STEP = .3
cfg.DATA.Y_STEP = .3
cfg.DATA.STEP = .3
cfg.DATA.FM_SCALE = .5
cfg.DATA.FM_HEIGHT = np.int32(((cfg.DATA.Y_MAX - cfg.DATA.Y_MIN)/cfg.DATA.Y_STEP)*cfg.DATA.FM_SCALE)
cfg.DATA.FM_WIDTH = np.int32(((cfg.DATA.X_MAX - cfg.DATA.X_MIN)/cfg.DATA.X_STEP)*cfg.DATA.FM_SCALE)
cfg.DATA.CANVAS_HEIGHT = np.int32((cfg.DATA.Y_MAX - cfg.DATA.Y_MIN)/cfg.DATA.Y_STEP)
cfg.DATA.CANVAS_WIDTH = np.int32((cfg.DATA.X_MAX - cfg.DATA.X_MIN)/cfg.DATA.X_STEP)

animal = np.array([.5,1,.5])/cfg.DATA.STEP
bicycle = np.array([.75,2,1.5])/cfg.DATA.STEP
bus = np.array([3,12.5,3.5])/cfg.DATA.STEP
car = np.array([2,5,1.75])/cfg.DATA.STEP
emergency_vehicle = np.array([2.5,6.5,2.5])/cfg.DATA.STEP
motorcycle = np.array([1,2.5,1.5])/cfg.DATA.STEP
other_vehicle = np.array([2.75,8.5,3.5])/cfg.DATA.STEP
pedestrian = np.array([.75,.75,1.75])/cfg.DATA.STEP
truck = np.array([3,10,3.5])/cfg.DATA.STEP
cfg.DATA.NUM_CLASSES = 9
cfg.DATA.ANCHOR_DIMS = [animal,animal,bicycle,bicycle,bus,bus,\
               car,car,emergency_vehicle,emergency_vehicle, \
               motorcycle,motorcycle,other_vehicle,other_vehicle,\
               pedestrian,pedestrian,truck,truck]


cfg.DATA.ANCHOR_YAWS = [0,90]*cfg.DATA.NUM_CLASSES
cfg.DATA.ANCHOR_ZS = [0]*2 +[.75]*2 + [1.5]*2 +[.75]*2 + [1.15]*2 + [.5]*2 + [1.15]*2 +[1]*2 +[1.5]*2
cfg.DATA.NUM_ANCHORS = len(cfg.DATA.ANCHOR_DIMS)
cfg.DATA.MAX_POINTS_PER_PILLAR = 100
cfg.DATA.MAX_PILLARS = 12000
cfg.DATA.REG_DIMS = 9
cfg.DATA.IOU_POS_THRESH = .6
cfg.DATA.IOU_NEG_THRESH = .45

# training set construction parameters
cfg.DATA.NUM_WORKERS = 4
cfg.DATA.TRAIN_DATA_FOLDER = osp.join(cfg.DATA.ROOT_DIR,'data/training_data')
cfg.DATA.VAL_DATA_FOLDER = osp.join(cfg.DATA.ROOT_DIR,'data/validation_data')
cfg.DATA.NAME_TO_IND = {'animal':0,'bicycle':1,'bus':2,'car':3,'emergency_vehicle':4,'motorcycle':5,'other_vehicle':6,'pedestrian':7,'truck':8}
#cfg.DATA.IND_TO_NAME = {0:'animal',1:'bicycle',2:'bus',3:'car',4:'emergency_vehicle',5:'motorcycle',6:'other_vehicle',7:'pedestrian',8:'truck'}
# model parameters

cfg.NET.FEATURE_NET_IN = 9
cfg.NET.FEATURE_NET_OUT = 64
cfg.NET.BATCH_SIZE = 4
cfg.NET.EPOCHS = 10
cfg.NET.LEARNING_RATE = 1e-5
cfg.NET.WEIGHT_DECAY = 1e-4
cfg.NET.NUM_WORKERS = 4
cfg.NET.B_ORT = .2
cfg.NET.B_REG = 2
cfg.NET.B_CLS = 1
cfg.NET.GAMMA = 2
