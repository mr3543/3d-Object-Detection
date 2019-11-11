from easydict import EasyDict as edict
import os.path as osp
import numpy as np

cfg = edict()

# data location paramers
cfg.DATA.ROOT_DIR = '/home/michaelregan/PointPillars'
cfg.DATA.CKPT_DIR = osp.join(cfg.DATA.ROOT_DIR,'ckpts')
cfg.DATA.DATA_PATH = 'home/michaelregan/data'
cfg.DATA.TRAIN_JSON_PATH = 'home/michaelregan/data/train_data'

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
cfg.DATA.FM_HEIGHT = np.int32(((cfg.DATA.Y_MAX - cfg.DATA.Y_MIN)/cfg.DATA.Y_STEP)*FM_SCALE)
cfg.DATA.FM_WIDTH = np.int32(((cfg.DATA.X_MAX - cfg.DATA.X_MIN)/cfg.DATA.X_STEP)*FM_SCALE)
cfg.DATA.CANVAS_HEIGHT = np.int32((cfg.DATA.Y_MAX - cfg.DATA.Y_MIN)/cfg.DATA.Y_STEP) + 2

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
cfg.DATA.MAX_POINTS_PER_PILLAR = 100
cfg.DATA.MAX_PILLARS = 12000
cfg.DATA.IOU_POS_THRESH = .6
cfg.DATA.IOU_NEG_THRESH = .45

# training set construction parameters
cfg.DATA.NUM_WORKERS = 2
cfg.DATA.TRAIN_DATA_FOLDER = osp.join(cfg.DATA.ROOT_DIR,'data/training_data')
cfg.DATA.VAL_DATA_FOLDER = osp.join(cfg.DATA.ROOT_DIR,'data/validation_data')
cfg.DATA.NAME_TO_IND = {'animal':0,'bicycle':1,'bus':2,'car':3,'emergency_vehicle':4,'motorcycle':5,'other_vehicle':6,'pedestrian':7,'truck':8}

