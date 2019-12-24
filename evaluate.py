import numpy as np
import os.path as osp
import torch
import pickle
import pdb
from tqdm import tqdm
from pyquaternion import Quaternion
from data.dataset import PPDataset
from config import cfg
from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions
from torchvision.ops import nms    


def make_box_dict(box,token,score):
    bd = {'sample_token': token,
         'translation' : list(box.center),
         'size'        : list(box.wlh),
         'rotation'    : list(box.orientation),
         'name'        : box.name}
    if score:
        bd['score'] = box.score
    
    return bd
    
def make_pred_boxes(inds,anchor_box_list,reg,classes,scores,token):
    
    out_box_list = []
    reg = reg.cpu().numpy()
    classes = classes.cpu().numpy()
    scores = scores.cpu().numpy()
    for j,i in enumerate(inds):
        
        a_box      = anchor_box_list[i]
        offsets    = reg[i,:] 
        a_box_diag = np.sqrt(a_box.wlh[0]**2 + a_box.wlh[1]**2)
        
        box_x = a_box.center[0] + offsets[0] * a_box_diag
        box_y = a_box.center[1] + offsets[1] * a_box_diag
        box_z = a_box.center[2] + offsets[2] * a_box_diag

        box_w = np.exp(offsets[3])*a_box.wlh[0]
        box_l = np.exp(offsets[4])*a_box.wlh[1]
        box_h = np.exp(offsets[5])*a_box.wlh[2]

        box_name = cfg.DATA.IND_TO_NAME[str(int(classes[i]))]
        box_yaw = np.arcsin(offsets[6]) + a_box.orientation.yaw_pitch_roll[0]
        #print('offsets: {}, yaw: {}, i: {}'.format(offsets[6],box_yaw,j))
        if offsets[7] > offsets[8]:
            box_ort = 1
        else:
            box_ort = -1
        
        box_yaw *= box_ort
        quat = Quaternion(axis=[0,0,1],radians = box_yaw) 
        box = Box(center=[box_x,box_y,box_z],
                  size = [box_w,box_l,box_h],
                  orientation=quat,
                  name=box_name,
                  score=scores[i],
                  token=token)

        out_box_list.append(box)

    return out_box_list

def move_box_to_car_space(box):
    x,y,z = box.center
    w,l,h = box.wlh
    
    canv_h = cfg.DATA.CANVAS_HEIGHT
    y_step = cfg.DATA.Y_STEP 
    x_step = cfg.DATA.X_STEP
    y_min  = cfg.DATA.Y_MIN
    x_min  = cfg.DATA.X_MIN

    car_y  = (canv_h - 1) - y
    car_y  = car_y*y_step + y_min
    car_x  = x*x_step + x_min

    car_w = w*y_step
    car_l = l*x_step

    box.wlh    = np.array([car_w,car_l,h])
    box.center = np.array([car_x,car_y,z]) 

def box_nms(pos_inds,anchor_xy,scores,thresh):
    
    nms_boxes = torch.from_numpy(anchor_xy.copy()).float()[pos_inds]
    nms_boxes[:,1] = (cfg.DATA.CANVAS_HEIGHT - 1) - nms_boxes[:,1]
    nms_boxes[:,3] = (cfg.DATA.CANVAS_HEIGHT - 1) - nms_boxes[:,3]
    scores    = scores.cpu()
    return nms(nms_boxes,scores,thresh)

def evaluate_single(pp_model,anchor_box_list,data_mean,device,
                    p,inds):

    data_dict_fp = osp.join(cfg.DATA.LIDAR_TRAIN_DIR,'data_dict.pkl')
    token_fp     = osp.join(cfg.DATA.TOKEN_TRAIN_DIR,'training_tokens.pkl')
    anch_xy_fp   = osp.join(cfg.DATA.ANCHOR_DIR,'anchor_xy.pkl')

    data_dict       = pickle.load(open(data_dict_fp,'rb'))
    data_mean       = pickle.load(open('pillar_means.pkl','rb')) 
    token_list      = pickle.load(open(token_fp,'rb'))
    anchor_xy       = pickle.load(open(anch_xy_fp,'rb'))

    cls,reg = pp_model(p,inds)
        
    cls            = cls.permute(0,2,3,1).reshape(-1,cfg.DATA.NUM_CLASSES)
    reg            = reg.permute(0,2,3,1).reshape(-1,cfg.DATA.REG_DIMS)
    cls            = torch.sigmoid(cls)
    reg[...,6]     = torch.tanh(reg[...,6])
    scores,classes = torch.max(cls,dim=-1)
    pos_inds       = torch.where(scores > cfg.DATA.VAL_POS_THRESH)[0]
    
    to_keep        = box_nms(pos_inds,anchor_xy,scores[pos_inds],cfg.DATA.VAL_NMS_THRESH)
    final_box_inds = pos_inds[to_keep]
     
    final_boxes    = make_pred_boxes(final_box_inds,anchor_box_list,reg,classes,scores,
                                     token_list[i])
    pdb.set_trace()
    gt_boxes_fp    = data_dict[lidar_filepaths[i]]['boxes']
    gt_boxes       = pickle.load(open(gt_boxes_fp,'rb'))
    i=0    
    for box in gt_boxes:
        move_box_to_car_space(box)
        box_dict = make_box_dict(box,token_list[i],score=False)
        gt_box_list.append(box_dict)

    for box in final_boxes:
        move_box_to_car_space(box)
        box_dict = make_box_dict(box,token_list[i],score=True)
        pred_box_list.append(box_dict)

    map_list = []
    for thresh in cfg.DATA.VAL_THRESH_LIST:
        thresh_ap = get_average_precisions(gt_box_list,pred_box_list,
                                        cfg.DATA.CLASS_NAMES,thresh)
        map_list.append(np.mean(thresh_ap))

    print('---------VALIDATON SET-----------')
    print('VAL mAP: ',np.mean(map_list))
    print('---------------------------------')



def evaluate(pp_model,anchor_box_list,data_mean,device):

    data_dict_fp = osp.join(cfg.DATA.LIDAR_VAL_DIR,'data_dict.pkl')
    token_fp     = osp.join(cfg.DATA.TOKEN_VAL_DIR,'val_tokens.pkl')
    anch_xy_fp   = osp.join(cfg.DATA.ANCHOR_DIR,'anchor_xy.pkl')

    data_dict       = pickle.load(open(data_dict_fp,'rb'))
    data_mean       = pickle.load(open('pillar_means.pkl','rb')) 
    token_list      = pickle.load(open(token_fp,'rb'))
    anchor_xy       = pickle.load(open(anch_xy_fp,'rb'))

    fn_in  = cfg.NET.FEATURE_NET_IN
    fn_out = cfg.NET.FEATURE_NET_OUT

    cls_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.NUM_CLASSES
    reg_channels = len(cfg.DATA.ANCHOR_DIMS)*cfg.DATA.REG_DIMS 

    pp_model.eval()
    pp_dataset = PPDataset(token_list,data_dict,None,None,None,
                           data_mean=data_mean,training=False)
                          
    dataloader = torch.utils.data.DataLoader(pp_dataset,batch_size=1,shuffle=False,
                                             num_workers=1)

    gt_box_list   = []
    pred_box_list = []

    for i,(p,inds) in tqdm(enumerate(dataloader),total=len(pp_dataset)):
        
        p = p.to(device)
        inds = inds.to(device)
        cls,reg = pp_model(p,inds)
        
        cls            = cls.permute(0,2,3,1).reshape(-1,cfg.DATA.NUM_CLASSES)
        cls            = torch.sigmoid(cls)
        reg[...,6]     = torch.tanh(reg[...,6])
        scores,classes = torch.max(cls,dim=-1)
        pos_inds       = torch.where(scores > cfg.DATA.VAL_POS_THRESH)[0]
        
        print(len(pos_inds))
        print(torch.max(scores))
        to_keep        = box_nms(pos_inds,anchor_xy,scores[pos_inds],cfg.DATA.VAL_NMS_THRESH)
        final_box_inds = pos_inds[to_keep]
         
        final_boxes    = make_pred_boxes(final_box_inds,anchor_box_list,reg,classes,scores,
                                         token_list[i])
        gt_boxes_fp    = data_dict[lidar_filepaths[i]]['boxes']
        gt_boxes       = pickle.load(open(gt_boxes_fp,'rb'))
        
        for box in gt_boxes:
            move_box_to_car_space(box)
            box_dict = make_box_dict(box,token_list[i],score=False)
            gt_box_list.append(box_dict)

        for box in final_boxes:
            move_box_to_car_space(box)
            box_dict = make_box_dict(box,token_list[i],score=True)
            pred_box_list.append(box_dict)
    
    map_list = []
    for thresh in cfg.DATA.VAL_THRESH_LIST:
        thresh_ap = get_average_precisions(gt_box_list,pred_box_list,
                                           cfg.DATA.CLASS_NAMES,thresh)
        map_list.append(np.mean(thresh_ap))

    print('---------VALIDATON SET-----------')
    print('VAL mAP: ',np.mean(map_list))
    print('---------------------------------')


def write_submission(boxes):
    sub = {}
    for i in range(len(boxes)):
        yaw = 2*np.arccos(boxes[i].rotation[0])
        pred = str(boxes[i].score) + ' ' + \
               str(boxes[i].center_x) + ' ' + \
               str(boxes[i].center_y) + ' ' + \
               str(boxes[i].center_z) + ' ' + \
               str(boxes[i].width) + ' ' + \
               str(boxes[i].length) + ' ' + \
               str(boxes[i].height) + ' ' + \
               str(yaw) + ' ' + \
               str(boxes[i].name) + ' '

        if boxes[i].sample_token in sub.keys():
            sub[boxes[i].sample_token] += pred
        else:
            sub[boxes[i].sample_token] = pred

    sub = pd.DataFrame(list(sub.items()))
    sub.columns = ['Id','PredictionString']
    sub.to_csv('lyft3d_pred.csv',index=False)


def move_boxes_to_world_space(pred_boxes,l5d):
    
    for box in pred_boxes:
        
        token = box.token
        sample = l5d.get('sample',token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = l5d.get("sample_data", sample_lidar_token)
        ego_pose = l5d.get("ego_pose", lidar_data["ego_pose_token"])
        translation = np.array(ego_pose['translation'])
        rotation = Quaternion(ego_pose['rotation'])
        box.translate(translation)
        box.rotate(rotation)


