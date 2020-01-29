import numpy as np
import os.path as osp
import torch
import gc
import pickle
import pdb
from tqdm import tqdm
from pyquaternion import Quaternion
from data.dataset import PPDataset
from config import cfg
from lyft_dataset_sdk.utils.data_classes import Box
from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions
from torchvision.ops import nms    
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def make_box_dict(box,token,score):
    """
    makes box dict for lyft mAP 
    """
    bd = {'sample_token': token,
         'translation'  : list(box.center),
         'size'         : list(box.wlh),
         'rotation'     : list(box.orientation),
         'name'         : box.name}
    if score:
        bd['score'] = box.score
    
    return bd
    
def make_pred_boxes(inds,anchor_box_list,reg,classes,scores,token):
    
    """
    takes a list of positively labeled anchor boxes and adjusts
    them using the network adjustment output. 

    `inds` contains the box indices of the positive anchor boxes
    `reg` contains the model outputs for the anchor box adjustments
    `classes` contains the classes of the anchors
    `scores` contains the model confidences
    """

    out_box_list = []
    reg = reg.cpu().numpy()
    classes = classes.cpu().numpy()
    scores = scores.cpu().numpy()
    # loop through the positive anchors

    for i in inds:
        # get the Box object and the box adjustments for box i
        a_box      = anchor_box_list[i]
        offsets    = reg[i,:] 
        a_box_diag = np.sqrt(a_box.wlh[0]**2 + a_box.wlh[1]**2)
        
        # compute new box xyz and wlh
        box_x = a_box.center[0] + offsets[0] * a_box_diag
        box_y = a_box.center[1] + offsets[1] * a_box_diag
        box_z = a_box.center[2] + offsets[2] * a_box.wlh[2]

        box_w = np.exp(offsets[3])*a_box.wlh[0]
        box_l = np.exp(offsets[4])*a_box.wlh[1]
        box_h = np.exp(offsets[5])*a_box.wlh[2]

        # get class & yaw of box
        box_name = cfg.DATA.IND_TO_NAME[str(int(classes[i]))]
        box_yaw = np.arcsin(offsets[6]) + a_box.orientation.yaw_pitch_roll[0]
   
        """
        if sigmoid(offsets[7]) < .5:
            box_ort = 1
        else:
            box_ort = -1
        
        box_yaw *= box_ort
        """
        # make new Box object
        quat = Quaternion(axis=[0,0,1],radians = box_yaw) 
        box = Box(center=[box_x,box_y,box_z],
                  size = [box_w,box_l,box_h],
                  orientation=quat,
                  name=box_name,
                  score=scores[i],
                  token=token)

        out_box_list.append(box)

    return out_box_list

def move_box_to_car_space(box,image=True):
    
    """
    takes a box in image space and moves it to the space
    of the ego vehicle
    """
    #pdb.set_trace()
    x,y,z = box.center
    w,l,h = box.wlh
    
    canv_h = cfg.DATA.CANVAS_HEIGHT
    y_step = cfg.DATA.Y_STEP 
    x_step = cfg.DATA.X_STEP
    y_min  = cfg.DATA.Y_MIN
    x_min  = cfg.DATA.X_MIN

    if image:
        y = (canv_h - 1) - y
        
    car_y  = y*y_step + y_min
    car_x  = x*x_step + x_min

    car_w = w*y_step
    car_l = l*x_step

    box_wlh    = np.array([car_w,car_l,h])
    box_center = np.array([car_x,car_y,z]) 

    car_box = Box(center=box_center,
                  size=box_wlh,
                  orientation=Quaternion(list(box.orientation)),
                  name=box.name,
                  token=box.token,
                  score=box.score)
    return car_box

def box_nms(pos_inds,anchor_xy,scores,thresh):
    """
    does NMS on selected boxes. boxes need to be in
    (x1,y1,x2,y2) format to use built-in torch NMS. 
    we also need to transform the boxes from image space
    to canvas space. 
    """
    
    nms_boxes = torch.from_numpy(anchor_xy.copy()).float()[pos_inds]
    nms_boxes[:,1] = (cfg.DATA.CANVAS_HEIGHT - 1) - nms_boxes[:,1]
    nms_boxes[:,3] = (cfg.DATA.CANVAS_HEIGHT - 1) - nms_boxes[:,3]
    scores    = scores.cpu()
    return nms(nms_boxes,scores,thresh)

def evaluate_single(cls_tensor,reg_tensor,token,anchor_box_list,data_dict):

    anch_xy_fp   = osp.join(cfg.DATA.ANCHOR_DIR,'anchor_xy.pkl')
    anchor_xy    = pickle.load(open(anch_xy_fp,'rb'))

    cls            = cls_tensor.permute(0,2,3,1).reshape(-1,cfg.DATA.NUM_CLASSES)
    reg            = reg_tensor.permute(0,2,3,1).reshape(-1,cfg.DATA.REG_DIMS)
    cls            = torch.sigmoid(cls)
    reg[...,6]     = torch.tanh(reg[...,6])
    scores,classes = torch.max(cls,dim=-1)
    # positive boxes are the boxes with classification scores above threshold
    pos_inds       = torch.where(scores > cfg.DATA.VAL_POS_THRESH)[0]

    # do nms on the positive boxes
    to_keep        = box_nms(pos_inds,anchor_xy,scores[pos_inds],cfg.DATA.VAL_NMS_THRESH)
    final_box_inds = pos_inds[to_keep[:100]]
     
    # adjust the boxes selected from nms
    final_boxes    = make_pred_boxes(final_box_inds,anchor_box_list,reg,classes,scores,
                                     token)

    # load the ground truth boxes from the validation sample
    gt_boxes_fp    = data_dict[token]['boxes']
    gt_boxes       = pickle.load(open(gt_boxes_fp,'rb'))

    # loop through ground truth boxes and add create gt_box_list to pass
    # to lyft_dataset_sdk evaluation function
    gt_box_list = []
    pred_box_list = []
    class_names = []
    for box in gt_boxes:
        car_box = move_box_to_car_space(box,image=False)
        box_dict = make_box_dict(car_box,token,score=False)
        gt_box_list.append(box_dict)
        class_names.append(box.name)

    # loop through anchor boxes and create pred_box_list to pass
    # to lyft_dataset_sdk evaluation function
    for box in final_boxes:
        car_box = move_box_to_car_space(box)
        box_dict = make_box_dict(car_box,token,score=True)
        pred_box_list.append(box_dict)

    map_list = []
    class_names = list(set(class_names))
    print('len pred boxes: ',len(pred_box_list))
    # get the average precision for each iou threshold - evaluation score
    # is the mean across all thresholds
    for thresh in cfg.DATA.VAL_THRESH_LIST:
        thresh_ap = get_average_precisions(gt_box_list,pred_box_list,
                                           class_names,thresh)
        map_list.append(np.mean(thresh_ap))

    gc.collect()    
    return map_list


def evaluate(pp_model,anchor_box_list,token_list,data_dict,device):

    """
    evaluates the model on the validation set
    """
    gc.collect()
    # load the data and tokens for the validation set
    anch_xy_fp   = osp.join(cfg.DATA.ANCHOR_DIR,'anchor_xy.pkl')

    data_mean    = pickle.load(open('pillar_means.pkl','rb')) 
    anchor_xy    = pickle.load(open(anch_xy_fp,'rb'))

    # set model to eval mode and create a dataloader with validation
    # data
    pp_model.eval()
    pp_dataset = PPDataset(token_list,data_dict,None,None,None,
                           data_mean=data_mean,training=False)
                          
    dataloader = torch.utils.data.DataLoader(pp_dataset,batch_size=1,shuffle=False,
                                             num_workers=0)

    gt_box_list   = []
    pred_box_list = []
    class_names = []
    # loop through validation data
    for i,(p,inds) in tqdm(enumerate(dataloader),total=len(pp_dataset)):
       
        # get model output
        p = p.to(device)
        inds = inds.to(device)
        cls,reg = pp_model(p,inds)
        
        # reshape output to extract classification and regression predictions
        cls            = cls.permute(0,2,3,1).reshape(-1,cfg.DATA.NUM_CLASSES)
        reg            = reg.permute(0,2,3,1).reshape(-1,cfg.DATA.REG_DIMS)
        cls            = torch.sigmoid(cls)
        reg[...,6]     = torch.tanh(reg[...,6])
        scores,classes = torch.max(cls,dim=-1)
        # positive boxes are the boxes with classification scores above threshold
        pos_inds       = torch.where(scores > cfg.DATA.VAL_POS_THRESH)[0]
    
        # do nms on the positive boxes
        to_keep        = box_nms(pos_inds,anchor_xy,scores[pos_inds],cfg.DATA.VAL_NMS_THRESH)
        final_box_inds = pos_inds[to_keep[:100]]
         
        # adjust the boxes selected from nms
        final_boxes    = make_pred_boxes(final_box_inds,anchor_box_list,reg,classes,scores,
                                         token_list[i])

        # load the ground truth boxes from the validation sample
        gt_boxes_fp    = data_dict[token_list[i]]['boxes']
        gt_boxes       = pickle.load(open(gt_boxes_fp,'rb'))

        # loop through ground truth boxes and add create gt_box_list to pass
        # to lyft_dataset_sdk evaluation function
        for box in gt_boxes:
            car_box = move_box_to_car_space(box,image=False)
            box_dict = make_box_dict(car_box,token_list[i],score=False)
            gt_box_list.append(box_dict)
            class_names.append(box.name)

        # loop through anchor boxes and create pred_box_list to pass
        # to lyft_dataset_sdk evaluation function
        for box in final_boxes:
            car_box = move_box_to_car_space(box)
            box_dict = make_box_dict(car_box,token_list[i],score=True)
            pred_box_list.append(box_dict)
        
        gc.collect()

    map_list = []
    class_names = list(set(class_names))
    # get the average precision for each iou threshold - evaluation score
    # is the mean across all thresholds
    for thresh in cfg.DATA.VAL_THRESH_LIST:
        thresh_ap = get_average_precisions(gt_box_list,pred_box_list,
                                           class_names,thresh)
        map_list.append(np.mean(thresh_ap))

    
    return map_list

def write_submission(boxes):
    """
    used for making kaggle submission
    """
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


