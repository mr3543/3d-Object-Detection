## 3D object Detection for Autonomous Driving

This repo implements a verison of [PointPillars](https://arxiv.org/abs/1812.05784) for detecting objects in 3d lidar point clouds. Our dataset is the [Lyft Level 5](https://level5.lyft.com/dataset/) dataset which contains over 17,000 lidar sweeps and full sensor readings. The full dataset is over 200gb. 

## Dataset Overview

The dataset is broken up into multiple scenes, each scene contains a number of samples. A sample is a group of sensor readings that occur roughly at the same time stamp. For example a sample may contain a full lidar sweep, front, back and side camera images, and a full radar sweep. Our implementation relies only on the lidar data. A sample also contains a set of 3D ground truth bounding boxes which denote objects in the environment. The Lyft Dataset API allows us the access the sample data through a unique hash string called a sample token. The lidar data that we use is given in a coordinate system with the lidar sensor as the origin. The API allows us to transform that coordinate system so the origin is at the center-back of the car. During our pre-processing we will transfor the lidar points to this frame of reference. 

## Data Pre-processing 

Our training-validation split is done on the basis of host car. This is arbitrary and other splits are equally valid. The script `train_prep.py` does a number of preprocessing steps. In  `train_prep.py` we loop through all the sample tokens in our training and validation sets and gather a list of the file paths to the lidar data and ground truth boxes. We store the transformation matrix from sensor to car space, along with the ground truth box file paths in a dict `data_dict` which we can index with the lidar data filepath, i.e

```python
data_dict[lidar_file] = {'boxes':box_filepath,'trans_matrix': car_from_sensor}.
```

We also store various quatities related to the anchor boxes as pickle files. These will be used later for computing IoUs with the ground truth boxes. 

## Model

Our Pytorch model is located in `model/model.py`. The main feature of the model that allows us to take the lidar data and repurpose it for object detection is implemented as `PPFeatureNet`. The idea is to group lidar points into pillars, thus forming a 2D grid on lidar space. From each point in a pillar we create a feature of length `D`. If we have `P` pillars and `N` points in each pillar then we can create a tensor of size `[D,P,N]` which is fed through a 1x1 convolutional network with `C` output channels giving us a `[C,P,N]` tensor. For each of the `P` pillars we rememeber its `(x,y)` location on the grid and scatter the `[C,P,N]` tensor back into a tensor of size `[H,W,C,N]`. Lastly we take the max over the last axis and we are left with a tensor of `[H,W,C]` (transposed to `[C,H,W]`), which we can feed into a standard 2D detection architecture.

In our implementation, constructing the `[D,P,N]` tensor is done when a new batch of data fetched from the dataloader. To do this quicly we use a python C++ extension `pybind11`. The code for transforming the raw lidar data to the network input is in `data/pillars.cpp`. 

## Loss

The loss function is similar to other 2D object detection losses in that there is a classification and a regression loss that are combined to form the total loss. The classification loss uses [Focal Loss](https://arxiv.org/abs/1708.02002) to handle the problem of a large imbalance between background and object boxes. The regression loss uses smooth L1 loss to learn bounding box offsets as well as the height of the bounding box and it's rotation. Unlike other implementations, notably [SECOND](https://www.mdpi.com/1424-8220/18/10/3337) we do not use an orientation classifier for the direction of the predicted box. Instead we rotate the ground truth boxes by 180 degrees during pre-processing so all the predicted yaws are positive. 

## Traning Loop

For making the training targets we again use `pybind11` to compute the anchor box/ground truth box IoUs. For this we utilize the [Boost Geometry](https://www.boost.org/doc/libs/1_65_1/libs/geometry/doc/html/index.html) library which implements the API to do arbitrary polygon IoU. Due to the fact that some ground truth boxes are not axis aligned, using arbitrary polygon IoU was a quick and simple choice. This code is implemented in `data/pillars.cpp`. We use the [one cycle learning policy](https://arxiv.org/abs/1803.09820) from Leslie Smith to train the model.  

## Results 

We do not implement any of the data augmentation from PointPillars or SECOND. After training for 20 epochs on 4 NVIDIA RTX 2080 ti GPUs we get a validation mAP of 0.078. This puts our implementation in the top 30/547 teams on kaggle. 
