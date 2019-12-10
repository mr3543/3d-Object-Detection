#!/bin/bash

pip install easydict
pip install lyft-dataset-sdk
apt-get install libboost-all-dev --yes
pip install --upgrade torch torchvision

g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` PointPillars/data/pillars.cpp -o pillars`python3-config --extension-suffix`

mv pillars* PointPillars/data

ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images
ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps
ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar


