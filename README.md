
A Pytorch implementation for Attention-based Domain Adaptation for Single Stage Detectors. This 
repositry is based on https://github.com/lufficc/SSD implementation of SSD. Please follow this repo
for installing the requirements.

### Dataset
For this work, we follow the same dataset setup as [EveryPixelMatters](https://github.com/chengchunhsu/EveryPixelMatters#dataset) 

Modify the [path_catalogs](https://github.com/vidit09/adass/blob/master/ssd/config/path_catlog.py) file in order to point to specific dataset location.

### Training
We train on a single NVIDIA V100 GPU
`python train.py --config-file configs/<adaptation_task>.yaml` 


