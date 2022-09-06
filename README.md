
Pytorch implementation for [Attention-based Domain Adaptation for Single Stage Detectors.](https://link.springer.com/article/10.1007/s00138-022-01320-y)

### Getting Started
This repositry is based on https://github.com/lufficc/SSD implementation of SSD. Please follow this repo
for installing the requirements and train/test procedure. Our code was run with following versions

1. Pytorch == 1.6

2. Python >=3.6

### Dataset
For this work, we follow the same dataset setup as [EveryPixelMatters](https://github.com/chengchunhsu/EveryPixelMatters#dataset). 

Modify the [path_catalogs](https://github.com/vidit09/adass/blob/master/ssd/config/path_catlog.py) file in order to point to specific dataset location.

### Training
We train on a single NVIDIA V100 GPU.

`python train.py --config-file configs/<adaptation_task>.yaml` 

### Attention Module

Our attention head [implementation](https://github.com/vidit09/adass/blob/master/ssd/modeling/backbone/vgg.py#L150) follows the [Detr's implementation](https://github.com/facebookresearch/detr) and used in the [domain classifier](https://github.com/vidit09/adass/blob/master/ssd/modeling/domain_classifier/domain_classifier.py#L62). The same arch design is followed in YOLO implementation. 


### Citation
```bibtex
@article{vidit2022attention,
  title={Attention-based domain adaptation for single-stage detectors},
  author={Vidit, Vidit and Salzmann, Mathieu},
  journal={Machine Vision and Applications},
  volume={33},
  number={5},
  pages={1--14},
  year={2022},
  publisher={Springer}
}
```
