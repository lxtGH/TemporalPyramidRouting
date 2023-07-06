
#   Improving Video Instance Segmentation via Temporal Pyramid Routing [[PAMI-2023]](https://arxiv.org/abs/2107.13155)


### Introduction 
To incorporate both temporal and scale information, we propose a Temporal Pyramid Routing (TPR) strategy to conditionally
align and conduct pixel-level aggregation from a feature pyramid pair of two adjacent frames. Specifically, TPR
contains two novel components, including Dynamic Aligned Cell Routing (DACR) and Cross Pyramid Routing (CPR),
where DACR is designed for aligning and gating pyramid features across temporal dimension, while CPR transfers
temporally aggregated features across scale dimension. 


![Figure](./fig/TPR.jpg) 
 

## Requirements

* CUDA 10.1 & cuDNN 7.6.3 & nccl 2.4.8 (optional)
* Python >= 3.6
* PyTorch >= 1.3
* torchvision >= 0.4.2
* OpenCV
* pycocotools
* GCC >= 4.9


## Get Started 

```shell

# Install and compile the requrements.txt and extra cuda operators. We add the masked convolution in extra ops.
pip install -r requirements.txt
python setup.py build develop

# Install customized pycocotools for supporting video instance segmentation
git clone https://github.com/hehao13/pycocotoolsvis.git
cd pycocotoolsvis
python setup.py build develop

# Preprare data path
ln -s /path/to/your/coco/dataset datasets/coco

# Enter a specific experiment dir 
cd playground/coco/blendmask/blendmask_r50_coco_1x
# Train
pods_train --num-gpus 8

# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional
  
  
# Then fine-tune on Youtube-VIS, you can choose different settings accordingly.
cd playground/vis/blendmask_vis/blendmask_r50_1x_track_1x_with_align_limit2_dynamic_bialign_offset_sepc_32211

# Multi node training
pods_train --num-gpus 8 MODEL.WEIGHTS /path/to/pretrained/blendmask


pods_test --num-gpus 1 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional
    
# submit the dumped json to youtube-vis server 

```


## Acknowledgement

cvpods is developed based on Detectron2. For more details about official detectron2, please check [DETECTRON2](https://github.com/facebookresearch/detectron2/blob/master/README.md)


If you find this codebase is useful to your research, please consider cite the paper and original codebase.

```BibTeX

@article{li2022improving,
  title={Improving video instance segmentation via temporal pyramid routing},
  author={Li, Xiangtai and He, Hao and Yang, Yibo and Ding, Henghui and Yang, Kuiyuan and Cheng, Guangliang and Tong, Yunhai and Tao, Dacheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={5},
  pages={6594--6601},
  year={2022},
  publisher={IEEE}
}

@misc{zhu2020cvpods,
  title={cvpods: All-in-one Toolbox for Computer Vision Research},
  author={Zhu*, Benjin and Wang*, Feng and Wang, Jianfeng and Yang, Siwei and Chen, Jianhu and Li, Zeming},
  year={2020}
}
```