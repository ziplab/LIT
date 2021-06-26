# LIT for Object Detection

This repo contains the supported code and configuration files to reproduce object detection results of [LIT](https://arxiv.org/abs/2105.14217). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).



## Usage

### Installation
1. Make sure you have created your environment with our provide [scripts](). We recommend you create a new environment for experiments with object detection.

   ```bash
   # Suppose you already have an env for training LIT on ImageNet.
   conda create -n lit-det --clone lit
   ```

2. Next, please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for mmdetection installation. 

3. Prepare COCO dataset.

   ```bash
   # Within this directory, do
   ln -s [path/to/coco] data/
   ```

4. Download our [pretrained weights]() on ImageNet and move the weights under `pretrained/`.



### Inference
```bash
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

For example, to evaluate mask-rcnn model with a lit-ti backbone, run:

```bash
tools/dist_test.sh configs/lit/mask_rcnn_lit_ti_fpn_1x_coco.py 1 --eval bbox segm
```



### Training

To train a detector with pre-trained models, run:
```bash
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Mask R-CNN model with a `lit-ti` backbone and 8 gpus, run:
```bash
tools/dist_train.sh configs/lit/mask_rcnn_lit_ti_fpn_1x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used for RetinaNet with LIT-S to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.



## Results

### RetinaNet

| Backbone | Params (M) | Lr schd | box mAP | config | Model | Log  |
| -------- | ---------- | ------- | ------- | ------ | ----- | ---- |
| LIT-Ti   | 30         | 1x      | 41.6    |        |       |      |
| LIT-S    | 39         | 1x      | 41.6    |        |       |      |


### Mask R-CNN

| Backbone | Params (M) | Lr schd | box mAP | mask mAP | config | Model | Log  |
| -------- | ---------- | ------- | ------- | -------- | ------ | ----- | ---- |
| LIT-Ti   | 40         | 1x      | 42.0    | 39.1     |        |       |      |
| LIT-S    | 48         | 1x      | 42.9    | 39.6     |        |       |      |



If you use this code for a paper please cite:

```
@article{pan2021less,
  title={Less is More: Pay Less Attention in Vision Transformers},
  author={Pan, Zizheng and Zhuang, Bohan and He, Haoyu and Liu, Jing and Cai, Jianfei},
  journal={arXiv preprint arXiv:2105.14217},
  year={2021}
}
```

