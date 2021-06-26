# LIT for Semantic Segmentation

This repo contains the supported code and configuration files to reproduce semantic segmentation results of [LIT](https://arxiv.org/abs/2105.14217). It is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).



## Usage

### Installation

1. Make sure you have created your environment with our provide [scripts](https://github.com/MonashAI/LIT/blob/main/setup_env.sh). We recommend you create a new environment for experiments with semantic segmentation.

   ```bash
   # Suppose you already have an env for training LIT on ImageNet.
   conda create -n lit-seg --clone lit
   ```

2. Next, please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for mmsegmentation installation.

3. Prepare ADE20K dataset.

   ```bash
   # Within this directory, do
   ln -s [path/to/ade20k] data/
   ```

4. Download our [pretrained weights](https://github.com/MonashAI/LIT/tree/main/classification) on ImageNet and move the weights under `pretrained/`.



### Inference

```bash
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU
```

For example, to evaluate a Semantic FPN model with a `lit-ti` backbone, run:

```bash
tools/dist_test.sh configs/lit/lit_ti_fpn_r50_512x512_80k_ade20k.py [path/to/checkpoint] 1 --eval mIoU
```



### Training

To train a detector with pre-trained models, run:

```bash
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```

For example, to train a Semantic FPN model with a `lit-ti` backbone and 8 gpus, run:

```bash
tools/dist_train.sh configs/lit/lit_ti_fpn_r50_512x512_80k_ade20k.py 8 --options model.pretrained=<PRETRAIN_MODEL>
```



## Results

### Semantic FPN

| Backbone | Params (M) | Iters | mIoU | Config                                                       | Model                                                        | Log                                                          |
| -------- | ---------- | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LIT-Ti   | 24         | 8k    | 41.3 | [config](https://github.com/MonashAI/LIT/blob/main/segmentation/configs/lit/lit_ti_fpn_r50_512x512_80k_ade20k.py) | [github](https://github.com/MonashAI/LIT/releases/download/v2.0/sem_fpn_lit_ti.pth) | [log](https://github.com/MonashAI/LIT/releases/download/v2.0/sem_fpn_lit_ti_log.json) |
| LIT-S    | 32         | 8k    | 41.7 | [config](https://github.com/MonashAI/LIT/blob/main/segmentation/configs/lit/lit_s_fpn_r50_512x512_80k_ade20k.py) | [github](https://github.com/MonashAI/LIT/releases/download/v2.0/sem_fpn_lit_s.pth) | [log](https://github.com/MonashAI/LIT/releases/download/v2.0/sem_fpn_lit_s_log.json) |



If you use this code for a paper please cite:

```
@article{pan2021less,
  title={Less is More: Pay Less Attention in Vision Transformers},
  author={Pan, Zizheng and Zhuang, Bohan and He, Haoyu and Liu, Jing and Cai, Jianfei},
  journal={arXiv preprint arXiv:2105.14217},
  year={2021}
}
```

