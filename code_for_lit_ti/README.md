# Less is More: Pay Less Attention in Vision Transformers

Training and evaluation code for LIT-Ti.



## Training

First, activate your conda virtual environment.

```bash
conda activate lit
```

Make sure you have the correct ImageNet `data_path` in `config/lit-ti.json`. 

To train LIT-Ti, run

```bash
bash scripts/train_lit.sh [GPUs]
```

You can set a different batch size by editing `batch_size` in `config/lit-ti.json`. 



## Evaluation

To evaluate LIT-Ti on ImageNet, run

```bash
bash scripts/eval_lit.sh [GPUs] [Checkpoint]
```

For example, to evaluate LIT-Ti with one GPU, you can run:

```bash
bash scripts/eval_lit.sh 1 checkpoint/lit_ti.pth
```

This should give

```
* Acc@1 81.124 Acc@5 95.544 loss 0.901
Accuracy of the network on the 50000 test images: 81.1%
```

> Result could be slightly different based on you environment.



## Results

| Name   | Params (M) | FLOPs (G) | Top-1 Acc. (%) | Model                                                        | Log                                                          |
| ------ | ---------- | --------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LIT-Ti | 19         | 3.6       | 81.1           | [google drive](https://drive.google.com/file/d/19X3u-0BtXXZRlWZeSe5e-Z0ocS6rWCFb/view?usp=sharing)/[github](https://github.com/MonashAI/LIT/releases/download/v1.0/lit_ti.pth) | [log](https://github.com/MonashAI/LIT/releases/download/v1.0/log_lit_ti.txt) |



