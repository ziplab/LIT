# Less is More: Pay Less Attention in Vision Transformers

Training and Evaluation code for LIT-S, LIT-M and LIT-B.



## Training

First, activate your python environment

```bash
conda activate lit
```

Make sure you have the correct ImageNet `DATA_PATH` in `config/*.yaml`. 

To train LIT-S:

```bash
bash scripts/lit-s.sh [GPUs] 
```

To train LIT-M:

```bash
bash scripts/lit-m.sh [GPUs] 
```

To train LIT-B:

```bash
bash scripts/lit-b.sh [GPUs] 
```



## Evaluation

We provide scripts to evaluate LIT-S, LIT-M and LIT-B. To evaluate a model, you can run

```bash
bash scripts/lit-b-eval.sh [GPUs] [path/to/checkpoint]
```

For example, to evaluate LIT-B with 1 GPU, you can run:

```bash
bash scripts/lit-b-eval.sh 1 checkpoint/lit_b.pth
```

This should give

```
* Acc@1 83.366 Acc@5 96.254
Accuracy of the network on the 50000 test images: 83.4%
```

> Result could be slightly different based on you environment.



## Results

| Model | Params (M) | FLOPs (G) | Top-1 Acc. (%) | Model                                                        | Log                                                          |
| ----- | ---------- | --------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LIT-S | 27         | 4.1       | 81.5           | [model](https://drive.google.com/file/d/1WbXspSpUFmiFEeJov4LNWEOLlgUO6eKs/view?usp=sharing) | [log](https://gist.github.com/MonashAI/1cfe7dc8474bb8d0537e160cc9cea971) |
| LIT-M | 48         | 8.6       | 83.0           | [model](https://drive.google.com/file/d/1HYJLmKSYO5rgGWPynzEMEG_TYEqFA0oy/view?usp=sharing) | [log](https://gist.github.com/MonashAI/8e6d37dbfd29cb1a061329b4ef6b1025) |
| LIT-B | 86         | 15.0      | 83.4           | [model](https://drive.google.com/file/d/1EX2CbCVUbc3IVFWdlnRoh7GBWov91iXb/view?usp=sharing) | [log](https://gist.github.com/MonashAI/17321a6b19ebabd5c2956c7d2019b237) |

