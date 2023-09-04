# SBNet: Segmentation-based Network for Natural Language-based Vehicle Search

CVPRW 2021

## requirements

* pytorch > 1.5
* yacs
* albumentations
* nltk (with nltk.download('stopwords'))

## train_dist

#### Using 4 GPU (RTX 1080ti)

Require 1~2 days for 10 epochs

```bash
python -m torch.distributed.launch train_dist.py
```

## test_dist

Requires 1 day for final submission

```bash
python -m torch.distributed.launch test_dist.py
```

```bash
python combine.py
```
