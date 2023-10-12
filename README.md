# Multiple Relations Classification using Imbalanced Predictions Adaptation

## Requirements
- Python 3.9
- TensorFlow 2.11
- TQDM 4

**Install Requirements**
```
python3 -m pip install tensorflow==2.11.*
python3 -m pip install tqdm
```

## Pre-trained Language Model
* RCEP achieves its best performance using **Glove word representation** (6B tokens, 400K vocab, uncased, 300d vectors)

[Download Glove](https://nlp.stanford.edu/projects/glove/)

## Datasets
Pre-processed data is available in `data` folder. Raw datasets can be downloaded using the links below:
- [NYT](https://github.com/xiangrongzeng/copy_re)
- [WEBNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)

## Run Training
* The below train/test scripts uses the reported hyperparameters in the paper.
* Please review `run.sh` to check the default arguments.

**WEBNLG**
```
./run.sh train webnlg_checkpoint data/webnlg > output.out &
```
**NYT**
```
./run.sh train nyt_checkpoint data/nyt > output.out &
```
