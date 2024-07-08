# CSFF-VD:Software Vulnerability Detection Based on Correlation of Structural Features between Functions

## Introduction

We proposed a new method named CSFF-VD, a new software vulnerability detection method which utilizes the correlation of structural features between functions. This method extracts the structural features of functions with GGNN and then calculates the correlation between them. Then method establishes new directed edges for nodes to represent the correlation between functions and constructs an association network among functions. Finally method uses GAT to extract structural similarity information between functions, thereby enhancing vulnerability detection performance.


## Dataset 

We conductd experiments on the following datasets:

|   Datasets    | #Samples | #Vul  | #Non-vul | Vul Ratio |
| :-----------: | :------: | :---: | :------: | :-------: |
|    FFmpeg     |  9,768   | 4,981 |  4,788   |  51.10%   |
|     Qemu      |  17,549  | 7,479 |  10,070  |  42.62%   |
| Chrome+Debian |  22,734  | 2,240 |  20,494  |   9.85%   |

These datasets are divided into three parts: training set, validation set, and test set and saved in the './data-process' folder.

## Implementation

### Environment Prerequisites

This experiment is implemented using conda v24.5.0 environment.

1. Create a conda environment with `environment.yml` file: `conda env create -f environment.yml`
2. Activate the environment: `conda activate CSFF-VD`
 
### Experiment

#### Download Dataset 

You can download the dataset from the following link:

https://drive.google.com/file/d/1M_OA3ZduIGVb0GvkxzKi3NkeJgCR1Omy/view?usp=drive_link

Then unzip the `embeding.zip` file and put it in the project folder.

#### Train GGNN(Actually done already, you can skip this step)
Before starting training GGNN, you need to choose the dataset you want to train GGNN. Here are the available datasets:

- ffmpeg
- qemu
- reveal (aka Chrome+Debian)

You can configure the dataset by changing the `dataset` variable in `train_GGNN_model.py`.

After changing the dataset, you can start training GGNN by running the following command:

```bash
python train_GGNN_model.py
```

The result can be found in the `./result/GGNN` folder.

#### Function Embedding(Actually done already, you can skip this step)

In this process, each function will be embedded into a vector using model trained in the previous step and edges between will be built after embedding. It is required since GAT will utilize these information to predict. Before starting training GAT, you need to choose the dataset you want. Here are the available datasets:

- ffmpeg
- qemu
- reveal (aka Chrome+Debian)

You can configure the dataset by changing the `dataset` variable in `construction.py`.

After changing the dataset, you can start training GAT by running the following command:

```bash
python construction.py
```

The embedded function vectors can be found in the `./embeding/` folder.

#### Train GAT and evaluate

After finishing the previous step, you can start training GAT and evaluate the performance of CSFF-VD.

But before starting training GAT, you need to choose the dataset you want. Here are the available datasets:
- ffmpeg
- qemu
- reveal (aka Chrome+Debian)

You can configure the dataset by changing the `opt_project` variable in `run_CSFF-VD.py`.

```bash
python run_CSFF-VD.py
```

The result can be found in the `./result/CSFF-VD` folder.
