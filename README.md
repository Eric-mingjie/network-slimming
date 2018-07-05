# Network Slimming (Pytorch)

This repository contains an official pytorch implementation for the following paper  
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  
[Zhuang Liu](https://liuzhuang13.github.io/), [Jianguo Li](https://sites.google.com/site/leeplus/), [Zhiqiang Shen](http://zhiqiangshen.com/), [Gao Huang](http://www.cs.cornell.edu/~gaohuang/), [Shoumeng Yan](https://scholar.google.com/citations?user=f0BtDUQAAAAJ&hl=en), [Changshui Zhang](http://bigeye.au.tsinghua.edu.cn/english/Introduction.html).

The code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming). We add support for `Resnet` and `Densenet`.  

Citation:
```
@InProceedings{Liu_2017_ICCV,
    author = {Liu, Zhuang and Li, Jianguo and Shen, Zhiqiang and Huang, Gao and Yan, Shoumeng and Zhang, Changshui},
    title = {Learning Efficient Convolutional Networks Through Network Slimming},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
}
```
Original torch implementation: [Torch](https://github.com/liuzhuang13/slimming) by [Zhuang Liu](https://liuzhuang13.github.io/).

## Dependencies
torch v0.3.1, torchvision v0.2.0

## Channel Selection layer
We introduce `channel selection` layer to help the pruning process of Resnet and Densenet. This layer is easy to implement. It stores a parameter `indexes` which is initialized to an all-1 vector. During pruning, it will set some places to 0 which correspond to the pruned channels.

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg`,`resnet` or
`densenet`. The depth is chosen to be the same as the networks used in the paper.
```shell
python main.py --dataset cifar10 --arch vgg --depth 19
python main.py --dataset cifar10 --arch resnet --depth 164
python main.py --dataset cifar10 --arch densenet --depth 40
```

## Train with Sparsity

```shell
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 164
python main.py -sr --s 0.00001 --dataset cifar10 --arch densenet --depth 40
```

## Prune

```shell
python vggprune.py --dataset cifar10 --arch vgg --depth 19 --percent 0.7 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python resprune.py --dataset cifar10 --arch resnet --depth 164 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python denseprune.py --dataset cifar10 --arch densenet --depth 40 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.

## Fine-tune

```shell
python main.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19 --epochs 160
```

## Results
### CIFAR10
|  CIFAR10-Vgg  | Baseline |  Sparsity (1e-4) | Prune (70%) | Fine-tune-160 |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  93.77   |            93.30            |        32.54        |         93.78         |
|    Parameters     |  20.04M  |            20.04M            |        2.25M        |         2.25M         |

|  CIFAR10-Resnet-164  | Baseline |    Sparsity (1e-5) | Prune(60%) | Fine-tune-160(0.6) |   Prune(40%)     |  Fine-tune-160(0.4)       |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |  :----------------:| :--------------------:|
| Top1 Accuracy (%) |  94.75   |            94.76             |        47.73       |         --         |      94.58       |     --     |
|    Parameters     |  1.71M  |             1.73M            |        1.12M        |         1.12M         |      1.45M          |   1.45M           |

|  CIFAR10-Densenet-40  | Baseline |  Sparsity (1e-5) | Prune (40%) | Fine-tune-160(0.4) |       Prune(60%)   | Fine-tune-160(0.6) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: | :--------------------: | :-----------------:|
| Top1 Accuracy (%) |  94.11   |           94.17             |        94.16       |         --         |      89.46       |     --     |
|    Parameters     |  1.07M  |            1.07M            |        0.69M       |         0.69M         |       0.49M      |    0.49M     |

### CIFAR100
|  CIFAR100-Vgg  | Baseline |   Sparsity (1e-4) | Prune (50%) | Fine-tune-160 |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |   72.12   |            72.05             |         5.31        |         73.32         |
|    Parameters     |  20.04M  |            20.04M            |        4.93M        |         4.93M         |

|  CIFAR100-Resnet-164  | Baseline |   Sparsity (1e-5) | Prune (40%) | Fine-tune-160 |    Prune(60%)  | Fine-tune-160(0.6) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  -----   |            ---             |        ---        |         ---         |  ---       |     --     |
|    Parameters     |  1.73M  |            1.73M            |        ---        |         ---         |---       |     --     |

|  CIFAR100-Densenet-40  | Baseline |    Sparsity (1e-5) | Prune (40%) | Fine-tune-160 | Prune(60%)  | Fine-tune-160(0.6) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  73.27   |          73.29            |        67.67        |         ---         |   19.18       |     --     |
|    Parameters     |  1.10M  |            1.10M            |        0.71M        |         0.71M         |  0.50M       |     0.50M    |

## Contact
liuzhuangthu at gmail.com  
sunmj15 at gmail.com 