# Network Slimming (Pytorch)

This repository contains an official pytorch implementation for the following paper  
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  
[Zhuang Liu](https://liuzhuang13.github.io/), [Jianguo Li](https://sites.google.com/site/leeplus/), [Zhiqiang Shen](http://zhiqiangshen.com/), [Gao Huang](http://www.cs.cornell.edu/~gaohuang/), [Shoumeng Yan](https://scholar.google.com/citations?user=f0BtDUQAAAAJ&hl=en), [Changshui Zhang](http://bigeye.au.tsinghua.edu.cn/english/Introduction.html).  

Original implementation: [slimming](https://github.com/liuzhuang13/slimming) in Torch.    
The code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming). We add support for ResNet and DenseNet.  

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


## Dependencies
torch v0.3.1, torchvision v0.2.0

## Channel Selection Layer
We introduce `channel selection` layer to help the  pruning of ResNet and DenseNet. This layer is easy to implement. It stores a parameter `indexes` which is initialized to an all-1 vector. During pruning, it will set some places to 0 which correspond to the pruned channels.

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
python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python resprune.py --dataset cifar10 --depth 164 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python denseprune.py --dataset cifar10 --depth 40 --percent 0.4 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.

## Fine-tune

```shell
python main.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19 --epochs 160
```

## Results

The results are fairly close to the original paper, whose results are produced by Torch. Note that due to different random seeds, there might be up to ~0.5%/1.5% fluctation on CIFAR-10/100 datasets in different runs, according to our experiences.
### CIFAR10
|  CIFAR10-Vgg  | Baseline |  Sparsity (1e-4) | Prune (70%) | Fine-tune-160(70%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  93.77   |            93.30            |        32.54        |         93.78         |
|    Parameters     |  20.04M  |            20.04M            |        2.25M        |         2.25M         |

|  CIFAR10-Resnet-164  | Baseline |    Sparsity (1e-5) | Prune(40%) | Fine-tune-160(40%) |   Prune(60%)     |  Fine-tune-160(60%)       |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |  :----------------:| :--------------------:|
| Top1 Accuracy (%) |  94.75   |            94.76             |        94.58       |         95.05         |      47.73       |     93.81     |
|    Parameters     |  1.71M  |             1.73M            |        1.45M        |         1.45M         |      1.12M          |   1.12M           |

|  CIFAR10-Densenet-40  | Baseline |  Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) |       Prune(60%)   | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: | :--------------------: | :-----------------:|
| Top1 Accuracy (%) |  94.11   |           94.17             |        94.16       |         94.32         |      89.46       |     94.22     |
|    Parameters     |  1.07M  |            1.07M            |        0.69M       |         0.69M         |       0.49M      |    0.49M     |

### CIFAR100
|  CIFAR100-Vgg  | Baseline |   Sparsity (1e-4) | Prune (50%) | Fine-tune-160(50%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |   72.12   |            72.05             |         5.31        |         73.32         |
|    Parameters     |  20.04M  |            20.04M            |        4.93M        |         4.93M         |

|  CIFAR100-Resnet-164  | Baseline |   Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) |    Prune(60%)  | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  76.79   |            76.87             |        48.0        |         77.36        |  ---       |     ---     |
|    Parameters     |  1.73M  |            1.73M            |        1.49M        |         1.49M         |---       |     ---     |

Note: For results of pruning 60% of the channels for resnet164-cifar100, in this implementation, sometimes some layers are all pruned and there would be error. However, we also provide a [mask implementation](https://github.com/Eric-mingjie/network-slimming/tree/master/mask-impl) where we apply a mask to the scaling factor in BN layer. For mask implementaion, when pruning 60% of the channels in resnet164-cifar100, we can also train the pruned network.

|  CIFAR100-Densenet-40  | Baseline |    Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) | Prune(60%)  | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  73.27   |          73.29            |        67.67        |         73.76         |   19.18       |     73.19     |
|    Parameters     |  1.10M  |            1.10M            |        0.71M        |         0.71M         |  0.50M       |     0.50M    |

## Contact
sunmj15 at gmail.com 
liuzhuangthu at gmail.com  
