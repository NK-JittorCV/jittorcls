# JittorCLS

## A codebase for image classification based on Jittor


## Getting Started
### Install Jittor
```shell
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
# If your computer contains an Nvidia graphics card, check the cudnn acceleration library
python3.7 -m jittor.test.test_cudnn_op
```
For more information on how to install jittor, you can check [here](https://cg.cs.tsinghua.edu.cn/jittor/download/).

### Install OpenMPI
```shell
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```
To obtain more information about OpenMPI, you can check [here](https://www.open-mpi.org/faq/?category=building#easy-build).

### Train
We provide scripts for single-machine single-gpu, single-machine multi-gpu training. Multi-gpu dependence can be referred to [here](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-44-distributed/)
```shell
# Single GPU
bash train.sh

# Multiple GPUs
bash dist_train.sh
```

### Test
```shell
# Single GPU
bash test.sh

# Multiple GPUs
bash dist_test.sh
```
