---
search:
  boost: 5
---
# Installation

## Required

### 1. Install PyTorch (prerequisite)

Sapsan can be run on both cpu and gpu. Below are the requirements for each version

| Device | | |
|-------------|-------------|-------------|
| CPU | torch>=1.9.0       | torchvision>=0.10.0       |
| GPU | torch>=1.9.0+cu111 | torchvision>=0.10.0+cu111 |

Please follow the instructions on [PyTorch](https://pytorch.org/get-started/locally/) to install either version. `CUDA>=11.1` can be installed directly with PyTorch as well.

#### Install CUDA-Toolkit

If you are planning to train on GPU, then you need to check that you have an `nvidia-cuda-toolkit` installed:
```
nvcc --version
```
if it is missing, you can install it via:
```
sudo apt install nvidia-cuda-toolkit
```

!!! Info "Device"
    If GPU is available, Sapsan will always try to run on a GPU by default (that includes the tests). However, you can specify the `device='cpu'` in model config.

### 2a. Install via pip (recommended)
```
pip install sapsan
```

### 2b. Clone from github (alternative)
```
git clone https://github.com/pikarpov-LANL/Sapsan.git
cd Sapsan/
python setup.py install
```

If you experience any issues, you can try installing packages individually with:
```
pip install -r requirements.txt
```

!!! Info "Version"
    Make sure you are using the latest release version!

<br/>

## Optional
### Install Graphviz
In order to create model graphs, Sapsan is using [graphviz](https://graphviz.org/). If you would like to utilize this functionality, then please install graphviz via:
```
conda install graphviz
```
or
```
sudo apt-get install graphviz
```

<br/>

### Install Docker

In order to run Sapsan through Docker or build your own container to share, you will need to install it
```
pip install docker
```

Next, you can build a docker setup with the following:

```
make build-container
```

this will create a container named `sapsan-docker`.

If you want to run the container, type:

```
make run-container
```
a Jupyter notebook will be launched at `localhost:7654`

## Troubleshooting

### libGL.so error
If you get the following error:
```shell
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
your `opencv-python` package has some dependency issues. To resolve, try the following:
```shell
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
```
### My Kernel is Dying! :material-ghost:
If when attempting to train the model (executing `Train.run()`) the kernel is dying without any errors, check if you have a GPU on your machine while not having a `nvidia-cuda-toolkit` installed. To resolve, you can either install it:
```
sudo apt install nvidia-cuda-toolkit
```
or specify `device='cpu'` in your model config. By defaul Sapsan always tries to train the models on a GPU if one is available.