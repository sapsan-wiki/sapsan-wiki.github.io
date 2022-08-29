---
search:
  boost: 5
---

# Custom Estimator

Sapsan makes it easy to get started on designing your own ML model layer-by-layer.

## Command-line Interface (CLI)
Here is the easiest way to get started, where you should replace `{name}` with your custom project name.

```shell
sapsan create -n {name}
```

This will create the full structure for your project, but in a template form. You will primarily focus on the designing your ML model (estimator). You will find the template for it in

```shell
{name}/{name}_estimator.py
```

The template is structured to utilize a custom backend `sapsan.lib.estimator.torch_backend.py`, hence it revolves around using PyTorch. In the template, you will define the layers your want to use, the order in which they should be executed, and a few custom model parameters (Optimizer, Loss Function, Scheduler). Since we are talking about PyTorch, refer to its [API to define your layers](https://pytorch.org/docs/stable/nn.html).

## Estimator Template

1. {name}Model
    1. define your ML layers
    1. forward function (layer order)
2. {name}Config
    1. set parameters (e.g. number of epochs) - usually set through a high-level interface (e.g. a [jupyter notebook](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/cnn_example.ipynb))
    1. add custom parameters to be tracked by MLflow
3. {name}
    1. Set the [Optimizer](https://pytorch.org/docs/stable/optim.html)
    1. Set the [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
    1. Set the [Scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
    1. Set the Model (based on {name}Model & {name}Config)

```python
"""
Estimator Template

Please replace everything between triple quotes to create
your custom estimator.
"""
import json
import numpy as np

import torch

from sapsan.core.models import EstimatorConfig
from sapsan.lib.estimator.torch_backend import TorchBackend
from sapsan.lib.data import get_loader_shape

class {name_upper}Model(torch.nn.Module):
    # input channels, output channels can be the input to define the layers
    def __init__(self):
        super({name_upper}Model, self).__init__()
        
        # define your layers
        """
        self.layer_1 = torch.nn.Linear(4, 8)
        self.layer_2 = torch.nn.Linear(8, 16)
        """

    def forward(self, x): 

        # set the layer order here
        
        """
        l1 = self.layer_1(x)
        output = self.layer_2(l1)
        """

        return output
    
    
class {name_upper}Config(EstimatorConfig):
    
    # set defaults to your liking, add more parameters
    def __init__(self,
                 n_epochs: int = 1,
                 batch_dim: int = 64,
                 patience: int = 10,
                 min_delta: float = 1e-5, 
                 logdir: str = "./logs/",
                 lr: float = 1e-3,
                 min_lr = None,                 
                 *args, **kwargs):
        self.n_epochs = n_epochs
        self.batch_dim = batch_dim
        self.logdir = logdir
        self.patience = patience
        self.min_delta = min_delta
        self.lr = lr
        if min_lr==None: self.min_lr = lr*1e-2
        else: self.min_lr = min_lr
        self.kwargs = kwargs
        
        #everything in self.parameters will get recorded by MLflow
        #by default, all 'self' variables will get recorded
        self.parameters = {{f'model - {{k}}': v for k, v in self.__dict__.items() if k != 'kwargs'}}
        if bool(self.kwargs): self.parameters.update({{f'model - {{k}}': v for k, v in self.kwargs.items()}})
    
    
class {name_upper}(TorchBackend):
    # Set your optimizer, loss function, and scheduler here
    
    def __init__(self, loaders,
                       config = {name_upper}Config(), 
                       model = {name_upper}Model()):
        super().__init__(config, model)
        self.config = config
        self.loaders = loaders
        
        #uncomment if you need dataloader shapes for model input
        #x_shape, y_shape = get_shape(loaders)
        
        self.model = {name_upper}Model()
        self.optimizer = """ optimizer """
        self.loss_func = """ loss function """
        self.scheduler = """ scheduler """        
        
    def train(self):
        
        trained_model = self.torch_train(self.loaders, self.model, 
                                         self.optimizer, self.loss_func, self.scheduler, 
                                         self.config)
                
        return trained_model
```

## Editing Catalyst Runner

For majority of applications, you won't need to touch Catalyst Runner settings, which located in `torch_backend.py`. However, in case you would like to dig further into more unique loss functions, optimizers, data distribution setups, then you can copy the `torch_backend.py` via `--get_torch_backend` or shorthand `--gtb` flag during the creation of the project:

```shell
sapsan create --gtb -n {name}
```
or just copy it to your current directory by:

```shell
sapsan gtb
```

For [`runner` types](https://catalyst-team.github.io/catalyst/api/runners.html) and extensive options please refer to [Catalyst Documentation](https://catalyst-team.github.io/catalyst/index.html).

As for runner adjustments to parallele your training, Sapsan's Wiki includes a page on [Parallel GPU Training](/tutorials/parallelgpu/).

### Loss

Catalyst includes a more extensive list of *losses*, i.e. *criterions*, than the standard PyTorch. Their implementations might require to include some extra `Callback`s to be specified in the runner ([Criterion Documentation](https://catalyst-team.github.io/catalyst/api/contrib.html#criterion)). Please refer to Catalyst examples to create your own loss functions.

### Optimizer

Similar deal is with the optimizer. While using standard (i.e. Adam) can be specified within Estimator Template, for a more complex or custom setup you will need to refer to the runner ([Optimizer Documentation](https://catalyst-team.github.io/catalyst/api/contrib.html#optimizers)).