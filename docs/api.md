---
hide:
  - navigation
---

# API Reference

## Glossary
| Variable | Definition |
| -------- | ---------- |
| N        | # of Batches |
|C~in~ | # of input channels (i.e. features) |
|D or D~b~ | Data or Batch depth (z) |
|H or H~b~| Data or Batch height (y) |
|W or W~b~| Data or Batch width (x) |

## Train/Evaluate

---

### Train
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.experiments.train.Train`_`(model: Estimator, data_parameters: dict, backend = FakeBackend(), show_log = True, run_name = 'train')`_

: Call `Train` to set up your run    

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `model`   | object | model to use for training| |
    | <nobr>`data_parameters`</nobr> | dict | data parameters from the data loader, necessary for tracking | |
    | `backend` | object | backend to track the experiment | FakeBackend() |
    | `show_log`| bool | show the loss vs. epoch progress plot (it will be save in mlflow in either case) | True |
    | `run_name`| str | 'run name' tag as recorded under MLflow | train |

!!! code ""
    `sapsan.lib.experiments.train.Train.run()`

: Run the model

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | pytorch or sklearn or custom type| trained model | 

---

### Evaluate
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.experiments.evaluate.Evaluate`_`(model: Estimator, data_parameters: dict, backend = FakeBackend(), cmap: str = 'plasma', run_name: str = 'evaluate', **kwargs)`_

: Call `Evaluate` to set up the testing of the trained model. Don't forget to update `estimator.loaders` with the new data for testing.

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `model` | object | model to use for testing |  |
    | <nobr>`data_parameters`</nobr> | dict | data parameters from the data loader, necessary for tracking |  |
    | `backend` | obejct | backend to track the experiment | FakeBackend() |
    | `cmap` | str | matplotlib colormap to use for slice plots | plasma |
    | `run_name` | str | 'run name' tag as recorded under MLflow | evaluate |
    | `pdf_xlim`| tuple | x-axis limits for the PDF plot | |
    | `pdf_ylim`| tuple | y-axis limits for the PDF plot | |

!!! code ""
    `sapsan.lib.experiments.evaluate.Evaluate.run()`

: Run the evaluation of the trained model

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | dict{'target' : np.ndarray, 'predict' : np.ndarray}  | target and predicted data | 


## Estimators

---

### CNN3d
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.CNN3d`_`(loaders, config, model)`_

: A model based on Pytorch's [3D Convolutional Neural Network]( /details/estimators/#convolution-neural-network-cnn)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `loaders` | dict | contains input and target data (loaders['train'], loaders['valid']). Datasets themselves have to be torch.tensor(s) | CNN3dConfig() |    
    | `config` | class | configuration to use for the model | CNN3dConfig() |
    | `model` | class | the model itself - should not be adjusted | CNN3dModel() |    

!!! code ""
    `sapsan.lib.estimator.CNN3d.save`_`(path: str)`_

: Saves model and optimizer states, as well as final epoch and loss

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively |  |

!!! code ""    
    `sapsan.lib.estimator.CNN3d.load`_`(path: str, estimator, load_saved_config = False)`_

: Loads model and optimizer states, as well as final epoch and loss

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively |  |    
    | `estimator` | estimator | need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further. |  |
    | `load_saved_config` | bool | updates config parameters from `{path}/params.json`. | False |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | pytorch model | loaded model | 

---

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.CNN3dConfig`_`(n_epochs, patience, min_delta, logdir, lr, min_lr, *args, **kwargs)`_

: Configuration for the CNN3d - based on pytorch and catalyst libraries

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `n_epochs` | int | number of epochs | 1 |    
    | `patience` | int | number of epochs with no improvement after which training will be stopped. Default | 10 |
    | `min_delta` | float | minimum change in the monitored metric to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement | 1e-5 |
    | `log_dir` | int | path to store the logs| ./logs/ |
    | `lr` | float | learning rate | 1e-3 |
    | `min_lr` | float | a lower bound of the learning rate  for [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) | lr\*1e-2 |
    | `device` | str | specify the device to run the model on | cuda (or switch to cpu)
    | <nobr>`loader_key`</nobr> | str | the loader to use for early stop: *train* or *valid* | first loader provided*, which is usually 'train' |    
    | <nobr>`metric_key`</nobr> | str | the metric to use for early stop | 'loss' |    
    | `ddp` | bool | turn on Distributed Data Parallel (DDP) in order to distribute the data and train the model across multiple GPUs.  This is passed to Catalyst to activate the `ddp` flag in `runner` (see more [Distributed Training Tutorial](https://catalyst-team.github.io/catalyst/tutorials/ddp.html); the `runner` is set up in [pytorch_estimator.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/lib/estimator/pytorch_estimator.py)). **Note: doesn't support jupyter notebooks - prepare a script!** | False |

---

### PIMLTurb
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.PIMLTurb`_`(activ, loss, loaders, ks_stop, ks_frac, ks_scale, l1_scale, l1_beta, sigma, config, model)`_

: Physics-informed machine learning model to predict Reynolds-like stress tensor, $Re$, for turbulence modeling. Learn more on the wiki: [PIMLTurb]( /details/estimators/#physics-informed-cnn-for-turbulence-modeling-pimlturb)

: A custom loss function was developed for this model combining spatial (SmoothL1) and statistical (Kolmogorov-Smirnov) losses.

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- | 
    | `activ` | str | activation function to use from PyTorch | Tanhshrink |
    | `loss` | str | loss function to use; accepts only custom | SmoothL1_KSLoss |
    | `loaders` | dict | contains input and target data (loaders['train'], loaders['valid']). Datasets themselves have to be torch.tensor(s) |  |    
    | `ks_stop` | float | early-stopping condition based on the KS loss value alone | 0.1 |
    | `ks_frac` | float | fraction the KS loss contributes to the total loss | 0.5 |
    | `ks_scale` | float | scale factor to prioritize KS loss over SmoothL1 (should not be altered) | 1 |
    | `l1_scale` | float | scale factor to prioritize SmoothL1 loss over KS | 1 |
    | `l1_beta` | float | $beta$ threshold for smoothing the L1 loss | 1 |
    | `sigma` | float | $sigma$ for the last layer of the network that performs a filtering operation using a Gaussian kernel | 1 |        
    | `config` | class | configuration to use for the model | PIMLTurbConfig() |
    | `model` | class | the model itself - should not be adjusted | PIMLTurbModel() |    

!!! code ""
    `sapsan.lib.estimator.PIMLTurb.save`_`(path: str)`_

: Saves model and optimizer states, as well as final epoch and loss

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively |  |

!!! code ""    
    `sapsan.lib.estimator.PIMLTurb.load`_`(path: str, estimator, load_saved_config = False)`_

: Loads model and optimizer states, as well as final epoch and loss

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively |  |    
    | `estimator` | estimator | need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further. |  |
    | `load_saved_config` | bool | updates config parameters from `{path}/params.json`. | False |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | pytorch model | loaded model | 

---

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.PIMLTurbConfig`_`(n_epochs, patience, min_delta, logdir, lr, min_lr, *args, **kwargs)`_

: Configuration for the PIMLTurb - based on pytorch (catalyst is not used)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `n_epochs` | int | number of epochs | 1 |    
    | `patience` | int | number of epochs with no improvement after which training will be stopped _(not used)_ | 10 |
    | `min_delta` | float | minimum change in the monitored metric to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement _(not used)_ | 1e-5 |
    | `log_dir` | int | path to store the logs| ./logs/ |
    | `lr` | float | learning rate | 1e-3 |
    | `min_lr` | float | a lower bound of the learning rate  for [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) | lr\*1e-2 |
    | `device` | str | specify the device to run the model on | cuda (or switch to cpu)
    | <nobr>`loader_key`</nobr> | str | the loader to use for early stop: *train* or *valid* | first loader provided*, which is usually 'train' |    
    | <nobr>`metric_key`</nobr> | str | the metric to use for early stop | 'loss' |    
    | `ddp` | bool | turn on Distributed Data Parallel (DDP) in order to distribute the data and train the model across multiple GPUs.  This is passed to Catalyst to activate the `ddp` flag in `runner` (see more [Distributed Training Tutorial](https://catalyst-team.github.io/catalyst/tutorials/ddp.html); the `runner` is set up in [pytorch_estimator.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/lib/estimator/pytorch_estimator.py)). **Note: doesn't support jupyter notebooks - prepare a script!** | False |

---

### PICAE
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.PICAE`_`(loaders, config, model)`_

: Convolutional Auto Encoder with Divergence-Free Kernel and with periodic padding. Further details can be found on the [PICAE page]( /details/estimators/#physics-informed-convolutional-autoencoder-picae)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `loaders` | dict | contains input and target data (loaders['train'], loaders['valid']). Datasets themselves have to be torch.tensor(s) |  |    
    | `config` | class | configuration to use for the model | PICAEConfig() |
    | `model` | class | the model itself - should not be adjusted | PICAEModel() |    

!!! code ""    
    `sapsan.lib.estimator.PICAE.save`_`(path: str)`_

: Saves model and optimizer states, as well as final epoch and loss

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively |  |    

!!! code ""    
    `sapsan.lib.estimator.PICAE.load`_`(path: str, estimator, load_saved_config = False)`_

: Loads model and optimizer states, as well as final epoch and loss

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively |  |    
    | `estimator` | estimator | need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further. |  |
    | <nobr>`load_saved_config`></nobr> | bool | updates config parameters from `{path}/params.json` | False |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | pytorch model | loaded model | 

---

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.PICAEConfig`_`(n_epochs, patience, min_delta, logdir, lr, min_lr, weight_decay, nfilters, kernel_size, enc_nlayers, dec_nlayers, *args, **kwargs)`_

: Configuration for the CNN3d - based on pytorch and catalyst libraries

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `n_epochs` | int | number of epochs | 1 | 
    | `batch_dim` | int | dimension of a batch in each axis | 64 |
    | `patience` | int | number of epochs with no improvement after which training will be stopped | 10 |
    | `min_delta` | float | minimum change in the monitored metric to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement | 1e-5 |
    | `log_dir` | str |  path to store the logs | ./logs/ |
    | `lr` | float | learning rate | 1e-3 |
    | `min_lr` | float | a lower bound of the learning rate  for [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) | lr\*1e-2 |
    | <nobr>`weight_decay`</nobr> | float | weight decay (L2 penalty) | 1e-5 |
    | `nfilters` | int | the output dim for each convolutional layer, which is the number of "filters" learned by that layer | 6 |
    | <nobr>`kernel_size`</nobr> | tuple | size of the convolutional kernel | (3,3,3) |
    | `enc_layers` | int | number of encoding layers | 3 |
    | `dec_layers` | int | number of decoding layers | 3 |
    | `device` | str | specify the device to run the model on | cuda (or switch to cpu)
    | <nobr>`loader_key`</nobr> | str | the loader to use for early stop: *train* or *valid* | first loader provided*, which is usually 'train' |    
    | <nobr>`metric_key`</nobr> | str | the metric to use for early stop | 'loss' |    
    | `ddp` | bool | turn on Distributed Data Parallel (DDP) in order to distribute the data and train the model across multiple GPUs.  This is passed to Catalyst to activate the `ddp` flag in `runner` (see more [Distributed Training Tutorial](https://catalyst-team.github.io/catalyst/tutorials/ddp.html); the `runner` is set up in [pytorch_estimator.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/lib/estimator/pytorch_estimator.py)). **Note: doesn't support jupyter notebooks - prepare a script!** | False |

---

### KRR
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.KRR`_`(loaders, config, model)`_

: A model based on sk-learn [Kernel Ridge Regression](/details/estimators/#kernel-ridge-regression-krr)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `loaders` | list | contains input and target data | |
    | `config` | class | configuration to use for the model | KRRConfig() |
    | `model` | class | the model itself - should not be adjusted | KRRModel() |

!!! code ""
    `sapsan.lib.estimator.KRR.save`_`(path: str)`_

: Saves the model

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively | |

!!! code "" 
    `sapsan.lib.estimator.KRR.load`_`(path: str, estimator, load_saved_config = False)`_

: Loads the model

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively | |
    | `estimator` | estimator | need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further. | |
    | <nobr>`load_saved_config`</nobr> | bool |  updates config parameters from `{path}/params.json` | False |      

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | sklearn model | loaded model |

---

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.KRRConfig`_`(alpha, gamma)`_

: Configuration for the KRR model

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `alpha` | float | regularization term, hyperparameter | None |
    | `gamma` | float | full-width at half-max for the RBF kernel, hyperparameter | None |

---

### load_estimator
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.load_estimator`_`()`_

: Dummy estimator to call `load()` to load the saved pytorch models

!!! code ""
    `sapsan.lib.estimator.load_estimator.load`_`(path: str, estimator, load_saved_config = False)`_

: Loads model and optimizer states, as well as final epoch and loss

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively | |
    | `estimator` | estimator | need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further | |
    | <nobr>`load_saved_config`</nobr> | bool | updates config parameters from `{path}/params.json` | False |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | pytorch model | loaded model |

---

### load_sklearn_estimator
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.load_sklearn_estimator`_`()`_

: Dummy estimator to call `load()` to load the saved sklearn models

!!! code ""
    `sapsan.lib.estimator.load_sklearn_estimator.load`_`(path: str, estimator, load_saved_config = False)`_

: Loads model

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively| |
    | `estimator` | estimator | need to provide an initialized model for which to load the weights. The estimator can include a new config setup to keep training the model further| |
    | <nobr>`load_saved_config`</nobr> | bool | updates config parameters from `{path}/params.json` | False |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | sklearn model | loaded model |

## Torch Modules

---

### Gaussian

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.torch_modules.Gaussian`_`(sigma: int)`_

: [3D] Applies a Guassian filter as a torch layer through a series of 3 separable 1D convolutions, utilizing [torch.nn.funcitonal.conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html). CUDA is supported.

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `sigma` | int | standard deviation $\sigma$ for a Gaussian kernel | 2 |

!!! code ""
    `sapsan.lib.estimator.torch_modules.Gaussian.forward`_`(tensor: torch.tensor)`_

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `tensor` | torch.tensor | input torch tensor of shape  [N, C~in~, D~in~, H~in~, W~in~]  | |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | torch.tensor | filtered 3D torch data |

---

### Interp1d
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.torch_modules.Interp1d`_`()`_

: Linear 1D interpolation done in native PyTorch. CUDA is supported. Forked from [@aliutkus](https://github.com/aliutkus/torchinterp1d)

!!! code ""
    `sapsan.lib.estimator.torch_modules.Interp1d.forward`_`(x: torch.tensor, y: torch.tensor, xnew: torch.tensor, out: torch.tensor)`_

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `x` | torch.tensor | 1D or 2D tensor | |
    | `y` | torch.tensor | 1D or 2D tensor; the length of `y` along its last dimension must be the same as that of `x` | |
    | `xnew` | torch.tensor | 1D or 2D tensor of real values. `xnew` can only be 1D if ^^both^^ `x` and `y` are 1D. Otherwise, its length along the first dimension must be the same as that of whichever `x` and `y` is 2D. | |
    | `out` | torch.tensor | Tensor for the output | If *None*, allocated automatically |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | torch.tensor | interpolated tensor | 


## Data Loaders

---

### HDF5Dataset
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.data.hdf5_dataset.HDF5Dataset`_`( path: str, features: List[str], target: List[str], checkpoints: List[int], batch_size: int = None, input_size: int = None, sampler: Optional[Sampling] = None, time_granularity: float = 1, features_label: Optional[List[str]] = None, target_label: Optional[List[str]] = None, flat: bool = False, shuffle: bool=False, train_fraction = None)`_

: HDF5 data loader class

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `path` | str | path to the data in the following format: `"data/t_{checkpoint:1.0f}/{feature}_data.h5"` | |
    | `features` | List[str] | list of train features to load | ['not_specified_data'] |
    | `target` | List[str] | list of target features to load | None |
    | `checkpoints` | List[int] | list of checkpoints to load (they will be appended as batches) | |
    | <nobr>`input_size`</nobr> | int | dimension of the loaded data in each axis | |
    | <nobr>`batch_size`</nobr> | int | dimension of a batch in each axis. If batch_size != input_size, the datacube will be evenly splitted | input_size (doesn't work with *sampler*) |
    | <nobr>`batch_num`</nobr> | int | the number of batches to be loaded at a time | 1 |
    | `sampler` | object | data sampler to use (ex: EquidistantSampling()) | None |
    | <nobr>`time_granularity`</nobr> | float | what is the time separation (dt) between checkpoints | 1 |
    | <nobr>`features_label`</nobr> | List[str] | hdf5 data label for the train features | list(file.keys())[-1], i.e. last one in hdf5 file |
    | <nobr>`target_label`</nobr> | List[str] | hdf5 data label for the target features | list(file.keys())[-1], i.e. last one in hdf5 file |
    | `flat` | bool | flatten the data into [C<sub>in</sub>, D\*H\*W]. Required for sk-learn models | False |
    | `shuffle` | bool | shuffle the dataset | False |
    | <nobr>`train_fraction`</nobr> | float or int | a fraction of the dataset to be used for training (accessed through loaders['train']). The rest will be used for validation (accessed through loaders['valid']). If *int* is provided, then that number of *batches* will be used for training. If *float* is provided, then it will try to split the data either by batch or by actually slicing the data cube into smaller chunks | None - training data will be used for validation, effectively skipping the latter |

!!! code ""
    `sapsan.lib.data.hdf5_dataset.HDF5Dataset.load_numpy()`

: HDF5 data loader method - call it to load the data as a numpy array. If *targets* are not specified, than only features will be loaded (hence you can just load 1 dataset at a time).

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray, np.ndarray | loaded a dataset as a numpy array |

!!! code ""
    `sapsan.lib.data.hdf5_dataset.HDF5Dataset.convert_to_torch([x, y])`

: Splits numpy arrays into batches and converts to torch dataloader

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `[x, y]` | list or np.ndarray | a list of input datasets to batch and convert to torch loaders | |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | OrderedDict{'train' : DataLoader, 'valid' : DataLoader } | Data in Torch Dataloader format ready for training |

!!! code ""
    `sapsan.lib.data.hdf5_dataset.HDF5Dataset.load()`

: Loads, splits into batches, and converts into torch dataloader. Effectively combines .load_numpy and .convert_to_torch

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray, np.ndarray | loaded train and target features: x, y |

---

### get_loader_shape
!!! code ""
    `sapsan.lib.data.data_functions.get_loader_shape()`

: Returns the shape of the loaded tensors - the loaded data that has been split into `train` and `valid` datasets.

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `loaders` | torch DataLoader | the loader of tensors passed for training | |
    | `name` | str | name of the dataset in the loaders; usually either `train` or `valid` | None - chooses the first entry in loaders |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | shape of the tensor |


## Data Manipulation

--- 

### EquidistantSampling

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.data.sampling.EquidistantSampling`_`(target_dim)`_

: Samples the data to a lower dimension, keeping separation between the data points equally distant

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | <nobr>`target_dim`</nobr> | np.ndarray | new shape of the input in the form [D, H, W] |

!!! code ""
    `sapsan.lib.data.sampling.EquidistantSampling.sample`_`(data)`_

: Performs sampling of the data

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `data` | np.ndarray | input data to be sampled - has the shape of [axis, D, H, W] | |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | Sampled data with the shape [axis, D, H, W]
 | 

---

### split_data_by_batch

!!! code ""
    `sapsan.utils.shapes.split_data_by_batch`_`(data: np.ndarray, size: int, batch_size: int, n_features: int, axis: int)`_

: [2D, 3D]: splits data into smaller cubes or squares of batches

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `data` | np.ndarray | input 2D or 3D data, [C<sub>in</sub>, D, H, W] | |
    | `size` | int | dimensionality of the data in each axis | |
    | <nobr>`batch_size`</nobr> | int | dimensionality of the batch in each axis | |
    | <nobr>`n_features`</nobr> | int | number of channels of the input data | |
    | `axis` | int | number of axes, 2 or 3 | |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | batched data: [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>] | 

---

### combine_data

!!! code ""
    `sapsan.utils.shapes.combine_data`_`(data: np.ndarray, input_size: tuple, batch_size: tuple, axis: int)`_

: [2D, 3D] - reverse of `split_data_by_batch` function

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `data` | np.ndarray | input 2D or 3D data, [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>] | |
    | <nobr>`input_size`</nobr> | tuple | dimensionality of the original data in each axis | |
    | <nobr>`batch_size`</nobr> | tuple | dimensionality of the batch in each axis | |
    | `axis` | int | number of axes, 2 or 3 | |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | reassembled data: [C<sub>in</sub>, D, H, W] | 

---

### slice_of_cube

!!! code ""
    `sapsan.utils.shapes.slice_of_cube`_`(data: np.ndarray, feature: Optional[int] = None, n_slice: Optional[int] = None)`_

: Select a slice of a cube (to plot later)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `data` | np.ndarray | input 3D data, [C<sub>in</sub>, D, H, W] | |
    | `feature` | int | feature to take the slice of, i.e. the value of C<sub>in</sub> | 1 |
    | `n_slice` | int | what slice to select, i.e. the value of D | 1 | 

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | data slice: [H, W] | 


## Filter

--- 

### spectral

!!! code ""
    `sapsan.utils.filter.spectral`_`(im: np.ndarray, fm: int)`_

: [2D, 3D] apply a spectral filter

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `im` | np.ndarray | input dataset (ex: [C<sub>in</sub>, D, H, W]) | |
    | `fm` | int | number of Fourier modes to filter down to | |  

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | filtered dataset | 

---

### box
!!! code ""
    `sapsan.utils.filter.box`_`(im: np.ndarray, ksize)`_

: [2D] apply a [box filter](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#boxfilter)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `im` | np.ndarray | input dataset (ex: [C<sub>in</sub>, H, W]) | |
    | `ksize` | tupple | kernel size (ex: ksize = (2,2)) | |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | filtered dataset | 

---

### gaussian
!!! code ""
    `sapsan.utils.filter.gaussian`_`(im: np.ndarray, sigma)`_

: [2D, 3D] apply a [gaussian filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)

: Note: Guassian filter assumes dx=1 between the points. Adjust sigma accordingly.

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `im` | np.ndarray | input dataset (ex: [H, W] or [D, H, W]) | |
    | `sigma` | float or tuple of floats | standard deviation for Gaussian kernel. Sigma can be defined for each axis individually. | |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | filtered dataset | 


## Backend (Tracking)

---

### MLflowBackend

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.backends.mlflow.MLflowBackend`_`(name, host, port)`_

: Initilizes [mlflow](https://www.mlflow.org/) and starts up [mlflow ui](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) at a given host:port

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `name` | str | name under which to record the experiment | "experiment" |
    | `host` | str | host of mlflow ui | "localhost" |
    | `port` | int | port of mlflow ui | 9000 |
    
!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.start_ui`_`()`_

: starts MLflow ui at a specified host and port

!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.start`_`(run_name: str, nested = False, run_id = None)`_

: Starts a tracking run

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `run_name` | str | name of the run | "train" for `Train()`, "evaluate" for `Evaluate()` |
    | `nested` | bool | whether or not to nest the recorded run | *False* for `Train()`, *True* for `Evaluate()` |
    | `run_id` | str | run id | None - a new will be generated |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | str | run_id | 

!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.resume`_`(run_id, nested = True)`_

: Resumes a previous run, so you can record extra parameters

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `run_id` | str | id of the run to resume | |
    | `nested` | bool | whether or not to nest the recorded run | True, since it will usually be an `Evaluate()` run |

!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.log_metric`_`()`_

: Logs a metric

!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.log_parameter`_`()`_

: Logs a parameter

!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.log_artifact`_`()`_

: Logs an artifact (any saved file such, e.g. .png, .txt)

!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.close_active_run`_`()`_

: Closes all active MLflow runs

!!! code ""
    `sapsan.lib.backends.mlflow.MLflowBackend.end`_`()`_

: Ends the most recent MLflow run

---

### FakeBackend

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.backends.fake.FakeBackend()`

: Pass to `train` in order to disable backend (tracking)


## Plotting

---

### plot_params

!!! code ""
    `sapsan.utils.plot.plot_params()`

: Contains the matplotlib parameters that format all of the plots (`font.size`, `axes.labelsize`, etc.)

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | dict | matplotlib parameters | 

: ??? cite "Default Parameters"
        ```python
        def plot_params():
            params = {'font.size': 14, 'legend.fontsize': 14, 
                    'axes.labelsize': 20, 'axes.titlesize': 24,
                    'xtick.labelsize': 17,'ytick.labelsize': 17,
                    'axes.linewidth': 1, 'patch.linewidth': 3, 
                    'lines.linewidth': 3,
                    'xtick.major.width': 1.5,'ytick.major.width': 1.5,
                    'xtick.minor.width': 1.25,'ytick.minor.width': 1.25,
                    'xtick.major.size': 7,'ytick.major.size': 7,
                    'xtick.minor.size': 4,'ytick.minor.size': 4,
                    'xtick.direction': 'in','ytick.direction': 'in',              
                    'axes.formatter.limits': [-7, 7],'axes.grid': True, 
                    'grid.linestyle': ':','grid.color': '#999999',
                    'text.usetex': False,}
            return params
        ```

---

### pdf_plot

!!! code ""
    `sapsan.utils.plot.pdf_plot`_`(series: List[np.ndarray], bins: int = 100, label: Optional[List[str]] = None, figsize: tuple, dpi: int, ax: matplotlib.axes, style: str)`_

: Plot a probability density function (PDF) of a single or multiple datasets

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `series` | List[np.ndarray] | input datasets | |
    | `bins` | int | number of bins to use for the dataset to generate the pdf | 100 |
    | `label` | List[str] | list of names to use as labels in the legend | None |
    | `figsize` | tuple | figure size as passed to matplotlib figure | (6,6) |
    | `dpi` | int | resolution of the figure | 60 |
    | `ax` | matplotlib.axes | axes object to use for plotting (if you want to define your own figure and subplots) | None - creates a separate figure |
    | `style` | str | accepts [matplotlib styles](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) | 'tableau-colorblind10'

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | matplotlib.axes object | ax | 

---

### cdf_plot
!!! code ""
    `sapsan.utils.plot.cdf_plot`_`(series: List[np.ndarray], bins: int = 100, label: Optional[List[str]] = None, figsize: tuple, dpi: int, ax: matplotlib.axes, ks: bool, style: str)`_

: Plot a cumulative distribution function (CDF) of a single or multiple datasets

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `series` | List[np.ndarray] | input datasets | |
    | `bins` | int | number of bins to use for the dataset to generate the pdf | 100 |
    | `label` | List[str] | list of names to use as labels in the legend | None |
    | `figsize` | tuple | figure size as passed to matplotlib figure | (6,6) |
    | `dpi` | int | resolution of the figure | 60 |
    | `ax` | matplotlib.axes | axes object to use for plotting (if you want to define your own figure and subplots) | None - creates a separate figure |
    | `ks` | bool | if _True_ prints out on the plot itself the [Kolomogorov-Smirnov Statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test). It will also be returned along with the _ax_ object | False |
    | `style` | str | accepts [matplotlib styles](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) | 'tableau-colorblind10'

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | matplotlib.axes object, float (if ks==True) | ax, ks (if ks==True) | 

---

### line_plot
!!! code ""
    `sapsan.utils.plot.line_plot`_`(series: List[np.ndarray], bins: int = 100, label: Optional[List[str]] = None, plot_type: str, figsize: tuple, dpi: int, ax: matplotlib.axes, style: str)`_

: Plot linear data of x vs y - same matplotlib formatting will be used as the other plots

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `series` | List[np.ndarray] | input datasets | |
    | `bins` | int | number of bins to use for the dataset to generate the pdf | 100 |
    | `label` | List[str] | list of names to use as labels in the legend | None |
    | `plot_type` | str | axis type of the matplotlib plot; options = ['plot', 'semilogx', 'semilogy', 'loglog'] | 'plot' |
    | `figsize` | tuple | figure size as passed to matplotlib figure | (6,6) |
    | `linestyle` | List[str] | list of linestyles to use for each profile for the matplotlib figure | ['-'] (solid line) |
    | `dpi` | int | resolution of the figure | 60 |
    | `ax` | matplotlib.axes | axes object to use for plotting (if you want to define your own figure and subplots) | None - creates a separate figure |
    | `style` | str | accepts [matplotlib styles](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) | 'tableau-colorblind10'

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | matplotlib.axes object | ax | 

---

### slice_plot
!!! code ""
    `sapsan.utils.plot.slice_plot`_`(series: List[np.ndarray], label: Optional[List[str]] = None, cmap = 'plasma', figsize: tuple, dpi: int, ax: matplotlib.axes)`_

: Plot 2D spatial distributions (slices) of your target and prediction datasets. Colorbar limits for both slices are set based on the minimum and maximum of the 2nd (target) provided dataset.

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `series` | List[np.ndarray] | input datasets | |
    | `label` | List[str] | list of names to use as labels in the legend | None |
    | `cmap` | str | matplotlib colormap to use | 'plasma' |
    | `figsize` | tuple | figure size as passed to matplotlib figure | (6,6) |
    | `dpi` | int | resolution of the figure | 60 |
    | `ax` | matplotlib.axes | axes object to use for plotting (if you want to define your own figure and subplots) <br> {==WARNING: only works if a ^^single^^ image is supplied to `slice_plot()`, otherwise will be ignored==} | None - creates a separate figure |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | matplotlib.axes object | ax | 

---

### log_plot
!!! code ""
    `sapsan.utils.plot.log_plot`_`(show_log = True, log_path = 'logs/logs/train.csv', valid_log_path = 'logs/logs/valid.csv', delimiter=',', train_name = 'train_loss', valid_name = 'valid_loss', train_column = 1, valid_column = 1, epoch_column = 0)`_

: Plots an interactive training log of train_loss vs. epoch with plotly

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `show_log` | bool | show the loss vs. epoch progress plot (it will be save in mlflow in either case) | True |
    | `log_path` | str | path to training log produced by the catalyst wrapper | 'logs/logs/train.csv' |
    | <nobr>`valid_log_path`</nobr> | str | path to validation log produced by the catalyst wrapper | 'logs/logs/valid.csv' |
    | `delimiter` | str | delimiter to use for numpy.genfromtxt data loading | ',' |
    | `train_name` | str | name for the training label | 'train_loss' |
    | `valid_name` | str | name for the validation label | 'valid_loss' |
    | `train_column` | int | column to load for training data from `log_path` | 1 |
    | `valid_column` | int | column to load for validation data from `valid_log_path` | 1 |
    | `epoch_column` | int | column to load the epoch index from `log_path`. If *None*, then epoch will be generated fro the number of entries | 0 |    

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | plotly.express object | plot figure | 

---

### model_graph

!!! code ""
    `sapsan.utils.plot.model_graph`_`(model, shape: np.array, transforms)`_

: Creates a graph of the ML model (needs graphviz to be installed). A tutorial is available on the wiki: [Model Graph](/tutorials/model_graph/)

: The method is based on [hiddenlayer](https://github.com/waleedka/hiddenlayer) originally written by Waleed Abdulla.

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `model` | object | initialized pytorch or tensorflow model | |
    | `shape` | np.array | shape of the input array in the form [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>], where C<sub>in</sub>=1 | |
    | <nobr>`transforms`</nobr> | list[methods] | a list of hiddenlayer transforms to be applied (*Fold, FoldId, Prune, PruneBranch, FoldDuplicates, Rename*) | <nobr>_See below_ :material-arrow-down-right:</nobr>|    

: ??? cite "Default Parameters"
        ```python
        import sapsan.utils.hiddenlayer as hl
        transforms = [
                        hl.transforms.Fold("Conv > MaxPool > Relu", "ConvPoolRelu"),
                        hl.transforms.Fold("Conv > MaxPool", "ConvPool"),    
                        hl.transforms.Prune("Shape"),
                        hl.transforms.Prune("Constant"),
                        hl.transforms.Prune("Gather"),
                        hl.transforms.Prune("Unsqueeze"),
                        hl.transforms.Prune("Concat"),
                        hl.transforms.Rename("Cast", to="Input"),
                        hl.transforms.FoldDuplicates()
                    ]
        ```

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | graphviz.Digraph object | graph of a model | 


## Physics

---

### ReynoldsStress

!!! code ""
    `sapsan.utils.physics.ReynoldsStress`_`(u, filt, filt_size, only_x_components=False)`_

: Calculates a stress tensor of the form 

$$
\tau_{ij} = \widetilde{u_i u_j} - \tilde{u}_i\tilde{u}_j
$$

: where $\tilde{u}$ is the filtered $u$

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `u` | np.ndarray | input velocity in 3D - [axis, D, H, W] | |
    | `filt` | sapsan.utils.filters | the type of filter to use (spectral, box, gaussian). Pass the filter itself by loading the appropriate one from `sapsan.utils.filters` | gaussian |
    | `filt_size` | int or float | size of the filter to apply. For different filter types, the size is defined differently. Spectral - fourier mode to filter to, Box - k_size (box size), Gaussian - sigma | 2 (sigma=2 for gaussian) |
    | <nobr>`only_x_component`</nobr> | bool | calculates and outputs only x components of the tensor in shape [row, D, H, W] - calculating all 9 can be taxing on memory | False |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | stress tensor of shape [column, row, D, H, W] | 

---

### PowerSpectrum

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.utils.physics.PowerSpectrum`_`(u: np.ndarray)`_

: Sets up to produce a power spectrum

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `u` | np.ndarray | input velocity in 3D - [axis, D, H, W] | |

!!! code ""
    `sapsan.utils.physics.PowerSpectrum.calculate()`

: Calculates the power spectrum

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray, np.ndarray | k_bins (fourier modes), Ek_bins (E(k)) | 

!!! code ""
    `sapsan.utils.physics.PowerSpectrum.spectrum_plot`_`(k_bins, Ek_bins, kolmogorov=True, kl_a)`_

: Plots the calculated power spectrum

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `k_bins` | np.ndarray | fourier mode values along x-axis | |
    | `Ek_bins` | np.ndarray | energy as a function of k: E(k) | |
    | `kolmogorov` | bool | plots scaled Kolmogorov's -5/3 spectrum alongside the calculated one | True |
    | `kl_A` | float | scaling factor of Kolmogorov's law | _np.amax(self.Ek_bins)*1e1_ |

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | matplotlib.axes object | spectrum plot | 

---

### GradientModel

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.utils.physics.GradientModel`_`(u: np.ndarray, filter_width, delta_u = 1)`_

: sets up to apply a gradient turbulence subgrid model: 

$$
\tau_{ij} = \frac{1}{12} \Delta^2 \,\delta_k u^*_i \,\delta_k u^*_j
$$

: where $\Delta$ is the filter width and $u^*$ is the filtered $u$

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `u` | np.ndarray | input **filtered** quantity in 3D - [axis, D, H, W] | |
    | <nobr>`filter_width`</nobr> | float | width of the filter which was applied onto `u` | |
    | `delta_u` |  | distance between the points on the grid to use for scaling | 1 |

!!! code ""
    `sapsan.utils.physics.GradientModel.gradient()`

: calculated the gradient of the given input data from GradientModel

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | gradient with shape [column, row, D, H, W] |

!!! code ""
    `sapsan.utils.physics.GradientModel.model()`

: calculates the gradient model tensor with shape [column, row, D, H, W]

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | gradient model tensor |

---

### DynamicSmagorinskyModel

!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.utils.physics.DynamicSmagorinskyModel`_`(u: np.ndarray, filt, original_filt_size, filt_ratio, du, delta_u)`_

: sets up to apply a Dynamic Smagorinsky (DS) turbulence subgrid model: 

$$
\tau_{ij} = -2(C_s\Delta^*)^2|S^*|S^*_{ij}
$$

: where $\Delta$ is the filter width and $S^*$ is the filtered $u$

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `u` | np.ndarray | input **filtered** quantity either in 3D [axis, D, H, W] or 2D [axis, D, H] | |
    | `du` | np.ndarray | gradient of `u` | None*: if `du` is not provided, then it will be calculated with `np.gradient()` |
    | `filt` | sapsan.utils.filters | the type of filter to use (spectral, box, gaussian). Pass the filter itself by loading the appropriate one from `sapsan.utils.filters` | spectral |
    | <nobr>`original_fil_size`</nobr> | int | width of the filter which was applied onto `u` | 15 (spectral, fourier modes = 15) |
    | `delta_u` | float | distance between the points on the grid to use for scaling | 1 |
    | `filt_ratio` | float | the ratio of additional filter that will be applied on the data to find the slope for Dynamic Smagorinsky extrapolation over `original_filt_size` | 0.5 |

!!! code ""
    `sapsan.utils.physics.DynamicSmagorinskyModel.model()`

: calculates the DS model tensor with shape [column, row, D, H, W]

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | np.ndarray | DS model tensor |