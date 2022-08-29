---
hide:
  - navigation
---

# API Reference

The following page is organized based on the method types. Feel free to jump through the navigation on the right -->

## Glossary
| Variable | Definition |
| -------- | ---------- |
| N        | # of Batches |
|C<sub>in</sub> | # of input channels (i.e. features) |
|D | Data depth (z) |
|H | Data height (y) |
|W | Data width (x) |
|D<sub>b</sub> | Batch depth (z) |
|H<sub>b</sub> | Batch height (y) |
|W<sub>b</sub> | Batch width (x) |

## Train/Evaluate

---

### Train
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.experiments.train.Train`_`(model: Estimator, data_parameters: dict, backend = FakeBackend(), show_log = True, run_name = 'train')`_

: call Train to set up your run    

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

: run the model

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

: call Evaluate to set up the testing of the trained model. Don't forget to update `estimator.loaders` with the new data for testing.

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

: run the evaluation of the trained model

: !!! code ""
        Return

    | Type | Description |
    | ---- | ----------- |
    | dict{'target' : np.ndarray, 'predict' : np.ndarray}  | target and predicted data | 

---

## Estimators

### CNN3d
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.CNN3d`_`(loaders: dict, config=CNN3dConfig(), model=CNN3dModel()`_

: a model based on Pytorch's [3D Convolutional Neural Network](/reference/estimators/#convolution-neural-network-cnn)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `loaders` | dict | contains input and target data (loaders['train'], loaders['valid']). Datasets themselves have to be torch.tensor(s) | CNN3dConfig() |    
    | `configure` | class | configuration to use for the model |  |
    | `model` | class | the model itself - should not be adjusted | CNN3dModel() |    

!!! code ""
    `sapsan.lib.estimator.CNN3d.save`_`(path: str)`_`

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

: configuration for the CNN3d - based on pytorch and catalyst libraries

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

### PICAE
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.estimator.PICAE`_`(loaders: dict, config=PICAEConfig(), model=PICAEModel())`_`

: Convolutional Auto Encoder with Divergence-Free Kernel and with periodic padding. Further details can be found on the [PICAE page](/reference/estimators/#physics-informed-convolutional-autoencoder-picae)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `loaders` | dict | contains input and target data (loaders['train'], loaders['valid']). Datasets themselves have to be torch.tensor(s) |  |    
    | `configure` | class | configuration to use for the model | PICAEConfig |
    | `model` | class | the model itself - should not be adjusted | PICAEModel |    

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

: configuration for the CNN3d - based on pytorch and catalyst libraries

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
    
    `sapsan.lib.estimator.KRR`_`(loaders: np.array or list, config=KRRConfig(), model=KRRModel())`_

: a model based on sk-learn [Kernel Ridge Regression](/estimators/#kernel-ridge-regression-krr)

: !!! code ""
        Parameters

    | Name | Type | Discription | Default |
    | ---- | ---- | ----------- | ------- |    
    | `loaders` | list | contains input and target data | |
    | `configure` | class | configuration to use for the model | KRRConfig |
    | `model` | class | the model itself - should not be adjusted | KRRModel |

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
    
    `sapsan.lib.estimator.KRRConfig`_`(alpha, gamma)`_`

: configuration for the KRR model

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

: dummy estimator to call `load()` to load the saved pytorch models

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

: dummy estimator to call `load()` to load the saved sklearn models

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


## Data Loaders

---

### HDF5Dataset
!!! code ""
    <span style="color:var(--class-color)">CLASS</span>
    
    `sapsan.lib.data.hdf5_dataset.HDF5Dataset`_`( path: str, features: List[str], target: List[str], checkpoints: List[int], batch_size: int = None, input_size: int = None, sampler: Optional[Sampling] = None, time_granularity: float = 1, features_label: Optional[List[str]] = None, target_label: Optional[List[str]] = None, flat: bool = False, shuffle: bool=False, train_fraction = None)`_

: hdf5 data loader class

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
    | `sampler` | object | data sampler to use (ex: EquidistantSampling()) | |
    | <nobr>`time_granularity`</nobr> | float | what is the time separation (dt) between checkpoints | 1 |
    | <nobr>`features_label`</nobr> | List[str] | hdf5 data label for the train features | list(file.keys())[-1], i.e. last one in hdf5 file |
    | <nobr>`target_label`</nobr> | List[str] | hdf5 data label for the target features | list(file.keys())[-1], i.e. last one in hdf5 file |
    | `flat` | bool | flatten the data into [C<sub>in</sub>, D\*H\*W]. Required for sk-learn models | False |
    | `shuffle` | bool | shuffle the dataset | False |
    | <nobr>`train_fraction`</nobr> | float or int | a fraction of the dataset to be used for training (accessed through loaders['train']). The rest will be used for validation (accessed through loaders['valid']). If *int* is provided, then that number of *batches* will be used for training. If *float* is provided, then it will try to split the data either by batch or by actually slicing the data cube into smaller chunks | None - training data will be used for validation, effectively skipping the latter |

!!! code ""
    `sapsan.lib.data.hdf5_dataset.HDF5Dataset.load_numpy()`

: hdf5 data loader method - call it to load the data as a numpy array. If *targets* are not specified, than only features will be loaded (hence you can just load 1 dataset at a time).

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

--- 

## Data Manipulation
### EquidistantSampling

<pre>
<b>CLASS</b> sapsan.lib.data.sampling.EquidistantSampling`_`(target_dim)`_
</pre> 

Samples the data to a lower dimension, keeping separation between the data points equally distant

`Parameters`

* __target_dim (np.ndarray)__ - new shape of the input in the form [D, H, W]


<pre>
sapsan.lib.data.sampling.EquidistantSampling.sample`_`(data)`_
</pre>

performs sampling of the data

`Parameters`

* __data (np.ndarray)__ - input data to be sampled - has the shape of [axis, D, H, W]

`Return`

sampled data with the shape [axis, D, H, W]

`Return Type`

np.ndarray

---

### split_data_by_batch

<pre>
sapsan.utils.shapes.split_data_by_batch`_`(data: np.ndarray, size: int, batch_size: int, n_features: int)`_
</pre>
[2D or 3D data]: splits data into smaller cubes or squares of batches

`Parameters`
* __data (np.ndarray)__ - input 2D or 3D data, [C<sub>in</sub>, D, H, W]
* __size (int)__ - dimensionality of the data in each axis
* __batch_size (int)__ - dimensionality of the batch in each axis
* __n_features (int)__ - number of channels of the input data

`Return`

batched data: [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>]

`Return type`

np.ndarray

---

### split_square_by_batch

<pre>
sapsan.utils.shapes.split_square_by_batch`_`(data: np.ndarray, size: int, batch_size: int, n_features: int)`_
</pre>
[2D] - splits big square into smaller ones - batches.

`Parameters`
* __data (np.ndarray)__ - input 2D data, [C<sub>in</sub>, H, W]
* __size (int)__ - dimensionality of the data in each axis
* __batch_size (int)__ - dimensionality of the batch in each axis
* __n_features (int)__ - number of channels of the input data

`Return`

batched data: [N, C<sub>in</sub>, H<sub>b</sub>, W<sub>b</sub>]

`Return type`

np.ndarray

---

### combine_data

<pre>
sapsan.utils.shapes.combine_data`_`(data: np.ndarray, input_size: int, batch_size: int)`_
</pre>
[3D] - reverse of `split_data_by_batch` function

`Parameters`
* __data (np.ndarray)__ - input 2D or 3D data, [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>]
* __input_size (int)__ - dimensionality of the original data in each axis
* __batch_size (int)__ - dimensionality of the batch in each axis

`Return`

reassembled data: [C<sub>in</sub>, D, H, W]

`Return type`

np.ndarray

---

### slice_of_cube

<pre>
sapsan.utils.shapes.slice_of_cube`_`(data: np.ndarray, feature: Optional[int] = None, n_slice: Optional[int] = None))`_
</pre>
select a slice of a cube (to plot later)

`Parameters`
* __data (np.ndarray)__ - input 3D data, [C<sub>in</sub>, D, H, W]
* __feature (int)__ - feature to take the slice of, i.e. the value of C<sub>in</sub> | 1*
* __n_slice (int)__ - what slice to select, i.e. the value of D | 1*

`Return`

data slice: [H, W]

`Return type`

np.ndarray

<br/>

## Filter

### spectral

<pre>
sapsan.utils.filter.spectral`_`(im: np.ndarray, fm: int)`_
</pre>

[2D, 3D] apply a spectral filter

`Parameters`

* __im (np.ndarray)__ - input dataset (ex: [C<sub>in</sub>, D, H, W])
* __fm (int)__ - number of Fourier modes to filter down to

`Return`

filtered dataset

`Return type`

np.ndarray

---

### box

<pre>
sapsan.utils.filter.box`_`(im: np.ndarray, ksize)`_
</pre>

[2D] apply a [box filter](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#boxfilter)

`Parameters`

* __im (np.ndarray)__ - input dataset (ex: [C<sub>in</sub>, H, W])
* __ksize (tupple)__ - kernel size (ex: ksize = (2,2))

`Return`

filtered dataset

`Return type`

np.ndarray

---

### gaussian

<pre>
sapsan.utils.filter.gaussian`_`(im: np.ndarray, sigma)`_
</pre>

[2D, 3D] apply a [gaussian filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)

Note: Guassian filter assumes dx=1 between the points. Adjust sigma accordingly.

`Parameters`

* __im (np.ndarray)__ - input dataset (ex: [H, W] or [D, H, W])
* __sigma (float or a sequence of floats)__ - standard deviation for Gaussian kernel. Sigma can be defined for each axis individually

`Return`

filtered dataset

`Return type`

np.ndarray

<br/>

## Backend (Tracking)

### MLflowBackend

<pre>
<b>CLASS</b> sapsan.lib.backends.mlflow.MLflowBackend`_`(name, host, port)`_
</pre> 

initilizes [mlflow](https://www.mlflow.org/) and starts up [mlflow ui](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) at a given host:port

`Parameters`

* __name (str)__ - name under which to record the experiment | "experiment"*
* __host (str)__ - host of mlflow ui | "localhost"*
* __port (int)__ - port of mlflow ui | 9000*


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.start_ui`_`()`_
</pre>

starts MLflow ui at a specified host and port


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.start`_`(run_name: str, nested = False, run_id = None)`_
</pre>

starts a tracking run

`Parameters`

* __run_name (str)__ - name of the run. Default "train" for *Train()* and "evaluate" for *Evaluate()*
* __nested (bool)__ - whether or not to nest the recorded run | False* (*False* for *Train()* and *True* for *Evaluate()*)
* __run_id (str)__ - run id | None* - a new will be generated

`Return`

run_id


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.resume`_`(run_id, nested = True)`_
</pre>

resumes a previous run

`Parameters`

* __run_id (str)__ - id of the run to resume
* __nested (bool)__ - whether or not to nest the recorded run | True*, since it will usually be an *Evaluate()* run

<pre>
sapsan.lib.backends.mlflow.MLflowBackend.log_metric`_`()`_
</pre>

logs a metric


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.log_parameter`_`()`_
</pre>

logs a parameter


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.log_artifact`_`()`_
</pre>

logs an artifact (any saved file such, e.g. .png, .txt)


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.close_active_run`_`()`_
</pre>

closes all active MLflow runs


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.end`_`()`_
</pre>

ends the most recent MLflow run

---

### FakeBackend

<pre>
<b>CLASS</b> sapsan.lib.backends.fake.FakeBackend()
</pre> 

pass to `train` in order to disable backend (tracking)

<br/>

## Plotting

### plot_params
<pre>
sapsan.utils.plot.plot_params()
</pre>
contains the matplotlib parameters that format all of the plots (font.size, axes.labelsize, etc.)

`Return`

matplotlib parameters

`Return type`

dict

---

### pdf_plot
<pre>
sapsan.utils.plot.pdf_plot`_`(series: List[np.ndarray], bins: int = 100, label: Optional[List[str]] = None, figsize: tuple, ax: matplotlib.axes)`_
</pre>
plot a probability density function (pdf) of a single or multiple dataset

`Parameters`

* __series (List[np.ndarray])__ - input datasets
* __bins (int)__ - number of bins to use for the dataset to generate the pdf | 100*.
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __figsize (tuple)__ - figure size as passed to matplotlib figure | (6,6)*
* __ax (matplotlib.axes)__ - axes object to use for plotting (if you want to define your own figure and subplots) | None* - creates a separate figure

`Return`

ax

`Return type`

matplotlib.axes object

---

### cdf_plot

<pre>
sapsan.utils.plot.cdf_plot`_`(series: List[np.ndarray], label: Optional[List[str]] = None, figsize: tuple, ax: matplotlib.axes, ks: Bool)`_
</pre>
plot a cumulative distribution function (cdf) of a single or multiple dataset

`Parameters`

* __series (List[np.ndarray])__ - input datasets
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __figsize (tuple)__ - figure size as passed to matplotlib figure | (6,6)*
* __ax (matplotlib.axes)__ - axes object to use for plotting (if you want to define your own figure and subplots) | None* - creates a separate figure
* __ks (bool)__ - if _True_ prints out on the plot itself the [Kolomogorov-Smirnov Statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test). It will also be returned along with the _ax_ object | False*

`Return`

ax, ks (if ks==True)

`Return type`

matplotlib.axes object, float (if ks==True)

---

### line_plot

<pre>
sapsan.utils.plot.line_plot`_`(series: List[np.ndarray], label: Optional[List[str]] = None, plot_type: str, figsize: tuple, ax: matplotlib.axes)`_
</pre>
plot linear data of x vs y - same matplotlib formatting will be used as the other plots

`Parameters`

* __series (List[np.ndarray])__ - input datasets in the format: [[x1,y1], [x2,y2], ... ]
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __plot_type (str)__ - axis type of the matplotlib plot; options = ['plot', 'semilogx', 'semilogy', 'loglog'] | 'plot'*
* __figsize (tuple)__ - figure size as passed to matplotlib figure | (6,6)*
* __linestyle (List[str])__ - list of linestyles to use for each profile for the matplotlib figure | '-'* (solid line)
* __ax (matplotlib.axes)__ - axes object to use for plotting (if you want to define your own figure and subplots) | None* - creates a separate figure

`Return`

ax

`Return type`

matplotlib.axes object

---

### slice_plot

<pre>
sapsan.utils.plot.slice_plot`_`(series: List[np.ndarray], label: Optional[List[str]] = None, cmap = 'plasma', figsize: tuple)`_
</pre>
plot 2D spatial distributions (slices) of your target and prediction datasets

`Parameters`

* __series (List[np.ndarray])__ - input datasets
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __cmap (str)__ - matplotlib colormap to use | plasma*.
* __figsize (tuple)__ - figure size as passed to matplotlib figure | (6,6)*

`Return`

ax

`Return type`

matplotlib.axes object

---

### log_plot

<pre>
sapsan.utils.plot.log_plot`_`(show_log = True, log_path = 'logs/logs/train.csv', valid_log_path = 'logs/logs/valid.csv', delimiter=',', train_name = 'train_loss', valid_name = 'valid_loss', train_column = 1, valid_column = 1, epoch_column = 0)`_
</pre>
plots the training log of train_loss vs. epoch

`Parameters`

* __show_log (bool)__ - show the loss vs. epoch progress plot (it will be save in mlflow in either case) | True*
* __log_path (str)__ - path to training log produced by the catalyst wrapper.  Default *'logs/logs/train.csv'*
* __valid_log_path (str)__ - path to validation log produced by the catalyst wrapper.  Default *'logs/logs/valid.csv'*
* __delimiter (str)__ - delimiter to use for numpy.genfromtxt data loading | ','*
* __train_name (str)__ - name for the training label | 'train_loss'*
* __valid_name (str)__ - name for the validation label | 'valid_loss'*
* __train_column (int)__ - column to load for training data from `log_path` | 1*
* __valid_column (int)__ - column to load for validation data from `valid_log_path` | 1*
* __epoch_column (int)__ - column to load the epoch index from `log_path`. If *None*, then epoch will be generated fro the number of entries | 0*

`Return`

plot figure

`Return type`

plotly.express object

---

### model_graph

<pre>
sapsan.utils.plot.model_graph`_`(model, shape: np.array, transforms)`_
</pre>
creates a graph of the ML model (needs graphviz to be installed)

The method is based on [hiddenlayer](https://github.com/waleedka/hiddenlayer) originally written by Waleed Abdulla.

`Parameters`

* __model (object)__ - initialized pytorch or tensorflow model
* __shape (np.array)__ - shape of the input array in the form [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>], where C<sub>in</sub>=1
* __transforms (list[methods])__ - a list of hiddenlayer transforms to be applied (*Fold, FoldId, Prune, PruneBranch, FoldDuplicates, Rename*). Default:
```python
> import sapsan.utils.hiddenlayer as hl
> transforms = [
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

`Return`

graph of a model

`Return type`

graphviz.Digraph object

<br/>

## Physics

### ReynoldsStress

<pre>
sapsan.utils.physics.ReynoldsStress`_`(u, filt, filt_size, only_x_components=False)`_
</pre>

calculates a stress tensor of the form *τ*<sub>*ij*</sub> = (u<sub>*i*</sub>u<sub>*j*</sub>)<sup>\*</sup>-u<sup>\*</sup><sub>*i*</sub>*u<sup>\*</sup><sub>*j*</sub>*

where u<sup>\*</sup> is the filtered u

`Parameters`
* __u (np.ndarray)__ - input velocity in 3D - [axis, D, H, W]
* __filt (sapsan.utils.filters)__ - the type of filter to use (spectral, box, gaussian). Pass the filter itself by loading the appropriate one from `sapsan.utils.filters` | gaussian*
* __filt_size (int or float)__ - size of the filter to apply. For different filter types, the size is defined differently. Spectral - fourier mode to filter to, Box - k_size (box size), Gaussian - sigma | 2* (sigma=2 for gaussian)
* __only_x_components (bool)__ - calculates and outputs only x components of the tensor in shape [row, D, H, W] - calculating all 9 can be taxing on memory | False*

`Return`

 stress tensor of shape [column, row, D, H, W]

`Return Type`

 np.ndarray

---

### PowerSpectrum

<pre>
<b>CLASS</b> sapsan.utils.physics.PowerSpectrum`_`(u: np.ndarray)`_
</pre>

sets up to produce a power spectrum

`Parameters`
* __u (np.ndarray)__ - input velocity in 3D - [axis, D, H, W]


<pre>
sapsan.utils.physics.PowerSpectrum.calculate()
</pre>

calculates the power spectrum

`Return`

 k_bins (fourier modes), Ek_bins (E(k))

`Return Type`

 np.ndarray, np.ndarray


<pre>
sapsan.utils.physics.PowerSpectrum.spectrum_plot`_`(k_bins, Ek_bins, kolmogorov=True, kl_a)`_
</pre>

plots the calculated power spectrum

`Parameters`
* __k_bins (np.ndarray)__ - fourier mode values along x-axis
* __Ek_bins (np.ndarray)__ -  energy as a function of k: E(k)
* __kolmogorov (bool)__ - plots scaled Kolmogorov's -5/3 spectrum alongside the calculated one | True*
* __kl_A (float)__ - scaling factor of Kolmogorov's law. Default _np.amax(self.Ek_bins)*1e1_

`Return`

 spectrum plot

`Return Type`

 matplotlib.axes object

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