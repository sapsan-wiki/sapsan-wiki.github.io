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

`CLASS sapsan.lib.experiments.train.Train(model: Estimator, data_parameters: dict, backend = FakeBackend(), show_log = True, run_name = 'train')`

: call Train to set up your run

`Parameters`

* __model (object)__ - model to use for training
* __data_parameters (dict)__ - data parameters from the data loader, necessary for tracking
* __backend (object)__ - backend to track the experiment. Default *FakeBackend()*
* __show_log (bool)__ - show the loss vs. epoch progress plot (it will be save in mlflow in either case). Default *True*
* __run_name (str)__ - 'run name' tag as recorded under MLflow. Default *train*


<pre>
sapsan.lib.experiments.train.Train.run()
</pre>

&nbsp; run the model

`Return`

&nbsp; trained model

`Return type`

&nbsp; pytorch or sklearn or custom type

---

<pre>
<b>CLASS</b> sapsan.lib.experiments.evaluate.Evaluate(<i>model: Estimator, data_parameters: dict, backend = FakeBackend(), cmap: str = 'plasma', axis: int = 3, flat: bool = False, run_name: str = 'evaluate'</i>)
</pre>

&nbsp; call Evaluate to set up the testing of the trained model. Don't forget to update `estimator.loaders` with the new data for testing.

`Parameters`
* __model (object)__ - model to use for testing
* __data_parameters (dict)__ - data parameters from the data loader, necessary for tracking
* __backend (object)__ - backend to track the experiment. Default *FakeBackend()*
* __cmap (str)__ - matplotlib colormap to use for slice plots. Default *plasma*.
* __axis (int)__ - dimensionality of the data (2D or 3D). Default *3*
* __run_name (str)__ - 'run name' tag as recorded under MLflow. Default *evaluate*


<pre>
sapsan.lib.experiments.evaluate.Evaluate.run()
</pre>

&nbsp; run the evaluation of the trained model

`Return`

&nbsp; target data, predicted data

`Return type`

&nbsp; np.ndarray, np.ndarray

<br/>

## Estimators

<pre>
<b>CLASS</b> sapsan.lib.estimator.CNN3d(<i>loaders: dict, config=CNN3dConfig(), model=CNN3dModel()</i>)
</pre>
&nbsp; a model based on Pytorch's [3D Convolutional Neural Network](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#convolution-neural-network-cnn)

`Parameters`

* __loaders (dict)__ - contains input and target data (loaders['train'], loaders['valid']). Datasets themselves have to be torch.tensor(s)
* __configure (class)__ - configuration to use for the model. Default *CNN3dConfig*
* __model (class)__ - the model itself - should not be adjusted. Default *CNN3dModel*

<pre>
sapsan.lib.estimator.CNN3d.save(<i>path: str</i>)
</pre>

&nbsp; Saves model and optimizer states, as well as final epoch and loss

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively


<pre>
sapsan.lib.estimator.CNN3d.load(<i>path: str, estimator, load_saved_config = False</i>)
</pre>

&nbsp; Loads model and optimizer states, as well as final epoch and loss

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively
* __estimator (estimator)__ - need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further.
* __load_saved_config (bool)__ - updates config parameters from `{path}/params.json`. Default *False*

`Return`

&nbsp; loaded model

`Return type`

&nbsp; pytorch model

---

<pre>
<b>CLASS</b> sapsan.lib.estimator.CNN3dConfig(<i>n_epochs, patience, min_delta, logdir, lr, min_lr, *args, **kwargs</i>)
</pre>
&nbsp; configuration for the CNN3d - based on pytorch and catalyst libraries

`Parameters`

* __n_epochs (int)__ - number of epochs. Default *1*
* __patience (int)__ - number of epochs with no improvement after which training will be stopped. Default *10*
* __min_delta (float)__ -  minimum change in the monitored metric to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement. Default *1e-5*
* __log_dir (int)__ - path to store the logs. Default *./logs/*
* __lr (float)__ - learning rate. Default *1e-3*
* __min_lr (float)__ - a lower bound of the learning rate  for [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html). Default *lr\*1e-2*
* __device (str)__ - specify the device to run the model on. Default *auto* - unless not available, will try to run on multi-GPU
* __loader_key (str)__ - the loader to use for early stop: *train* or *valid*. Default *first loader provided*, which is usually *'train'*
* __metric_key (str)__ - the metric to use for early stop. Default *'loss'*
* __ddp (bool)__ - turn on Distributed Data Parallel (DDP) in order to distribute the data and train the model across multiple GPUs.  This is passed to Catalyst to activate the `ddp` flag in `runner` (see more [Distributed Training Tutorial](https://catalyst-team.github.io/catalyst/tutorials/ddp.html); the `runner` is set up in [pytorch_estimator.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/lib/estimator/pytorch_estimator.py)). **Note: doesn't support jupyter notebooks - prepare a script!** Default *False*


---


<pre>
<b>CLASS</b> sapsan.lib.estimator.PICAE(<i>loaders: dict, config=PICAEConfig(), model=PICAEModel()</i>)
</pre>
&nbsp; Convolutional Auto Encoder with Divergence-Free Kernel and with periodic padding. Further details can be found on the [PICAE page](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators/_edit#physics-informed-convolutional-autoencoder-picae)

`Parameters`

* __loaders (dict)__ - contains input and target data (loaders['train'], loaders['valid']). Datasets themselves have to be torch.tensor(s)
* __configure (class)__ - configuration to use for the model. Default *PICAEConfig*
* __model (class)__ - the model itself - should not be adjusted. Default *PICAEModel*

<pre>
sapsan.lib.estimator.PICAE.save(<i>path: str</i>)
</pre>

&nbsp; Saves model and optimizer states, as well as final epoch and loss

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively


<pre>
sapsan.lib.estimator.PICAE.load(<i>path: str, estimator, load_saved_config = False</i>)
</pre>

&nbsp; Loads model and optimizer states, as well as final epoch and loss

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively
* __estimator (estimator)__ - need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further.
* __load_saved_config (bool)__ - updates config parameters from `{path}/params.json`. Default *False*

`Return`

&nbsp; loaded model

`Return type`

&nbsp; pytorch model

---

<pre>
<b>CLASS</b> sapsan.lib.estimator.PICAEConfig(<i>n_epochs, patience, min_delta, logdir, lr, min_lr, weight_decay, nfilters, kernel_size, enc_nlayers, dec_nlayers, *args, **kwargs</i>)
</pre>
&nbsp; configuration for the CNN3d - based on pytorch and catalyst libraries

`Parameters`

* __n_epochs (int)__ - number of epochs. Default *1*
* __batch_dim (int)__ - dimension of a batch in each axis. Default *64*
* __patience (int)__ - number of epochs with no improvement after which training will be stopped. Default *10*
* __min_delta (float)__ -  minimum change in the monitored metric to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement. Default *1e-5*
* __log_dir (int)__ - path to store the logs. Default *./logs/*
* __lr (float)__ - learning rate. Default *1e-3*
* __min_lr (float)__ - a lower bound of the learning rate  for [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html). Default *lr\*1e-2*
* __weight_decay (float)__ -  weight decay (L2 penalty). Default *1e-5*
* __nfilters (int)__ - the output dim for each convolutional layer, which is the number of "filters" learned by that layer. Default *6*
* __kernel_size (tuple)__ - size of the convolutional kernel. Default *(3,3,3)*
* __enc_layers (int)__ - number of encoding layers. Default *3*
* __dec_layers (int)__ - number of decoding layers. Default *3*
* __device (str)__ - specify the device to run the model on. Default *auto* - unless not available, will try to run on multi-GPU
* __loader_key (str)__ - the loader to use for early stop: *train* or *valid*. Default *first loader provided*, which is usually *'train'*
* __metric_key (str)__ - the metric to use for early stop. Default *'loss'*
* __ddp (bool)__ - turn on Distributed Data Parallel (DDP) in order to distribute the data and train the model across multiple GPUs.  This is passed to Catalyst to activate the `ddp` flag in `runner` (see more [Distributed Training Tutorial](https://catalyst-team.github.io/catalyst/tutorials/ddp.html); the `runner` is set up in [pytorch_estimator.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/lib/estimator/pytorch_estimator.py)). **Note: doesn't support jupyter notebooks - prepare a script!** Default *False*

---

<pre>
<b>CLASS</b> sapsan.lib.estimator.KRR(<i>loaders: np.array or list, config=KRRConfig(), model=KRRModel()</i>)
</pre>
&nbsp; a model based on sk-learn [Kernel Ridge Regression](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#kernel-ridge-regression-krr)

`Parameters`

* __loaders (np.array or list)__ - contains input and target data.
* __configure (class)__ - configuration to use for the model. Default *KRRConfig*
* __model (class)__ - the model itself - should not be adjusted. Default *KRRModel*

<pre>
sapsan.lib.estimator.KRR.save(<i>path: str</i>)
</pre>

&nbsp; Saves model

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively


<pre>
sapsan.lib.estimator.KRR.load(<i>path: str, estimator, load_saved_config = False</i>)
</pre>

&nbsp; Loads model

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively
* __estimator (estimator)__ - need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further.
* __load_saved_config (bool)__ - updates config parameters from `{path}/params.json`. Default *False*

`Return`

&nbsp; loaded model

`Return type`

&nbsp; sklearn model

---

<pre>
<b>CLASS</b> sapsan.lib.estimator.KRRConfig(<i>alpha, gamma</i>)
</pre>
&nbsp; configuration for the KRR model

`Parameters`

* __alpha (float)__ - regularization term, hyperparameter. Default *None*
* __gamma (float)__ - full-width at half-max for the RBF kernel, hyperparameter. Default *None*

---

<pre>
<b>CLASS</b> sapsan.lib.estimator.load_estimator(<i></i>)
</pre>
&nbsp; dummy estimator to call `load()` to load the saved pytorch models

<pre>
sapsan.lib.estimator.load_estimator.load(<i>path: str, estimator, load_saved_config = False</i>)
</pre>

&nbsp; Loads model and optimizer states, as well as final epoch and loss

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively
* __estimator (estimator)__ - need to provide an initialized model for which to load the weights. The estimator can include a new config setup, changing `n_epochs` to keep training the model further
* __load_saved_config (bool)__ - updates config parameters from `{path}/params.json`. Default *False*

`Return`

&nbsp; loaded model

`Return type`

&nbsp; pytorch model

---

<pre>
<b>CLASS</b> sapsan.lib.estimator.load_sklearn_estimator(<i></i>)
</pre>
&nbsp; dummy estimator to call `load()` to load the saved sklearn models

<pre>
sapsan.lib.estimator.load_sklearn_estimator.load(<i>path: str, estimator, load_saved_config = False</i>)
</pre>

&nbsp; Loads model

`Parameters`

* __path (str)__ - save path of the model and its config parameters, `{path}/model.pt` and `{path}/params.json` respectively
* __estimator (estimator)__ - need to provide an initialized model for which to load the weights. The estimator can include a new config setup to keep training the model further
* __load_saved_config (bool)__ - updates config parameters from `{path}/params.json`. Default *False*

`Return`

&nbsp; loaded model

`Return type`

&nbsp; sklearn model

---


<br/>

## Data Loaders
<pre>
<b>CLASS</b> sapsan.lib.data.hdf5_dataset.HDF5Dataset(<i> path: str, features: List[str], target: List[str], checkpoints: List[int], batch_size: int = None, input_size: int = None, sampler: Optional[Sampling] = None, time_granularity: float = 1, features_label: Optional[List[str]] = None, target_label: Optional[List[str]] = None, flat: bool = False, shuffle: bool=False, train_fraction = None</i>)
</pre>
&nbsp; hdf5 data loader class

`Parameters`

* __path (str)__ - path to the data in the following format: `"data/t_{checkpoint:1.0f}/{feature}_data.h5"`
* __features (List[str])__ - list of train features to load. Default *['not_specified_data']* 
* __target (List[str])__ - list of target features to load. Default *None*
* __checkpoints (List[int])__ - list of checkpoints to load (they will be appended as batches)
* __input_size (int)__ - dimension of the loaded data in each axis
* __batch_size (int)__ - dimension of a batch in each axis. If batch_size != input_size, the datacube will be evenly splitted. Default *batch_size = input_size* (doesn't work with *sampler*)
* __batch_num (int)__ - the number of batches to be loaded at a time. Default *1*
* __sampler (object)__ - what sampler to use (ex: EquidistantSampling(...))
* __time_granularity (float)__ - what is the time separation (dt) between checkpoints. Default *1*
* __features_label ([List[str])__ - hdf5 data label for the train features. Default *list(file.keys())[-1], i.e. last one in hdf5 file*
* __target_label (List[str])__ - hdf5 data label for the target features. Default *list(file.keys())[-1], i.e. last one in hdf5 file*
* __flat (bool)__ - flatten the data into [C<sub>in</sub>, D\*H\*W]. Required for sk-learn models. Default *False*
* __shuffle (bool)__ - shuffle the dataset. Default *False*
* __train_fraction (float or int)__ - a fraction of the dataset to be used for training (accessed through loaders['train']). The rest will be used for validation (accessed through loaders['valid']). If *int* is provided, then that number of *batches* will be used for training. If *float* is provided, then it will try to split the data either by batch or by actually slicing the data cube into smaller chunks. Default *None* - training data will be used for validation, effectively skipping the latter.

<pre>
sapsan.lib.data.hdf5_dataset.HDF5Dataset.load_numpy()
</pre>
&nbsp; hdf5 data loader method - call it to load the data as a numpy array. If *targets* are not specified, than only features will be loaded (hence you can just load 1 dataset at a time).

`Return`

&nbsp; loaded a dataset as a numpy array

`Return type`

&nbsp; np.ndarray, np.ndarray

<pre>
sapsan.lib.data.hdf5_dataset.HDF5Dataset.convert_to_torch([x, y])
</pre>
&nbsp; Splits numpy arrays into batches and converts to torch dataloader

`Parameters`

* __[x, y] (list or np.ndarray)__ - a list of input datasets to batch and convert to torch loaders

`Return`

&nbsp; loaders{'train':training_data, 'valid':validation_data}

`Return type`

&nbsp; collections.OrderedDict{'train': torch.utils.data.dataloader.DataLoader,'valid': torch.utils.data.dataloader.DataLoader }


<pre>
sapsan.lib.data.hdf5_dataset.HDF5Dataset.load()
</pre>
&nbsp; Loads, splits into batches, and converts into torch dataloader. Effectively combines .load_numpy and .convert_to_torch

`Return`

&nbsp; loaded train and target features: x, y

`Return type`

&nbsp; np.ndarray, np.ndarray

<pre>
sapsan.lib.data.data_functions.get_loader_shape()
</pre>
&nbsp; Returns the shape of the loaded tensors - the loaded data that has been split into `train` and `valid` datasets.

`Parameters`
* __loaders (torch Dataloader)__ - the loader of tensors passed for training
* __name (str)__ - name of the dataset in the loaders; usually either `train` or `valid`. Default *None* - chooses the first entry in loaders.

`Return`

&nbsp; shape of the tensor

`Return type`

&nbsp; np.ndarray

<br/>

## Data Manipulation
<pre>
<b>CLASS</b> sapsan.lib.data.sampling.EquidistantSampling(<i>target_dim</i>)
</pre> 

Samples the data to a lower dimension, keeping separation between the data points equally distant

`Parameters`

* __target_dim (np.ndarray)__ - new shape of the input in the form [D, H, W]


<pre>
sapsan.lib.data.sampling.EquidistantSampling.sample(<i>data</i>)
</pre>

&nbsp; performs sampling of the data

`Parameters`

* __data (np.ndarray)__ - input data to be sampled - has the shape of [axis, D, H, W]

`Return`

sampled data with the shape [axis, D, H, W]

`Return Type`

&nbsp; np.ndarray

---

<pre>
sapsan.utils.shapes.split_data_by_batch(<i>data: np.ndarray, size: int, batch_size: int, n_features: int</i>)
</pre>
&nbsp; [2D or 3D data]: splits data into smaller cubes or squares of batches

`Parameters`
* __data (np.ndarray)__ - input 2D or 3D data, [C<sub>in</sub>, D, H, W]
* __size (int)__ - dimensionality of the data in each axis
* __batch_size (int)__ - dimensionality of the batch in each axis
* __n_features (int)__ - number of channels of the input data

`Return`

&nbsp; batched data: [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>]

`Return type`

&nbsp; np.ndarray

---

<pre>
sapsan.utils.shapes.split_square_by_batch(<i>data: np.ndarray, size: int, batch_size: int, n_features: int</i>)
</pre>
&nbsp; [2D] - splits big square into smaller ones - batches.

`Parameters`
* __data (np.ndarray)__ - input 2D data, [C<sub>in</sub>, H, W]
* __size (int)__ - dimensionality of the data in each axis
* __batch_size (int)__ - dimensionality of the batch in each axis
* __n_features (int)__ - number of channels of the input data

`Return`

&nbsp; batched data: [N, C<sub>in</sub>, H<sub>b</sub>, W<sub>b</sub>]

`Return type`

&nbsp; np.ndarray

---

<pre>
sapsan.utils.shapes.combine_data(<i>data: np.ndarray, input_size: int, batch_size: int</i>)
</pre>
&nbsp; [3D] - reverse of `split_data_by_batch` function

`Parameters`
* __data (np.ndarray)__ - input 2D or 3D data, [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>]
* __input_size (int)__ - dimensionality of the original data in each axis
* __batch_size (int)__ - dimensionality of the batch in each axis

`Return`

&nbsp; reassembled data: [C<sub>in</sub>, D, H, W]

`Return type`

&nbsp; np.ndarray

---

<pre>
sapsan.utils.shapes.slice_of_cube(<i>data: np.ndarray, feature: Optional[int] = None, n_slice: Optional[int] = None)</i>)
</pre>
&nbsp; select a slice of a cube (to plot later)

`Parameters`
* __data (np.ndarray)__ - input 3D data, [C<sub>in</sub>, D, H, W]
* __feature (int)__ - feature to take the slice of, i.e. the value of C<sub>in</sub>. Default *1*
* __n_slice (int)__ - what slice to select, i.e. the value of D. Default *1*

`Return`

&nbsp; data slice: [H, W]

`Return type`

&nbsp; np.ndarray

<br/>

## Filter

<pre>
sapsan.utils.filter.spectral(<i>im: np.ndarray, fm: int</i>)
</pre>

&nbsp; [2D, 3D] apply a spectral filter

`Parameters`

* __im (np.ndarray)__ - input dataset (ex: [C<sub>in</sub>, D, H, W])
* __fm (int)__ - number of Fourier modes to filter down to

`Return`

&nbsp; filtered dataset

`Return type`

&nbsp; np.ndarray

---

<pre>
sapsan.utils.filter.box(<i>im: np.ndarray, ksize</i>)
</pre>

&nbsp; [2D] apply a [box filter](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#boxfilter)

`Parameters`

* __im (np.ndarray)__ - input dataset (ex: [C<sub>in</sub>, H, W])
* __ksize (tupple)__ - kernel size (ex: ksize = (2,2))

`Return`

&nbsp; filtered dataset

`Return type`

&nbsp; np.ndarray

---

<pre>
sapsan.utils.filter.gaussian(<i>im: np.ndarray, sigma</i>)
</pre>

&nbsp; [2D, 3D] apply a [gaussian filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)

&nbsp; Note: Guassian filter assumes dx=1 between the points. Adjust sigma accordingly.

`Parameters`

* __im (np.ndarray)__ - input dataset (ex: [H, W] or [D, H, W])
* __sigma (float or a sequence of floats)__ - standard deviation for Gaussian kernel. Sigma can be defined for each axis individually

`Return`

&nbsp; filtered dataset

`Return type`

&nbsp; np.ndarray

<br/>

## Backend (Tracking)

<pre>
<b>CLASS</b> sapsan.lib.backends.mlflow.MLflowBackend(<i>name, host, port</i>)
</pre> 

&nbsp; initilizes [mlflow](https://www.mlflow.org/) and starts up [mlflow ui](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) at a given host:port

`Parameters`

* __name (str)__ - name under which to record the experiment. Default *"experiment"*
* __host (str)__ - host of mlflow ui. Default *"localhost"*
* __port (int)__ - port of mlflow ui. Default *9000*


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.start_ui(<i></i>)
</pre>

&nbsp; starts MLflow ui at a specified host and port


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.start(<i>run_name: str, nested = False, run_id = None</i>)
</pre>

&nbsp; starts a tracking run

`Parameters`

* __run_name (str)__ - name of the run. Default "train" for *Train()* and "evaluate" for *Evaluate()*
* __nested (bool)__ - whether or not to nest the recorded run. Default *False* (*False* for *Train()* and *True* for *Evaluate()*)
* __run_id (str)__ - run id. Default *None* - a new will be generated

`Return`

&nbsp; run_id


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.resume(<i>run_id, nested = True</i>)
</pre>

&nbsp; resumes a previous run

`Parameters`

* __run_id (str)__ - id of the run to resume
* __nested (bool)__ - whether or not to nest the recorded run. Default *True*, since it will usually be an *Evaluate()* run

<pre>
sapsan.lib.backends.mlflow.MLflowBackend.log_metric(<i></i>)
</pre>

&nbsp; logs a metric


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.log_parameter(<i></i>)
</pre>

&nbsp; logs a parameter


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.log_artifact(<i></i>)
</pre>

&nbsp; logs an artifact (any saved file such, e.g. .png, .txt)


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.close_active_run(<i></i>)
</pre>

&nbsp; closes all active MLflow runs


<pre>
sapsan.lib.backends.mlflow.MLflowBackend.end(<i></i>)
</pre>

&nbsp; ends the most recent MLflow run

---

<pre>
<b>CLASS</b> sapsan.lib.backends.fake.FakeBackend()
</pre> 

pass to `train` in order to disable backend (tracking)

<br/>

## Plotting

<pre>
sapsan.utils.plot.plot_params()
</pre>
&nbsp; contains the matplotlib parameters that format all of the plots (font.size, axes.labelsize, etc.)

`Return`

&nbsp; matplotlib parameters

`Return type`

&nbsp; dict

---

<pre>
sapsan.utils.plot.pdf_plot(<i>series: List[np.ndarray], bins: int = 100, label: Optional[List[str]] = None, figsize: tuple, ax: matplotlib.axes</i>)
</pre>
&nbsp; plot a probability density function (pdf) of a single or multiple dataset

`Parameters`

* __series (List[np.ndarray])__ - input datasets
* __bins (int)__ - number of bins to use for the dataset to generate the pdf. Default *100*.
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __figsize (tuple)__ - figure size as passed to matplotlib figure. Default *(6,6)*
* __ax (matplotlib.axes)__ - axes object to use for plotting (if you want to define your own figure and subplots). Default *None* - creates a separate figure

`Return`

&nbsp; ax

`Return type`

&nbsp; matplotlib.axes object

---

<pre>
sapsan.utils.plot.cdf_plot(<i>series: List[np.ndarray], label: Optional[List[str]] = None, figsize: tuple, ax: matplotlib.axes, ks: Bool</i>)
</pre>
&nbsp; plot a cumulative distribution function (cdf) of a single or multiple dataset

`Parameters`

* __series (List[np.ndarray])__ - input datasets
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __figsize (tuple)__ - figure size as passed to matplotlib figure. Default *(6,6)*
* __ax (matplotlib.axes)__ - axes object to use for plotting (if you want to define your own figure and subplots). Default *None* - creates a separate figure
* __ks (bool)__ - if _True_ prints out on the plot itself the [Kolomogorov-Smirnov Statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test). It will also be returned along with the _ax_ object. Default *False*

`Return`

&nbsp; ax, ks (if ks==True)

`Return type`

&nbsp; matplotlib.axes object, float (if ks==True)

---

<pre>
sapsan.utils.plot.line_plot(<i>series: List[np.ndarray], label: Optional[List[str]] = None, plot_type: str, figsize: tuple, ax: matplotlib.axes</i>)
</pre>
&nbsp; plot linear data of x vs y - same matplotlib formatting will be used as the other plots

`Parameters`

* __series (List[np.ndarray])__ - input datasets in the format: [[x1,y1], [x2,y2], ... ]
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __plot_type (str)__ - axis type of the matplotlib plot; options = ['plot', 'semilogx', 'semilogy', 'loglog']. Default *'plot'*
* __figsize (tuple)__ - figure size as passed to matplotlib figure. Default *(6,6)*
* __linestyle (List[str])__ - list of linestyles to use for each profile for the matplotlib figure. Default *'-'* (solid line)
* __ax (matplotlib.axes)__ - axes object to use for plotting (if you want to define your own figure and subplots). Default *None* - creates a separate figure

`Return`

&nbsp; ax

`Return type`

&nbsp; matplotlib.axes object

---

<pre>
sapsan.utils.plot.slice_plot(<i>series: List[np.ndarray], label: Optional[List[str]] = None, cmap = 'plasma', figsize: tuple</i>)
</pre>
&nbsp; plot 2D spatial distributions (slices) of your target and prediction datasets

`Parameters`

* __series (List[np.ndarray])__ - input datasets
* __label (List[str])__ - list of names to use as labels in the legend.  Default *None*.
* __cmap (str)__ - matplotlib colormap to use. Default *plasma*.
* __figsize (tuple)__ - figure size as passed to matplotlib figure. Default *(6,6)*

`Return`

&nbsp; ax

`Return type`

&nbsp; matplotlib.axes object

---

<pre>
sapsan.utils.plot.log_plot(<i>show_log = True, log_path = 'logs/logs/train.csv', valid_log_path = 'logs/logs/valid.csv', delimiter=',', train_name = 'train_loss', valid_name = 'valid_loss', train_column = 1, valid_column = 1, epoch_column = 0</i>)
</pre>
&nbsp; plots the training log of train_loss vs. epoch

`Parameters`

* __show_log (bool)__ - show the loss vs. epoch progress plot (it will be save in mlflow in either case). Default *True*
* __log_path (str)__ - path to training log produced by the catalyst wrapper.  Default *'logs/logs/train.csv'*
* __valid_log_path (str)__ - path to validation log produced by the catalyst wrapper.  Default *'logs/logs/valid.csv'*
* __delimiter (str)__ - delimiter to use for numpy.genfromtxt data loading. Default *','*
* __train_name (str)__ - name for the training label. Default *'train_loss'*
* __valid_name (str)__ - name for the validation label. Default *'valid_loss'*
* __train_column (int)__ - column to load for training data from `log_path`. Default *1*
* __valid_column (int)__ - column to load for validation data from `valid_log_path`. Default *1*
* __epoch_column (int)__ - column to load the epoch index from `log_path`. If *None*, then epoch will be generated fro the number of entries. Default *0*

`Return`

&nbsp; plot figure

`Return type`

&nbsp; plotly.express object

---

<pre>
sapsan.utils.plot.model_graph(<i>model, shape: np.array, transforms</i>)
</pre>
&nbsp; creates a graph of the ML model (needs graphviz to be installed)

&nbsp; The method is based on [hiddenlayer](https://github.com/waleedka/hiddenlayer) originally written by Waleed Abdulla.

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

&nbsp; graph of a model

`Return type`

&nbsp; graphviz.Digraph object

<br/>

## Physics

<pre>
sapsan.utils.physics.ReynoldsStress(<i>u, filt, filt_size, only_x_components=False</i>)
</pre>

&nbsp; calculates a stress tensor of the form *τ*<sub>*ij*</sub> = (u<sub>*i*</sub>u<sub>*j*</sub>)<sup>\*</sup>-u<sup>\*</sup><sub>*i*</sub>*u<sup>\*</sup><sub>*j*</sub>*

&nbsp; where u<sup>\*</sup> is the filtered u

`Parameters`
* __u (np.ndarray)__ - input velocity in 3D - [axis, D, H, W]
* __filt (sapsan.utils.filters)__ - the type of filter to use (spectral, box, gaussian). Pass the filter itself by loading the appropriate one from `sapsan.utils.filters`. Default *gaussian*
* __filt_size (int or float)__ - size of the filter to apply. For different filter types, the size is defined differently. Spectral - fourier mode to filter to, Box - k_size (box size), Gaussian - sigma. Default *2* (sigma=2 for gaussian)
* __only_x_components (bool)__ - calculates and outputs only x components of the tensor in shape [row, D, H, W] - calculating all 9 can be taxing on memory. Default *False*

`Return`

&nbsp;  stress tensor of shape [column, row, D, H, W]

`Return Type`

&nbsp;  np.ndarray

---

<pre>
<b>CLASS</b> sapsan.utils.physics.PowerSpectrum(<i>u: np.ndarray</i>)
</pre>

&nbsp; sets up to produce a power spectrum

`Parameters`
* __u (np.ndarray)__ - input velocity in 3D - [axis, D, H, W]


<pre>
sapsan.utils.physics.PowerSpectrum.calculate()
</pre>

&nbsp; calculates the power spectrum

`Return`

&nbsp;  k_bins (fourier modes), Ek_bins (E(k))

`Return Type`

&nbsp;  np.ndarray, np.ndarray


<pre>
sapsan.utils.physics.PowerSpectrum.spectrum_plot(<i>k_bins, Ek_bins, kolmogorov=True, kl_a</i>)
</pre>

&nbsp; plots the calculated power spectrum

`Parameters`
* __k_bins (np.ndarray)__ - fourier mode values along x-axis
* __Ek_bins (np.ndarray)__ -  energy as a function of k: E(k)
* __kolmogorov (bool)__ - plots scaled Kolmogorov's -5/3 spectrum alongside the calculated one. Default *True*
* __kl_A (float)__ - scaling factor of Kolmogorov's law. Default _np.amax(self.Ek_bins)*1e1_

`Return`

&nbsp;  spectrum plot

`Return Type`

&nbsp;  matplotlib.axes object

---

<pre>
<b>CLASS</b> sapsan.utils.physics.GradientModel(<i>u: np.ndarray, filter_width, delta_u = 1</i>)
</pre>

&nbsp; sets up to apply a gradient turbulence subgrid model: 

&nbsp; *τ*<sub>*ij*</sub> = 1/12 Δ<sup>2</sup> ∂<sub>*k*</sub>u<sup>\*</sup><sub>i</sub> ∂<sub>*k*</sub>u<sup>\*</sup><sub>j</sub>

&nbsp; where Δ is the filter width and u<sup>\*</sup> is the filtered u


`Parameters`
* __u (np.ndarray)__ - input **filtered** quantity in 3D - [axis, D, H, W]
* __filter_width (float)__ - width of the filter which was applied onto `u`
* __delta_u (float)__ - distance between the points on the grid to use for scaling. Default *1*


<pre>
sapsan.utils.physics.GradientModel.gradient()
</pre>

&nbsp; calculated the gradient of the given input data from GradientModel 

`Return`

gradient with shape [column, row, D, H, W]

`Return Type`

&nbsp; np.ndarray


<pre>
sapsan.utils.physics.GradientModel.model()
</pre>

&nbsp; calculates the gradient model tensor with shape [column, row, D, H, W]

`Return`

gradient model tensor

`Return Type`

&nbsp; np.ndarray

---

<pre>
<b>CLASS</b> sapsan.utils.physics.DynamicSmagorinskyModel(<i>u: np.ndarray, filt, original_filt_size, filt_ratio, du, delta_u</i>)
</pre>

&nbsp; sets up to apply a Dynamic Smagorinsky (DS) turbulence subgrid model: &nbsp; *τ*<sub>*ij*</sub> = -2(C<sub>s</sub>Δ<sup>\*</sup>)<sup>2</sup>|S<sup>\*</sup>|S<sup>\*</sup><sub>*ij*</sub>

&nbsp; where Δ is the filter width and S<sup>\*</sup> is the filtered u


`Parameters`
* __u (np.ndarray)__ - input **filtered** quantity either in 3D [axis, D, H, W] or 2D [axis, D, H]
* __du (np.ndarray)__ - gradient of `u`. Default *None*: if `du` is not provided, then it will be calculated with `np.gradient()`
* __filt (sapsan.utils.filters)__ - the type of filter to use (spectral, box, gaussian). Pass the filter itself by loading the appropriate one from `sapsan.utils.filters`. Default *spectral*
* __original_filt_size (int)__ - width of the filter which was applied onto `u`. Default *15* (spectral, fourier modes = 15)
* __delta_u (float)__ - distance between the points on the grid to use for scaling. Default *1*
* __filt_ratio (float)__ - the ratio of additional filter that will be applied on the data to find the slope for Dynamic Smagorinsky extrapolation over `original_filt_size`. Default *0.5*

<pre>
sapsan.utils.physics.DynamicSmagorinskyModel.model()
</pre>

&nbsp; calculates the DS model tensor with shape [column, row, D, H, W]

`Return`

DS model tensor

`Return Type`

&nbsp; np.ndarray