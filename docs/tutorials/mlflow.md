---
search:
  boost: 5
---

# MLflow Tracking

## Default MLflow Tracking in Sapsan

### Starting MLflow Server
`mlflow ui` server will <ins>automatically</ins> start locally if a designated port is open. If not, Sapsan assumes the `mLflow ui` server is already running on that local port and will direct mlflow to write to it. Also you can start `mlflow ui` manually via:
```bash
mlflow ui --host localhost --port 9000
```

### Structure
By default, Sapsan will keep the following structure in MLflow:



- Train 1
     * Evaluate 1
      * Evaluate 2
- Train 2
      * Evaluate 1
      * Evaluate 2

where all evaluation runs are _nested_ under the trained run entry. This way all evaluations are grouped together under the model that was just trained.

Every _Train_ checks for other active runs, terminates them, and starts a new run. At the end of the _Train_ method, the run does not terminate, awaiting _Evaluate_ runs to be nested under it. Thus, _Evaluate_ runs start and end at the end of the method. However, one can still add extra metrics, artifacts and etc by resuming the previously closed run and writing to it, as discussed in the [later section](#after-training-or-evaluation).

### Tracked Parameters

Evaluation runs include the training model parameters and metrics to make it easier to parse through. Here is a complete list of what is tracked by default after running _Train_ or _Evaluate_ loop.

|Parameter|Train|Evaluate|
|-----| :---: | :---: |
|Everything passed to `ModelConfig()` <br/> (including new parameters passed to `kwargs`) | :material-check-circle: | :material-check-circle:|
|`model - {parameter}` - device, logdir, lr, min_delta, <br/> min_lr, n_epochs, patience <br/> | :material-check-circle: | :material-check-circle:|
|`data - {parameter}` - features, features_label, <br/> target, target_label, axis, path, path, shuffle | :material-check-circle: | :material-check-circle:|
|`chkpnt - {parameter}` - initial_size, sample to size, <br/> batch_size, batch_num, time, time_granularity <br/> | :material-check-circle: | :material-check-circle:|


Since _Train_ metrics are recorded for _Evaluate_ runs, they are prefixed as `train - {metric}`. Subsequently, all _Evaluate_ metrics are written as `eval - {metric}`

Metrics|Train|Evaluate|
|-----| :---: | :---: |
|`eval - MSE Loss` - Mean Squared Error (if the target is provided)| | :material-check-circle: |
|`eval - KS Stat` - Kolmogorov-Smirnov Statistic (if the target is provided)| | :material-check-circle: |
|`train - final epoch` - final training epoch| :material-check-circle: | :material-check-circle: |
|All model metrics `model.metrics()` <br/> (provided by Catalyst and PyTorch) | :material-check-circle: | :material-check-circle: |
|Runtime| :material-check-circle: | :material-check-circle: |

|Artifacts|Train|Evaluate|
|-----| :---: | :---: |
|`model_details.txt` - model layer init & optimizer settings | :material-check-circle: | |
|`model_forward.txt` - Model.forward() function| :material-check-circle: | |
|`runtime_log.html` - loss vs. epoch training progress| :material-check-circle: | |
|`pdf_cdf.png` - Probability Density Function (PDF) & <br/> Cummulative Distribution Function (CDF) plots| | :material-check-circle: |
|`slices_plot.png` - 2D Spatial Distribution (slice snapshots)| | :material-check-circle: |

## Adding extra parameters
### Before Training

In order to add a `new_parameter` to be tracked with [MLflow](https://www.mlflow.org/docs/latest/index.html) per your run, simply pass it to config as such: `ModelConfig(new_parameter=value)`. 

Since it will be initialized under `ModelConfig().kwargs['new_parameter']`, the parameter name can be anything. You will see it in MLflow recorded as `model - new_parameter`.

Internally, everything in `ModelConfig().parameters` gets recorded to MLflow. By default, all `ModelConfig()` variables, including `kwargs` are passed to it. Here is the implementation from the [CNN3d estimator](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/lib/estimator/cnn/spacial_3d_encoder.py).

```python
class CNN3dConfig(EstimatorConfig):
    def __init__(self,
                 n_epochs: int = 1,
                 patience: int = 10,
                 min_delta: float = 1e-5,
                 logdir: str = "./logs/",
                 lr: float = 1e-3,
                 min_lr = None,
                 *args, **kwargs):
        self.n_epochs = n_epochs
        self.logdir = logdir
        self.patience = patience
        self.min_delta = min_delta
        self.lr = lr
        if min_lr==None: self.min_lr = lr*1e-2
        else: self.min_lr = min_lr
        self.kwargs = kwargs
        
        #everything in self.parameters will get recorded by MLflow
        #by default, all 'self' variables will get recorded
        self.parameters = {f'model - {k}': v for k, v in self.__dict__.items() if k != 'kwargs'}
        if bool(self.kwargs): self.parameters.update({f'model - {k}': v for k, v in self.kwargs.items()})
```

Note: MLflow doesn't like labels that contain `/` symbol. Please avoid or you might encounter an error.

### After Training or Evaluation

If you want to perform some extra analysis on your model or predictions and record additional metrics after you have called `Train.run()` or `Evaluation.run()`, Sapsan has an interface to do so in **3 steps**:

1. Resume MLflow run
   * Since MLflow run is closed at the end of `Evaluation.run()`, it will need to be resumed first before attempting to write to it. For that reason, both _Train_ and _Evaluate_ classes have a parameter `run_id` which contains the MLflow run_id. You can use it to resume the run, and record new metrics.

2. Record new parameters
   * To add extra parameters to the most recent _Train_ or _Evaluate_ entry in MLflow, simply use either the `backend()` interface or the standard MLflow interface.

3. End the run
   * In order to keep MLflow tidy, it is advised to call `backend.end()` after you are done.

```python
eval = Evaluate(...)
cubes = eval.run()

#do something with the prediction and/or target cube
new_metric = np.amax(cubes['pred_cube'] / cubes['target_cube'])

backend.resume(run_id = eval.run_id)
backend.log_metric('new_metric', new_metric) #or use backend.log_parameter() or backend.log_artifact()
backend.end()
```
Feel free to review the full [API Reference: Backend (Tracking)](/api/#mlflowbackend) for the full description of MLflow-related functions built into Sapsan. 