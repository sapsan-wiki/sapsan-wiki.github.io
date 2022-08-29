# Parallel GPU Training

## Automatic

Sapsan relies on Catalyst to implement [Distributed Data Parallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). You can specify `ddp=True` in `ModelConfig`, which in turn will set `ddp=True` parameter for the Catalyst [runner.train()](https://catalyst-team.github.io/catalyst/api/runners.html#runner). Let's take a look at how it could be done by adjusting [cnn_example](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/cnn_example.ipynb):

```python title="cnn_example.ipynb"
estimator = CNN3d(config = CNN3dConfig(n_epochs=5, 
                                       patience=10, 
                                       min_delta=1e-5, 
                                       ddp=True),
                  loaders = loaders)
```

**DDP is not supported on Jupyter Notebooks! You will have to prepare a script.**

Thus, it is advised to start off developing and testing your model on a single CPU or GPU in a jupyter notebook, then downloading it as a python script to run on multiple GPUs locally or on HPC. In addition, you will have to add the following statement to the beginning of your script in order for torch.multiprocessing to work correctly:

```python
if __name__ == '__main__':
```

Even though Training will be performed on the GPUs, evaluation will be done on the CPU.

## Customizing

For more information and further customization of your parallel setup, see [DDP Tutorial from Catalyst](https://catalyst-team.github.io/catalyst/tutorials/ddp.html). It might come in useful if you want, among other things, to take control over what portion of the data is copied onto which node. The runner itself, [torch_backend.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/lib/estimator/torch_backend.py), can be copied to the project directory and accessed when creating a new project by invoking `--get_torch_backend` or `--gtb` flag as such:

```
sapsan create --gtb -n {name}
```

The `torch_backend.py` contains lots of important functions, but for customizing `DDP` you will need to focus on `TorchBackend.torch_train()` as shown below. Most likely you will need to adjust `self.runner` to either another Catalyst runner or your own, custom runner. Next, you will need to edit `self.runner.train()` parameters accordingly.

```python title="torch_backend.py"
class TorchBackend(Estimator):
    def __init__(self, config: EstimatorConfig, model):
        super().__init__(config)
        self.runner = SupervisedRunner() # (1)
        .
        .
        .
    def torch_train(self):
        .
        .
        .
        self.runner.train(model=model,
                          criterion=self.loss_func,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          loaders=loaders,
                          logdir=self.config.logdir,
                          num_epochs=self.config.n_epochs,
                          callbacks=[EarlyStoppingCallback(patience=self.config.patience,
                                                           min_delta=self.config.min_delta,
							   loader_key=self.loader_key,
							   metric_key=self.metric_key,
							   minimize=True),
                                    SchedulerCallback(loader_key=self.loader_key,
                                                      metric_key=self.metric_key,),
                                    SkipCheckpointCallback(logdir=self.config.logdir)
                                    ],
                          verbose=False,
                          check=False,
                          engine=DeviceEngine(self.device),
                          ddp=self.ddp # (2)
                          )
```

1.  Adjust the Runner here. Check [Catalyst's documentation](https://catalyst-team.github.io/catalyst/tutorials/ddp.html)
2. Controls automatic [Distributed Data Parallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
