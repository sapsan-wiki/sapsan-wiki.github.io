# Save & Load Models

All estimators in Sapsan depend either on `torch_backend` or `sklearn_backend`, depending on the model architecture. Both backends have save and load functions. Thus, no matter whether you are using included estimators or designing your own, both methods will be available. Loaded models can be used either for evaluation or to continue training. In the case of the latter either old or new config parameters can be set.

## Saving the Model
To save the model, call:

```python
estimator.save(path = {save_path})
```

For PyTorch, the states of the model and optimizer will be saved, along with the last epoch and loss in `{save_path}/model.pt`. The config parameters will be saved in `{save_path}/params.json`

For Sklearn, only the model itself will be saved, in `{save_path}/model.pt`

## Loading the Model

Even though all Sapsan estimators have `load` method, you can use a dummy estimator to load your model.


### PyTorch
Import [load_estimator()](https://github.com/pikarpov-LANL/Sapsan/wiki/API-Reference#estimators) to load your  PyTorch model. You can pass new ModelConfig() parameters as well if you intend to continue training your model.

```python
from sapsan.lib.estimator import load_estimator

estimator = CNN3d(config = CNN3dConfig(n_epoch=100),
                  loaders = loaders)

loaded_estimator = load_estimator.load({path_to_model}, 
                                       estimator = estimator)
```

### Sklearn
Sklearn uses a different interface, so you will need to call [load_sklearn_estimator()](https://github.com/pikarpov-LANL/Sapsan/wiki/API-Reference#estimators)

```python
from sapsan.lib.estimator import load_sklearn_estimator

estimator = KRR(config = KRRConfig(),
                loaders = loaders)

loaded_estimator = load_sklearn.estimator.load({path_to_model}, 
                                               estimator = estimator)
```