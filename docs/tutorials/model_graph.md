# Model Graph

## How to construct a graph of the model
This a page describing in detail how to construct nice-looking graphs of your model automatically.

!!! bug
    Temporarily doesn't work due to a compatability issue between ONNX and PyTorch>=1.12.0

## Example

```python
from sapsan.lib.estimator.cnn.cnn3d_estimator import CNN3d, CNN3dConfig
from sapsan.utils.plot import model_graph
from sapsan.lib.data import get_loader_shape

# load your data into loaders

estimator = CNN3d(config = CNN3dConfig(),
                  loaders = loaders)

shape_x, shape_y = get_loader_shape(loaders)

model_graph(model = estimator.model, shape = shape_x)
```
Considering that `shape_x = (8,1,8,8,8)`, the following graph will be produced:

<p align="center">
  <img src="/assets/cnn_model_graph.png#only-light" alt="cnn_model_graph" width=200px>
</p>

<p align="center">
  <img src="/assets/cnn_model_graph_dark.png#only-dark" alt="cnn_model_graph" width=200px>
</p>

## Details

`shape` of the input data is in the format [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>]. You can either grab it from the loader as shown above or provide your own, as long as the number of channels C<sub>in</sub> matches the data your model was initialized with.
  

`transforms` allow you to adjust the graph to your liking. For example, they can allow you to combine layers to be displayed in a single box, instead of separate. Please refer to the [API of model_graph](/api/#model_graph) to see what options are available for transformations.

## Limitations

* `model` input param must be a PyTorch, TensorFlow, or Keras-with-TensorFlow-backend model.

## API for model_graph

`sapsan.utils.plot.model_graph(<i>model, shape: np.array, transforms</i>)`

: Creates a graph of the ML model (needs graphviz to be installed). The method is based on [hiddenlayer](https://github.com/waleedka/hiddenlayer) originally written by Waleed Abdulla.

`Parameters`

: model (object) - initialized pytorch or tensorflow model
: shape (np.array) - shape of the input array in the form [N, C<sub>in</sub>, D<sub>b</sub>, H<sub>b</sub>, W<sub>b</sub>], where C<sub>in</sub>=1
: transforms (list[methods]) - a list of hiddenlayer transforms to be applied (*Fold, FoldId, Prune, PruneBranch, FoldDuplicates, Rename*). Default:
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

: graph of a model

`Return type`

: graphviz.Digraph object