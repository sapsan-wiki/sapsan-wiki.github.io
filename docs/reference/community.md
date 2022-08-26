# Community Guidelines

Sapsan welcomes contributions from the community, looking to improve the pipeline and to grow its library of models. Let's make ML models more accessible together!

## General Suggestions
Please feel free to post any bugs you run into or make suggestions through the [Issues](https://github.com/pikarpov-LANL/Sapsan/issues). I will do my best to address them as soon as possible.

If you would like to contribute directly, then a [Pull Request](https://github.com/pikarpov-LANL/Sapsan/pulls) would be the most straightforward way to do so. Once approved, you will be added as a contributor to Sapsan on GitHub.

## Adding a Model
You would like to contribute to Sapsan's 'model zoo'? That's great! Here are the steps to do so
1. Create a new folder under `sapsan/lib/estimator` with the name to reflect your model (`custom_model` for now).
2. Place your python script with the model into that folder, adhering to the format outlined in the template (see [Custom Estimator](https://github.com/pikarpov-LANL/Sapsan/wiki/Custom-Estimator) for details)
   * make sure you initialize the model with `sapsan/lib/estimator/custom_model/__init__.py`
   * add to `sapsan/lib/estimator/__init__.py` a line to access your model, such as
```python
from .custom_model.custom_model import Custom_Model, Custom_ModelConfig
```
3. Set up a Jupyter notebook example and include it under `sapsan/examples`. Make sure the example data is either randomly generated, provided in a small batch, or can be auto-downloaded.
4. Write a short description of your model for the [Estimators'](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators) page on the Wiki. It is a good idea to provide a graph to show the structure of your model ([graph example](https://github.com/pikarpov-LANL/Sapsan/wiki/Model-Graph)), along with the links to any publications of the model if such exist.
5. Pull Request it!

Once approved, your model will be included in automatic testing on _push_ for all future Sapsan releases.

## Analytical Tools
We use a huge variety of tools to analyze our results depending on the problem at hand. Sapsan certainly won't be able to cover everything, but it tries to cover the most general ones (e.g. power spectrum). If there is something major missing, please write about it in the [Issues](https://github.com/pikarpov-LANL/Sapsan/issues) or create a [Pull Request](https://github.com/pikarpov-LANL/Sapsan/pulls). For the latter, the tools should be added into `sapsan/utils`. You can further add to either
1. `plot.py` as a separate `function` if it is a visual analysis (e.g. plotting probability density function) 
2. `physics.py` as a separate `Class` for any type of physics-based calculations
3. Anything custom is fine too



### Analytical Turbulence Models
I am looking to expand a library of analytical turbulence models (i.e. gradient model) to compare ones results with. There are lots of flavors of such, hence a pull request would be highly appreciated. Analytical models should be added as a separate `Class` in `sapsan/utils/physics.py`. In addition, please prepare a short description of it for the Wiki.