# Structure
## Dependencies

Sapsan is a python-based framework. Dependencies can be associated with logical modules of the project. The core module does not have any particular dependencies as all classes are implemented using native Python. Lib modules relying heavily on PyTorch with a Catalyst wrapper, as well as scikit-learn for regression-based ML models. CLI module depends on the click library for implementing command line interfaces.

Sapsan is integrated with MLflow to provide for easy and automatic tracking during the experiment stage, as well as saving the trained model. This gives direct access to run history and performance, which in turn gives the user ability to analyze and further tweak their model.

## Structure & flexibility

To provide flexibility and scalability of the project a number of abstraction classes were introduced. Core abstractions include:

|Core Abstraction|Description|
|-----|-----|
|Experiment | main abstraction which encapsulates execution of algorithms, experiments, tests, etc. |
|Algorithm | a base class which all models are extended from |
|BaseAlgorithm| base class for all algorithms that do not need to be trained and has only run method |
|Estimator| an algorithm that has *train* and *predict* methods, like regression model or classifier|
|Dataset|high level wrapped over dataset loaders|

<br/>

Next Sapsan has utility abstractions responsible for all-things tracking:

|Utility Abstractions |Description|
|-----|-----|
|Metric|a single instance of metric emitted during the experiment run|
|Parameter|a parameter used in the experiment|
|Artifact|artifacts for an algorithm (model weights, images, etc.)|
|TrackingBackend|adapter for tracking systems to save metrics, parameters, and artifact|

<br/>

The project is built around those abstractions to make it easier to reason about. In order to extend the project with new models/algorithms, the user will inherit from Estimator(or BaseAlgorithm) and implement required methods.