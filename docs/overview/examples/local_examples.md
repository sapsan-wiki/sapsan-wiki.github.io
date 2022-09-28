# Built-in Examples
## Jupyter Notebook Examples

You can run the included examples ([CNN](/details/estimators/#convolution-neural-network-cnn), [PIMLTurb](/details/estimators/#physics-informed-cnn-for-turbulence-modeling-pimlturb), or [PICAE](/details/estimators/#physics-informed-convolutional-autoencoder-picae) on 3D data, and [KRR](/details/estimators/#kernel-ridge-regression-krr) on 2D data). To copy the examples, type:

```
sapsan get_examples
```
This will create a folder `./sapsan_examples` with appropriate example jupyter notebooks and GUI. For starters, to launch a CNN example:

```
jupyter notebook ./sapsan_examples/cnn_example.ipynb
```

## GUI Examples

In order to try out Sapsan's GUI, start a streamlit instance to open in a browser. After the examples have been compied into your working directory as described above, you will be able to find the GUI example. The entry point:

```
streamlit run ./sapsan_examples/GUI/Welcome.py
```

The scripts for the pages you see ([welcome](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/GUI/Welcome.py) and [examples](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/GUI/pages/Examples.py)) are located in the subsequent directory: `./sapsan_examples/GUI/pages/`. 

If you want to build your own demo, then look into [Examples.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/GUI/pages/Welcome.py) to get started. Ideally, you would only need to import your `Estimator`, `EstimatorConfig`, `EstimatorModel` and adjust the `run_experiment()` function, which has a nearly identical setup to a standard Sapsan's jupyter notebook interface.

![Sapsan GUI](/assets/GUI_light.png#only-light){ align=center }
![Sapsan GUI](/assets/GUI_dark.png#only-dark){ align=center }

## Sample Data

The [data](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/data/cnn_krr/t0) for the CNN and KRR examples has been sourced from [JHTDB](http://turbulence.pha.jhu.edu/). Specifically the [Forced MHD Dataset](http://turbulence.pha.jhu.edu/Forced_MHD_turbulence.aspx) (1024<sup>3</sup>) has been used as a starting point.

|Data|Description|
|---|---|
|u_dim128_2d.h5|velocity field sampled down to [128,128,128] and using the 1st slice|
|u_dim32_fm15.h5| velocity field sampled down to [32,32,32], and spectrally filtered down to 15 modes|