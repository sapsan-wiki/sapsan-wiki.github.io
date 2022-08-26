# Built-in Examples
## Jupyter Notebook Examples

You can run the included examples ([CNN](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#convolution-neural-network) or [PICAE](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#physics-informed-convolutional-autoencoder) on 3D data, and [KRR](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators#kernel-ridge-regression) on 2D data). To copy the examples, type:

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
streamlit run ./sapsan_examples/GUI/st_intro.py
```

The scripts for the pages you see ([welcome](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/GUI/pages/st_welcome.py) and [examples](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/GUI/pages/st_cnn3d.py)) are located in the subsequent directory: `./sapsan_examples/GUI/pages/`. 

If you want to build your own demo, then look into [st_cnn3d.py](https://github.com/pikarpov-LANL/Sapsan/blob/master/sapsan/examples/GUI/pages/st_cnn3d.py) to get started. Ideally, you would only need to import your `Estimator`, `EstimatorConfig`, `EstimatorModel` and adjust the `run_experiment()` function, which has a nearly identical setup to a standard Sapsan's jupyter notebook interface.

## Sample Data

The [data](sapsan/examples/data/t0) for the CNN and KRR examples has been sourced from [JHTDB](http://turbulence.pha.jhu.edu/). Specifically the [Forced MHD Dataset](http://turbulence.pha.jhu.edu/Forced_MHD_turbulence.aspx) (1024<sup>3</sup>) has been used as a starting point.

|Data|Description|
|---|---|
|u_dim128_2d.h5|velocity field sampled down to [128,128,128] and using the 1st slice|
|u_dim32_fm15.h5| velocity field sampled down to [32,32,32], and spectrally filtered down to 15 modes|