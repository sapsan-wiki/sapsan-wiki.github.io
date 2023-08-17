---
search:
  boost: 10
---

# Getting Started

## Command Line Interface (CLI) & Jupyter Notebooks

CLI allows users to create new projects leveraging the structure and abstractions of Sapsan providing a unified interface of interaction with the experiments. In addition, you can test your installation and play around with a few included examples.

### Testing
To make sure everything is working correctly and Sapsan was installed without issues, run:
```
sapsan test
```

### Running Examples
To get started and familiarize yourself with the Jupyter Notebook interface, feel free to run the included examples ([CNN](/details/estimators/#convolution-neural-network-cnn), [PICAE](/details/estimators/#physics-informed-convolutional-autoencoder-picae), [PIMLTurb](/details/estimators/#physics-informed-cnn-for-turbulence-modeling-pimlturb) on 3D data, [PIMLTurb1D](/details/estimators/#physics-informed-cnn-for-1d-turbulence-modeling-pimlturb1d) on 1D data, and [KRR](/details/estimators/#kernel-ridge-regression-krr) on 2D data). There is also a notebook with examples of plotting routines and ML network visualization. To copy the examples, type:

```
sapsan get_examples
```
This will create a folder `./sapsan_examples` with appropriate example jupyter notebooks.

### Custom Projects
In order to get started on your own project, proceed as follows:

```
sapsan create --name {name}
```
where `{name}` should be replaced with your custom project name. This will result in creation of the following file structure:

```
Project Folder:             {name}/
Data Folder:                {name}/data/
Estimator Template:         {name}/{name}_estimator.py
Jupyter Notebook Template:  {name}/{name}.ipynb
Docker Template:            {name}/Dockerfile  
Docker Makefile:            {name}/Makefile  
```

This structure allows you to focus on the designing your network structure itself in `{name}_estimator.py`. At the same time, you can quickly jump into Jupyter Notebook and start running your custom setup. Lastly, `Dockerfile` is already pre-filled to easily share your work with your collaborators or as part of a publication.

<br/>

## Graphical User Interface (GUI) - *beta*

In the aim to provide a user-friendly experience, best suited for demonstrations of your models at talks and conferences, while attempting to not sacrifice too much on customization we have designed a GUI for Sapsan. By utilizing [Streamlit](https://www.streamlit.io/), a python library to build web applications, Sapsan can be fully interacted in the browser running locally. A user can tweak the parameters, edit the portion of the code responsible for the ML model, perform visual layer-by-layer analysis, train/validate, analyze the results, and more.

Lastly, Sapsan can be tried out in the demo-mode directly on the website - [sapsan.app](https://sapsan.app). There, one has limited editing capabilities but can explore the hyper-parameters and get a general understanding of what the framework is capable of.

!!! bug "Offline"
    [sapsan.app](https://sapsan.app) is temporarily offline while transitioning to a new hosting service. Please refer to [local GUI example](/overview/examples/local_examples/#gui-examples) for the demo.

### Running GUI
In order to run it type in the following and follow the instructions - the interface will be opened in your browser
```
sapsan get_examples
streamlit run ./sapsan-examples/GUI/st_intro.py
```

Learn more at [GUI Examples](/overview/examples/local_examples/#gui-examples).

### Troubleshooting 

If you encounter the following error when launching streamlit

```
upper limit on inotify watches reached!
```

Then follow the following [instruction by Shivani Bhardwaj](https://unixia.wordpress.com/2018/04/28/inotify-watch-limit-reached-wait-what/) to increase the watchdog limit (it won't hog your RAM)