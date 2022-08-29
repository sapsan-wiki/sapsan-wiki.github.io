---
search:
  boost: 5
---

# Custom Docker

![Docker](/assets/docker_logo2.png){ align=center }

The best way to release your code along with the publication is through [Docker](https://www.docker.com/). This will ensure the reproducibility of your paper's results, making your methods easily accessible to the readers and general public.

While you can write and handle Docker containers in any fashion you want, Sapsan includes a ready-to-go template to make this process easier. Here are the steps in the Docker template:

1. Setup a virtual environment 
2. Install requirements (Sapsan)
3. Launch a Jupyter notebook to reproduce the results

That's it! Now there won't be any struggle or emails to you, the author, about the setup and configuration of your methods!

In order to make this work, we will need to set up a Dockerfile, build a container, and run it. The latter steps are combined into a Makefile. When it comes to publishing your Docker, share the Docker setup files for the container to be built on-site. In this article, we will first discuss the Docker setup and then the release options.

!!! info "Note"
    Make sure Docker is installed on your machine ([Installation](/overview/installation/))

## Docker Setup
### Dockerfile

The template below is will be created when starting a project via `sapsan create -n {name}`, where `{name}` is your custom project name. Feel free to edit it to your liking, such as adding further packages to install outside of Sapsan, name of working directories and etc.

```shell
FROM python:3.8.5-slim

# remember to expose the port your app will run on
EXPOSE 7654

ENV GIT_PYTHON_REFRESH=quiet
RUN pip install -U pip

RUN pip install sapsan=={version}

# copy the notebook and data into a directory of its own (so it isn't in the top-level dir)
COPY {name}_estimator.py {name}_docker/
COPY {name}.ipynb {name}_docker/
COPY ./data/ {name}_docker/data/
WORKDIR /{name}_docker

# run it!
ENTRYPOINT ["jupyter", "notebook", "{name}.ipynb", "--port=7654", 
            "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", 
            "--NotebookApp.password=''", "--no-browser"]
```
Here is a working [Dockerfile](https://github.com/pikarpov-LANL/Sapsan/blob/master/Dockerfile) to *dockerize* the Sapsan's included CNN example.

### Makefile to build and run the container
The `Makefile` is also created upon initializing a project. It makes it straightforward to build and run your Docker container, launching a Jupyter Notebook as a result.

```shell
# to build and start the container 
build-container:
	@docker build . -t {name}-docker

# to run existing the container created above
# (jupyter notebook will be started at --port==7654)
run-container:
	@docker run -p 7654:7654 {name}-docker:latest
```

Thus, the user will need to type the following to build and run the Docker container:
```shell
make build-container
make run-container
```
Here is a working [Makefile](https://github.com/pikarpov-LANL/Sapsan/blob/master/Makefile) for Sapsan's included CNN example.

## Release Your Docker

### Provide the Setup Files

In order for someone to reproduce your results, you will need to provide:

  1. Dockerfile
  2. Makefile
  3. Jupyter Notebook
  4. Training Data

The virtual environment will be built from the ground up on the user's local machine. Besides the training data, the other files won't weigh anything. The only pre-requisite is to have the Docker installed, which can be done through `pip`.




