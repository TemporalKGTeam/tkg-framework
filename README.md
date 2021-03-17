<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>\
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

A framework for learning-based temporal knowledge graph embedding. This framework should be the very ##first## one for temporal knowledge graph embedding tasks, 
under which you can be free from the nuisance, e.g. data processing, training configuring, hyperparameter tuning metrics evaluating, and the like.

Here's why:
* All the experiments should be reproducible and comparable under a unified framework particularly for temporal knowledge graph embedding.
* Hyper-parameter searching and model tuning should be least pivotal when training new models.






<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites
* python3
* pytorch
* pyyaml
* numba
* arrow

### Installation
1. Set up your python environment with e.g. [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) (see prerequisites).
2. Clone the repo to your local directory via `git clone https://github.com/TemporalKGTeam/tkg-framework.git` or download the zip.

<!-- USAGE EXAMPLES -->
## Usage

### Run options
There are 4 different options to run the tkg-framework:
1. Training of a model
2. Resumption of training of a model
3. Evaluation of a trained model
4. Search of an appropriate hyperparameter setting for a model
 ```
 # 1. & 2. train a model
 python tkge.py train --config config-default.yaml

 # 3. evaluate a model
 python tkge.py eval --config config-default.yaml
 
 # 4. search for appropriate setting of a model
 python tkge.py search --config config-default.yaml
 ```

### Execution environment
You just need to specify an execution directory in the config file where all the checkpoints and logfiles of the executions will be maintained automatically.<br>
If you do not specify any directory in the config file, the execution directory tkg-framework/execution/ will be created automatically inside the tkg-framework directory.<br>
In the config file:
```
# maintain checkpoints and logfiles in a specified directory
console:
  execution_dir: path/to/directory
  
# maintain checkpoints and logfiles in the default directory (i.e. tkg-framework/execution)
console:
  execution_dir: ~
```
If you execute a train task with a given configuration for the first time, you'll find the results in:<br><br>
`path/to/<execution_directory>/<execution-type>/<model-type>/<execution-id>/log`<br>
`path/to/<execution_directory>/<execution-type>/<model-type>/<execution-id>/ckpt`
- `<execution-directory>` is the specified (or default) execution directory.
- `<execution-type>` is the specified task (train, eval, search).
- `<model-type>` is the specified model type/name (e.g. ttranse).
- `<execution-id>` is a unique id for the execution. This can be used to resume the training for a model with the same configuration.
- `log` maintains the logfiles identified by the start timestamp of the execution (useful for training resumption so one can differ between the different executions).
- `ckpt` maintains the checkpoints of a model identified by the epoch number.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Gengyuan Zhang - gengyuanmax@gmail.com




<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [PyKEEN](https://github.com/pykeen/pykeen)
* [AllenNLP](https://github.com/allenai/allennlp)
* [LibKGE](https://github.com/uma-pi1/kge)
