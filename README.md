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

1. git clone the repo to your local directory.
2. Make sure to provide an environemnt with all the prequisites listet above.


<!-- USAGE EXAMPLES -->
## Usage

1. Edit the `config-default.yaml` file with the configuration you like to train (see comments of the file for further information about supported models and hyperparameters).
2. Make sure to specify:
    1. Path to datasets (`tkg-framework/data` as default)
    2. Path for logging (tkg-framework will create it, if not already exist)
    3. Path for model checkpoints (tkg-framework will create it, if not already exist)
    4. Model & hyperparameters (see comments in `config-default.yaml` for more information)
3. Run the framework with the following commands:
    ```
    # definition
    python tkge.py [train, eval] [--config <config-file>] 
    
    # examples 
    # test a model
    python tkge.py train --config config-default.yaml
    
    # eval a model
    python tkge.py eval --config config-default.yaml
    ```




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
