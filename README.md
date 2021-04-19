<<<<<<< HEAD
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

* python3 >= 3.7
* pytorch
* pyyaml
* numba
* arrow




### Installation

1. git clone the repo to your local derictory



<!-- USAGE EXAMPLES -->
## Usage

 ```
 # test a model
 python tkge.py train --config config-default.yaml

 # eval a model
 python tkge.py eval --config config-default.yaml

 # hyper-parameter optimization
 python tkge.py hpo --config config-default.yaml

 # resume training
 python tkge.py resume --ex /path/to/folder
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
=======
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
* python3 >= 3.7
* pytorch
* pyyaml
* numba
* arrow




### Installation

1. git clone the repo to your local derictory



<!-- USAGE EXAMPLES -->
## Usage

 ```
 # test a model
 python tkge.py train --config config-default.yaml

 # eval a model
 python tkge.py eval --config config-default.yaml

 # resume training 
 python tkge.py resume --ex ~/experiment000123 [--checkpoint latest.ckpt]

 # hyper-parameter optimization
 python tkge.py hpo --config config-hpo-default.yaml
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
>>>>>>> 0786332879ecc099044795f62c5b4bfe9355c677
