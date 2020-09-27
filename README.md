# Evaluation of Neural Image Captions Based on Caption-Image Retrieval

Term project for the course **PM Computational Semantics with Pictures** at the Universität Potsdam in the summer semester 2020, taught by Prof. Dr. David Schlangen.

Developed by **Alexander Koch, Meryem Karalioglu** and **Rodrigo Lopez Portillo Alcocer**.

## What it does
Copy abstract?

## Setup
Is this section necessary? <br>
To install Pytorch Lightning check out their official GitHub repo [here](https://github.com/PyTorchLightning/pytorch-lightning), and Test-tube's pip installation command [here](https://pypi.org/project/test-tube/).

## Demos
In the `notebooks/` directory you can find some already run IPython Notebooks with the key parts of our models and evaluations. If you want to execute the ones related to our caption generator, please use the ones in `models/caption_generator/` and `models/level_generator`.

## Looking a bit deeper
If you want to look more closely at the scripts and models we used, see the corresponding **.py** files in `models/`

## Reproducibility

### Data
Most of our generated data can be found in the `data/` directory.<br> Heavier files like our neural network's weights and checkpoints can be downloaded from <https://drive.google.com/drive/folders/1UK1CIVG-ASd9VSmCN0_hYsaUsB3drJWK?usp=sharing>. <br>
The versions we used from the Flickr8k and COCO val2014 data sets can be downloaded from <https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb> and <https://cocodataset.org/#download> respectively.

### Training
**Pytorch Lightning** makes **Pytorch** code device-independent. If you want to re-train or run some of our models in CPU, simply comment out the following arguments from the **Trainer** function:
- gpus
- num_nodes
- auto_select_gpus
- distributed_backend
