# Evaluation of Neural Image Captions Based on Caption-Image Retrieval

Term project for the course **PM Computational Semantics with Pictures** at the Universität Potsdam in the summer semester 2020, taught by Prof. Dr. David Schlangen.

Developed by **Alexander Koch, Meryem Karalioglu** and **Rodrigo Lopez Portillo Alcocer**.

## Abstract
Currently a growing reliance on automatic image captioning systems can be observed. These captions are oftentimes very generic and apply to a multitude of images. Furthermore the evaluation of the quality of these captions is a difficult task. In this paper we evaluate the quality of automatically generated image captions by how discriminative they are. We trained an image captioning system and a multimodal version of it on the Flickr8K dataset. With them we conducted different experiments with varying levels of difficulty. Their implementation and theoretical foundations are described. <br><br>
The generated captions turn out to be not sufficiently discriminative which is demonstrated by the retrieval evaluation method. We show that this is an applicable method of automatic evaluation.

## Setup
To install Pytorch Lightning check out their official GitHub repo [here](https://github.com/PyTorchLightning/pytorch-lightning), and Test-tube's pip installation command [here](https://pypi.org/project/test-tube/).

## Demos
In the `notebooks/` directory you can find some already run IPython Notebooks with the key parts of our models and evaluations. If you want to execute the ones related to our caption generator, please use the ones in `models/caption_generator/` and `models/level_generator`.

## Looking a bit deeper
If you want to look more closely at the scripts and models we used, see the corresponding **.py** files in `models/`

## Reproducibility

### Data
Most of our generated data can be found in the `data/` directory.<br> Heavier files like our neural network's weights and checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1UK1CIVG-ASd9VSmCN0_hYsaUsB3drJWK?usp=sharing>). <br>
The final weights and embedding for the multimodal model can be downloaded from [here](https://drive.google.com/drive/folders/13Mw5i6ygAkrDwfLvF1PrE8GYwcxGE1aF?usp=sharing). <br>
The versions we used from the Flickr8k and COCO val2014 data sets can be downloaded from [kaggle](https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb) and [cocodataset.org](https://cocodataset.org/#download) respectively.

### Training
**Pytorch Lightning** makes **Pytorch** code device-independent. If you want to retrain or run some of our models in CPU, simply comment out the following arguments from the **Trainer** function:
- gpus
- num_nodes
- auto_select_gpus
- distributed_backend

### Other
For a quick review on how to transform **Pytorch** code into **Pytorch Lightning** models, <https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09> is a good place to start. <br>
If you need help adapting our models to non-SLURM computational clusters, please contact us at rodrigolpa@protonmail.com or check the official **Pytorch Lightning** documentation at <https://pytorch-lightning.readthedocs.io/en/latest/>.
