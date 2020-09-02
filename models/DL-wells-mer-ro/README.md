# Deep Learning SoSe20 Final Project

This project is based on the paper: [Vinyals et al. Show and Tell: A Neural Image Caption Generator. CVPR 2015.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)

As recommended, we implemented the image captioning system following an encoder-decoder architecture, with a pre-trained CNN as our encoder and a LSTM as our decoder. We used the COCO2017 training set as our whole dataset, and divided it into training and testing ourselves. 

Because we had access to a SLURM-running GPU cluster we decided to implement our system using *Pytorch Lightning*'s framework, and ran small grid-search parameter optimization using *Test-tube*. 

Installing these libraries is NOT necessary to run or see our evaluation files, but they are necessary to reproduce our results.

To install Pytorch Lightning check out they official GitHub repo [here](https://github.com/PyTorchLightning/pytorch-lightning), and Test-tube's pip installation command [here](https://pypi.org/project/test-tube/).

## Important files

All evaluation notebooks are available in this repo under *evaluation-experiment_name*.ipynb

If you want to see how our training script looked before transforming it into a Pytorch Lightning module, check *non_pl_train.py*

The final version of our model can be found in *pl_model.py*

If you want to reproduce our results, you can download our experiment's checkpoints from [here](https://drive.google.com/drive/folders/1mB9OuKhlkDSrenjq5uRoKf-BqX9lyeqQ?usp=sharing).
Once you download these checkpoints, place the lightning_logs folder on your local copy of this repo's master folder. All of our cluster-related outputs can be found in *slurm_files/*

Our main training script can be found as *pl_train_grid.py*

The script we used to divide our dataset into training and testing is also here at it is called *splitter.py*

Our BLEU-4 score generating class can be found as *bleu.py*

## Contact

Because of the big size of our dataset, we can only provide it on request. If you would like to get a copy, as well as if you have any remarks or questions about our implementation, please contact us at rodrigolpa@protonmail.com

Project by Wellesley Boboc, Meryem Karalioglu, & Rodrigo Lopez Portillo Alcocer
