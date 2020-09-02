import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from utilities import get_dataset
from pl_model import CaptionGenerator, CocoDataModule
import random
random.seed(163)
import pytorch_lightning as pl
import os


transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

dataset, _ = get_dataset(
                        "../data/images",
                        "../data/Captiones.tsv",
                        transform)


unique = list(np.unique(dataset.imgs))
total = len(unique)
ten_percent = total//10
testing_samples = random.sample(unique, ten_percent)

images = np.asarray(dataset.df['image'])
belongs = [(im in testing_samples) for im in images]
not_belongs = [not be for be in belongs]
testing = dataset.df[belongs]

training = dataset.df[not_belongs]

#print(len(unique))

#print(testing.shape[0])
#print(training.shape[0])
#print(dataset.df.shape[0])

testing.shape[0] + training.shape[0] == dataset.df.shape[0] # good to go!

testing.to_csv('../data/testing_captions.tsv', index = False, sep = '\t')
training.to_csv('../data/training_captions.tsv', index = False, sep = '\t')
