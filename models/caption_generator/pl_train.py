import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from get_loader import get_dataset
from pl_model import CaptionGenerator, FlickrDataModule
import pytorch_lightning as pl
import os
from PIL import Image

transform = transforms.Compose(
                    [
                        transforms.Resize((356, 356)),
                        transforms.RandomCrop((299, 299)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
        )

train, pad_idx = get_dataset(
                        "../data/flickr8k/images",
                        "../data/flickr8k/training_captions.txt",
                        transform)

# Hyperparameters
embed_size = 250
hidden_size = 250
vocab_size = len(train.vocab)
num_layers = 1
learning_rate = 3e-4
#num_epochs = 1

#Training parameters
num_nodes = 2
gpus = 2 #2 GPUs/node

#for loader
batch_size = 32
num_workers = 4


dm = FlickrDataModule(batch_size, num_workers)
dm.setup()

# initialize model
model = CaptionGenerator(embed_size,
                    hidden_size,
                    vocab_size,
                    num_layers,
                    pad_idx)

#CPU version for fast testing
#trainer = pl.Trainer(max_epochs = 10, profiler = True, early_stop_callback=False)


trainer = pl.Trainer(gpus = gpus, num_nodes = num_nodes, max_epochs = 10,  auto_select_gpus = True, profiler = True, distributed_backend='ddp', early_stop_callback=False)

trainer.fit(model, dm)
