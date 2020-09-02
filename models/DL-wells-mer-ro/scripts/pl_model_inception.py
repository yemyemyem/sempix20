import pickle
import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import pytorch_lightning as pl
import en_core_web_sm
from utilities import get_dataset, Collate
import torch.optim as optim
import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

#======================================================
#Data Module
class CocoDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, num_workers,
                        ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.root_folder = "../data/images"
        self.annotation_file = "../data/Captiones.tsv"

    def setup(self, stage = None):
        self.transform = transforms.Compose(
                    [
                        transforms.Resize((356, 356)),
                        transforms.RandomCrop((299, 299)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
        )

        self.dataset, _ = get_dataset(
                                "../data/images",
                                "../data/Captiones.tsv",
                                self.transform)

        self.train, _ = get_dataset(
                                "../data/images",
                                "../data/training_captions.tsv",
                                self.transform)

    def train_dataloader(self):
        return DataLoader(dataset = self.train,
        batch_size = self.batch_size,
        pin_memory = True,
        num_workers = self.num_workers,
        shuffle = True,
        collate_fn = MyCollate(pad_idx = 0),
        )

#======================================================

#======================================================
# MODEL MODULE

class CaptionGenerator_inception(pl.LightningModule):

    #MODELS:

    def __init__(self,
                embed_size,
                hidden_size,
                vocab_size,
                num_layers):

        super(CaptionGenerator_inception, self).__init__()

        # Encoder CNN
        self.incept = models.inception_v3(pretrained=True, aux_logits = False)
        self.incept.fc = nn.Linear(self.incept.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

        # Decoder LSTM
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

        self.save_hyperparameters('embed_size', 'hidden_size' , 'vocab_size' , 'num_layers')

    def encode(self, images):
        features = self.incept(images)
        return self.drop(self.relu(features))

    def decode(self, features, captions):
        embeds = self.dropout(self.embed(captions))
        embeds = torch.cat((features.unsqueeze(0), embeds), dim=0)
        hiddens, _ = self.lstm(embeds)
        return self.linear(hiddens)

    def forward(self, images, captions):

        features = self.encode(images)
        outputs = self.decode(features, captions)

        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            #encoding
            x = self.encode(image).unsqueeze(0)
            states = None

            #decoding
            for _ in range(max_length):
                hiddens, states = self.lstm(x, states)
                output = self.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

#======================================================

#======================================================
# OPTIMIZER
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
#======================================================

#======================================================
# LOSS
    def cross_entropy_loss(self):
        return nn.CrossEntropyLoss(ignore_index=0)

#======================================================

#======================================================
# TRAINING
    def training_step(self, batch, batch_idx):
        #Unloading batch
        imgs, captions = batch

        #Forward pass
        outputs = self.forward(imgs, captions[:-1])
        loss = self.cross_entropy_loss()
        loss = loss(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        result = pl.TrainResult(loss)
        return result
