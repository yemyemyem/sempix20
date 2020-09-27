import pickle
import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import pytorch_lightning as pl
import en_core_web_sm
from utilities import get_dataset, MyCollate
import torch.optim as optim
import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms


#Data Module
class FlickrDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, num_workers,
                        ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        

    def setup(self, stage = None):
        
        self.root_folder = "../../data/flickr8k/images"
        self.annotation_file = "../../data/flickr8k/training_captions.txt"
        
        #transform needed for inputing images into inception
        self.transform = transforms.Compose(
        [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        )

        self.train, self.pad_idx = get_dataset(self.root_folder, 
                                    self.annotation_file, 
                                    self.transform)


    def train_dataloader(self):
        return DataLoader(dataset = self.train,
        batch_size = self.batch_size,
        pin_memory = True,
        num_workers = self.num_workers,
        shuffle = True,
        collate_fn = MyCollate(pad_idx = self.pad_idx),
        )


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
    
#======================================================
# MODEL MODULE

class CaptionGenerator(pl.LightningModule):


    def __init__(self,
                embed_size,
                hidden_size,
                vocab_size,
                num_layers,
                batch_size,
                pad_idx,
                ):

        super(CaptionGenerator, self).__init__()
        self.pad_idx = pad_idx
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cnntornn = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
    
        
        #Tuning the last layer between our encoder and the decoder
        for name, parameter in self.cnntornn.encoderCNN.inception.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                parameter.requires_grad = True

            else:
                parameter.requires_grad = False
        
        
        
        self.save_hyperparameters('embed_size', 'hidden_size' , 'vocab_size' , 'num_layers', 'batch_size')

    
    def forward(self, images, captions):

        return self.cnntornn(images, captions)


    def caption_image(self, image, vocabulary, max_length=50):
        return self.cnntornn.caption_image(image, vocabulary, max_length)

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
        return nn.CrossEntropyLoss(ignore_index=self.pad_idx)

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
