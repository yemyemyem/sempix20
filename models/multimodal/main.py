#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import spacy
import pandas as pd
from gensim.models import KeyedVectors
from PIL import Image
import os

spacy_eng = spacy.load("en")

def encode_caption(w2i, caption):
    v = np.zeros((len(caption)+2), dtype=np.long)
    v[0] = w2i["<SOS>"]
    for i, token in enumerate(caption):
        if token in w2i:
            v[i+1] = w2i[token]
        else:
            v[i+1] = w2i["<UNK>"]
    v[len(caption)+1] = w2i["<EOS>"]
    return v

def embed_caption(embedding, w2i, caption):
    caption0 = encode_caption(w2i, caption)
    return torch.tensor([embedding[idx] for idx in caption0])

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, embedding, w2i, transform):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.w2i = w2i
        self.embedding = embedding
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        tokens = [tok.text for tok in spacy_eng.tokenizer(caption.lower())]

        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        return self.transform(img), embed_caption(self.embedding, self.w2i,caption)

class MultiModalModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.inception = models.inception_v3(pretrained=True, aux_logits=False, init_weights=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)

    def forward_cnn(self, img):
        return torch.norm(self.inception(img), p=2, dim=1)

    def forward_cap(self, cap):
        _, hidden = self.gru(cap.float())
        return torch.norm(hidden[-1], p=2, dim=1)

    def forward(self, x):
        img, cap = x
        embd_img = self.forward_cnn(img).view(-1,1)
        embd_cap = self.forward_cap(cap).view(-1,1)
        return torch.cat((embd_img, embd_cap), dim=1)

class Collate:
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.idx)
        return imgs, targets

def main():
    captions = pd.read_csv("flickr8k/captions.txt")
    tokenized_captions = list()
    freq_dict = dict()
    
    # Analyse caption vocab
    for image, caption in captions.values:
        tokens = [tok.text for tok in spacy_eng.tokenizer(caption.lower())]
        for token in tokens:
            if token in freq_dict:
                freq_dict[token] += 1
            else:
                freq_dict[token] = 1

        tokenized_captions.append(tokens)
    
    # Truncate vocab
    vocab = set()
    for key,value in freq_dict.items():
        if value >= 4:
            vocab.add(key)
    
    # Build word index
    words = list(vocab)
    words.extend(["<SOS>", "<EOS>", "<UNK>", "<PAD>"])
    print("Actual vocab size:",len(freq_dict.keys()))
    print("Truncated vocab size:", len(words))

    w2i = { w:i for i,w in enumerate(words) }
    i2w = { i:w for i,w in enumerate(words) }

    glove = KeyedVectors.load_word2vec_format("glove.6B.100d.bin.word2vec", binary=True)
    embedding = np.zeros((len(words), 100))
    for i, word in enumerate(words):
        if word in glove:
            embedding[i] = glove[word]
        else:
            embedding[i] = np.random.uniform(-1,1,100)
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    flickr = FlickrDataset("flickr8k/images", "flickr8k/captions.txt", embedding, w2i, transform)
    loader = DataLoader(dataset=flickr, batch_size=8, shuffle=True, collate_fn=Collate(w2i["<PAD>"]))

    model = MultiModalModel(100, 100) 

    for x in loader:
        out = model(x)
        print(out.shape)

if __name__ == "__main__":
    main()
