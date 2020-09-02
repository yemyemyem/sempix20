#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
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
import argparse

spacy_eng = spacy.load("en")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    return torch.FloatTensor([embedding[idx] for idx in caption0]).to(device)

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
        out_img = self.transform(img).to(device)
        out_cap = embed_caption(self.embedding, self.w2i,caption).to(device)
        return out_img, out_cap

class MultiModalModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        #self.inception = models.inception_v3(pretrained=True, aux_logits=False, init_weights=False)
        #self.inception.fc = nn.Linear(self.inception.fc.in_features, hidden_dim)
        self.vgg16 = models.vgg16(pretrained=True)
        self.linear = nn.Linear(4096, hidden_dim)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-4])
        
        self.gru = nn.GRU(embedding_dim, hidden_dim)

        #for name, param in self.inception.named_parameters():
        #    if "fc.weight" in name or "fc.bias" in name:
        #        param.requires_grad = True
        #    else:
        #        param.requires_grad = False

    def forward_cnn(self, img):
        with torch.no_grad():
            result = self.vgg16(img)
        result = self.linear(result)
        return result / torch.norm(result, p=2, dim=1).view(-1,1)

    def forward_cap(self, cap):
        _, hidden = self.gru(cap)
        return hidden[-1] / torch.norm(hidden[-1], p=2, dim=1).view(-1,1)

    def forward(self, x):
        img, cap = x
        embd_img = self.forward_cnn(img)
        embd_cap = self.forward_cap(cap)
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

class ContrastiveLoss:
    def __init__(self, margin=0):
        self.margin = margin

    def __call__(self, output, target):
        img, cap = torch.chunk(output, 2, dim=1)
        batch_size = img.shape[0]
        zeros = torch.zeros(batch_size).to(device)

        img = img.unsqueeze(0)
        cap = cap.unsqueeze(0).view(batch_size,1,cap.shape[1])
        errors = torch.square(img - cap).sum(axis=2)
        diagonal = torch.diagonal(errors, 0)
        cost_captions = torch.max(zeros, self.margin -errors + diagonal)
        cost_images = torch.max(zeros, self.margin -errors + diagonal.reshape((-1,1)))
        cost = cost_captions + cost_images
        cost.fill_diagonal_(0)
        return cost.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model")
    args = parser.parse_args()

    captions = pd.read_csv("flickr8k/captions_train.txt")
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
    
    glove = KeyedVectors.load_word2vec_format("glove.6B.100d.bin.word2vec", binary=True)

    # Truncate vocab
    vocab = set()
    for key,value in freq_dict.items():
        if value >= 4 and key in glove:
            vocab.add(key)
    
    # Build word index
    words = list(vocab)
    words.extend(["<SOS>", "<EOS>", "<UNK>", "<PAD>"])
    print("Actual vocab size:",len(freq_dict.keys()))
    print("Truncated vocab size:", len(words))

    w2i = { w:i for i,w in enumerate(words) }
    i2w = { i:w for i,w in enumerate(words) }
    
    # Load glove embedding for the word in the vocabulary
    k = glove.vector_size
    embedding_size = glove.vector_size+4
    embedding = np.zeros((len(words), embedding_size))
    for i, word in enumerate(words):
        if word in glove:
            embedding[i] = np.concatenate((glove[word], np.zeros(4)))
        else:
            embedding[i] = np.zeros(embedding_size)
            embedding[i][k] = 1
            k += 1

    # Hyper parameters
    batch_size = 16
    hidden_size = 1024
    
    # Flickr dataset loading
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    flickr = FlickrDataset("flickr8k/images", "flickr8k/captions_train.txt", embedding, w2i, transform)
    loader = DataLoader(dataset=flickr, batch_size=batch_size, shuffle=True, collate_fn=Collate(w2i["<PAD>"]))
    
    # Model, loss and optimizer definition
    model = MultiModalModel(embedding_size, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = ContrastiveLoss()

    #model.load_state_dict(torch.load("bin/model_39.pth"))

    if args.train:
        running_loss = 0
        target = torch.zeros(batch_size, 2)
        for epoch in range(3):
            print("Epoch:", epoch)
            for i, x in enumerate(loader):
                optimizer.zero_grad()

                out = model(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    print("Loss:", running_loss / 200)
                    running_loss = 0

            torch.save(model.state_dict(), f"bin/model_{epoch}.pth")
            print(f"Saved to: bin/model_{epoch}.pth")
    else:
        model.eval()
        model.load_state_dict(torch.load("bin/model_2.pth"))

        sample_df = pd.read_csv("sample.txt")
        caption = sample_df.iloc[0]["caption"]
        print(caption)
        tokenized_caption = [tok.text for tok in spacy_eng.tokenizer(caption.lower())]
        print(tokenized_caption)
        embedded_caption = embed_caption(embedding, w2i, caption)
        print(embedded_caption)
        embedded_caption = embedded_caption.unsqueeze(1)
        print(embedded_caption.shape)
        
        result = model.forward_cap(embedded_caption)
        print("Embedded vector:", result)

if __name__ == "__main__":
    main()
