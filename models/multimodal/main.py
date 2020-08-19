#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False, init_weights=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        out = self.dropout(self.relu(features))
        return torch.norm(out, p=2)

class EncoderCaption(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)

    def forward(self, caption):
        embd = self.embedding(caption)
        _, hidden = self.gru(embd)
        out = hidden[-1]
        return torch.norm(out, p=2)

class MultiModalModel(nn.Module):
    def __init__(self, cnn_embed_size, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder_cnn = EncoderCNN(cnn_embed_size)
        self.encoder_cap = EncoderCaption(vocab_size, embedding_dim, hidden_dim)

    def forward(self, x):
        img, cap = x
        embd_img = self.encoder_cnn(img)
        embd_cap = self.encoder_cap(cap)
        return torch.cat((embd_img, embd_cap), 0)

def main():
    pass

if __name__ == "__main__":
    main()
