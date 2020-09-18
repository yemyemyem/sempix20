#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from PIL import Image
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from model import MultiModalModel
from vectorizer import CaptionVectorizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vectorizer, transform):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vectorizer = vectorizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        out_img = self.transform(img).to(device)
            
        caption = self.captions[index]
        out_cap = self.vectorizer(caption).to(device)
        return out_img, out_cap

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
    def __init__(self, margin=1):
        self.margin = margin

    def __call__(self, output, target):
        img, cap = torch.chunk(output, 2, dim=1)
        batch_size = img.shape[0]
        zeros = torch.zeros(batch_size).to(device)

        img = img.unsqueeze(0)
        cap = cap.unsqueeze(1)

        errors = -torch.square(img - cap).sum(axis=2)
        diagonal = torch.diagonal(errors, 0)
        cost_captions = torch.max(zeros, self.margin - errors + diagonal)
        cost_images = torch.max(zeros, self.margin - errors + diagonal.reshape((-1,1)))

        cost = cost_captions + cost_images
        cost.fill_diagonal_(0)
        return cost.sum()

def evaluate(model, transform, vectorizer, sample_size=1000):
    # Test set
    test_df = pd.read_csv("flickr8k/captions_test.txt").sample(sample_size).reset_index(drop=True)
    test_img_ids = test_df["image"].unique()
    test_img_vecs = np.zeros((len(test_img_ids), model.hidden_dim))
    print("Computing image vectors...")
    for i, img_id in enumerate(tqdm(test_img_ids)):
        img = Image.open(os.path.join("flickr8k/images", img_id)).convert("RGB")
        img_transformed = transform(img).unsqueeze(0).to(device)
        test_img_vecs[i] = model.forward_cnn(img_transformed).squeeze(0).cpu().detach().numpy()

    print("Computing caption vectors...")
    cap_vecs = np.zeros((len(test_df), model.hidden_dim))
    for i, caption in enumerate(tqdm(test_df["caption"])):
        embedded_caption = vectorizer(caption).unsqueeze(1).to(device)
        cap_vecs[i] = model.forward_cap(embedded_caption).squeeze(0).cpu().detach().numpy()

    print("Retrieving captions based on images...")
    errors = np.zeros(len(test_df))
    im2cap_recall1 = 0
    im2cap_recall10 = 0
    for i, img in enumerate(tqdm(test_img_ids)):
        img_vec = test_img_vecs[i]
        gold_img_caps = test_df[test_df["image"] == img]["caption"].values
        for j, cap_vec in enumerate(cap_vecs):
            errors[j] = -np.square(np.linalg.norm(cap_vec - img_vec))
        
        best = test_df["caption"][np.argmin(errors)]
        if best in gold_img_caps:
            im2cap_recall1 += 1 / len(test_img_ids)

        best_10 = np.argsort(errors)[:10]
        num_found = 0
        for i, idx in enumerate(best_10):
            caption = test_df["caption"][idx]
            if caption in gold_img_caps:
                num_found += 1
        r10 = num_found / len(gold_img_caps)
        im2cap_recall10 += r10 / len(test_img_ids)
    print("=> Recall@1:", im2cap_recall1)
    print("=> Recall@10:", im2cap_recall10)

    print("Retrieving images based on captions...")
    errors = np.zeros(len(test_img_ids))
    cap2im_recall1 = 0
    cap2im_recall10 = 0
    for i, gold_image_name in enumerate(tqdm(test_df["image"])):
        cap_vec = cap_vecs[i]
        for j, img_vec in enumerate(test_img_vecs):
            errors[j] = -np.square(np.linalg.norm(cap_vec - img_vec))

        img_name = test_img_ids[np.argmin(errors)]
        if img_name == gold_image_name:
            cap2im_recall1 += 1 / len(test_df)

        best_10 = np.argsort(errors)[:10]
        found = 0
        for idx in best_10:
            img_name = test_img_ids[idx]
            if img_name == gold_image_name:
                found = 1
                break
        cap2im_recall10 += found / len(test_df)
     
    print("=> Recall@1:", cap2im_recall1)
    print("=> Recall@10:", cap2im_recall10)

    return (im2cap_recall1, im2cap_recall10), (cap2im_recall1, cap2im_recall10)

def evaluate_sample(model, transform, vectorizer):
    # Custom sample
    sample_df = pd.read_csv("sample.txt")

    # Sample a caption and compute its combined-space embedding vector
    caption = sample_df.iloc[0]["caption"]
    embedded_caption = vectorizer(caption).unsqueeze(1).to(device)
    cap_vec = model.forward_cap(embedded_caption).squeeze(0).cpu().detach().numpy()

    # For each of the images compute the combined-space embedding vectors
    # Calculate the distance to the caption vector (KNN)
    img_ids = sample_df["image"].unique()
    errors = np.zeros(len(img_ids))
    for i, img_id in enumerate(img_ids):
        img = Image.open(os.path.join("flickr8k/images", img_id)).convert("RGB")
        img_transformed = transform(img).unsqueeze(0).to(device)
        img_vec = model.forward_cnn(img_transformed).squeeze(0).cpu().detach().numpy()
        
        errors[i] = -np.square(np.linalg.norm(cap_vec - img_vec))
    
    min_idx = np.argmin(errors)
    print("Arg min:", min_idx)
    print("Best image:", img_ids[min_idx])
    
    print("Top 10:")
    best_10 = np.argsort(errors)[:10]
    for i, idx in enumerate(best_10):
        print(i, img_ids[idx], errors[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--epoch", type=int, help="start from specified epoch")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--margin", type=float, default=0.1, help="margin contrastive loss")
    parser.add_argument("--hidden", type=int, default=1024, help="embedding vector size")
    parser.add_argument("--threshold", type=int, default=4, help="vocab threshold")
    args = parser.parse_args()

    print("Margin:", args.margin)
    print("Batch size:", args.batch)
    print("Final vec size:", args.hidden)
    
    captions = pd.read_csv("flickr8k/captions_train.txt")["caption"]
    vectorizer = CaptionVectorizer()
    path = Path("bin/embedding.pkl")
    if path.exists():
        print("Loading embedding from file...")
        vectorizer.load_embedding(path)
    else:
        print("Generating embedding...")
        vectorizer.generate_embedding(captions, args.threshold)
        vectorizer.write_embedding(path)
    embedding_size = vectorizer.embedding_size

    # Flickr dataset loading
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    flickr = FlickrDataset("flickr8k/images", "flickr8k/captions_train.txt", vectorizer, transform)
    loader = DataLoader(dataset=flickr, batch_size=args.batch, shuffle=True, collate_fn=Collate(vectorizer.w2i["<PAD>"]))
    
    # Model, loss and optimizer definition
    model = MultiModalModel(embedding_size, args.hidden, bidirectional=True).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = ContrastiveLoss(margin=args.margin)
    
    j = 0
    if args.epoch:
        model.load_state_dict(torch.load(f"bin/model_{args.epoch}.pth"))
        j = int(args.epoch)+1

    if args.train:
        writer = SummaryWriter()
        k = 0

        running_loss = 0
        target = torch.zeros(args.batch, 2)
        for epoch in range(j,j+24):
            print("Epoch:", epoch)
            for i, x in enumerate(loader):
                optimizer.zero_grad()

                out = model(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print("Loss:", running_loss / 100)
                    running_loss = 0

                writer.add_scalar("loss", loss.item(), k)
                k += 1

            torch.save(model.state_dict(), f"bin/model_{epoch}.pth")
            print(f"Saved to: bin/model_{epoch}.pth")
    else:
        model.eval()
        model.load_state_dict(torch.load("bin/model_22.pth"))
        evaluate(model, transform, vectorizer)
        evaluate_sample(model, transform, vectorizer)

if __name__ == "__main__":
    main()
