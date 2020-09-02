import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN

#setting seed in order to have same splits training and testing
#torch.manual_seed(163)

import os
d = os.path.dirname(os.getcwd())


def train():

    #normalization for resnet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, dataset = get_loader(
        root_folder=d+"/project/data/images",
        annotation_file=d+"/project/data/Captiones.txt",
        transform=transform,
        num_workers=0,
    )
    
    
    
    #dividing dataset into train and test
    #train_set, test_set = torch.utils.data.random_split(dataset, [107287, 11000]) #training, test
    #OLD VERSION

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 5
    hidden_size = 5
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 1

    # for tensorboard
    writer = SummaryWriter("runs/coco")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #learning scheduler
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.1, verbose = True)
    
    
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)
        
        #losses = []
        
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            
            #losses.append(loss.item())

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
        
        #getting mean loss for this batch to test scheduler
        #mean_loss = sum(losses)/len(losses)
        #scheduler.step(mean_loss)


if __name__ == "__main__":
    train()
