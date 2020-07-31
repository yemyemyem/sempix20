import pandas as pd
import torchvision.transforms as transforms
import pickle
from model import CNNtoRNN
from PIL import Image
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 256
hidden_size = 256
num_layers = 1

def load_vocab():
    with open("vocab.pkl", "rb") as f:
        return pickle.load(f)

def sanitize_tokens(tokens):
    return list(filter(lambda x: x not in ["<SOS>", "<EOS>"], tokens))

def main():
    df = pd.read_csv("flickr8k/captions_test.txt")
    vocab = load_vocab()
    
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    vocab_size = len(vocab)
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    checkpoint = torch.load("my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    references = list()
    hypotheses = list()
    for index, row in tqdm(df.iterrows(), total=len(df)):
        path = "flickr8k/images/"+row["image"]
        img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        tokens = sanitize_tokens(model.caption_image(img.to(device), vocab))
        #print("Gold:", row["caption"].lower())
        #print("Generated:", " ".join(tokens))

        references.append([row["caption"].lower().split()])
        hypotheses.append(tokens)

    weight_list = [(1.0/1.0, ),
            (1.0/2.0, 1.0/2.0,),
            (1.0/3.0, 1.0/3.0, 1.0/3.0,),
            (1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0,)]
    for i, weights in enumerate(weight_list):
        print(f"BLEU-{i+1}:",corpus_bleu(references, hypotheses, weights))

if __name__ == "__main__":
    main()
