import os 
import pandas as pd
import spacy 
import torch
from torch.nn.utils.rnn import pad_sequence  
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import en_core_web_sm
import torchvision.transforms as transforms

spacy_eng = en_core_web_sm.load()

class Vocabulary:
    '''
    Class description:
        This class will help us generate our dataset's vocabulary. Once we encounter a word more than the frequency threshold number of times we will add it and get it's corresponding numerical value. The two-way dictionaries between indices and strings will allow us to convert the output vector of our network into natural language. 
    
    Class variables:
        itos: Index to string dictionary
        stoi: String to index dictionary
        freq_threshold: int, minimum number of time that a token has to appear in our training dataset in order to add it to our vocabulary.
        
        Special tokens:
        <PAD> : Padding token
        <SOS> : Start of sentence token
        <EOS> : End of sentence token
        <UNK> : Token reserved for token not appearing in our vocabulary
    
    '''
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(str(text))]

    def build_vocab(self, sentences):
        freqs = {}
        idx = 4

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1
                # if the token has appeared at least a number of times equal to the frequency threshold, we add it to our vocabulary.
                if freqs[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        '''
        Returns a list with the numerical representations of each token belonging to the input sentence. 
        '''
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class CocoDataset(Dataset):
    '''
    Class description:
        This class will create our dataset from the images and captions files located in ../data/
    
    '''
    def __init__(self, root_dir,
                        captions_file,
                        transform=None,
                        freq_threshold=5,
                        ):

        self.root_dir = root_dir
        # Reading dataset from tsv file
        self.df = pd.read_csv(captions_file, sep = '\t')
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        '''
        For a given index, it returns the image and its corresponding numericalized caption.
        '''
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class Collate:
    '''
    Pytorch's DataLoader function allows us to use our own collate function. For every batch we pass into the training loop, we need all its elements to have the same size, namely the size of the longest caption.
    
    In order to achieve that we will add the special <PAD> token to every caption. We will specify <PAD>'s index when we call our loss function, so that it knows that it should not count them as generated tokens. 
    '''
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        '''
        For the given batch provided by the DataLoader this function will return the tensor containing all concatenated images and its corresponding padded captions. 
        '''
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_dataset(root_folder, annotation_file, transform):
    '''
    Given the location of our images and annotation file, it returns the corresponding CocoDataset object. 
    '''
    dataset = CocoDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]
    return dataset, pad_idx


def print_examples(model, dataset):
    '''
    This function serves as a reality check for our results. It captions 5 images and outputs both the golden captions and the generated ones.
    '''
    
    # Setup 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            ]
        )

    model.eval()
    
    # Testing
    test_img1 = transform(Image.open("../data/test_examples/man_on_phone.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: A man talking on his phone in the public")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("../data/test_examples/giraffe.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: A giraffe walking in the grass near a fence")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("../data/test_examples/women.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: A group of women in a small kitchen.")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("../data/test_examples/stuffed.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A group of stuffed animals are lined up on a bed.")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("../data/test_examples/bowl.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A bowl filled with vegetables and noodles on a table.")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
