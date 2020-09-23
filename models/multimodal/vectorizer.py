import spacy
import torch
import numpy as np
from gensim.models import KeyedVectors
import pickle

class CaptionVectorizer:
    def __init__(self):
        self.w2i = dict()
        self.embedding = None
        self.embedding_size = 0
        self.spacy_eng = spacy.load("en")

    def generate_embedding(self, captions, threshold):
        """Generates a vocabulary with matrix embedding.

        Args:
            captions: A list of captions
            threshold: Cut-off threshold for words
        """

        glove = KeyedVectors.load_word2vec_format("glove.6B.100d.bin.word2vec", binary=True)
        freq_dict = dict()

        # Analyse caption vocab
        for caption in captions:
            tokens = [tok.text for tok in self.spacy_eng.tokenizer(caption.lower())]
            for token in tokens:
                if token in freq_dict:
                    freq_dict[token] += 1
                else:
                    freq_dict[token] = 1

        # Truncate vocab
        vocab = set()
        for key,value in freq_dict.items():
            if value >= threshold and key in glove:
                vocab.add(key)
        
        # Build word index
        words = list(vocab)
        words.extend(["<SOS>", "<EOS>", "<UNK>", "<PAD>"])
        print("Actual vocab size:",len(freq_dict.keys()))
        print("Truncated vocab size:", len(words))

        self.w2i = { w:i for i,w in enumerate(words) }

        # Load glove embedding for the word in the vocabulary
        k = glove.vector_size
        self.embedding_size = glove.vector_size+4
        self.embedding = np.zeros((len(words), self.embedding_size))
        for i, word in enumerate(words):
            if word in glove:
                self.embedding[i] = np.concatenate((glove[word], np.zeros(4)))
            else:
                self.embedding[i] = np.zeros(self.embedding_size)
                self.embedding[i][k] = 1
                k += 1

    def write_embedding(self, path):
        with open(path, "wb") as f:
            pickle.dump({"embedding": self.embedding,
                "w2i": self.w2i}, f)

    def load_embedding(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.embedding = data["embedding"]
            self.embedding_size = self.embedding.shape[1]
            self.w2i = data["w2i"]

    def encode_caption(self, caption):
        v = np.zeros((len(caption)+2), dtype=np.long)
        v[0] = self.w2i["<SOS>"]
        for i, token in enumerate(caption):
            if token in self.w2i:
                v[i+1] = self.w2i[token]
            else:
                v[i+1] = self.w2i["<UNK>"]
        v[len(caption)+1] = self.w2i["<EOS>"]
        return v

    def embed_caption(self, caption):
        caption0 = self.encode_caption(caption)
        return torch.FloatTensor([self.embedding[idx] for idx in caption0])

    def __call__(self, caption):
        tokens = [tok.text for tok in self.spacy_eng.tokenizer(caption.lower())]
        return self.embed_caption(tokens)
