from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import pickle
import pandas as pd
import numpy as np

class BLEU():
    def __init__(self, output_captions_file, weight=(.25, .25, .25, .25)):
        
        #This file contains a dictionary form of the gold captions for the testing dataset.
        with open('../data/gold_dict_testing.pkl', 'rb') as output:
            self.gold_dict = pickle.load(output)
        self.output_captions_file = output_captions_file
        self.weight = weight 

        with open(self.output_captions_file, 'r') as f:
            self.df_output = pd.read_csv(f, sep='\t')

    
    def make_dict(self, df):
        d = {}
        try:
            for image in df['image']:
                if image not in d.keys():
                    d[image] = [word_tokenize(i) for i in list(df[df['image'] == image]['caption'])]
        except TypeError:
            pass

        return d


    def make_bleu_dict(self):
        '''
        return dictionary with image id as key and bleu score as value
         {'000000203564.jpg': 0.5410822690539396}
        
        '''
        bleu_dict = {}
        self.output_dict = self.make_dict(self.df_output)
        
        for image in self.df_output['image']:
            reference = self.gold_dict[image]
            candidate = self.output_dict[image][0]
            
            bleu_dict[image] = sentence_bleu(reference, candidate, weights=self.weight)
            
        return bleu_dict

    
    def get_bleu_score(self):
        '''

        returns average of BLEU-4 score over all output captions
        
        '''
        bleu_dict = self.make_bleu_dict()
        scores = np.fromiter(bleu_dict.values(), dtype=float)

        return np.mean(scores)


# azul = BLEU('captions.txt', 'tiny_output_captions.txt')

# score = azul.get_bleu_score()
# print(score)

# #0.6458