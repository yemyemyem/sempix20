import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import pickle
import matplotlib.pyplot as plt
import skimage.io as io
from pycocotools.coco import COCO
from utilities import get_dataset, get_examples
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_model import CaptionGenerator, FlickrDataModule
from PIL import Image
from bleu import BLEU
import torchvision.transforms as transforms

#setting random seed
np.random.seed(163)

#loading training dataset
transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ]
    )

dataset, pad_idx = get_dataset(
                            '../../data/flickr8k/images',
                            '../../data/flickr8k/training_captions.csv',
                            transform)

#load COCO API
coco = COCO('../level_generator/annotations/instances_val2014.json')
coco_caps = COCO('../level_generator/annotations/captions_val2014.json')

#load model
epoch_file = '../../data/caption_generator/version_24/checkpoints/epoch=998.ckpt'
model = CaptionGenerator.load_from_checkpoint(checkpoint_path = epoch_file, pad_idx = pad_idx)

#load levels
with open('../../data/levels.pkl', 'rb') as input:
    levels = pickle.load(input)

# Retrieval task for every generated level

#to keep track of results
results = {}
for n in levels.keys():
    results[n] = {}

    for c in levels[n].keys():
        results[n][c] = []

        for j in range(50):
            print('Evaluating: ', n, c, j)

            #loading level to evaluate
            level = levels[n][c][j]

            #checking that the level is not one of the 8 non-functioning levels

            if level['selected_set'] != 0:

                #select target image and its golden caption
                rand_index = np.random.randint(n)

                im = io.imread(level['imgs'][rand_index]['coco_url'])

                #selecting one caption randomly
                capId = coco_caps.getAnnIds(level['imgs'][rand_index]['id'])[np.random.randint(5)]
                cap = coco_caps.loadAnns(capId)[0]['caption'].split()

                target = [im, cap]

                #for every image in the level generate caption
                gen_captions = []
                for i in range(n):
                    #transforming every image
                    im = transform(Image.fromarray(io.imread(level['imgs'][i]['coco_url'])).convert('RGB')).unsqueeze(0)
                    #generating caption
                    gen_captions.append(model.caption_image(im, dataset.vocab)[1:-1])

                #get pair-wise BLEU scores
                pw_scores = [sentence_bleu(target[1], gen_captions[i], weights=(0.25, 0.25, 0.25, 0.25)) for i in range(n)]

                #return argmax
                argmax = np.argmax(pw_scores)

                #evaluate if correct or not and register result
                if argmax == rand_index:
                    results[n][c].append(1)
                else:
                    results[n][c].append(0)

with open('eval_results.pkl', 'wb') as input:
    pickle.dump(results, input)

with open('eval_results.pkl', 'wb') as output:
    pickle.dump(results, output)
