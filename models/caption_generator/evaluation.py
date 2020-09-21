import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import nltk
import pickle
import torch
import torch.nn as nn
import yaml
from utilities import get_dataset, get_examples
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_model import CaptionGenerator, FlickrDataModule
from PIL import Image
from bleu import BLEU
import torchvision.transforms as transforms


#Loading testing dataset

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

test, pad_idx = get_dataset(
                        "../../data/flickr8k/images",
                        "../../data/caption_generator/testing_captions.csv",
                        transform)


file_names = np.unique(np.asarray(test.df['image']))

imgs = []
for name in file_names:
    path = '../../data/flickr8k/images/'+name
    imgs.append(transform(Image.open(path).convert('RGB')).unsqueeze(0))

def model_eval(epoch_file):
    print('=============================================')
    print()
    model_version_number = epoch_file.split('/')[5].split('_')[1]
    print('Testing model version: ', model_version_number)

    model = CaptionGenerator.load_from_checkpoint(checkpoint_path = epoch_file, pad_idx = pad_idx)
    model.eval()

    with open(r'../../data/caption_generator/lightning_logs/version_'+model_version_number+'/hparams.yaml') as file:
        parameters = yaml.load(file, Loader = yaml.FullLoader)
    print('With parameters: ', parameters)

    #captions = [" ".join(model.caption_image(image, dataset.vocab)[1:-1]) for image in imgs]

    # Putting the file names and their corresponding captions together in a DataFrame to then save as .tsv
    #df = pd.DataFrame(data = {'image':file_names, 'caption':captions})
    #df.to_csv('../../data/caption_generator/version_'+model_version_number+'_outputs.tsv', index = False, sep = '\t')

    evaluation = BLEU('../../data/caption_generator/version_'+model_version_number+'_outputs.tsv')
    azul = evaluation.get_bleu_score()

    examples = get_examples(model, dataset)

    print('The model achieved the following performance on the test set: ')
    print('BLEU-4 average (rounded) score: ' + '{:.3f}'.format(azul))
    print()

    print('=============================================')
    print()

    return model_version_number, parameters, azul, examples



#retrieving versions found in lightning_logs folder
path = '../../data/caption_generator/lightning_logs/'
version_names = [x for x in os.listdir(path) if x != '.DS_Store']
version_numbers = np.sort([int(name.split('_')[1]) for name in version_names])

#retrieving epoch files
epoch_files = []
for num in version_numbers:
    num = str(num)
    epoch_path = '../../data/caption_generator/lightning_logs/version_' + num + '/checkpoints/'
    epoch_files.append(epoch_path+os.listdir(epoch_path)[0])

#Running evaluation and saving relevant data in a dictionary with key:pairs as model_version_number:[mean_smooth, mean_l1, mean_mse]
results = {}


for epoch_file in epoch_files:
    number, params, azul, examples = model_eval(epoch_file)
    results[int(number)] = {'parameters': params, 'bleu':azul, 'examples': examples}

with open('evaluation_results_finalModel.pkl', 'wb') as input:
    pickle.dump(results, input)

#with open('evaluation_results.pkl', 'rb') as input:
#   test = pickle.load(input)
#test.keys()
#test[1]
