#Imports
import os
os.getcwd()
#os.chdir('PythonAPI/')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import pickle
import sys
random.seed(163)
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
import itertools



class levelGenerator():

    def __init__(self):

        annFile='../annotations/instances_val2014.json'
        # initialize COCO api for instance annotations
        self.coco=COCO(annFile);

        #getting COCO categories
        cats = self.coco.loadCats(self.coco.getCatIds())

        #selecting 20 objects of the 80
        names = set([cat['name'] for cat in cats])

        #mapping of category Id to its name
        self.catId_to_cat = {}
        for name in names:
            current_id = self.coco.getCatIds(catNms = [name])[0]
            self.catId_to_cat[current_id] = name


        current_testbed = random.sample(names, 20)

        categories_ids = self.coco.getCatIds(catNms = current_testbed)


        #first generate all possible combinations and then select from them, to make things faster while testing
        # 5 because we limit the level generation to images having up to 5 elements in common
        self.combinations = {}
        for i in range(5):
            self.combinations[i+1] = []
            for x in itertools.combinations(categories_ids, i+1):
                self.combinations[i+1].append(x)


    def generate(self, n, c, display_selected_cats = 0):
        '''
        Function generating the sets of images that we will present to our testing systems.
        Arg:
            n: 10 >= int, Number of images in the level.
            c: 5 >=int >= 1, Number of shared objects between all images.
            display_selected_cats: bool, tool for debugging.
        '''
        not_enough_images = True
        i=0
        #Randomly selecting c classes from the current testbed sample
        #Sampling 10000 times before giving up
        while not_enough_images and i < 10000:
            cats = random.sample(self.combinations[c], 1)[0]

            #Randomly selecting n images
            options = self.coco.getImgIds(catIds = cats)

            if len(options) >= n:
                    not_enough_images = False
                    imgIds = random.sample(options, n)
                    imgs = [self.coco.loadImgs(imgIds[i])[0] for i in range(len(imgIds))]
                    if display_selected_cats == 1:
                        print('Selected categories: ')
                        for cat in cats:
                            print(self.catId_to_cat[cat])
                    selected_cats = []
                    for cat in cats:
                        selected_cats.append(self.catId_to_cat[cat])
                    return selected_cats, imgs
            i+=1
        #print('Desired combination of images not found, please try again with different parameters.')
        return [0,0]

#Generating common set of levels for evaluation
lg = levelGenerator()

levels = {}
for n in range(2, 11): #for each desired number of images in the set
    levels[n] = {}
    for c in range(1, 6): #for each desired number of objects in common
        levels[n][c] = []
        for i in range(50): #try to get 50 images in each category
            set, imgs = lg.generate(n, c, 0)
            levels[n][c].append({'selected_set':set, 'imgs':imgs})

#Saving levels file
with open('../../../data/levels.pkl', 'wb') as output:
    pickle.dump(levels, output)
