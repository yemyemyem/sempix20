#Importing
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
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())

    #selecting 20 objects of the 80
    names = set([cat['name'] for cat in cats])

    #mapping of category Id to its name
    self.catId_to_cat = {}
    for name in names:
        current_id = coco.getCatIds(catNms = [name])[0]
        self.catId_to_cat[current_id] = name


    current_testbed = random.sample(names, 20)

    categories_ids = coco.getCatIds(catNms = current_testbed)


    #first generate all possible combinations and then select from them, to make things faster while testing
    # 5 because we limit the level generation to images having up to 5 elements in common
    self.combinations = {}
    for i in range(5):
        self.combinations[i+1] = []
        for x in itertools.combinations(categories_ids, i+1):
            self.combinations[i+1].append(x)


    def level_generator(self, n, c):
        '''
        Function generating the sets of images that we will present to our testing systems.
        Arg:
            n: int, Number of images in the level.
            c: 5 >=int >= 1, Number of shared objects between all images.
        '''
        not_enough_images = True
        i=0
        #Randomly selecting c classes from the current testbed sample
        while not_enough_images and i < 10000:
            cats = random.sample(combinations[c], 1)[0]

            #Randomly selecting n images
            options = coco.getImgIds(catIds = cats)

            if len(options) >= n:
                    not_enough_images = False
                    imgIds = random.sample(options, n)
                    imgs = [coco.loadImgs(imgIds[i])[0] for i in range(len(imgIds))]
                    print('Selected categories: ')
                    for cat in cats:
                        print(catId_to_cat[cat])
                    return imgs
            i+=1
        print('Set of images not found, please try again with different parameters.')
        return 0

    def save(self.):
        '''
        Saved the generated levels in a .pkl file
        '''
        with open('level.pkl', 'wb') as input:
            pickle.dump(self.level_generator(int(sys.argv[1]), int(sys.argv[2])), input)
