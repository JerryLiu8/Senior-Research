from __future__ import print_function
import keras
import os
from keras import backend as K
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from numpy import *
from sklearn.utils import shuffle

#source: https://www.youtube.com/watch?v=2pQOXjpO_u0
pieces = ['empty', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
for piece in pieces:
    path1 = f'C:\\Users\\Jerry\\sys\\Data\\test\\{piece}'
    path2 = f'C:\\Users\\Jerry\\sys\\resized\\{piece}'
    print(path2)

    img_rows = 500
    img_cols = 500

    listing = os.listdir(path1)

    num_samples = len(listing)
    print(num_samples)
    #Input image dimensions
    for file in listing:
        im = Image.open(path1 + '\\' + file)
        img = im.resize((img_rows,img_cols))
        gray = img.convert('L')

        gray.save(path2 + '\\' + file, "JPEG")
