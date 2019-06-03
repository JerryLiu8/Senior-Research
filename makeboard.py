#source - https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
##########################
import keras
import os
import sys
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import pickle
from classify import predict
folder = sys.argv[1]
####################################################3
pieces = [] # from H8 = pieces[0] to A1 = pieces[63]
path = f'C:\\Users\\Jerry\\sys\\{folder}'
listing = os.listdir(path)
for file in listing:
	filepath = f'C:\\Users\\Jerry\\sys\\{folder}\\{file}'
	pieces.append(predict(filepath))
squares = ['A1','B1','C1','D1','E1','F1','G1','H1',
		   'A2','B2','C2','D2','E2','F2','G2','H2',
		   'A3','B3','C3','D3','E3','F3','G3','H3',
		   'A4','B4','C4','D4','E4','F4','G4','H4',
		   'A5','B5','C5','D5','E5','F5','G5','H5',
		   'A6','B6','C6','D6','E6','F6','G6','H6',
		   'A7','B7','C7','D7','E7','F7','G7','H7',
		   'A8','B8','C8','D8','E8','F8','G8','H8']
i = 0
final_out = []
for piece in reversed(pieces): #A1 to H8
	final_out.append((squares[i], piece))
	i+=1
for i in range(len(final_out)):
	print(final_out[i]," ")
	if (i + 1) % 8 == 0:print()
