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
file = sys.argv[1]
####################################################3
def predict(infile):
	image = cv2.imread(infile)
	#cv2.imshow('image',image)
	output = imutils.resize(image,width=400)
	model = load_model('chess.h5' )

	image = cv2.resize(image, (100, 100))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	#score = model.evaluate(X_test,labels_test,batch_size=None,verbose=0)
	classes = ['empty', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
	proba = list(model.predict(image)[0])
	ind = proba.index(max(proba))
	return classes[ind]


def main():
	print("prediction: ",predict(file))
	image = cv2.imread(file)
	#cv2.imshow('image',image)
	output = imutils.resize(image,width=400)
	model = load_model('chess.h5' )

	image = cv2.resize(image, (100, 100))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	#score = model.evaluate(X_test,labels_test,batch_size=None,verbose=0)
	proba = model.predict(image)[0]
	print(proba)
	idxs = np.argsort(proba)[::-1]
	classes = ['empty', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
	# loop over the indexes of the high confidence class labels
	for (i, j) in enumerate(idxs):
		# build the label and draw the label on the image
		label = "{}: {:.2f}%".format(classes[j], proba[j] * 100)
		cv2.putText(output, label, (10, (i * 30) + 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

	# show the probabilities for each of the individual labels
	for (label, p) in zip(classes, proba):
		print("{}: {:.2f}%".format(label, p * 100))

	# show the output image
	cv2.imshow("Output", output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Print test accuracy
	#print('\n', 'Test accuracy:', score[1])
	#
if __name__ == "__main__":
    main()
