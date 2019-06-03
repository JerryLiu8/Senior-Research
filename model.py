import matplotlib
matplotlib.use('Agg')

from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

##########################
import keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


##########################
pickle_in = open("x.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

pickle_in = open("x_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

print(len(X))
print(len(y))

labels = keras.utils.to_categorical(y, num_classes=6) #one hot vectors
labels_test = keras.utils.to_categorical(y_test, num_classes=6) #one hot vectors

X = X/255.0
X_test = X_test/255.0
###########################333
model = Sequential()
#model.add(Dropout(0.2))
# Must define the input shape in the first layer of the neural network

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))

#model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.35))

#model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))

model.add(Conv2D(filters=4024, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
#model.add(Dropout(0.8))

model.add(Dense(6, activation='softmax'))

# Take a look at the model summary
model.summary()
######################
model.compile(loss='categorical_crossentropy', # binary crossentropy is only for 2 classes
             optimizer=SGD(lr=0.05),
             metrics=['accuracy'])
#######################

history = model.fit(X, labels, batch_size = 32, epochs = 50, validation_split = 0.3)

print(history.history.keys())

# summarize history for accuracy   
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ion()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ion()

# Evaluate the model on test set
score = model.evaluate(X_test,labels_test,batch_size=None,verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])

model.save('chess.h5')
del model
print('done')
