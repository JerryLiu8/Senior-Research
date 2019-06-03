import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

DATADIR = "C:\\Users\\Jerry\\sys\\Data\\test\\"

CATEGORIES = ["empty", "pawn", "knight", "bishop", "rook", "queen", "king"]

training_data = []

IMG_SIZE = 500

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #print(os.path.join(path,img))
                img_array = cv2.imread(os.path.join(path,img))
                #cv2.imshow("image",new_array)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)
for sample in training_data[:10]:
    #cv2.imshow("image",sample[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(sample[1])

x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)

#print(len(x))
#print(len(y))

print("before: \nShape: ", x[0].shape, "\n", x[0], "\n")


x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #confusing line

print("after: \nShape: ", x[0].shape, "\n", x[0], "\n")

print(len(x))
print(len(y))

pickle_out = open("x_test.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print('done')
