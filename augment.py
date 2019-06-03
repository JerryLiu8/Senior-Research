from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image
import os

# source: https://www.codesofinterest.com/2018/02/using-data-augmentations-in-keras.html
pieces = ['empty', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
for piece in pieces:
    path = f'C:\\Users\\Jerry\\sys\\resized\\{piece}'
    listing = os.listdir(path)
    num_samples = len(listing)
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    for file in listing:
        img = Image.open(path + '\\' + file)
        img_arr = img_to_array(img)
        # convert to numpy array with shape (1, 3, width, height)
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `data/augmented` directory
        i = 0
        num = 0
        for batch in datagen.flow(
            img_arr,
            batch_size=1,
            save_to_dir=f'C:\\Users\\Jerry\\sys\\data\\training\\{piece}',
            save_prefix='',
            save_format='jpeg'):
            num +=1
            i += 1
            if i > 15:
                break  # otherwise the generator would loop indefinitely
