import numpy as np
import cv2
import tensorflow as tf
import sklearn
from matplotlib import pyplot as plt
from shutil import copyfile
import shutil
import random
import albumentations as A

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score, binary_accuracy
import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()

from os import listdir, mkdir, getcwd
from os.path import isfile, join

# # Importing dataset paths

labels_path = "data/ai4mars-dataset-merged-0.1/msl/labels/train/"
imgs_path = "data/ai4mars-dataset-merged-0.1/msl/images/edr/"

labels_files = [f for f in listdir(labels_path) if isfile(join(labels_path, f))]
imgs_files = [f for f in listdir(imgs_path) if isfile(join(imgs_path, f))]


# # Selecting images and cleaning labels with big rocks[3]

# I will select a subset of images where big_rocks appear and another one where they don't appear. In this way, I will be able to create balanced datasets for training and testing.
# 
# I will label the background as 0 and the big_rocks as 1.
# 
# I will save the original images normalized to the [0,1] range.
def selecting_images():
    max_images = 200
    label_id = 3
    imgs_contain_rocks = []
    labels_contain_rocks = []
    imgs_contain_no_rocks = []
    labels_contain_no_rocks = []

    labels_only_rocks = "data/ai4mars-dataset-merged-0.1/msl/labels/only_rocks/"
    imgs_only_rocks = "data/ai4mars-dataset-merged-0.1/msl/images/only_rocks/"

    #randomly shuffled image names to make sure there is no order dependency
    random.shuffle(labels_files)

    count = 0
    print(getcwd())
    try:
        mkdir(labels_only_rocks)
    except:
        print("Folder already exists")
        max_images = 0
        count = len(labels_files)
    try:
        mkdir(imgs_only_rocks)
    except:
        print("Folder already exists")
        max_images = 0
        count = len(labels_files)

    while (len(imgs_contain_rocks) < max_images or len(imgs_contain_no_rocks) < max_images) and count < len(labels_files)-1:
        label_name = labels_files[count]
        labels = cv2.imread(labels_path+label_name)
        count += 1
        
        if label_id in np.unique(labels):
            if len(imgs_contain_rocks) < max_images:
                labels[labels!=label_id] = 255
                labels[labels==label_id] = 1
                labels[labels!=label_id] = 0
                imgs_contain_rocks.append(imgs_only_rocks+label_name[:-4]+".JPG")
                labels_contain_rocks.append(labels_only_rocks+label_name)

                cv2.imwrite(labels_only_rocks+label_name, labels)
                copyfile(imgs_path+label_name[:-4]+".JPG", imgs_only_rocks+label_name[:-4]+".JPG")

        elif len(imgs_contain_no_rocks) < max_images:
            labels[labels!=label_id] = 0
            imgs_contain_no_rocks.append(imgs_only_rocks+label_name[:-4]+".JPG")
            labels_contain_no_rocks.append(labels_only_rocks+label_name)

            cv2.imwrite(labels_only_rocks+label_name, labels)
            copyfile(imgs_path+label_name[:-4]+".JPG", imgs_only_rocks+label_name[:-4]+".JPG")
        print("num_rocks: ", len(imgs_contain_rocks), "num_no_rocks", len(imgs_contain_no_rocks))

    if max_images > 0:
        with open('src/contain_rocks/imgs_contain_no_rocks.txt', 'w') as f:
            for item in imgs_contain_no_rocks:
                f.write("%s\n" % item)

        with open('src/contain_rocks/labels_contain_no_rocks.txt', 'w') as f:
            for item in labels_contain_no_rocks:
                f.write("%s\n" % item)
            

        with open('src/contain_rocks/imgs_contain_rocks.txt', 'w') as f:
            for item in imgs_contain_rocks:
                f.write("%s\n" % item)

        with open('src/contain_rocks/labels_contain_rocks.txt', 'w') as f:
            for item in labels_contain_rocks:
                f.write("%s\n" % item)


# # Splitting data into train, validation and test datasets

def split_dataset(num_imgs_rocks, num_imgs_no_rocks,split=0.6):
    imgs_contain_rocks = np.loadtxt('src/contain_rocks/imgs_contain_rocks.txt', dtype=str)
    labels_contain_rocks = np.loadtxt('src/contain_rocks/labels_contain_rocks.txt', dtype=str)
    labels_contain_no_rocks = np.loadtxt('src/contain_rocks/labels_contain_no_rocks.txt', dtype=str)
    imgs_contain_no_rocks = np.loadtxt('src/contain_rocks/imgs_contain_no_rocks.txt', dtype=str)

    num_rock_images_train = int(num_imgs_rocks * split)
    num_no_rock_images_train = int(num_imgs_no_rocks * split)

    indexes_rocks_train = random.sample(range(0, num_imgs_rocks), num_rock_images_train)
    indexes_rocks_no_train = np.setxor1d(range(0,num_imgs_rocks), indexes_rocks_train)
    indexes_rocks_validation = indexes_rocks_no_train[random.sample(range(0,len(indexes_rocks_no_train)), len(indexes_rocks_no_train)//2)]
    indexes_rocks_test = np.setxor1d(indexes_rocks_no_train, indexes_rocks_validation)

    indexes_no_rocks_train = random.sample(range(0, num_imgs_no_rocks), num_rock_images_train)
    indexes_no_rocks_no_train = np.setxor1d(range(0,num_imgs_no_rocks), indexes_no_rocks_train)
    indexes_no_rocks_validation = indexes_no_rocks_no_train[random.sample(range(0,len(indexes_no_rocks_no_train)), len(indexes_no_rocks_no_train)//2)]
    indexes_no_rocks_test = np.setxor1d(indexes_no_rocks_no_train, indexes_no_rocks_validation)

    imgs_paths_train = np.concatenate((imgs_contain_rocks[indexes_rocks_train], imgs_contain_no_rocks[indexes_no_rocks_train]))
    labels_paths_train = np.concatenate((labels_contain_rocks[indexes_rocks_train], labels_contain_no_rocks[indexes_no_rocks_train]))
    imgs_paths_test = np.concatenate((imgs_contain_rocks[indexes_rocks_test], imgs_contain_no_rocks[indexes_no_rocks_test]))
    labels_paths_test = np.concatenate((labels_contain_rocks[indexes_rocks_test], labels_contain_no_rocks[indexes_no_rocks_test]))
    imgs_paths_validation = np.concatenate((imgs_contain_rocks[indexes_rocks_validation], imgs_contain_no_rocks[indexes_no_rocks_validation]))
    labels_paths_validation = np.concatenate((labels_contain_rocks[indexes_rocks_validation], labels_contain_no_rocks[indexes_no_rocks_validation]))

    return imgs_paths_train, labels_paths_train, imgs_paths_test, labels_paths_test, imgs_paths_validation, labels_paths_validation

# ## Applying some data augmentation to the data
# To the dataset used, some images are duplicated but transformed. The possible transformations that can be applied to the images are horizonal flip, brigthness and contrast, gaussian blur and sharppenning of the image.
# 
# This transformations will help the model to generalize more and be able to detect a similar object in different capturing conditions.

def apply_random_data_augmentation(img, labels, prob=0.5):
    if random.uniform(0,1) <= prob:
        transform = A.Compose([ A.HorizontalFlip(p=0.5),
                                A.RandomBrightnessContrast(p=0.2),
                                A.GaussianBlur(p=0.2),
                                A.Sharpen(p=0.2)
                                ])
        transformed = transform(image=img, mask=labels)
        return transformed['image'], transformed['mask']
    else:
        return np.array([]), np.array([])

# ## Loading selected images and apply the data augmentation
# 
# When this step is done, the data will be augmented according to the different transformations and split into training, validation and testing dataset. These 3 different datasets will have the same distribution of the classes in it.

def load_imgs_from_indexes(imgs_paths, labels_paths):
    imgs_list = []
    labels_list = []
    for indx in random.sample(range(0, len(imgs_paths)), len(imgs_paths)):
        img_path = imgs_paths[indx]
        labels_path = labels_paths[indx]

        img = cv2.imread(img_path)#cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        labels = cv2.imread(labels_path)[:,:,0]
        imgs_list.append(img)
        labels_list.append(labels)

        img_augm, labels_augm = apply_random_data_augmentation(img=img, labels=labels)
        if img_augm.shape[0] != 0:
            imgs_list.append(img_augm)
            labels_list.append(labels_augm)
    return imgs_list, labels_list

def load_data(split=0.6):
    imgs_contain_rocks = np.loadtxt('src/contain_rocks/imgs_contain_rocks.txt', dtype=str)
    labels_contain_rocks = np.loadtxt('src/contain_rocks/labels_contain_rocks.txt', dtype=str)
    labels_contain_no_rocks = np.loadtxt('src/contain_rocks/labels_contain_no_rocks.txt', dtype=str)
    imgs_contain_no_rocks = np.loadtxt('src/contain_rocks/imgs_contain_no_rocks.txt', dtype=str)

    imgs_paths_train, labels_paths_train, imgs_paths_test, labels_paths_test, imgs_paths_validation, \
        labels_paths_validation = split_dataset(len(imgs_contain_rocks), len(imgs_contain_no_rocks),split=split)
        
    
    #split the data set using the indexes and return 6 np.arrays of loaded images
    x_train, y_train = load_imgs_from_indexes(imgs_paths=imgs_paths_train, labels_paths=labels_paths_train)
    x_test, y_test = load_imgs_from_indexes(imgs_paths=imgs_paths_test, labels_paths=labels_paths_test)
    x_val, y_val = load_imgs_from_indexes(imgs_paths=imgs_paths_validation, labels_paths=labels_paths_validation)

    return x_train, y_train, x_test, y_test, x_val, y_val   
    
# # Training model

# load your data
def tf_dataset():
    x_train, y_train, x_test, y_test, x_val, y_val = load_data(split=0.6)

    x_train = tf.convert_to_tensor(x_train[:200])
    y_train = tf.convert_to_tensor(y_train[:200])
    y_train = tf.cast(y_train, tf.float32)

    x_test = tf.convert_to_tensor(x_test[:100])
    y_test = tf.convert_to_tensor(y_test[:100])
    y_test = tf.cast(y_test, tf.float32)

    x_val = tf.convert_to_tensor(x_val[:100])
    y_val = tf.convert_to_tensor(y_val[:100])
    y_val = tf.cast(y_val, tf.float32)
    return x_train, y_train, x_test, y_test, x_val, y_val

if __name__ == "__main__":
    to_train = False
    selecting_images()

    x_train, y_train, x_test, y_test, x_val, y_val = tf_dataset()
    
    #Following this tutorial https://segmentation-models.readthedocs.io/en/latest/tutorial.html

    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    if to_train:
        print("TRAINING!!")
        # preprocess input
        X_train = preprocess_input(x_train)
        X_val = preprocess_input(x_val)

        shape = X_train[0].shape
        # define model
        model = Unet(BACKBONE, encoder_weights='imagenet', classes=1, input_shape=(shape[0], shape[1], shape[2]),encoder_freeze=True, activation='sigmoid')
        model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score, binary_accuracy])

        # fit model
        model.fit(
            x=X_train,
            y=y_train,
            batch_size=2,
            epochs=10,
            validation_data=(X_val, y_val),
        )
        model.save_weights('src/checkpoints/my_checkpoint')

    # Create a new model instance
    model2 = Unet(BACKBONE, encoder_weights='imagenet', classes=1, input_shape=(shape[0], shape[1], shape[2]),encoder_freeze=True)
    model2.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

    # Restore the weights
    model2.load_weights('src/checkpoints/my_checkpoint')
    
    x_test = preprocess_input(x_test)
    loss, acc = model2.evaluate(x_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))