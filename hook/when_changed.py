import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2

def changed():
    print("There's some change in data folder")
    preprocessing_train()

def preprocessing_train():
    from glob import glob

    print (os.getcwd())
    train_path = "100_1"
    test_path = "100_2"


    train_imgs_path = []
    test_imgs_path = []

    X_train = []
    y_train = []

    X_test = []
    y_test = []
    for id in range(10):
        current_id_train_imgs_path = glob (f"{train_path}/{id}/*.jpg")
        current_id_train_imgs_size = len(current_id_train_imgs_path)

        train_imgs_path += current_id_train_imgs_path

        current_id_test_imgs_path = glob(f"{test_path}/{id}/*.jpg")
        current_id_test_imgs_size = len(current_id_test_imgs_path)

        test_imgs_path += current_id_test_imgs_path


        y_train += [id] * current_id_train_imgs_size
        y_test += [id] * current_id_test_imgs_size
    y_train = np.array(y_train).astype('float32')
    y_test = np.array(y_test).astype('float32')
    len(train_imgs_path), len(test_imgs_path), len (y_train), len(y_test)


    train_imgs_list = []
    test_imgs_list = []
    for path in train_imgs_path:
        train_imgs_list.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    for path in test_imgs_path:
        test_imgs_list.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    X_train = np.array(train_imgs_list).reshape(-1, 28, 28, 1).astype('float32')
    X_test = np.array(test_imgs_list).reshape(-1, 28, 28, 1).astype('float32')

    from keras import layers
    from keras import Sequential
    from keras.callbacks import EarlyStopping

    model = Sequential([
        layers.Conv2D(8, (2, 2), activation='relu'),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics= ['accuracy']
    )
    es = EarlyStopping(patience=10)
    history = model.fit(X_train, y_train, epochs = 100, validation_split = 0.2, callbacks=[es])
if __name__ == "__main__":
    changed()