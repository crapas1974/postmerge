import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
from glob import glob


from keras import layers
from keras import Sequential
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
def changed():
    print("There's some change in data folder")
    ic = ImageClassification()
    ic.preprocess()
    ic.train()
    ic.evaluate()
class ImageClassification():
    def __init__(self, train_path = "100_1", test_path = "100_2"):
        self.train_path = train_path
        self.test_path = test_path

        self.X_train = []
        self.y_train = []

        self.X_test = []
        self.y_test = []

        self.X_val = []
        self.y_val = []

        self.history = None
        self.model = None
    def preprocess(self):
        train_imgs_path = []
        test_imgs_path = []
        train_imgs_list = []
        test_imgs_list = []

        for id in range(10):
            current_id_train_imgs_path = glob (f"{self.train_path}/{id}/*.jpg")
            current_id_train_imgs_size = len(current_id_train_imgs_path)

            train_imgs_path += current_id_train_imgs_path

            current_id_test_imgs_path = glob(f"{self.test_path}/{id}/*.jpg")
            current_id_test_imgs_size = len(current_id_test_imgs_path)

            test_imgs_path += current_id_test_imgs_path


            self.y_train += [id] * current_id_train_imgs_size
            self.y_test += [id] * current_id_test_imgs_size
        self.y_train = np.array(self.y_train).astype('float32')
        self.y_test = np.array(self.y_test).astype('float32')
        len(train_imgs_path), len(test_imgs_path), len (self.y_train), len(self.y_test)


        for path in train_imgs_path:
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize = (28, 28))
            train_imgs_list.append(img)
    
        for path in test_imgs_path:
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize = (28, 28))
            test_imgs_list.append(img)

        self.X_train = np.array(train_imgs_list).reshape(-1, 28, 28, 1).astype('float32')
        self.X_test = np.array(test_imgs_list).reshape(-1, 28, 28, 1).astype('float32')

        print (self.X_train.shape)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size = 0.2, stratify=self.y_train)
    def train(self):
        self.model = Sequential([
            layers.Conv2D(8, (3, 3), input_shape = (28, 28, 1), activation='relu'),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.ZeroPadding2D(padding=(1,1)),

            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.ZeroPadding2D(padding=(1,1)),

            layers.Flatten(),
            layers.Dense(500, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        self.model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics= ['accuracy']
        )
        es = EarlyStopping(patience=10)

        print (self.model.summary())
        self.history = self.model.fit(self.X_train, self.y_train, batch_size = 500, epochs = 100, validation_data=(self.X_val, self.y_val), callbacks=[es])
    def evaluate(self):
        print (self.model.evaluate(self.X_test, self.y_test))
        self.model.save('models')
if __name__ == "__main__":
    changed()