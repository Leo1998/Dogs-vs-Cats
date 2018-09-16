import plaidml.keras
plaidml.keras.install_backend()

import numpy as np 
import os
import cv2
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

MODEL_NAME = "dogsncats-{}".format(int(time.time()))

CATEGORIES = ["Dog", "Cat"]

X = np.load("X.npy")
Y = np.load("Y.npy")

X = np.divide(X, 255.0)
Y = keras.utils.to_categorical(Y, 2) 

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#model.add(Dense(128, activation="relu"))
#model.add(Dropout(0.5))

model.add(Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=5, batch_size=64, validation_split=0.2)

model.save("{}.model".format(MODEL_NAME))

