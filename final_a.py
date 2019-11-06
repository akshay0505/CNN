from keras.models import model_from_json
from numpy.random import seed
import random
import pandas as pd
import numpy as np
from keras import initializers
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Conv2D, Conv3D, Flatten, MaxPool2D, MaxPooling2D, Dropout, Add, Input, AveragePooling2D, UpSampling2D, ZeroPadding2D
import os
from keras import initializers

import sys
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.utils import to_categorical
from keras.optimizers import rmsprop, adam, Adagrad, SGD
from keras import callbacks
from keras.layers import BatchNormalization
from keras.initializers import glorot_uniform
import keras
from keras.utils import np_utils
from keras import regularizers, optimizers
from tensorflow import set_random_seed
set_random_seed(1234)
seed(1234)
train_data = pd.read_csv(sys.argv[1], header=None, delimiter=" ")
test_data = pd.read_csv(sys.argv[2], header=None, delimiter=" ")

train_image = train_data.iloc[:, :-1].values
R = train_image[:, 0:1024].reshape(train_image.shape[0], 32, 32)
G = train_image[:, 1024:2048].reshape(train_image.shape[0], 32, 32)
B = train_image[:, 2048:3072].reshape(train_image.shape[0], 32, 32)
images = np.stack((R.T, G.T, B.T), axis=0).T
print(train_data.shape, test_data.shape, images.shape)
k = 40000
x_train = images[:k]
x_val = images[k:]
y_train = train_data.iloc[:k, -1].values
y_val = train_data.iloc[k:, -1].values
print(R.shape, G.shape, B.shape, x_train.shape,
      y_train.shape, x_val.shape, y_val.shape)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7)
x_val = (x_val-mean)/(std+1e-7)

y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
weight_decay = 1e-4


def model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu',strides=1, padding="same", input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu',padding="same", strides=1))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(10, activation="softmax"))
    return model
model = model()
model.summary()
optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = [],shuffle=True,epochs=6,batch_size=64)

test_image = test_data.iloc[:, :-1].values
R = test_image[:, 0:1024].reshape(test_image.shape[0], 32, 32)
G = test_image[:, 1024:2048].reshape(test_image.shape[0], 32, 32)
B = test_image[:, 2048:3072].reshape(test_image.shape[0], 32, 32)
images = np.stack((R.T, G.T, B.T), axis=0).T
mean = np.mean(images, axis=(0, 1, 2, 3))
std = np.std(images, axis=(0, 1, 2, 3))
images = (images-mean)/(std+1e-7)
Y_pred = model.predict_classes(images.astype("float32"))
# a = np.array([np.where(out == np.amax(out))[0][0] for out in Y_pred])

np.savetxt(sys.argv[3], Y_pred)

