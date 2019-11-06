import random
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
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
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.initializers import glorot_uniform
import keras
from keras.utils import np_utils
from keras import regularizers, optimizers
from tensorflow import set_random_seed
set_random_seed(1234)
from numpy.random import seed
seed(1234)

data_dir = "/home/mech/btech/me2160793/COL341/A2_B/data/"
#sys.argv = []
#sys.argv.append(data_dir+"cnn_a.py")
#sys.argv.append(data_dir+"train.csv")
#sys.argv.append(data_dir+"test.csv")
#sys.argv.append(data_dir+"output.txt")
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
print(R.shape, G.shape, B.shape, x_train.shape,y_train.shape, x_val.shape, y_val.shape)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7)
x_val = (x_val-mean)/(std+1e-7)

y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
weight_decay = 1e-4

def vgg_model(x_train):
  inputs = Input(shape=x_train[0].shape)
  x = Conv2D(filters=32, kernel_size=(3,3),padding="same",kernel_initializer=initializers.glorot_uniform(seed=1234) ,kernel_regularizer=regularizers.l2(weight_decay))(inputs)
  x = Activation(activation="relu")(x)
  x = BatchNormalization()(x)
  x = Conv2D(filters=32, kernel_size=(3,3),padding="same",kernel_initializer=initializers.glorot_uniform(seed=1234) ,kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = Activation(activation="relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2,2))(x)
  x = Dropout(0.2)(x)
  x = Conv2D(filters=64, kernel_size=(3,3),padding="same",kernel_initializer=initializers.glorot_uniform(seed=1234) ,kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = Activation(activation="relu")(x)
  x = BatchNormalization()(x)
  x = Conv2D(filters=64, kernel_size=(3,3),padding="same",kernel_initializer=initializers.glorot_uniform(seed=1234) ,kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = Activation(activation="relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2,2))(x)
  x = Dropout(0.3)(x)
  x = Conv2D(filters=128, kernel_size=(3,3),padding="same",kernel_initializer=initializers.glorot_uniform(seed=1234) ,kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = Activation(activation="relu")(x)
  x = BatchNormalization()(x)
  x = Conv2D(filters=128, kernel_size=(3,3),padding="same",kernel_initializer=initializers.glorot_uniform(seed=1234) ,kernel_regularizer=regularizers.l2(weight_decay))(x)
  x = Activation(activation="relu")(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2,2))(x)
  x = Dropout(0.4)(x)
  x = Flatten()(x)
  x = Dense(10,kernel_initializer=initializers.glorot_uniform(seed=1234) ,activation="softmax")(x)
  model = Model(inputs = inputs, output=x)
  return model

  
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)
datagen.fit(x_train)

batch_size = 64
epochs = 125
#opt_rms = keras.optimizers.rmsprop(lr=learning_rate(0), decay=1e-6)

def learning_rate(epoch):
  lr = 0.001
  if(epoch>100):
    lr = 0.0003
  elif(epoch>75):
    lr = 0.0005
  elif(epoch>0):
    lr = 0.001
  return lr
opt_rms = keras.optimizers.rmsprop(lr=learning_rate(0), decay=1e-6)
model = vgg_model(x_train)
model.summary()
lrate = callbacks.LearningRateScheduler(learning_rate)
#es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
filepath="weights-improvement.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.compile(loss='categorical_crossentropy',optimizer=opt_rms,metrics=['accuracy'])
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),callbacks=[lrate,checkpoint], validation_data=(x_val, y_val),steps_per_epoch=x_train.shape[0]/ batch_size, epochs=epochs, verbose=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
load_model = model_from_json(loaded_model_json)
load_model.load_weights("weights-improvement.hdf5")
print("Loaded model from disk")
print("working on prediction")
test_image = test_data.iloc[:,:-1].values
R = test_image[:,0:1024].reshape(test_image.shape[0],32,32)
G = test_image[:,1024:2048].reshape(test_image.shape[0],32,32)
B = test_image[:,2048:3072].reshape(test_image.shape[0],32,32)
images = np.stack((R.T,G.T,B.T),axis=0).T
mean = np.mean(images,axis=(0,1,2,3))
std = np.std(images,axis=(0,1,2,3))
images = (images-mean)/(std+1e-7)
Y_pred = load_model.predict(images.astype("float32"))
a = np.array([ np.where(out==np.amax(out))[0][0] for out in Y_pred])

np.savetxt(sys.argv[3],a)

