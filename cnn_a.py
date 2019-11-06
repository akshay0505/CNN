import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation , Conv2D, Conv3D,Flatten , MaxPool2D,MaxPooling2D, Dropout
import os
import sys
from keras.utils import to_categorical
from keras.optimizers import rmsprop , adam, Adagrad
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
seed_value = 1234
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
data_dir = "/content/drive/My Drive/Colab Notebooks/"
sys.argv = []
sys.argv.append(data_dir+"cnn_a.py")
sys.argv.append(data_dir+"train.csv")
sys.argv.append(data_dir+"test.csv")
sys.argv.append(data_dir+"output.txt")
train = pd.read_csv(sys.argv[1],header=None)
test  = pd.read_csv(sys.argv[2],header=None)
k = 40000
val = 10000

X = train_data.iloc[:k,:-1].values
X_train = (X-np.mean(X,axis=0)).reshape(k,32,32,3)
Y_train = to_categorical(train_data.iloc[:k,-1].values).reshape(k,10) 

X = train_data.iloc[k:k+val,:-1].values
X_val = (X-np.mean(X,axis=0)).reshape(val,32,32,3)
Y_val = to_categorical(train_data.iloc[k:k+val,-1].values).reshape(val,10)

X = test_data.iloc[:,:-1].values
X_test = (X-np.mean(X,axis=0)).reshape(test_data.shape[0],32,32,3)

X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

print(X_train.shape, Y_train.shape,X_val.shape,Y_val.shape, X_test.shape)
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu',strides=1,padding="same", input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=3, activation='relu',padding="same",strides=1))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(10,activation="softmax"))
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# optimizer = Adagrad(lr=0.001, decay=1e-6,momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val),callbacks = [],shuffle=True,epochs=30,batch_size=50)
Y_pred = model.predict_classes(X_test)
np.savetxt("predection.txt",Y_pred)