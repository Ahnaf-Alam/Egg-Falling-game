#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:40:19 2020

@author: simanto
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.engine.topology import get_source_inputs
from keras.layers import Add
from keras.layers import Dropout, Flatten, Activation
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling2D, MaxPooling2D
from keras.models import Model
from keras.models import load_model
from keras.regularizers import l2

dataset = pd.read_csv('train.csv')


x = dataset.iloc[:,1:785].values
y = dataset.iloc[:,0].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


x_train=x_train.reshape(33600,28,28,1)
x_test=x_test.reshape(-1,28,28,1)


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

input_layer = Input(shape=[28, 28, 1])
layer1 = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(input_layer)
layer2 = Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')(layer1)
layer3 = MaxPool2D(pool_size=2)(layer2)
layer4 = Dropout(0.25)(layer3)
layer5 = Conv2D(filters=64, kernel_size=5, padding='same', activation='relu')(layer4)
layer6 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(layer5)
layer7 = Conv2D(filters=64, kernel_size=5, padding='same', activation='relu')(layer4)
layer8 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(layer7)
added = Add()([layer6, layer8])
layer9 = MaxPool2D(pool_size=2)(added)
layer10 = Dropout(rate=0.25)(layer9)
flatten = Flatten()(layer10)
fullyConnectedLayer11 = Dense(1280)(flatten)
layer12 = Dropout(rate=0.25)(fullyConnectedLayer11)
outputLayer = Dense(10, activation='softmax')(layer12)
model = Model([input_layer], outputLayer)


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(x_train,y_train,epochs=1,batch_size=100,validation_split=0.1)

model.save('digit_recognizer.h5')
print(model.evaluate(x_test,y_test))