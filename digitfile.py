# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:58:54 2020

@author: Ahnaf
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model




dataset=pd.read_csv('train.csv')

dataset.head()

x=dataset.iloc[:,1:785].values
y=dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train=x_train.reshape(33600,28,28,1)

from keras.utils import to_categorical
y_train=to_categorical(y_train)

classifier=Sequential()
classifier.add(Convolution2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(10, activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,epochs=10,batch_size=500,validation_split=0.2)
classifier.save('digit_recognizer1.h5')

