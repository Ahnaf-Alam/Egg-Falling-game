#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 23:05:29 2020

@author: simanto
"""


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('digit_recognizer1.h5')

def plot(filename):
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    plt.imshow(img,cmap='gray')
    plt.show()
    img=img_to_array(img)
    return img.reshape(-1,28,28,1)

def load_image(filename):
	
	img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
	
	img = img_to_array(img)
	img = img.reshape(1, 28, 28, 1)
	
	img = img.astype('float32')
	img = img / 255.0
	return img


def run_example(file):
	img= plot(file)
    #img = load_image(file)
	print(model.predict(img))
	digit = model.predict(img)
	print(np.argmax(digit))



file='8.png'
run_example(file)
