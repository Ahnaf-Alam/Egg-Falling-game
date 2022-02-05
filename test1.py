#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 00:34:38 2020

@author: simanto
"""


import cv2
import math
import numpy as np
import operator
from keras.models import load_model



model = load_model('digit_recognizer.h5')    


roi = cv2.imread('sample_image.png',0)

roi = cv2.resize(roi,(28,28))>0
roi = 255 - roi
cv2.imshow('as',roi)

val = model.predict(roi.reshape(1, 28, 28, 1))
print(val.argmax())
cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
cv2.destroyAllWindows()  #
