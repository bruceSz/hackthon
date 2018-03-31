# -*- coding: utf-8 -*-
# Author: bruceSz

import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

class VGGNET:
    def __init__(self):
        self.input_shape = (224,224,3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights=self.weight,
                           input_shape=(self.input_shape[0],self.input_shape[1],self.input_shape[2]),
                           pooling=self.pooling,include_top=False)
        self.model.predict(np.zeros((1,224,224,3)))

        '''
        Use vgg16 model to extract features
        Output normalized feature vector
        '''
    def extract_ft(self,img_path):
        img = image.load_img(img_path,target_size=(self.input_shape[0],self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img = preprocess_input(img)
        ft = self.model.predict(img)
        norm_ft = ft[0]/LA.norm(ft[0])
        return norm_ft

