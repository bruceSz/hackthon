
import abc
import numpy as np

from keras.applications.vgg16 import VGG16

class FeatureGenerator(object):

    @abc.ABCMeta
    def ft_gen(self,img):
        pass


class VGGFeatureGenerator(FeatureGenerator):
    VGG_INPUT_SHAPE = (224,224,3)
    VGG_WEIGHTS = 'imagenet'
    VGG_POOLING = "max"


    def __init__(self):
        self.input_shape = VGGFeatureGenerator.VGG_INPUT_SHAPE
        self.weight = VGGFeatureGenerator.VGG_WEIGHTS
        self.pooling = VGGFeatureGenerator.VGG_POOLING
        self.model = VGG16(weights=self.weight,
                           input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                           pooling=self.pooling, include_top=False)
        self.model.predict(np.zeros((1, 224, 224, 3)))