
import abc
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from numpy import linalg as LA
from PIL import Image

class FeatureGenerator(object):
    #__metaclass__ = abc.ABCMeta
    # TODO. should rename extract_ft with ft_gen.
    @abc.abstractmethod
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

    def extract_ft(self,img_path):
        print(type(img_path))
        #img = Image.open(img_path)
        #img = img.resize((self.input_shape[0],self.input_shape[1]))
        img = image.load_img(img_path,target_size=(self.input_shape[0],self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img = preprocess_input(img)
        ft = self.model.predict(img)
        norm_ft = ft[0]/LA.norm(ft[0])
        return norm_ft