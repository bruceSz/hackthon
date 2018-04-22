
from mod import indexer
#from utils import misc
import h5py
from mod import ft_generator
import numpy as np

import abc

class SearchEngineBase(object):
    __meta__ = abc.ABCMeta
    @abc.abstractmethod
    def search(self, img):
        pass


class SearchEngineVGGML(SearchEngineBase):

    def __init__(self):
        self.vgg_ = ft_generator.VGGFeatureGenerator()
        # TODO should be a class_label mapper
        self.model_ = indexer.SvmIndexer.load_model()
        self.class_label_2name_ = indexer.SvmIndexer.load_index2name()


    def search(self, img):
        # 1 extract fts with VGG
        query_v = self.vgg_.extract_ft(img)
        print(query_v.shape)
        # 2 model predict
        class_label = self.model_.predict(query_v)
        img_id = self._class_output_img_list(class_label)
        return img_id

    def _class_output_img_list(self, class_label_idx):
        print(type(class_label_idx))
        print(class_label_idx)
        return [self.class_label_2name_[class_label_idx[0]]]

    def _class_output2json(self,class_label):
        # TODO class_label map to real result.
        pass

#@misc.Singleton
# TODO.
class SearchEngine(SearchEngineBase):
    _INDEX_FILE = "data/online/in/data.index"
    def __init__(self):
        #self.conf_ = {}
        h5f = h5py.File(SearchEngine._INDEX_FILE, 'r')
        self.pic_fts = h5f['ft_data'][:]
        self.pic_names = h5f['name_data'][:]
        h5f.close()
        self._init_index()
        self.vgg_model_ = ft_generator.VGGFeatureGenerator()

    def _init_index(self):
        pass

    def search(self, img):
        # 1 extact fts.
        # 2 dot multiply and compute the the score.
        # 3 simlest model , compute the cosine distance between given img and those img in repos.
        query_v = self.vgg_model_.extract_ft(img)
        print(query_v.shape)
        scores = np.dot(query_v, self.pic_fts.T)
        r_idx = np.argsort(scores)[::-1]
        max_ret = 3
        img_list = [self.pic_names[index] for i, index in enumerate(r_idx[0:max_ret])]
        return img_list



