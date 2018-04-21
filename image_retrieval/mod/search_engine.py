
import indexer
#from utils import misc
import h5py
import ft_generator
import numpy as np

#@misc.Singleton
# TODO.
class SearchEngine(object):
    _INDEX_FILE = "data/online/in/data.index"
    def __init__(self):
        #self.conf_ = {}
        h5f = h5py.File(SearchEngine._INDEX_FILE, 'r')
        self.pic_fts = h5f['ft_data'][:]
        self.pic_names = h5f['name_data'][:]
        h5f.close()
        self.init_index()

        self.vgg_model_ = ft_generator.VGGFeatureGenerator()

    def init_index(self):
        pass

    def search(self, img):
        query_v = self.vgg_model_.extract_ft(img)
        scores = np.dot(query_v, self.pic_fts.T)
        r_idx = np.argsort(scores)[::-1]
        max_ret = 3
        img_list = [self.pic_names[index] for i, index in enumerate(r_idx[0:max_ret])]
        return img_list



