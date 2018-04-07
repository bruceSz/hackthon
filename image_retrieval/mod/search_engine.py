
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
        #self.index_reader_ = indexer.FileIndexReader()#indexer.IndexReader()
        self.init_index()

        self.vgg_model_ = ft_generator.VGGFeatureGenerator()

    def init_index(self):
        # here index is the name of the picture and
        #index_name = self.conf_['index_file']
        #self.index_ = self.index_reader_.get_index()
        #self.fts_ = self.index_['ft_data'][:]
        #self.names_ = self.index_['name_data'][:]
        pass

    def search(self, img):
        query_v = self.vgg_model_.extract_ft(img)
        scores = np.dot(query_v, self.pic_fts.T)
        r_idx = np.argsort(scores)[::-1]
        #r_score = scores[r_idx]

        max_ret = 3
        img_list = [self.pic_names[index] for i, index in enumerate(r_idx[0:max_ret])]
        return img_list



