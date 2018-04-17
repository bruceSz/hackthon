import h5py
import abc
import os
import numpy as np

import ft_generator

class IndexReader(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def read(self):
        pass


class FileIndexReader(IndexReader):
    INDEX_FILE = "./data/offline/in/index.da"
    KEY = "ft"
    VAL = "output_f_path"

    def __init__(self):
        h5f = h5py.File(IndexReader.INDEX_FILE,'r')
        fts = h5f['fts'][:]
        names = h5f['names'][:]
        self.index_d = dict(zip(fts,names))

    def read(self):
        return self.index_d


class IndexWriter(object):
    @abc.abstractmethod
    def write(self):
        pass


class IndexBuilder(object):
    #_LOCAL_FILE_DIR = "data/offline/in"
    _LOCAL_FILE_DIR = "static/web/"
    _LOCAL_INDEX_PATH = "data/online/in/data.index"

    # TODO.
    # 1 DATA gen from hdfs.
    # 2 index hot-patch.
    # 3 index mem.

    def get_img_list(self, dir_p):
        """
        Return a list of files for all jpg images in dir.
        """
        #TODO. change fetch jpg dir.
        return [os.path.join(dir_p,f) for f in os.listdir(dir_p) if f.endswith(".jpg")]

    def build(self):
        img_list = self.get_img_list(IndexBuilder._LOCAL_FILE_DIR)

        fts = []
        names = []
        print('ft gen:')
        model = ft_generator.VGGFeatureGenerator()
        for i, img_p in enumerate(img_list):
            norm_ft = model.extract_ft(img_p)
            img_name = os.path.split(img_p)[1]
            fts.append(norm_ft)
            names.append(img_name)

        fts = np.array(fts)
        output = IndexBuilder._LOCAL_INDEX_PATH
        print("index created.")
        h5f = h5py.File(output, 'w')
        h5f.create_dataset("ft_data", data=fts)
        h5f.create_dataset("name_data", data=names)
        h5f.close()