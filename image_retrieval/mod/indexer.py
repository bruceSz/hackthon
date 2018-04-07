import h5py
import abc

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

