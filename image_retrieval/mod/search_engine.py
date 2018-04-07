
import indexer
from utils import misc

@misc.Singleton
class SearchEngine(object):
    def __init__(self):
        self.conf_ = {}
        self.index_reader_ = indexer.FileIndexReader()#indexer.IndexReader()
        self.init_index()

    def init_index(self):
        # here index is the name of the picture and
        index_name = self.conf_['index_file']
        self.index_ = self.index_reader_.get_index()
        self.fts_ = self.index_['ft_data'][:]
        self.names_ = self.index_['name_data'][:]




