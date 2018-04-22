import h5py
import abc
import os
import pickle
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from mod import ft_generator

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

class ClassPicManager(object):
    _FILE_P = "static/web"
    class_name2path = "data/offline/out/classn2path.pcl"

    @staticmethod
    def Pic2class():
        class_label2path = {}

        class_name_set = set([])

        for d in os.listdir(ClassPicManager._FILE_P):
            p_ = ClassPicManager._FILE_P + "/" + d
            if os.path.isdir(p_):
                pic_p = p_ + "/trans"
                if os.path.isdir(pic_p):
                    class_name = d
                    class_name_set.add(class_name)
                    pics_dir = pic_p
                    for pic_f in os.listdir(pics_dir):
                        pic_f_p = pics_dir + "/" + pic_f
                        # print(pic_f_p)
                        if not class_name in class_label2path:
                            class_label2path[class_name] = []
                        class_label2path[class_name].append(pic_f_p)
                    # gen map and pickle it.

        class_name_l = list(class_name_set)
        class_name2idx = dict(zip(class_name_l, range(len(class_name_l))))
        class_idx2name = dict(zip(range(len(class_name_l)), class_name_l))

        ret = dict()
        ret['name2idx'] = class_name2idx
        ret['idx2name'] = class_idx2name
        ret['class_name2path'] = class_label2path

        with open(ClassPicManager.class_name2path, 'wb') as f:
            pic_str = pickle.dump(ret, f)
            # print(d,pic_p)

    def loadClassPics(self):
        with open(ClassPicManager.class_name2path, 'rb') as f:
            class_map = pickle.load(f)
            return class_map


class SvmIndexer(object):
    _dishes_ft_f = "data/offline/out/picft.pcl"
    _dishes_labelidx_f = "data/offline/out/pic_label_idx.pcl"
    _dishes_svm_f = "data/offline/out/pic_svm.pcl"
    _dish_idx_name_f = "data/offline/out/pic_idx_name.pcl"
    _dishes = ["liangbanxihongshi", "qingzhengyu","rouchaobaocai","rouchaohuanggua","rouchaoxilanhua","tudoujikuai"]

    @staticmethod
    def load_index2name():
        with open(SvmIndexer._dish_idx_name_f, 'rb') as f:
            idx2name = pickle.load(f)
            return idx2name

    @staticmethod
    def load_model():
        with open(SvmIndexer._dishes_svm_f, 'rb') as f:
            model = pickle.load(f)
            return model

    def __init__(self):
        self.pm_ = ClassPicManager()
        self.ft_model_ = ft_generator.VGGFeatureGenerator()

    def prepare_train_data(self):
        paths_map = self.pm_.loadClassPics()
        X = []
        y = []
        idx2path = {}
        path2ft = {}
        path2label = {}
        idx2name = {}
        idx = 0
        for dish_name in SvmIndexer._dishes:
            print("Processing " + dish_name)
            img_paths = paths_map['class_name2path'][dish_name]
            img_idx = paths_map['name2idx'][dish_name]
            idx2name[img_idx] = dish_name
            print("Total number of img: " + str(len(img_paths)))
            idx__ = 0
            for p_ in img_paths:
                n_ft = self.ft_model_.extract_ft(p_)
                path2ft[p_] = n_ft
                path2label[p_] = img_idx
                X.append(n_ft)
                y.append(img_idx)
                idx2path[idx] = p_
                idx += 1
                idx__ += 1
                if idx__ % 20 == 0:
                    print("processed " + str(idx__) + " for " + dish_name)
        with open(SvmIndexer._dishes_ft_f, 'wb') as f:
            pic_str = pickle.dump(X, f)
        with open(SvmIndexer._dishes_labelidx_f, 'wb') as f:
            pic_str = pickle.dump(y, f)
        with open(SvmIndexer._dish_idx_name_f, 'wb') as f:
            pic_str = pickle.dump(idx2name, f)


    def train_model(self):

        with open(SvmIndexer._dishes_ft_f, 'rb') as f:
            X = pickle.load(f)
        with open(SvmIndexer._dishes_labelidx_f, 'rb') as f:
            y = pickle.load(f)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        model = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
        y_train_pred = (model.predict(X_train))
        print(accuracy_score(y_train, y_train_pred))
        y_test_pred = (model.predict(X_test))
        print(accuracy_score(y_test, y_test_pred))


        with open(SvmIndexer._dishes_svm_f, 'wb') as f:
            pic_str = pickle.dump(model, f)

class IndexBuilder(object):
    #_LOCAL_FILE_DIR = "data/offline/in"
    _LOCAL_FILE_DIR = "static/web/"
    _LOCAL_INDEX_PATH = "data/online/in/data.index"

    # TODO.
    # 1 DATA gen from hdfs.
    # 2 index hot-patch.
    # 3 index mem.

    def __init__(self):
        self.model_ = ft_generator.VGGFeatureGenerator()


    def _get_img_list(self, dir_p):
        """
        Return a list of files for all jpg images in dir.
        """
        ret = []
        for d in os.listdir(dir_p):
            p_ = dir_p+"/"+d
            if os.path.isdir(p_):
                for f in os.listdir(p_):
                    if f.endswith(".jpg"):
                       ret.append(os.path.join(p_,f))
        #return [os.path.join(dir_p,f) for f in os.listdir(dir_p) if f.endswith(".jpg")]
        return ret

    def build(self,img_p,class_name):
        norm_ft = self.model_.extract_ft(img_p)
        return class_name,norm_ft

    def batch_build(self):
        img_list = self._get_img_list(IndexBuilder._LOCAL_FILE_DIR)
        print(img_list)
        fts = []
        names = []

        model = ft_generator.VGGFeatureGenerator()
        for i, img_p in enumerate(img_list):
            norm_ft = model.extract_ft(img_p)
            print(norm_ft.shape)
            img_name = os.path.split(img_p)[1]
            fts.append(norm_ft)
            names.append(img_p.encode())

        fts = np.array(fts)
        output = IndexBuilder._LOCAL_INDEX_PATH
        print("index created.")
        h5f = h5py.File(output, 'w')
        h5f.create_dataset("ft_data", data=fts)
        h5f.create_dataset("name_data", data=names)
        h5f.close()