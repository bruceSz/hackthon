
from keras.preprocessing.image import ImageDataGenerator

def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


@Singleton
class A(object):
    def __init__(self):
        print("Init A.")


def main():
    print("Begin init test")

    a = A()
    print("After init the first one")
    b = A()
    print("Init finished.")


def image_preprocess(img_path,output_dir,out_size):
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing.image import array_to_img
    from keras.preprocessing.image import  img_to_array
    from keras.preprocessing.image import load_img
    #img_name = img_path.split("jpg")[0]
    #print img_name
    import os
    img_name= os.path.basename(img_path).split(".")[0]
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    img_ = load_img(img_path)
    x = img_to_array(img_)
    x = x.reshape((1,) + x.shape)

    i = 0
    for b in datagen.flow(x,batch_size=1,save_to_dir=output_dir,
                          save_prefix=img_name,save_format='jpeg'):
        i+=1
        if i>out_size:
            break



if __name__ == "__main__":
    import sys
    import os

    #["../static/web/doufu","../static/web/jiliu","../static/web/liangbandoupi",
    dirs = [ "../static/web/jiliu", "../static/web/liangbandoupi",
            "../static/web/mao","../static/web/qincaihuashengmi","../static/web/ribendoufu",
            "../static/web/roucaipinfan","../static/web/shoufengqin","../static/web/zhuti"]
    for dir in dirs:
        for d_ in os.listdir(dir):
            if d_ == "tofu1.jpg":
                continue
            path = os.path.join(dir,d_)
            trans_path = os.path.join(dir,"trans")
            if not os.path.isdir(trans_path):
                os.mkdir(trans_path)
            if os.path.isfile(path):
                print(path)
                print(trans_path)
                image_preprocess(path,trans_path,100)
    #image_preprocess("../static/web/doufu/tofu1.jpg",
    #                 "../static/web/doufu/trans",80)