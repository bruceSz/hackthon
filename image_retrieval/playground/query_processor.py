from vgg_ft_extractor import VGGNET
import numpy as np
import h5py

import matplotlib.pyplot as plt

import matplotlib.image as pimg
import argparse

def query_parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("-query", required=True,
                    help="Path to query which contains image to be queried")
    ap.add_argument("-index", required=True,
                    help="Path to index")
    ap.add_argument("-result", required=True,
                    help="Path for output retrieved images")

    ret = ap.parse_args()
    return ret



def main():
    args = query_parse_arg()
    h5f = h5py.File(args.index,'r')
    fts = h5f['ft_data'][:]
    names = h5f['name_data'][:]
    h5f.close()

    print("*"*10)
    print("** start search **")
    print("*"*10)

    q_dir = args.query
    q_img = pimg.imread(q_dir)
    plt.title("Q image")
    plt.imshow(q_img)
    plt.show()

    model = VGGNET()
    query_v = model.extract_ft(q_dir)
    # TODO, figure out the format of fts.
    scores = np.dot(query_v,fts.T)
    r_idx = np.argsort(scores)[::-1]
    r_score = scores[r_idx]

    max_ret = 3
    img_list = [names[index] for i,index in enumerate(r_idx[0:max_ret])]
    print("top {} images in order are: {}".format(max_ret,img_list))

    for i,im in enumerate(img_list):
        im = pimg.imread(args.result+"/"+im)
        plt.title("search output %d"%(i+1))
        plt.imshow(im)
    #plt.imshow()


if __name__ == "__main__":
    main()
