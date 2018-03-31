# -*- coding: utf-8 -*-
# Author: bruceSz

import os
import h5py
import  numpy as np
import argparse

from vgg_ft_extractor import VGGNET

def index_parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("-database",required=True,
                    help = "Path to database which contains images to be indexed")
    ap.add_argument("-index",required=True,
                    help="Name of the index file")

    args = vars(ap.parse_args())
    return ap.parse_args()


def get_img_list(path):
    """
    Return a list of files for all jpg images in dir.
    """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]


def main():
    """
    Extract fts and index the images
    :return:
    """
    args = index_parse_arg()
    print(args)
    db = args.database
    img_list = get_img_list(db)
    print("*"*10)
    print("** start ft extraction **")
    print("*"*10)

    fts = []
    names = []
    model = VGGNET()
    for i,img_p in enumerate(img_list):
        norm_ft = model.extract_ft(img_p)
        img_name = os.path.split(img_p)[1]
        fts.append(norm_ft)
        names.append(img_name)

    fts = np.array(fts)
    output = args.index
    print("*" * 10)
    print("** write index file  **")
    print("*" * 10)
    h5f = h5py.File(output,'w')
    h5f.create_dataset("ft_data",data=fts)
    h5f.create_dataset("name_data",data=names)
    h5f.close()


if __name__ == "__main__":
    main()
