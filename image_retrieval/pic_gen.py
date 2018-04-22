
import os
from utils import misc

def main():
    ret = []
    dir_p = "static/web/"
    for d in os.listdir(dir_p):
        p_ = dir_p + "/" + d
        if os.path.isdir(p_):
            for f in os.listdir(p_):
                if f.endswith(".jpg"):
                    ret.append(os.path.join(p_, f))
    for raw in ret:
        p_ = os.path.split(raw)
        trans_p = p_[0] + "/"+"trans"
        root_p = p_[0]
        print (raw)
        misc.image_preprocess(raw,trans_p,10)

if __name__ == "__main__":
    main()