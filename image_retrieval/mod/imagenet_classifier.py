import re
import numpy as np
import tensorflow as tf

class NodeLookup(object):
    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = 'data/models/imagenet_2012_challenge_label_map_proto.pbtxt'
        if not uid_lookup_path:
            uid_lookup_path = "data/models/imagenet_synset_to_human_label_map.txt"
        self.node_lookup = self.load(label_lookup_path,uid_lookup_path)

    def load(self, label_lookup_path,uid_lookup_path):
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal("File does not exist. %s",uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal("File does not exist. %s", label_lookup_path)
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*','')
        for l in proto_as_ascii_lines:
            parsed_items = p.findall(l)
            uid = parsed_items[0]
            human_str  = parsed_items[2]
            uid_to_human[uid] = human_str

        node_id_2uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for l in proto_as_ascii:
            if l.starswith(" target_class:"):
                target_class = int(l.split(": ")[1])
            if l.startswith(" target_class_string:"):
                target_class_string = l.split(": ")[1]
                node_id_2uid[target_class] = target_class_string[1:-2]

        for k,v in node_id_2uid.items():
            if v not in uid_to_human:
                tf.logging.fatal("Failed to locate: %s",v)
            name = uid_to_human[v]
            node_id_2uid[k] = name

    def id_2_string(self,node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]



class Classifier(object):
    def __init__(self):
        with tf.gfile.FastGFile('models/classify_image_graph_def.pb','rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def,name='')

        node_lookup = NodeLookup()
        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')

    def classify(self,img_data):
        pred = self.sess.run(self.softmax_tensor,
                             {'DecodeJpeg/contents:0':img_data})
        pred = np.squeeze(pred)
