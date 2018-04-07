

import tensorflow as tf
import numpy as np

def maxPoolLayer(x, kH, kW, strideX,strideY,name, padding = 'SAME'):
    """ """
    # TODO. try to understand `padding`
    return tf.nn.max_pool(x,ksize = [1,kH,kW,1],
                          strides=[1,strideX,strideY,1],
                          padding=padding,
                          name = name)


def dropout(x, kp, name = None):
    """ dropput"""
    return tf.nn.dropout(x,kp,name)


def fcLayer(x, inputD, outputD, reluFlag, name):
    """ fully-connected"""
    with tf.varibale_scope(name) as scope:
        w = tf.get_variable('w',shape=[inputD,outputD], dtype='float')
        b = tf.get_variable('b',[outputD],dtype='float')
        out = tf.nn.xw_plus_b(x,w,b,name=scope.name)

        if reluFlag:
            return  tf.nn.relu(out)
        else:
            return out

def convLayer(x, kH,kW, strideX, strideY,
              ftNum, name, padding="Same"):
    """ conv """
    chann = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        # w is the filter / conv kernel
        # where the ftNum is the number of feature map
        # where it is the out channel.
        w = tf.get_variable('w',shape=[kH,kW,chann,ftNum])
        b = tf.get_variable('b',shape=[ftNum])

        ftMap = tf.nn.conv2d(x,w,strides=[1,strideY,strideX,1],
                             padding=padding)
        out = tf.nn.bias_add(ftMap,b)
        return tf.nn.relu(tf.reshape(out,ftMap.get_shape().as_list()),
                          name = scope.name)


class VGG19(object):
    """ VGG model"""
    def __init__(self, x, kp, classNum, skip, model = "vgg19.npy"):
        self.X = x
        self.kp = kp
        self.classNum = classNum
        self.skip = skip
        self.model = model
        # build cnn
        self.buildCNN()

    def buildCNN(self):
        """ build model"""
        conv1_1 = convLayer(self.X,3,3,1,1,64,'conv1_1')
        conv1_2 = convLayer(conv1_1,3,3,1,1,64,'conv1_2')
        pool1 = maxPoolLayer(conv1_2,2,2,2,2,"pool1")

        conv2_1 = convLayer(pool1,3,3,1,1,128,'conv2_1')
        conv2_2 = convLayer(conv2_1,3,3,1,1,128,'conv2_2')
        pool2 = maxPoolLayer(conv2_2,2,2,2,2,'pool2')

        # each below conv layers has four/three conv connected to each other.
        conv3_1 = convLayer(pool2,3,3,1,1,256,'conv3_1')
        conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, 'conv3_2')
        conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, 'conv3_3')
        conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, 'conv3_4')
        pool3 = maxPoolLayer(conv3_4,2,2,2,2,'pool3')

        conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, 'conv4_1')
        conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, 'conv4_2')
        conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, 'conv4_3')
        conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, 'conv4_4')
        pool4 = maxPoolLayer(conv4_4,2,2,2,2,'pool4')

        conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, 'conv5_1')
        conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, 'conv5_2')
        conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, 'conv5_3')
        conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, 'conv5_4')
        pool5 = maxPoolLayer(conv5_4,2,2,2,2,"pool5")
        # flatten the pooling output
        fcIn  = tf.reshape(pool5,[-1,7*7*512])
        fc6 = fcLayer(fcIn,7*7*512, 4096, True, "fc6")
        drop1 = dropout(fc6,self.kp)

        fc7 = fcLayer(drop1,4096,4096,True,'fc7')
        drop2 = dropout(fc7,self.kp)

        self.fc8 = fcLayer(drop2,4096,self.classNum,True,'fc8')

    def loadModel(self, sess):
        """ load model"""
        d = np.load(self.model, encoding='bytes').item()
        for name in d:
            if name not in self.skip:
                with tf.variable_scope(name,reuse=True):
                    for p in d[name]:
                        if len(p.shape) == 1:
                            # bias
                            sess.run(tf.get_variable('b',trainable=False).assign(p))
                        else:
                            # weights

                            sess.run(tf.get_variable('w',trainable=False).assign(p))



