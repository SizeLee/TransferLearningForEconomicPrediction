import tensorflow as tf
import numpy as np
from VGG import vgg16

class EconomicRegression:
    def __init__(self, im_size_channel, label_dim, imgs_mean, vggweighfile, regressionweightfile=None):
        self.vgg = vgg16(im_size_channel, label_dim, imgs_mean, weights=vggweighfile)
        self.featureSize = self.vgg.getfeatureSize()
        self.regressionG = tf.Graph()
        self.sess = tf.Session(graph=self.regressionG)
        self.parameters = []
        self.fc_layer()
        self.regression_layer()
        self.train_node()
        if regressionweightfile is not None:
            self.load_weights(regressionweightfile, self.sess)


    def fc_layer(self):
        with self.regressionG.as_default():
            with tf.name_scope('input'):
                self.input_feature = tf.placeholder(tf.float32, [None, self.featureSize], 'feature')

            with tf.name_scope('feature map'):
                self.fcW = tf.Variable(tf.truncated_normal([self.featureSize, self.featureSize],
                                                           dtype=tf.float32,
                                                           stddev=1e-1), name='weights')
                self.fcb = tf.Variable(tf.constant(1, shape=[self.featureSize], dtype=tf.float32),
                                       trainable=True, name='biases')

                self.fcl = tf.nn.bias_add(tf.matmul(self.input_feature, self.fcW), self.fcb)
                self.fc = tf.nn.relu(self.fcl)
            self.parameters += [self.fcW, self.fcb]
        return

    def regression_layer(self):
        with self.regressionG.as_default():
            with tf.name_scope('regression'):
                self.W = tf.Variable(tf.truncated_normal([self.featureSize, 1],
                                                    dtype=tf.float32,
                                                    stddev=1e-1), name='weights')
                self.b = tf.Variable(tf.constant(1, shape=[1], dtype=tf.float32), trainable=True, name='bias')
                self.out = tf.nn.bias_add(tf.matmul(self.fc, self.W), self.b)
            self.parameters += [self.W, self.b]
        return

    def train_node(self):
        with self.regressionG.as_default():
            with tf.name_scope('train'):
                self.y = tf.placeholder(tf.float32, [None, 1], name='y')
                self.reg_lambda = tf.placeholder(tf.float32, name='reg_lambda')
                para_loss = tf.nn.l2_loss(self.fcW) + tf.nn.l2_loss(self.fcb) + tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
                self.regression_loss = tf.reduce_mean(tf.square(self.out - self.y))
                self.loss = self.regression_loss + self.reg_lambda * para_loss
                optimizer = tf.train.AdamOptimizer()
                self.train_step = optimizer.minimize(self.loss)

            with tf.name_scope('summary'):
                self.regression_loss_summary = tf.summary.scalar('regression_loss', self.regression_loss)
                self.loss_summary = tf.summary.scalar('regularized_loss', self.loss)
                self.whole_summary = tf.summary.merge([self.regression_loss_summary, self.loss_summary])

        return

    def train(self):
        
        return
        # input data [sample, images, im_height, im_weight, channels] ,
        # calculate by vgg get[sample, images, features] data
        # and take mean by images,
        # then get the [sample, features] data

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys(), key=lambda x: int(x.split('_')[1]))
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def save_weights(self, savefilename):
        ###save model
        weights = []
        for each in self.parameters:
            weight = self.sess.run(each)
            weights.append(weight)
        np.savez(savefilename, *weights)

if __name__ == '__main__':
    myregression = EconomicRegression([400, 400, 3], 6, [98, 98, 98], 'tweights.npz')
