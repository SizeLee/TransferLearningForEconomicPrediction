import tensorflow as tf
import numpy as np
import time
import json
import datapreprocess
from VGG import vgg16

class EconomicRegression:
    def __init__(self, im_size_channel, label_dim, imgs_mean, vggstructurefile, vggweighfile, regressionweightfile=None):
        with open(vggstructurefile, 'r') as f:
            net_paras = json.load(f)
        self.vgg = vgg16(im_size_channel, label_dim, imgs_mean, net_paras, weights=vggweighfile)
        self.featureSize = self.vgg.getfeatureSize()
        # print(self.featureSize)
        self.regressionG = tf.Graph()
        self.sess = tf.Session(graph=self.regressionG)
        self.parameters = []
        self.layer_reduce_ratio = 1
        self.fc_layer()
        self.regression_layer()
        self.train_node()
        with self.regressionG.as_default():
            self.init_variables = tf.global_variables_initializer()

        if regressionweightfile is not None:
            self.load_weights(regressionweightfile, self.sess)


    def fc_layer(self):
        self.layer_reduce_ratio = 4
        with self.regressionG.as_default():
            with tf.name_scope('input'):
                self.input_feature = tf.placeholder(tf.float32, [None, self.featureSize], 'feature')

            with tf.name_scope('feature_map'):
                self.fcW = tf.Variable(tf.truncated_normal([self.featureSize, self.featureSize//self.layer_reduce_ratio],
                                                           dtype=tf.float32,
                                                           stddev=1e-1), name='weights')
                self.fcb = tf.Variable(tf.constant(1, shape=[self.featureSize//self.layer_reduce_ratio], dtype=tf.float32),
                                       trainable=True, name='biases')

                self.fcl = tf.nn.bias_add(tf.matmul(self.input_feature, self.fcW), self.fcb)
                
            ### here to decide linear or non-linear
                self.fc = tf.nn.relu(self.fcl)
                # self.fc = self.fcl
            
            self.parameters += [self.fcW, self.fcb]
        return

    def regression_layer(self):
        with self.regressionG.as_default():
            with tf.name_scope('regression'):
                self.W = tf.Variable(tf.truncated_normal([self.featureSize//self.layer_reduce_ratio, 1],
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
                self.r2 = 1 - self.regression_loss/(tf.reduce_mean(tf.square(tf.reduce_mean(self.y) - self.y)))
                self.loss = self.regression_loss + self.reg_lambda * para_loss
                optimizer = tf.train.AdamOptimizer()
                self.train_step = optimizer.minimize(self.loss)

            with tf.name_scope('summary'):
                self.regression_loss_summary = tf.summary.scalar('regression_loss', self.regression_loss)
                self.loss_summary = tf.summary.scalar('regularized_loss', self.loss)
                self.r2_summary = tf.summary.scalar('r2', self.r2)
                self.whole_summary = tf.summary.merge([self.regression_loss_summary, self.loss_summary, self.r2_summary])

        return

    def feature_reduce(self, features_of_a_sample, method):
        if method == 'Mean':
            return np.mean(features_of_a_sample, axis=0)


    def feature_map(self, images_of_a_sample):
        features_of_a_sample = self.vgg.getfeature(images_of_a_sample)
        return self.feature_reduce(features_of_a_sample, 'Mean')


    def train(self, images_of_samples, y_of_samples, val_imgs, val_ys, epoch_num, reg_lambda, log_dir, y_name, save_weights_filename):
        feature_of_samples = self.feature_map(images_of_samples[0])
        for each in images_of_samples[1:]:
            feature_of_samples = np.vstack((feature_of_samples, self.feature_map(each)))

        val_feature = self.feature_map(val_imgs[0])
        for each in val_imgs[1:]:
            val_feature = np.vstack((val_feature, self.feature_map(each)))

        self.sess.run(self.init_variables)
        train_writer = tf.summary.FileWriter(log_dir + '/regression/%s/train'%y_name, self.sess.graph)
        val_writer = tf.summary.FileWriter(log_dir + '/regression/%s/val'%y_name)
        for i in range(epoch_num):
            if i % 10 != 9:
                self.sess.run(self.train_step, feed_dict={self.input_feature: feature_of_samples,
                                                          self.y: y_of_samples,
                                                          self.reg_lambda: reg_lambda})
            else:
                _, loss, summary = self.sess.run([self.train_step, self.loss, self.whole_summary],
                                                 feed_dict={self.input_feature: feature_of_samples,
                                                            self.y: y_of_samples,
                                                            self.reg_lambda: reg_lambda})
                val_summary = self.sess.run(self.whole_summary, feed_dict={self.input_feature: val_feature,
                                                                             self.y: val_ys,
                                                                             self.reg_lambda: reg_lambda})
                train_writer.add_summary(summary, i)
                val_writer.add_summary(val_summary, i)
                # print(i)
                # print(time.strftime('%Y-%m-%d %H:%M:%S'))
                # print(loss)
                # print()

        train_writer.close()
        self.save_weights(save_weights_filename)
        return
        # or input data format: list of [images, im_height, im_weight, channels] which represent a sample
        # input data [sample, images, im_height, im_weight, channels] ,
        # calculate by vgg get[sample, images, features] data
        # and take mean by images,
        # then get the [sample, features] data

    def test_loss(self, images_of_samples, y_of_samples):
        feature_of_samples = self.feature_map(images_of_samples[0])
        for each in images_of_samples[1:]:
            feature_of_samples = np.vstack((feature_of_samples, self.feature_map(each)))

        y_predict, loss, r2 = self.sess.run([self.out, self.regression_loss, self.r2], feed_dict={self.input_feature: feature_of_samples, 
                                                                                     self.y: y_of_samples})
        return y_predict, loss, r2

    def predict(self, images_of_samples):
        feature_of_samples = self.feature_map(images_of_samples[0])
        for each in images_of_samples[1:]:
            feature_of_samples = np.vstack((feature_of_samples, self.feature_map(each)))

        y_predict = self.sess.run(self.out, feed_dict={self.input_feature: feature_of_samples})
        return y_predict

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
    rdc = datapreprocess.RegressionDataContainer(0.7)
    img_mean = np.load('data/img_mean.npz')['arr_0']
    # print(img_mean)
    gdpregression = EconomicRegression(rdc.getimgsize(), 3, img_mean, 'net_structure.json', 'tweights.npz')
    gdpregression.train(rdc.train_data, rdc.train_y[:, 0:1], rdc.test_data, rdc.test_y[:, 0:1], 10000, 0.8, 'tflog', 'gdp', 'gdpregression_weights.npz')
    # y_predict, loss, r2 = gdpregression.test_loss(rdc.test_data, rdc.test_y[:, 0:1])
    # print('test_y_label:\n', rdc.test_y[:, 0:1])
    # print('y_predict:\n', y_predict)
    # print('gdp test_loss:', loss)
    # print('gdp test_r2:', r2)
    faregression = EconomicRegression(rdc.getimgsize(), 3, img_mean, 'net_structure.json', 'tweights.npz')
    faregression.train(rdc.train_data, rdc.train_y[:, 1:2], rdc.test_data, rdc.test_y[:, 1:2], 10000, 0.8, 'tflog', 'fa', 'faregression_weights.npz')
    crvregression = EconomicRegression(rdc.getimgsize(), 3, img_mean, 'net_structure.json', 'tweights.npz')
    crvregression.train(rdc.train_data, rdc.train_y[:, 2:], rdc.test_data, rdc.test_y[:, 2:], 10000, 0.8, 'tflog', 'crv', 'crvregression_weights.npz')
    print('training done!')

