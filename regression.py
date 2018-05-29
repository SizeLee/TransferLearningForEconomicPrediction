import tensorflow as tf
import numpy as np
import time
import datapreprocess
from VGG import vgg16

class EconomicRegression:
    def __init__(self, im_size_channel, label_dim, imgs_mean, vggweighfile, regressionweightfile=None):
        self.vgg = vgg16(im_size_channel, label_dim, imgs_mean, weights=vggweighfile)
        self.featureSize = self.vgg.getfeatureSize()
        # print(self.featureSize)
        self.regressionG = tf.Graph()
        self.sess = tf.Session(graph=self.regressionG)
        self.parameters = []
        self.fc_layer()
        self.regression_layer()
        self.train_node()
        with self.regressionG.as_default():
            self.init_variables = tf.global_variables_initializer()

        if regressionweightfile is not None:
            self.load_weights(regressionweightfile, self.sess)


    def fc_layer(self):
        with self.regressionG.as_default():
            with tf.name_scope('input'):
                self.input_feature = tf.placeholder(tf.float32, [None, self.featureSize], 'feature')

            with tf.name_scope('feature_map'):
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

    def feature_reduce(self, features_of_a_sample, method):
        if method == 'Mean':
            return np.mean(features_of_a_sample, axis=0)


    def feature_map(self, images_of_a_sample):
        features_of_a_sample = self.vgg.getfeature(images_of_a_sample)
        return self.feature_reduce(features_of_a_sample, 'Mean')


    def train(self, images_of_samples, y_of_samples, epoch_num, reg_lambda, log_dir, save_weights_filename):
        feature_of_samples = self.feature_map(images_of_samples[0])
        for each in images_of_samples[1:]:
            feature_of_samples = np.vstack((feature_of_samples, self.feature_map(each)))

        self.sess.run(self.init_variables)
        train_writer = tf.summary.FileWriter(log_dir + '/regression_train', self.sess.graph)
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
                train_writer.add_summary(summary, i)
                print(i)
                print(time.strftime('%Y-%m-%d %H:%M:%S'))
                print(loss)
                print()

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

        y_predict, loss = self.sess.run([self.out, self.regression_loss], feed_dict={self.input_feature: feature_of_samples, 
                                                                                     self.y: y_of_samples})
        return y_predict, loss

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
    myregression = EconomicRegression(rdc.getimgsize(), 3, img_mean, 'tweights.npz')
    myregression.train(rdc.train_data, rdc.train_y, 1000, 1e-8, 'tflog', 'regression_weights.npz')
    _, loss = myregression.test_loss(rdc.test_data, rdc.test_y)
    print('test_loss:', loss)
