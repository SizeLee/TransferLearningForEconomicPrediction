########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import datapreprocess
import time
import json

class vgg16:
    def __init__(self, im_size_channel, labeldim, imgs_mean, net_structure, weights=None):
        self.sess = None
        self.im_size_channel = im_size_channel
        self.labeldim = labeldim
        self.net_structure = net_structure
        self.parameters = []
        self.graph = tf.Graph()
        # batchsize = 2

        ###here set dataset node structure
        # self.features_placeholder = tf.placeholder(tf.float32, [None, im_size_channel[0], im_size_channel[1], im_size_channel[2]])
        # self.labels_placeholder = tf.placeholder(tf.float32, [None, self.labeldim])
        # self.data = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        # self.batched_data = self.data.batch(batchsize)
        # self.iterator = self.batched_data.make_initializable_iterator()
        # self.next_element = self.iterator.get_next()

        ###here set vgg net structure
        with self.graph.as_default():
            self.imgs = tf.placeholder(tf.float32, [None, im_size_channel[0], im_size_channel[1], im_size_channel[2]])
            self.convlayers(imgs_mean)
            self.fc_layers()
            self.probs = tf.nn.softmax(self.fc3l)
            self.probs_distribution = tf.summary.histogram('probs_distribution', self.probs)

        ###here set vgg training node
            self.init_train_node()
            self.summary_node()
            self.init_variables = tf.global_variables_initializer()

        if weights is not None:
            self.sess = tf.Session(graph=self.graph)
            self.load_weights(weights, self.sess)

    def summary_node(self):
        with tf.name_scope('summary_node'):
            self.whole_loss_node = tf.placeholder(tf.float32)
            self.whole_loss_summary = tf.summary.scalar('whole_loss', self.whole_loss_node)
            self.whole_accuracy_node = tf.placeholder(tf.float32)
            self.whole_accuracy_summary = tf.summary.scalar('whole_accuracy', self.whole_accuracy_node)
            self.validation_accuracy_node = tf.placeholder(tf.float32)
            self.validation_accuracy_summary = tf.summary.scalar('val_accuracy', self.validation_accuracy_node)
            self.whole_summary = tf.summary.merge([self.whole_loss_summary, self.whole_accuracy_summary, self.validation_accuracy_summary])

    def init_train_node(self):
        with tf.name_scope('train'):
            self.labels = tf.placeholder(tf.float32, name='labels', shape=[None, self.labeldim])
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3l, labels=self.labels))
            self.batch_loss_summary = tf.summary.scalar('batch_loss', self.loss)
            optimizer = tf.train.AdamOptimizer()
            self.train_step = optimizer.minimize(self.loss)
            correct_prediction = tf.equal(tf.argmax(self.probs, axis=1), tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.batch_accuracy_summary = tf.summary.scalar('batch_accuracy', self.accuracy)
            self.batch_summary = tf.summary.merge([self.batch_loss_summary, self.batch_accuracy_summary])

    def convlayers(self, imgs_mean):
        # zero-mean input  ###todo mean value should be satellite data set or can be transferred, need test
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant(imgs_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean') #imgnet[123.68, 116.779, 103.939]
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            stru_p = self.net_structure['conv1_1']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            stru_p = self.net_structure['conv1_2']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        stru_p = self.net_structure['pool1']
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=stru_p['ksize'],
                               strides=stru_p['strides'],
                               padding=stru_p['padding'],
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            stru_p = self.net_structure['conv2_1']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            stru_p = self.net_structure['conv2_2']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        stru_p = self.net_structure['pool2']
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=stru_p['ksize'],
                               strides=stru_p['strides'],
                               padding=stru_p['padding'],
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            stru_p = self.net_structure['conv3_1']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            stru_p = self.net_structure['conv3_2']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            stru_p = self.net_structure['conv3_3']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        stru_p = self.net_structure['pool3']
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=stru_p['ksize'],
                               strides=stru_p['strides'],
                               padding=stru_p['padding'],
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            stru_p = self.net_structure['conv4_1']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            stru_p = self.net_structure['conv4_2']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            stru_p = self.net_structure['conv4_3']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        stru_p = self.net_structure['pool4']
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=stru_p['ksize'],
                               strides=stru_p['strides'],
                               padding=stru_p['padding'],
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            stru_p = self.net_structure['conv5_1']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            stru_p = self.net_structure['conv5_2']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            stru_p = self.net_structure['conv5_3']
            kernel = tf.Variable(tf.truncated_normal(stru_p['kernel_shape'], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, stru_p['strides'], padding=stru_p['padding'])
            biases = tf.Variable(tf.constant(0.0, shape=stru_p['biases_shape'], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        stru_p = self.net_structure['pool5']
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=stru_p['ksize'],
                               strides=stru_p['strides'],
                               padding=stru_p['padding'],
                               name='pool4')

    def fc_layers(self):
        # fc1
        stru_p = self.net_structure['fc']
        midsize = stru_p['midsize'] #4096 todo here to switch to 4096 size when working with better gpu card
        neural_reduce_ratio = stru_p['neural_reduce_ratio']
        with tf.name_scope('dropout_parameter'):
            self.dropout_keepprob = tf.placeholder(tf.float32, name='keepprob')

        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, midsize],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[midsize], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.fc1_drop_out = tf.nn.dropout(self.fc1, self.dropout_keepprob)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([midsize, int(midsize/neural_reduce_ratio)],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[int(midsize/neural_reduce_ratio)], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1_drop_out, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.fc2_drop_out = tf.nn.dropout(self.fc2, self.dropout_keepprob)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([int(midsize/neural_reduce_ratio), self.labeldim],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.labeldim], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2_drop_out, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]


    def training(self, epoch, imgs, labels, batch_size, drop_keepprob, savefilename, log_dir, val_imgs, val_labels):
        if self.sess is not None:
            self.sess.close()

        self.sess = tf.Session(graph=self.graph)

        # Compute for trainround epochs.
        self.sess.run(self.init_variables)
        self.sess.graph.finalize()

        train_writer = tf.summary.FileWriter(log_dir + '/train', self.sess.graph)

        sampleNum = labels.shape[0]
        train_steps = int(sampleNum / batch_size)
        best_val_accuracy = 0.
        for i in range(epoch):   ####i is epoch iterator
            # todo unknown reason, gpu memory leak out
            # self.sess.run(self.iterator.initializer, feed_dict={self.features_placeholder: imgs,
            #                                                     self.labels_placeholder: labels})
            # count = 0
            # while True:
            #     try:
            #         count += 1
            #         print(count)
            #         x, y = self.sess.run(self.next_element)
            #         self.sess.run(self.train_step, feed_dict={self.imgs: x,
            #                                                   self.labels: y,
            #                                                   self.dropout_keepprob: 0.5})
            #     except tf.errors.OutOfRangeError:
            #         break

            for j in range(train_steps):  ###j is batch iterator
                if j%5 != 4:
                    self.sess.run(self.train_step,
                              feed_dict={self.imgs: imgs[j * batch_size: (j + 1) * batch_size, :, :, :],
                                         self.labels: labels[j * batch_size: (j + 1) * batch_size, :],
                                         self.dropout_keepprob: drop_keepprob})

                else:
                    _, batch_summary = self.sess.run([self.train_step, self.batch_summary],
                                      feed_dict={self.imgs: imgs[j * batch_size: (j + 1) * batch_size, :, :, :],
                                                 self.labels: labels[j * batch_size: (j + 1) * batch_size, :],
                                                 self.dropout_keepprob: drop_keepprob})
                    train_writer.add_summary(batch_summary, i * (sampleNum) + (j + 1) * batch_size)

                    # _, batch_loss_summary, batch_accuracy_summary = \
                    #     self.sess.run([self.train_step, self.batch_loss_summary, self.batch_accuracy_summary],
                    #               feed_dict={self.imgs: imgs[j * batch_size: (j + 1) * batch_size, :, :, :],
                    #                          self.labels: labels[j * batch_size: (j + 1) * batch_size, :],
                    #                          self.dropout_keepprob: drop_keepprob})
                    # train_writer.add_summary(batch_loss_summary, i * (sampleNum) + (j + 1) * batch_size)
                    # train_writer.add_summary(batch_accuracy_summary, i * (sampleNum) + (j + 1) * batch_size)


            if train_steps * batch_size != sampleNum:
                self.sess.run(self.train_step, feed_dict={self.imgs: imgs[train_steps * batch_size:, :, :, :],
                                                          self.labels: labels[train_steps * batch_size:, :],
                                                          self.dropout_keepprob: drop_keepprob})

            if i%1 == 0:
                batch_loss = []
                batch_accuracy = []
                for j in range(train_steps):
                    loss, accuracy, probs_distribution = self.sess.run([self.loss, self.accuracy, self.probs_distribution],
                                         feed_dict={self.imgs: imgs[j * batch_size: (j + 1) * batch_size, :, :, :],
                                                    self.labels: labels[j * batch_size: (j + 1) * batch_size, :],
                                                    self.dropout_keepprob: 1})
                    batch_loss.append(loss)
                    batch_accuracy.append(accuracy)
                    train_writer.add_summary(probs_distribution, i*train_steps+j)
                whole_loss = np.array(batch_loss).mean()
                whole_accuracy = np.array(batch_accuracy).mean()

                if train_steps * batch_size != sampleNum:
                    loss, accuracy = self.sess.run([self.loss, self.accuracy],
                                         feed_dict={self.imgs: imgs[train_steps * batch_size:, :, :, :],
                                                    self.labels: labels[train_steps * batch_size:, :],
                                                    self.dropout_keepprob: 1})
                    whole_loss = (whole_loss * train_steps * batch_size +
                                 loss * (sampleNum - train_steps * batch_size))/sampleNum
                    whole_accuracy = (whole_accuracy * train_steps * batch_size +
                                     accuracy * (sampleNum - train_steps * batch_size))/sampleNum

                val_accuracy = self.testaccuracy(val_imgs, val_labels, 8)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_weights(savefilename)

                whole_summary = self.sess.run(self.whole_summary, feed_dict={self.whole_loss_node: whole_loss,
                                                                             self.whole_accuracy_node: whole_accuracy,
                                                                             self.validation_accuracy_node: val_accuracy})
                train_writer.add_summary(whole_summary, i*sampleNum)
                # probs_distribution = self.sess.run(self.probs_distribution, feed_dict={self.imgs:imgs, self.dropout_keepprob:1})
                # train_writer.add_summary(probs_distribution, i*sampleNum)

                print(i)
                print(time.strftime('%Y-%m-%d %H:%M:%S'))
                print(whole_accuracy, whole_loss, val_accuracy)
                print()

        train_writer.close()
        # self.save_weights(savefilename)

        return


    def testaccuracy(self, imgs, labels, batch_size):
        if self.sess is None:
            print('no session contain model')
            return

        sampleNum = labels.shape[0]
        test_steps = int(sampleNum / batch_size)
        batch_accuracy = []
        for j in range(test_steps):
            accuracy = self.sess.run(self.accuracy,
                                     feed_dict={self.imgs: imgs[j * batch_size: (j + 1) * batch_size, :, :, :],
                                                self.labels: labels[j * batch_size: (j + 1) * batch_size, :],
                                                self.dropout_keepprob: 1})
            batch_accuracy.append(accuracy)
        whole_accuracy = np.array(batch_accuracy).mean()

        if test_steps * batch_size != sampleNum:
            accuracy = self.sess.run(self.accuracy,
                                     feed_dict={self.imgs: imgs[test_steps * batch_size:, :, :, :],
                                                self.labels: labels[test_steps * batch_size:, :],
                                                self.dropout_keepprob: 1})
            whole_accuracy = (whole_accuracy * test_steps * batch_size +
                             accuracy * (sampleNum - test_steps * batch_size)) / sampleNum

        return whole_accuracy

    def getfeature(self, imgs):
        feature = self.sess.run(self.fc2, feed_dict={self.imgs: imgs, self.dropout_keepprob: 1})
        return feature

    def getfeatureSize(self):
        return int(self.fc2.get_shape()[1])

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
    # imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with open('net_structure.json', 'r') as f:
        net_paras = json.load(f)
    # print(type(net_paras['conv1_1']['kernel_shape'][0]))
    data = datapreprocess.datacontainer(0.7)
    vgg = vgg16(data.getimgsize(), data.getlabeldim(), data.getTrainMean(), net_paras)  # 'vgg16_weights.npz'

    start = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    vgg.training(100, data.trainimgs, data.trainlabels, 8, 1, 'tweights.npz', 'tflog', data.testimgs, data.testlabels)
    print(vgg.testaccuracy(data.testimgs, data.testlabels, 8))
    end = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print('total time: ', end - start, 's')

    # img1 = imread('laska.png', mode='RGB')
    # img1 = imresize(img1, (224, 224))
    #
    # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print(class_names[p], prob[p])

