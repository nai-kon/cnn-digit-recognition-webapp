import os
import sys
import time
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageDraw
from tensorflow.examples.tutorials.mnist import input_data


# CNN
class ConvNeuralNet:

    # initialize computational graph
    def __init__(self):

        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # No.1 conv layer
        W_conv1 = self.__weight_variable([5, 5, 1, 32])
        b_conv1 = self.__bias_variable([32])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.__conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.__max_pool_2x2(h_conv1)  # 28x28 -> 14x14

        # No.2 conv layer
        W_conv2 = self.__weight_variable([5, 5, 32, 64])
        b_conv2 = self.__bias_variable([64])
        h_conv2 = tf.nn.relu(self.__conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.__max_pool_2x2(h_conv2)  # 14x14 -> 7x7

        # fully connected layer
        W_fc1 = self.__weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.__bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # drop out
        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # softmax output layer
        W_fc2 = self.__weight_variable([1024, 10])
        b_fc2 = self.__bias_variable([10])
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # loss function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def __del__(self):
        self.sess.close()

    # check network is already trained
    def isTrained(self):
        return tf.train.checkpoint_exists("./model/WeightFile.ckpt")

    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # initialize bias
    def __bias_variable(self, shape):
        initial = tf.constant(0, 1, shape=shape)
        return tf.Variable(initial)

    # conv
    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # max pooling(2x2)
    def __max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # centering input digit
    def __centering_img(self, img):

        width, height = img.size
        left, top, right, bottom = width, height, -1, -1
        imgpix = img.getdata()

        for y in range(height):
            yoffset = y * width
            for x in range(width):
                if imgpix[yoffset + x] < 255:

                    # do not use GetPixel and SetPixel, it is so slow.
                    if x < left:
                        left = x
                    if y < top:
                        top = y
                    if x > right:
                        right = x
                    if y > bottom:
                        bottom = y

        shiftX = int((left + (right - left) / 2) - width / 2)
        shiftY = int((top + (bottom - top) / 2) - height / 2)

        return ImageChops.offset(img, -shiftX, -shiftY)

    # training model
    def train(self):

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.sess.run(tf.global_variables_initializer())

        for i in range(20000):
            batch = mnist.train.next_batch(50)

            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: 1.0})
                print("mini batch:{0}/20000 accur:{1:1.2f}".format(i, train_accuracy))

            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

        self.saver.save(self.sess, "./model/WeightFile.ckpt")

    # load model
    def loadmodel(self):
        self.saver = tf.train.Saver()
        if self.isTrained():
            self.saver.restore(self.sess, "./model/WeightFile.ckpt")
            print("model loaded")
        else:
            print("no model for loading")

    # predict digit
    def predict(self, imgpath):
        try:
            img = Image.open(imgpath).convert('L')

        except IOError:
            print("image not found")
            return None

        # centering input digit
        img = self.__centering_img(img)

        img.thumbnail((28, 28))
        img = np.array(img, dtype=np.float32)
        img = 1 - np.array(img / 255)
        img = img.reshape(1, 784)

        # predict
        res = self.sess.run(self.y_conv, feed_dict={self.x: img, self.y_: [[0.0] * 10], self.keep_prob: 1.0})[0]
        return res
