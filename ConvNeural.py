import os,sys
import time
import tensorflow as tf
import numpy as np

from PIL import Image, ImageChops, ImageDraw
from tensorflow.examples.tutorials.mnist import input_data

# CNNクラス
class ConvNeuralNet:

    # 計算グラフの設定
    def __init__(self):

        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape = [None, 784])
        self.y_ = tf.placeholder(tf.float32, shape = [None, 10])        

        # 第1層
        W_conv1 = self.__weight_variable([5, 5, 1, 32])    # 5x5フィルタで32特徴を出力 (入力ch=1 出力ch=32)
        b_conv1 = self.__bias_variable([32])
        # 入力画像を四次元テンソルに変換
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        # 第1層の畳み込み
        h_conv1 = tf.nn.relu(self.__conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.__max_pool_2x2(h_conv1) # 28x28 -> 14x14

        # 第2層
        W_conv2 = self.__weight_variable([5, 5, 32, 64])    # 5x5フィルタで64特徴を出力 (入力ch=32 出力ch=64)
        b_conv2 = self.__bias_variable([64])
        # 第2層の畳み込み
        h_conv2 = tf.nn.relu(self.__conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.__max_pool_2x2(h_conv2) # 14x14 -> 7x7

        # 全結合
        W_fc1 = self.__weight_variable([7*7*64, 1024])
        b_fc1 = self.__bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # ドロップアウト
        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 出力層
        W_fc2 = self.__weight_variable([1024, 10])
        b_fc2 = self.__bias_variable([10])
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # ソフトマックス関数で確率計算

        # 訓練関数
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def __del__(self):
        self.sess.close()

    # 訓練済みか(ネットワークが保存済みか)
    def isTrained(self):
        return tf.train.checkpoint_exists("./model/WeightFile.ckpt")

    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    # バイアスの初期化(一様値)
    def __bias_variable(self, shape):
        initial = tf.constant(0,1, shape = shape)
        return tf.Variable(initial)

    # 畳み込み
    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

    # マックスプーリング(2x2)
    def __max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')    

    # 文字位置のセンタリング
    def __centering_img(self, img):

        # 文字の位置を取得
        width, height = img.size
        left,top,right,bottom = width, height, -1, -1
        imgpix = img.getdata()

        for y in range(height):
            yoffset = y*width
            for x in range(width):
                if imgpix[yoffset + x] < 255:
                #if img.getpixel((x,y)) < 255: # 遅いので使わない
                    if x < left: left   = x
                    if y < top: top     = y
                    if x > right: right = x
                    if y > bottom: bottom = y   
                        
        # (経過表示用)文字を囲む枠を描画
        #detect_img = img.copy();
        #draw = ImageDraw.Draw(detect_img)
        #draw.rectangle((left-3, top-3, right+3, bottom+3), outline=(0))

        # センタリングのシフト量を計算
        shiftX = int((left + (right - left) / 2) - width / 2)
        shiftY = int((top + (bottom - top) / 2) - height / 2)

        # センタリング(シフト量はマイナスとなる)
        return ImageChops.offset(img, -shiftX, -shiftY)#, detect_img

    # 訓練処理
    def train(self):
        if self.isTrained():
            if input("Network is already trained. Retrain it? (y/n) ") != "y":
                print("exit")
                return

        mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
        self.sess.run(tf.global_variables_initializer())

        for i in range(20000):
            batch = mnist.train.next_batch(50)

            if i % 100 == 0:
               train_accuracy = self.accuracy.eval(feed_dict = {self.x : batch[0], self.y_:batch[1], self.keep_prob:1.0})
               print("mini batch:{0}/20000 accur:{1:1.2f}".format(i, train_accuracy))

            self.train_step.run(feed_dict = {self.x:batch[0], self.y_:batch[1], self.keep_prob:0.5})
        
        self.saver.save(self.sess,"./model/WeightFile.ckpt")

    # モデルの読み込み
    def loadmodel(self):
        self.saver = tf.train.Saver()
        if self.isTrained():
            self.saver.restore(self.sess, "./model/WeightFile.ckpt")
            print("load model")
        else:
            print("no model for load")
        

    # 予測処理
    def predict(self, imgpath):
        try:
            img = Image.open(imgpath).convert('L')
        
        except IOError:
            print("file not found")
            return None
        
        # 文字のセンタリング
        start = time.time()
        img = self.__centering_img(img)
        print("time:{}".format(time.time()-start))
        #img = center_img.copy()

        img.thumbnail((28, 28))
        img = np.array(img, dtype=np.float32)
        img = 1 - np.array(img / 255)
        img = img.reshape(1, 784)

        res = self.sess.run(self.y_conv, feed_dict = {self.x : img, self.y_:[[0.0] * 10], self.keep_prob:1.0})[0]
        return res#, center_img, detect_img
