import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#時間計測の開始
start = time.time()

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# 重みの初期化(ランダム値)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# バイアスの初期化(一様値)
def bias_variable(shape):
    initial = tf.constant(0,1, shape = shape)
    return tf.Variable(initial)

# 畳み込み
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# マックスプーリング(2x2)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


# 第1層
W_conv1 = weight_variable([5, 5, 1, 32])    # 5x5フィルタで32特徴を出力 (入力ch=1 出力ch=32)
b_conv1 = bias_variable([32])
# 入力画像を四次元テンソルに変換
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 第1層の畳み込みを行う
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # 28x28 -> 14x14


# 第2層
W_conv2 = weight_variable([5, 5, 32, 64])    # 5x5フィルタで64特徴を出力 (入力ch=32 出力ch=64)
b_conv2 = bias_variable([64])
# 第2層の畳み込みを行う
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # 14x14 -> 7x7

# 全結合
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# ドロップアウト
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 出力層
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # ソフトマックス関数で確率計算

# 訓練処理
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x : batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5})


saver.save(sess, "./WeightFile.ckpt")
print("test accuracy %g" %  accuracy.eval(feed_dict = {x : mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

#時間計測#
sess.close()
time = time.time() - start
print ("実行時間:{0}".format(time))
