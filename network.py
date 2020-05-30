import tensorflow as tf


# CNN
class ConvNeuralNet:

    # initialize computational graph
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])  # 28x28
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])  # classification 0-9
        input = tf.reshape(self.x, [-1, 28, 28, 1])

        # No.1 conv layer
        W_conv1 = self.__weight_variable([5, 5, 1, 32])
        b_conv1 = self.__bias_variable([32])
        h_conv1 = tf.nn.relu(self.__conv2d(input, W_conv1) + b_conv1)
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

    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # initialize bias
    def __bias_variable(self, shape):
        initial = tf.constant(0, 1, shape=shape)
        return tf.Variable(initial)

    # convolution
    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # max pooling(2x2)
    def __max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
