import tensorflow as tf
from network import ConvNeuralNet
from tensorflow.examples.tutorials.mnist import input_data

MODEL_PATH = "model/WeightFile.ckpt"


if __name__ == '__main__':

    if tf.train.checkpoint_exists(MODEL_PATH):
        if input('Trained model already exists. Continue? y/n ') != "y":
            exit(0)

    with tf.Session() as sess:
        model = ConvNeuralNet()
        sess.run(tf.global_variables_initializer())
        correct_prediction = tf.equal(tf.argmax(model.y_conv, 1), tf.argmax(model.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        # download MNIST dataset
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # training
        best_eval_acc = 0
        for step in range(20000 + 1):
            # mini batch training
            batch = mnist.train.next_batch(50)
            model.train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.5})

            # evaluation
            if step % 100 == 0:
                eval_acc = accuracy.eval(feed_dict={model.x: mnist.test.images, model.y_: mnist.test.labels, model.keep_prob: 1.0})
                print("training... :{}/20000 eval_acc:{:.4f}".format(step, eval_acc))

                # save best accuracy model
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    saver.save(sess, MODEL_PATH)

        print("training finished. best_eval_acc:{:.4f}".format(best_eval_acc))
