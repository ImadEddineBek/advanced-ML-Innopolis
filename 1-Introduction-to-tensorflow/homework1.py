import sys

import tensorflow as tf
from functools import reduce
from numpy import unique, array, vectorize
from sklearn.metrics import accuracy_score, f1_score


class SVMClassifier:

    def __init__(self, train_data=None):
        self.train_step = None
        self.sess = None
        self.accuracy = None
        self.loss = None
        self.X = None
        self.y = None
        self.prediction = None
        data, labels = train_data
        labels = self._transform_labels(labels)
        print(labels.shape)
        data = self._flatten_input(data)
        print(data.shape)
        self.train_data = (data, labels)
        self._open_session()

        self.assemble_graph()

        if train_data:
            self.train()

    def assemble_graph(self, learning_rate=0.02):
        self.X = tf.placeholder(name="input", dtype=tf.float32, shape=(None, 784))
        self.y = tf.placeholder(name="label", dtype=tf.float32, shape=(None, 1))

        self.A = tf.Variable(tf.random_normal(shape=[784, 1]))
        self.b = tf.Variable(tf.random_normal(shape=[1, 1]))

        model_output = tf.subtract(tf.matmul(self.X, self.A), self.b)

        self.loss = tf.reduce_mean(tf.maximum(0., 1 - self.y * model_output)) + 0.001 * tf.norm(self.A)

        self.prediction = tf.sign(model_output)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.y),
                                               tf.float32))

        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_step = my_opt.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, epochs=30, minibatch_size=256):

        data = self._create_minibatches(minibatch_size)
        train_loss = 0
        for i in range(epochs * len(data)):
            xx, yy = data[i % len(data)]
            self.sess.run(self.train_step, feed_dict={self.X: xx, self.y: yy})
            train_loss += self.sess.run(self.loss, feed_dict={self.X: xx, self.y: yy})
            accuracy = self.sess.run(self.accuracy, feed_dict={self.X: xx, self.y: yy})
            if i % len(data) == 0:
                print("Epoch %d, loss: %.2f accuracy %.2f." % (
                    i // (len(data)), train_loss / len(data), accuracy))
                # one epoch is an iteration over the whole data set
                train_loss = 0

    def predict(self, data):
        data = self._flatten_input(data)
        preds = self.sess.run(self.prediction, feed_dict={self.X: data}).flatten()
        preds[preds == -1] = 0
        return preds

    def _create_minibatches(self, minibatch_size):
        pos = 0

        data, labels = self.train_data
        n_samples = len(labels)

        batches = []
        while pos + minibatch_size < n_samples:
            batches.append((data[pos:pos + minibatch_size, :], labels[pos:pos + minibatch_size]))
            pos += minibatch_size

        if pos < n_samples:
            batches.append((data[pos:n_samples, :], labels[pos:n_samples, :]))

        return batches

    def _transform_labels(self, labels):
        labels[labels == 0] = -1
        return labels.reshape((-1, 1))

    def _flatten_input(self, data):
        size, w, h = data.shape
        return data.reshape((size, w * h))

    def _open_session(self):
        self.sess = tf.Session()


if __name__ == "__main__":
    # tf.set_random_seed(41)
    def mnist_to_binary(train_data, train_label, test_data, test_label):

        binarized_labels = []
        for labels in [train_label, test_label]:
            remainder_2 = vectorize(lambda x: x % 2)
            binarized_labels.append(remainder_2(labels))

        train_label, test_label = binarized_labels

        return train_data, train_label, test_data, test_label


    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data, train_labels, test_data, test_labels = mnist_to_binary(train_data, train_labels, eval_data, eval_labels)

    svm = SVMClassifier((train_data, train_labels))
    print("Testing score f1: {}".format(f1_score(test_labels, svm.predict(test_data))))
