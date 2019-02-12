import sys

import tensorflow as tf
import collections, numpy
import random


class ModelInception:
    def __init__(self, pre_trained_graph_path='InceptionV3.pb'):
        self.sess = tf.Session()
        with tf.gfile.GFile(pre_trained_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            self.input_ = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
            self.cnn_embedding_ = tf.get_default_graph().get_tensor_by_name("Model_Output:0")

            print(self.input_)
            print(self.cnn_embedding_)
        print("[INFO] Model loaded")

    def get_latent_space(self, img):
        if len(img.shape) == 3:
            img = img.reshape((1, 299, 299, 3))
        return self.sess.run(self.cnn_embedding_, feed_dict={self.input_: img})


class ModelSiamese:
    def __init__(self, alpha=0.2, T=0.8, lr=0.0001):
        self.first = True
        self.T = T
        self.counts = None
        self.sess = tf.Session()
        self.anchor = tf.placeholder(name="anchor", dtype=tf.float32, shape=(None, 2048))
        self.positive = tf.placeholder(name="positive", dtype=tf.float32, shape=(None, 2048))
        self.negative = tf.placeholder(name="negative", dtype=tf.float32, shape=(None, 2048))

        self.compare = tf.placeholder(name="compared_image", dtype=tf.float32, shape=(None, 2048))
        with tf.variable_scope('siamese') as scope:
            self.latent_anchor = self.siamese(self.anchor)
            scope.reuse_variables()

            self.latent_positive = self.siamese(self.positive)
            self.latent_negative = self.siamese(self.negative)

            self.latent_compare = self.siamese(self.compare)

        self.pos_loss = tf.nn.l2_loss(self.latent_anchor - self.latent_positive)
        self.neg_loss = -alpha * tf.nn.l2_loss(self.latent_anchor - self.latent_negative)

        self.loss = tf.reduce_mean(tf.maximum(0.0, self.pos_loss + self.neg_loss))
        self.comparison = tf.norm(self.latent_anchor - self.latent_compare, axis=1)*1000
        self.decision = self.comparison < self.T
        self.lr = tf.placeholder(name="lr", dtype=tf.float32)
        my_opt = tf.train.AdamOptimizer(self.lr)
        self.train_step = my_opt.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        writer.close()

    def siamese(self, input):
        layer1 = tf.layers.dense(name="d1", inputs=input, units=512, activation=tf.nn.sigmoid)
        layer2 = tf.layers.dense(name="d2", inputs=layer1, units=256, activation=tf.nn.sigmoid)
        return tf.nn.l2_normalize(layer2, name="l2", axis=1)

    def train(self, X, y, test_anchor=None, test_pos=None, test_neg=None, epochs=500):
        self.counts = collections.Counter(y)
        lr = 0.0001
        reduction = 0.9999
        for epoch in range(epochs):
            losses = 0
            c = 0
            for an, po, ne, la in self.generate_batches(X, y):
                an = numpy.squeeze(an.reshape((-1, 2048)))
                po = numpy.squeeze(po.reshape((-1, 2048)))
                ne = numpy.squeeze(ne.reshape((-1, 2048)))
                _, loss, pos_loss, neg_loss, laan = self.sess.run(
                    [self.train_step, self.loss, self.pos_loss, self.neg_loss, self.latent_anchor],
                    feed_dict={self.anchor: an, self.positive: po,
                               self.negative: ne, self.lr: lr})
                losses += loss
                # print(laan)
                # print(pos_loss)
                c += len(an)
                # print("laan_loss", laan)
            lr *= reduction
            test_acc, test_loss = self.validate(test_anchor, test_pos, test_neg, lr)
            print(
                "Epoch %d, test acc %.4f, test batch %.4f loss  %.4f," % (epoch, test_acc, test_loss, losses))
            # print("epoch %d, %d," % (epoch, losses))

    def getPositiveNegative(self, X, Y, x, y):
        batch_count = self.check_nb_pictures(y)
        x = x[batch_count > 1]
        y = y[batch_count > 1]
        pos_batch_index = []
        neg_batch_index = []
        for img, label in zip(x, y):
            y1, y2, y_neg = self.getThreeRandomIndexes(Y, label)
            if not numpy.array_equal(img, X[y1]):
                pos_batch_index.append(y1)
            else:
                pos_batch_index.append(y2)
            neg_batch_index.append(y_neg)
        return x, y, X[pos_batch_index], X[neg_batch_index]

    def check_nb_pictures(self, y):
        batch_count = []
        for label in y:
            batch_count.append(self.counts.get(label))
        return numpy.array(batch_count)

    def generate_batches(self, X, y, batch_size=256):
        pos = 0

        data, labels = X, y
        n_samples = len(labels)

        batches = []
        while pos + batch_size < n_samples:
            anchor_batch = data[pos:pos + batch_size, :]
            anchor_labels = labels[pos:pos + batch_size]
            anchor_batch, anchor_labels, posi, neg = self.getPositiveNegative(X, y, anchor_batch, anchor_labels)
            batches.append((anchor_batch, posi, neg, anchor_labels))
            pos += batch_size

        if pos < n_samples:
            anchor_batch = data[pos:n_samples, :]
            anchor_labels = labels[pos:n_samples]
            anchor_batch, anchor_labels, posi, neg = self.getPositiveNegative(X, y, anchor_batch, anchor_labels)
            batches.append(
                (anchor_batch.reshape((-1, 2048)), numpy.squeeze(posi.reshape((-1, 2048))),
                 numpy.squeeze(neg.reshape((-1, 2048))), anchor_labels))

        return batches

    def getThreeRandomIndexes(self, Y, label):
        indexes = numpy.argwhere(Y == label)
        indexes_neg = numpy.argwhere(Y != label)
        index1 = random.randint(0, len(indexes) - 1)
        index2 = random.randint(0, len(indexes) - 1)
        index3 = random.randint(0, len(indexes_neg) - 1)

        return indexes[index1], indexes[index2], indexes_neg[index3]

    def validate(self, X_anchor, X_pos, X_neg, lr):
        X_anchor = numpy.squeeze(X_anchor.reshape((-1, 2048)))
        X_pos = numpy.squeeze(X_pos.reshape((-1, 2048)))
        X_neg = numpy.squeeze(X_neg.reshape((-1, 2048)))
        decision_pos, dec = self.sess.run([self.decision, self.comparison],
                                          feed_dict={self.anchor: X_anchor, self.compare: X_pos})
        print("dec",dec.mean())
        decision_neg, dec = self.sess.run([self.decision, self.comparison],
                                          feed_dict={self.anchor: X_anchor, self.compare: X_neg})
        print("dec",dec.mean())

        _, loss = self.sess.run([self.train_step, self.loss],
                                feed_dict={self.anchor: X_anchor, self.positive: X_pos, self.negative: X_neg,
                                           self.lr: lr})
        accuracy = decision_pos.sum() + (1 - decision_neg).sum()
        return accuracy / (2 * len(X_anchor)), loss
