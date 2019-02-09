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
    def __init__(self, alpha=0.2, T=-0.8, lr=0.0001):
        self.T = T
        self.counts = None
        self.sess = tf.Session()
        self.anchor = tf.placeholder(name="anchor", dtype=tf.float32, shape=(None, 2048))
        self.positive = tf.placeholder(name="positive", dtype=tf.float32, shape=(None, 2048))
        self.negative = tf.placeholder(name="negative", dtype=tf.float32, shape=(None, 2048))

        self.compare = tf.placeholder(name="compared_image", dtype=tf.float32, shape=(None, 2048))

        self.latent_anchor = self.siamese(self.anchor)
        self.latent_positive = self.siamese(self.positive)
        self.latent_negative = self.siamese(self.negative)

        self.latent_compare = self.siamese(self.compare)

        pos_loss = tf.nn.l2_loss(self.latent_anchor - self.latent_positive)
        neg_loss = alpha * -1 * tf.nn.l2_loss(self.latent_anchor - self.latent_negative)
        self.loss = tf.reduce_mean(tf.maximum(0., pos_loss + neg_loss))
        self.comparison = -1 * tf.nn.l2_loss(self.latent_anchor - self.latent_compare)
        self.decision = self.comparison > self.T
        my_opt = tf.train.AdamOptimizer(lr)
        self.train_step = my_opt.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def siamese(self, input):
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE):
            x = input
            layer1 = tf.layers.dense(inputs=x, units=512, activation=tf.nn.sigmoid)
            layer2 = tf.layers.dense(inputs=layer1, units=256, activation=tf.nn.sigmoid)
            output = tf.nn.l2_normalize(layer2, axis=1)
            return output

    def train(self, X, y, epochs=500):
        self.counts = collections.Counter(y)
        for epoch in range(epochs):
            losses = 0
            c = 0
            for an, po, ne, la in self.generate_batches(X, y):
                _, loss = self.sess.run([self.train_step, self.loss],
                                        feed_dict={self.anchor: an, self.positive: po, self.negative: ne})
                losses += loss
                c += len(an)
            losses /= c
            print("epoch [%d/%d] train_loss: %.5f" % (epoch, epochs, losses))

    def getPositiveNegative(self, X, Y, x, y):
        batch_count = self.check_nb_pictures(y)
        x = x[batch_count > 1]
        y = y[batch_count > 1]
        pos_batch_index = []
        neg_batch_index = []
        for img, label in zip(x, y):
            y1, y2, y_neg = self.getThreeRandomIndexes(Y, label)
            if not numpy.array_equal(img,X[y1]):
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
            anchor_batch, anchor_labels, pos, neg = self.getPositiveNegative(X, y, anchor_batch, anchor_labels)
            batches.append((anchor_batch, pos, neg, anchor_labels))
            pos += batch_size

        if pos < n_samples:
            anchor_batch = data[pos:n_samples, :]
            anchor_labels = labels[pos:n_samples]
            anchor_batch, anchor_labels, pos, neg = self.getPositiveNegative(X, y, anchor_batch, anchor_labels)
            batches.append((anchor_batch.reshape((-1,2048)), pos.reshape((-1,2048)), neg.reshape((-1,2048)), anchor_labels))

        return batches

    def getThreeRandomIndexes(self, Y, label):
        indexes = numpy.argwhere(Y == label)
        indexes_neg = numpy.argwhere(Y != label)
        index1 = random.randint(0, len(indexes)-1)
        index2 = random.randint(0, len(indexes)-1)
        index3 = random.randint(0, len(indexes_neg)-1)

        return indexes[index1], indexes[index2], indexes_neg[index3]
