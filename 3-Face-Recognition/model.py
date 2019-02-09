import tensorflow as tf


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
    def __init__(self, alpha):
        self.anchor = tf.placeholder(name="anchor", dtype=tf.float32, shape=(None, 2048))
        self.positive = tf.placeholder(name="positive", dtype=tf.float32, shape=(None, 2048))
        self.negative = tf.placeholder(name="negative", dtype=tf.float32, shape=(None, 2048))

        self.latent_anchor = self.siamese(self.anchor)
        self.latent_positive = self.siamese(self.positive)
        self.latent_negative = self.siamese(self.negative)
        pos_loss = tf.nn.l2_loss(self.latent_anchor - self.latent_positive)
        neg_loss = alpha * -1 * tf.nn.l2_loss(self.latent_anchor - self.latent_negative)
        self.loss = tf.reduce_mean(tf.maximum(0., pos_loss + neg_loss))
        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
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
