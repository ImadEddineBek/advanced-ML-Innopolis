import tensorflow as tf


class Model:
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
    def get_latent_space(self,img):
        if len(img.shape) == 3:
            img = img.reshape((1, 299, 299, 3))
        return self.sess.run(self.cnn_embedding_, feed_dict={self.input_: img})

pre_trained_graph_path = 'InceptionV3.pb'
