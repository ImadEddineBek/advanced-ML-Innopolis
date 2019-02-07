import argparse
import os
import cv2
import glob
import numpy as np


def main(config):
    names = os.listdir(config.dataset)
    print("nb of people:", len(names))
    names2label = {name: label for name, label in zip(names, range(len(names)))}
    print(names2label)

    im_size = 299
    load_image = lambda im_path: cv2.resize(cv2.imread(im_path), (im_size, im_size))
    latent_space = []
    print("generate latent representations of images")
    progress = 0
    for name in names:
        print("progress %d/%d" % (progress, len(names)))
        progress += 1
        for path in glob.glob(config.dataset + "/" + name + "/*.jpg"):
            img = load_image(path)
            import tensorflow as tf

            pre_trained_graph_path = 'InceptionV3.pb'
            with tf.Session() as sess:
                with tf.gfile.GFile(pre_trained_graph_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')
                    input_ = tf.get_default_graph().get_tensor_by_name("Model_Input:0")
                    cnn_embedding_ = tf.get_default_graph().get_tensor_by_name("Model_Output:0")

                    latent = sess.run(cnn_embedding_, feed_dict={input_: img.reshape((1, 299, 299, 3))})
                    latent_space.append([latent.flatten(), names2label[name]])

    print(latent_space)
    print(np.array(latent_space).shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('dataset')
    config = parser.parse_args()
    print(config)
    main(config)
