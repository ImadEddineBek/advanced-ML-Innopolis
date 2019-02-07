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

print(input_)
print(cnn_embedding_)
