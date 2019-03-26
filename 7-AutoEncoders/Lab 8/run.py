from utils import load_data, get_current_time, create_dirs, \
    create_minibatches, write_to_tensorboard, \
    create_summary_and_projector, create_evaluation_tensor
import tensorflow as tf
import os

# this project uses tensorboard. You can launch tensorboard by executing
# "tensorboard --logdir=log" in your project folder

# Set parameters
learning_rate = 0.001
minibatch_size = 125
num_epochs = 20
latent_space_size = 30
log_dir = "log"
current_run = get_current_time()

# Create necessary directories
log_path, run_path = create_dirs(log_dir, current_run)

# Load MNIST data
imgs, lbls = load_data()
mbs = create_minibatches(imgs, lbls, minibatch_size)

# Prepare evaluation set
# this set is used to visualize embedding space and decoding results
evaluation_set = mbs[0]
evaluation_shape = (minibatch_size, latent_space_size)


def create_model(input_shape):
    """
    Create a simple autoencoder model. Input is assumed to be an image
    :param input_shape: expects the input in format (height, width, n_channels)
    :return: dictionary with tensors required to train and evaluate the model
    """
    h, w, c = input_shape

    ### START CODE HERE --------

    ### input is a placeholder for your data
    ### set up shape and dtype
    input = tf.placeholder(???)
    l0 = ???
    l1 = ???
    l2 = ???
    l3 = ???
    ### encoding is a bottle neck layer of the NN
    ### this layer has no activation
    encoding = tf.layers.dense(l3, units=latent_space_size, activation=None, name="encoded")
    l5 = ???
    l6 = ???
    l7 = ???
    l8 = ???

    ### any layer without activation could be named as logits
    logits = tf.reshape(l8, [-1, h, w, c], name="logits")
    decode = tf.nn.sigmoid(logits, name="decoded")
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=logits, name='loss')
    cost = tf.reduce_mean(loss, name="cost")

    ### END CODE HERE -------

    model = {'cost': cost,
             'input': input,
             'enc': encoding,
             'dec': decode
             }
    return model

# Create model and tensors for evaluation
input_shape = (28, 28, 1)
model = create_model(input_shape)
evaluation = create_evaluation_tensor(model, evaluation_shape)

# Create optimizer
opt = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

# Create tensors for visualizing with tensorboard
# https://www.tensorflow.org/programmers_guide/saved_model
saver = tf.train.Saver()
for_tensorboard = create_summary_and_projector(model, evaluation, evaluation_set, run_path)

tf.set_random_seed(1)
with tf.Session() as sess:
    # Save graph
    # https: // www.tensorflow.org / programmers_guide / graph_viz
    train_writer = tf.summary.FileWriter(run_path, sess.graph)

    print("Initializing model")
    sess.run(tf.global_variables_initializer())

    for e in range(num_epochs):
        # iterate through minibatches
        for mb in mbs:
            batch_cost, _ = sess.run([model['cost'], opt],
                                     feed_dict={model['input']: mb[0]})

        # write current results to log
        write_to_tensorboard(sess, train_writer, for_tensorboard, evaluation_set, evaluation, e)
        # save trained model
        saver.save(sess, os.path.join(run_path, "model.ckpt"))

        print("Epoch: {}/{}".format(e+1, num_epochs),
              "batch cost: {:.4f}".format(batch_cost))
