#######################################################
#
# This template is the starting point for your homework
# You can modify this file in any way you want
#
#######################################################

import tensorflow as tf
import pandas
import numpy as np
from sklearn.metrics import accuracy_score
import sys

if len(sys.argv) != 3:
    print("Usage:")
    print("\tmain.py train.csv test.csv")
    sys.exit()

data_path = sys.argv[1]
test_data_path = sys.argv[2]

# Set parameters
letter_embedding_size = 5
lstm_hidden_size = 5
epochs = 100
minibatch_size = 256

# Load data
p_train_data = pandas.read_csv(data_path)
p_test_data = pandas.read_csv(test_data_path)

p_train_data.Name = p_train_data.Name.str.lower()
p_test_data.Name = p_test_data.Name.str.lower()

print(p_train_data.describe())
print(p_test_data.describe())
print(p_train_data[p_train_data.Name == 'Trelynn'])

# Convert data to numpy arrays
train = p_train_data.to_numpy()
test = p_test_data.to_numpy()

# Sort by name length
# np.random.shuffle(train)
train = np.stack(sorted(list(train), key=lambda x: len(x[0])))


def transform_data(data, max_len):
    """
    Transform the data into machine readable format. Substitute character with
    letter ids, replace gender according to the mapping M->0, F->1
    :param data: ndarray where first column is names, and the second is gender
    :param max_len: maximum length of a name
    :return: names, labels, vocab
    where
    - names: ndarray with shape [?,max_len]
    - labels: ndarray with shape [?,1]
    - vocab: dictionary with mapping from letters to integer IDs
    """

    unique = list(set("".join(data[:, 0])))
    unique.sort()
    vocab = dict(zip(unique, range(1, len(unique) + 1)))  # start from 1 for zero padding

    classes = list(set(data[:, 1]))
    classes.sort()
    class_map = dict(zip(classes, range(len(unique))))

    names = list(data[:, 0])
    labels = list(data[:, 1])

    def transform_name(name):
        point = np.zeros((1, max_len), dtype=int)
        name_mapped = np.array(list(map(lambda l: vocab[l], name)))
        point[0, 0: len(name_mapped)] = name_mapped
        return point

    transform_label = lambda lbl: np.array([[class_map[lbl]]])

    names = list(map(transform_name, names))
    labels = list(map(transform_label, labels))

    names = np.concatenate(names, axis=0)
    labels = np.concatenate(labels, axis=0)

    return names, labels, vocab


def get_minibatches(names, labels, mb_size):
    """
    Split data in minibatches
    :param names: ndarray of shape [?, max_name_len]
    :param labels: ndarray of shape [?, 1]
    :param mb_size: minibatch size
    :return: list of batches
    """
    batches = []

    position = 0

    while position + mb_size < len(labels):
        batches.append((names[position: position + mb_size], labels[position: position + mb_size]))
        position += mb_size

    batches.append((names[position:], labels[position:]))

    return batches


def create_model(emb_size, vocab_size, lstm_hidden_size, T, learning_rate=0.001):
    """
    Assemble tensorflow LSTM model for gender recognition
    :param emb_size: size of trainable letter embeddings
    :param vocab_size: number of unique letters
    :param lstm_hidden_size: passed as argument to LSTM cell
    :param T: maximum name length, the maximum length of input sequence
    :return: dictionary with tensors
    - train: tensor for updating weights
    - input: placeholder for input data
    - labels': placeholder for true labels
    - loss: tensor that returns loss
    - classify: tensor that returns sigmoid activations
    """

    pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32, name="zero_padding")
    symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

    symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

    input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
    labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    embedded = tf.nn.embedding_lookup(symbol_embedding, input_)

    lstm = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)
    outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=embedded, dtype=tf.float32)
    output = outputs[:, -1, :]
    logits = tf.layers.dense(output, 1)

    classify = tf.nn.sigmoid(logits)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

    # train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

    print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    return {
        'train': train,
        'input': input_,
        'labels': labels_,
        'loss': loss,
        'classify': classify
    }


# Find longest name length
max_len = p_train_data['Name'].str.len().max()

print(max_len)
train_data, train_labels, voc = transform_data(train, max_len)
print(train[0], train_data[0], train_labels[0], voc)
test_data, test_labels, _ = transform_data(test, max_len)
batches = get_minibatches(train_data, train_labels, minibatch_size)
terminals = create_model(letter_embedding_size, len(voc), lstm_hidden_size, max_len)

train_ = terminals['train']
input_ = terminals['input']
labels_ = terminals['labels']
loss_ = terminals['loss']
classify_ = terminals['classify']


def evaluate(tf_session, tf_loss, tf_classify, data, labels):
    """
    Evaluate loss and accuracy on a single minibatch
    :param tf_session: current opened session
    :param tf_loss: tensor for calculating loss
    :param tf_classify: tensor for calculating sigmoid activations
    :param data: data from the current batch
    :param labels: labels from the current batch
    :return: loss_value, accuracy_value
    """

    loss_val, predict = tf_session.run([tf_loss, tf_classify], {
        input_: data,
        labels_: labels
    })
    acc_val = accuracy_score(labels, np.where(predict > 0.5, 1, 0))

    return loss_val, acc_val


def base_line_lstm():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch

                sess.run([train_], {
                    input_: names,
                    labels_: labels
                })

            # Performance on the first training batch
            # but the first batch contains only the shortest names
            # comparing different batches can be used to see how zero paddign affects the performance
            names, labels = batches[0]
            train_loss, train_acc = evaluate(sess, loss_, classify_, names, labels)

            # Performance on the test set
            test_loss, test_acc = evaluate(sess, loss_, classify_, test_data, test_labels)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))


base_line_lstm()


def build_neural_netwrok(vocab_size, T, learning_rate=0.001):
    input_ = tf.placeholder(shape=[None, T * vocab_size], dtype=tf.int32)
    labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    layer1 = tf.layers.dense(name="l1", inputs=input_, units=52, activation=tf.nn.sigmoid)
    layer2 = tf.layers.dense(name="l1", inputs=layer1, units=12, activation=tf.nn.sigmoid)
    logits = tf.layers.dense(layer2, 1, activation=tf.nn.sigmoid)