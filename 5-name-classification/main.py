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

# if len(sys.argv) != 3:
#     print("Usage:")
#     print("\tmain.py train.csv test.csv")
#     sys.exit()
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

seed = 7
np.random.seed(seed)
data_path = "new_train.csv"  # sys.argv[1]
test_data_path = "new_test.csv"  # sys.argv[2]

# Set parameters
letter_embedding_size = 4
lstm_hidden_size = 4
epochs = 500
minibatch_size = 1000

# Load data
p_train_data = pandas.read_csv(data_path, usecols=['Name', 'Sex']).dropna()
p_test_data = pandas.read_csv(test_data_path, usecols=['Name', 'Sex']).dropna()

# p_train_data.Name = p_train_data.Name.str.lower()
# p_test_data.Name = p_test_data.Name.str.lower()

# print(p_train_data.describe())
# print(p_test_data.describe())
# print(p_train_data[p_train_data.Name == 'trelynn'])

# Convert data to numpy arrays
train = p_train_data.to_numpy()
test = p_test_data.to_numpy()
# print(np.unique(train[:, 1], return_counts=True)[1] / len(train[:, 1]))
#
# i_class0 = np.where(train[:, 1] == 'M')[0]
# i_class1 = np.where(train[:, 1] == 'F')[0]
# i_class2 = np.where(train[:, 1] == 'B')[0]
#
# # Number of observations in each class
# n_class0 = len(i_class0)
# n_class1 = len(i_class1)
# n_class2 = len(i_class2)
#
# # For every observation in class 1, randomly sample from class 0 with replacement
# i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
# i_class2_upsampled = np.random.choice(i_class2, size=n_class1, replace=True)
#
# # Join together class 0's upsampled target vector with class 1's target vector
# train = np.concatenate((train[i_class0_upsampled], train[i_class1], train[i_class2_upsampled]))
#
# i_class0 = np.where(test[:, 1] == 'M')[0]
# i_class1 = np.where(test[:, 1] == 'F')[0]
# i_class2 = np.where(test[:, 1] == 'B')[0]
#
# # Number of observations in each class
# n_class0 = len(i_class0)
# n_class1 = len(i_class1)
# n_class2 = len(i_class2)
#
# # For every observation in class 1, randomly sample from class 0 with replacement
# i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
# i_class2_upsampled = np.random.choice(i_class2, size=n_class1, replace=True)
#
# # Join together class 0's upsampled target vector with class 1's target vector
# test = np.concatenate((test[i_class0_upsampled], test[i_class1], test[i_class2_upsampled]))

# Sort by name length
np.random.shuffle(train)
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
        point[0, -len(name_mapped):] = name_mapped
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


def create_model(emb_size, vocab_size, lstm_hidden_size, T, learning_rate=0.0005):
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

    pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
    symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

    symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

    input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
    labels_ = tf.placeholder(shape=[None,1], dtype=tf.int32)

    embedded = tf.nn.embedding_lookup(symbol_embedding, input_)

    lstm = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)
    outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=embedded, dtype=tf.float32)
    output = outputs[:, -1, :]
    logits = tf.layers.dense(output, 3)

    classify = tf.nn.softmax(logits, axis=1)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(tf.reshape(labels_, [-1]), 3, dtype=tf.int32), dim=1))

    # train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train = tf.train.AdamOptimizer(learning_rate,beta1=0.5).minimize(loss)

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
np.random.shuffle(train)
train_data, train_labels, voc = transform_data(train, max_len)
print(np.unique(train_labels, return_counts=True)[1] / len(train_labels))
print(train[0], train_data[0], train_labels[0], voc)
print(np.unique(train_labels))
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
    acc_val = accuracy_score(labels, np.argmax(predict, axis=1))

    return loss_val, acc_val


from random import randint


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

            names, labels = batches[0]#randint(0, len(batches) - 1)]
            train_loss, train_acc = evaluate(sess, loss_, classify_, names, labels)

            # Performance on the test set
            test_loss, test_acc = evaluate(sess, loss_, classify_, test_data, test_labels)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))


base_line_lstm()


def build_neural_netwrok(vocab_size, T, learning_rate=0.001):
    with tf.name_scope('simple'):
        input_ = tf.placeholder(shape=[None, T], dtype=tf.float32)
        labels_ = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        layer1 = tf.layers.dense(inputs=input_, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 3)

        classify = tf.nn.softmax(logits, axis=1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network():
    terminals = build_neural_netwrok(len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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
        print(np.argmax(predict, axis=1), labels.flatten())
        acc_val = accuracy_score(labels, np.argmax(predict, axis=1))

        return loss_val, acc_val

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_simple_neural.npy', np.array(losses))
    np.save('accuracies_simple_neural.npy', np.array(accuracies))


# run_neural_network()


def build_neural_netwrok_embedding(emb_size, vocab_size, T, learning_rate=0.001):
    with tf.name_scope('embedded_network'):
        pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
        symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

        symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

        input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
        labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)
        embedded = tf.reshape(embedded, shape=(-1, T * emb_size))
        layer1 = tf.layers.dense(inputs=embedded, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 1)

        classify = tf.nn.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network_embeddoing():
    terminals = build_neural_netwrok_embedding(letter_embedding_size, len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_emb_neural.npy', np.array(losses))
    np.save('accuracies_emb_neural.npy', np.array(accuracies))


# run_neural_network_embeddoing()


def build_neural_netwrok_embedding_maxpool_1(emb_size, vocab_size, T, learning_rate=0.001):
    with tf.name_scope('max_network'):
        pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
        symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

        symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

        input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
        labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)
        embedded = tf.reduce_max(embedded, axis=1)

        layer1 = tf.layers.dense(inputs=embedded, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 1)

        classify = tf.nn.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network_embeddoing_maxpool_1():
    terminals = build_neural_netwrok_embedding_maxpool_1(letter_embedding_size, len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_emb_neural.npy', np.array(losses))
    np.save('accuracies_emb_neural.npy', np.array(accuracies))


# run_neural_network_embeddoing_maxpool_1()


def build_neural_netwrok_embedding_maxpool_2(emb_size, vocab_size, T, learning_rate=0.001):
    with tf.name_scope('max_network'):
        pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
        symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

        symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

        input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
        labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)
        embedded = tf.reduce_max(embedded, axis=2)

        layer1 = tf.layers.dense(inputs=embedded, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 1)

        classify = tf.nn.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network_embeddoing_maxpool_2():
    terminals = build_neural_netwrok_embedding_maxpool_2(letter_embedding_size, len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_emb_neural.npy', np.array(losses))
    np.save('accuracies_emb_neural.npy', np.array(accuracies))


# run_neural_network_embeddoing_maxpool_2()


def build_neural_netwrok_embedding_avg_1(emb_size, vocab_size, T, learning_rate=0.001):
    with tf.name_scope('avg_network1'):
        pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
        symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

        symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

        input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
        labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)
        embedded = tf.reduce_mean(embedded, axis=1)

        layer1 = tf.layers.dense(inputs=embedded, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 1)

        classify = tf.nn.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network_embeddoing_avg_1():
    terminals = build_neural_netwrok_embedding_avg_1(letter_embedding_size, len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_emb_neural.npy', np.array(losses))
    np.save('accuracies_emb_neural.npy', np.array(accuracies))


# run_neural_network_embeddoing_avg_1()


def build_neural_netwrok_embedding_avg_2(emb_size, vocab_size, T, learning_rate=0.001):
    with tf.name_scope('avg_network2'):
        pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
        symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

        symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

        input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
        labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)
        embedded = tf.reduce_mean(embedded, axis=2)

        layer1 = tf.layers.dense(inputs=embedded, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 1)

        classify = tf.nn.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network_embeddoing_avg_2():
    terminals = build_neural_netwrok_embedding_avg_2(letter_embedding_size, len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_emb_neural.npy', np.array(losses))
    np.save('accuracies_emb_neural.npy', np.array(accuracies))


# run_neural_network_embeddoing_avg_2()


def build_neural_netwrok_embedding_avg_3(emb_size, vocab_size, T, learning_rate=0.001):
    with tf.name_scope('avg_network1'):
        pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
        symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

        symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

        input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
        labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)

        filter_ = tf.get_variable("filter", shape=(1, T, 1))
        filtered = tf.reduce_mean(embedded * filter_, axis=1)

        layer1 = tf.layers.dense(inputs=filtered, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 1)

        classify = tf.nn.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network_embeddoing_avg_3():
    terminals = build_neural_netwrok_embedding_avg_3(letter_embedding_size, len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_emb_neural.npy', np.array(losses))
    np.save('accuracies_emb_neural.npy', np.array(accuracies))


# run_neural_network_embeddoing_avg_3()


def build_neural_netwrok_embedding_avg_4(emb_size, vocab_size, T, learning_rate=0.001):
    with tf.name_scope('avg_network2'):
        pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32)
        symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

        symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

        input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
        labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)
        embedded = tf.nn.embedding_lookup(symbol_embedding, input_)

        filter_ = tf.get_variable("filter", shape=(1, T, 1))
        filtered = tf.reduce_mean(embedded * filter_, axis=2)
        layer1 = tf.layers.dense(inputs=filtered, units=20, activation=tf.nn.leaky_relu)
        layer2 = tf.layers.dense(inputs=layer1, units=12, activation=tf.nn.sigmoid)
        layer3 = tf.layers.dense(inputs=layer2, units=20, activation=tf.nn.leaky_relu)
        layer4 = tf.layers.dense(inputs=layer3, units=12, activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(layer4, 1)

        classify = tf.nn.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

        train = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss)
        # train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

        print("trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        return {
            'train': train,
            'input': input_,
            'labels': labels_,
            'loss': loss,
            'classify': classify
        }


def run_neural_network_embeddoing_avg_4():
    terminals = build_neural_netwrok_embedding_avg_4(letter_embedding_size, len(voc), max_len)

    train_ = terminals['train']
    input_ = terminals['input']
    labels_ = terminals['labels']
    loss_ = terminals['loss']
    classify_ = terminals['classify']
    epochs = 100
    losses = []
    accuracies = []

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for batch in batches:
                names, labels = batch
                # print(names.shape)
                # names = names.reshape(names.shape[0],-1)
                # print(names.shape)
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
            losses.append(train_loss)
            accuracies.append(test_acc)

            print("Epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test accuracy %.5f" % (
                e, train_loss, train_acc, test_loss, test_acc))

    np.save('losses_emb_neural.npy', np.array(losses))
    np.save('accuracies_emb_neural.npy', np.array(accuracies))


# run_neural_network_embeddoing_avg_4()


# TODO

def three_label_data(data):
    names = data[:, 0]
    non_unique = np.array([len(np.where(names == n)[0]) != 1 for n in names])
    data[non_unique, 1] = 'B'
    return data


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
#
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# dummy_y = np_utils.to_categorical(encoded_Y)
# results = cross_val_score(estimator, train[], dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
