from collections import Counter
import numpy as np
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sig
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


class Vocabulary:
    def __init__(self, corpus_location, n):
        corpus_file = open(corpus_location, "r")
        counter = Counter()
        self.lines = corpus_file.readlines()
        # while line != None:
        for line in self.lines:
            counter.update(line.split())
        most_freq = counter.most_common(n)
        self.words, self.frequencies = zip(*most_freq)
        self.words, self.frequencies = np.array(self.words), np.array(self.frequencies)
        self.words_to_ints = {w: i for w, i in zip(self.words, range(len(self.words)))}
        unigram = self.frequencies / sum(self.frequencies)

        modified_unigram = np.power(unigram, 3 / 4)
        self.modified_unigram_weighs = modified_unigram / sum(modified_unigram)
        self.line_index = 0
        self.in_line_index = 0

    def word_freq(self, word):
        index = self.word_to_int(word)
        return self.frequencies[index]

    def get_neg_sample(self, context_size=5):
        return np.random.choice(range(len(self.words)), 2 * context_size, p=self.modified_unigram_weighs)

    def get_next_batch(self, n_contexts=200, context_size=5, k=None):
        batch = []
        for i in range(n_contexts):
            central, pos = self.get_next_context(context_size)
            neg = self.get_neg_sample(context_size)
            neg = np.delete(neg, np.argwhere(neg == central))
            for p in pos:
                batch.append([central, p, 1.0])
            for n in neg:
                batch.append([central, n, 0.0])
        return np.array(batch)

    def get_next_context(self, context_size):
        line = self.lines[self.line_index]
        center = line[self.in_line_index]
        center = self.word_to_int(center)
        if center != -1:
            beg_index = self.in_line_index - context_size if self.in_line_index - context_size >= 0 else 0
            end_index = self.in_line_index
            prev = line[beg_index:end_index]
            beg_index = self.in_line_index + 1 if self.in_line_index + 1 < len(line) else -1
            end_index = self.in_line_index + 1 + context_size if self.in_line_index + 1 + context_size < len(
                line) else -1
            nex = line[beg_index:end_index]
            pos = prev + nex
            pos = [self.word_to_int(w) for w in pos]
            while -1 in pos:
                pos.remove(-1)
            self.update_indecies()
            return center, pos
        else:
            self.update_indecies()
            return self.get_next_context(context_size)

    def update_indecies(self):
        lengh_of_current_line = len(self.lines[self.line_index])
        self.in_line_index += 1
        if self.in_line_index < lengh_of_current_line:
            return
        self.in_line_index = 0
        self.line_index += 1
        if self.line_index < len(self.lines):
            return
        self.line_index = 0

    def word_to_int(self, word):
        return self.words_to_ints.get(word, -1)

    def int_to_word(self, i):
        return self.words[i]

    def is_most_common(self, word):
        return word in self.words

    def get_words(self):
        return self.words


class Model:
    def __init__(self, vocab, vocabulary_size=5000, embedding_size=50):
        # self.W_IN = tf.get_variable("W_IN",[None,])
        self.vocab = vocab
        self.LOG_DIR = 'imad_logs'
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        self.metadata = os.path.join(self.LOG_DIR, 'metadata.tsv')
        with open(self.metadata, 'w') as metadata_file:
            for row in vocab.get_words():
                metadata_file.write(str(row + "\n"))
        self.sess = tf.Session()
        with tf.name_scope('inputs'):
            self.center = tf.placeholder(tf.int32, shape=[None, ], name="center")
            self.word = tf.placeholder(tf.int32, shape=[None, ], name="word")
            self.label = tf.placeholder(tf.float32, shape=[None, ], name="label")
        with tf.name_scope('embeddings'):
            self.W_IN = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W_IN")
            self.W_OUT = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W_OUT")
            self.u = tf.nn.embedding_lookup(params=self.W_IN, ids=self.center, name='embeddings/W_IN')
            self.v = tf.nn.embedding_lookup(params=self.W_OUT, ids=self.word, name='embeddings/W_OUT')
            logits = tf.reduce_sum(self.u * self.v, axis=1)

        self.loss = tf.reduce_mean(sig(logits=logits, labels=self.label))
        self.merged = tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver()

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        writer = tf.summary.FileWriter(self.LOG_DIR, self.sess.graph)
        writer.close()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.LOG_DIR,
                                                  graph=self.sess.graph)

    def train(self, epochs=10):
        for i in range(epochs):
            batch = self.vocab.get_next_batch()
            center, words, labels = batch[:, 0], batch[:, 1], batch[:, 2]
            _, loss, merged, _ = self.sess.run(
                [self.optimizer, self.loss, self.merged, self.emb],
                feed_dict={self.center: center, self.word: words, self.label: labels})
            print(loss)
            self.train_writer.add_summary(merged, i)

        self.saver.save(self.sess, os.path.join(self.LOG_DIR, 'model.ckpt'))
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embeddings/W_IN'
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata.tsv'

        embedding = config.embeddings.add()
        embedding.tensor_name = 'embeddings/W_OUT'
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata.tsv'
        # # Saves a config file that TensorBoard will read during startup


        projector.visualize_embeddings(tf.summary.FileWriter(self.LOG_DIR), config)

    def log(self):
        pass


def main():
    vocab = Vocabulary("wikipedia_sample_tiny.txt", 5000)
    model = Model(vocab)
    model.train()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('weights')
    # parser.add_argument('dataset')
    # config = parser.parse_args()
    # print(config)
    main()
