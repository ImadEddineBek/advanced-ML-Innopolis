from collections import Counter
import numpy as np


class Vocabulary:
    def __init__(self, corpus_location, n):
        corpus_file = open(corpus_location, "r")
        counter = Counter()
        lines = corpus_file.readlines()
        # while line != None:
        for line in lines:
            counter.update(line.split())
        most_freq = counter.most_common(n)
        self.words, self.frequencies = zip(*most_freq)
        self.words, self.frequencies = np.array(self.words), np.array(self.frequencies)
        self.words_to_ints = {w: i for w, i in zip(self.words, range(len(self.words)))}
        unigram = self.frequencies / sum(self.frequencies)

        modified_unigram = np.power(unigram, 3 / 4)
        self.modified_unigram_weighs = modified_unigram / sum(modified_unigram)

    def word_freq(self, word):
        index = self.word_to_int(word)
        return self.frequencies[index]

    def get_neg_sample(self, context_size=5):
        return np.random.choice(range(len(self.words)), context_size, p=self.modified_unigram_weighs)

    def generate_neg(self):
        pass

    def word_to_int(self, word):
        return self.words_to_ints.get(word)

    def int_to_word(self, i):
        return self.words[i]


def main():
    vocab = Vocabulary("wikipedia_sample_tiny.txt", 50)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('weights')
    # parser.add_argument('dataset')
    # config = parser.parse_args()
    # print(config)
    main()
