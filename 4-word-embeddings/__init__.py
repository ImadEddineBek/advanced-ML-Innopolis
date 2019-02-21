from collections import Counter


class Vocabulary:
    def __init__(self, corpus_location, n):
        corpus_file = open(corpus_location, "r")
        counter = Counter()
        lines = corpus_file.readlines()
        # while line != None:
        for line in lines:
            #     for token in line.strip().split():
            #         print(token)
            counter.update(line.split())
            #
            # line = corpus_file.readline()
            # pass
        # print(corpus_file.readline())
        most_freq = counter.most_common(n)
        self.words, self.frequencies = zip(*most_freq)
        self.words_to_ints = {w: i for w, i in zip(self.words, range(len(self.words)))}
        # print(self.words_to_ints)
        # print(self.word_freq("the"))

    def word_freq(self,word):
        index = self.word_to_int(word)
        return self.frequencies[index]

    def generate_neg(self):
        pass

    def word_to_int(self,word):
        return self.words_to_ints.get(word)


def main():
    vocab = Vocabulary("wikipedia_sample_tiny.txt", 5)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('weights')
    # parser.add_argument('dataset')
    # config = parser.parse_args()
    # print(config)
    main()
