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
        words, frequencies = zip(*most_freq)
        print(words)

    def cal_freq(self):
        pass

    def generate_neg(self):
        pass

    def word_to_int(self):
        pass


def main():
    vocab = Vocabulary("wikipedia_sample_tiny.txt", 50)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('weights')
    # parser.add_argument('dataset')
    # config = parser.parse_args()
    # print(config)
    main()
