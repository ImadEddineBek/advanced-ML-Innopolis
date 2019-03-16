import pandas as pd
import numpy as np


def read_data():
    train_words = []
    train_tags = []
    with open('train_pos.txt') as f:
        lines = f.readlines()
        line_word_rep = []
        line_tag_rep = []
        for line in lines:
            line = line.rstrip('\n')
            if line != ' ':
                splitted = line.split(' ')
                line_word_rep.append(splitted[0])
                line_tag_rep.append(splitted[1])
        train_words.append(line_word_rep)
        train_tags.append(line_tag_rep)
    test_words = []
    test_tags = []
    with open('test_pos.txt') as f:
        lines = f.readlines()
        line_word_rep = []
        line_tag_rep = []
        for line in lines:
            line = line.rstrip('\n')
            if line != ' ':
                splitted = line.split(' ')
                line_word_rep.append(splitted[0])
                line_tag_rep.append(splitted[1])
        test_words.append(line_word_rep)
        test_tags.append(line_tag_rep)

    tag_dist_words = pd.read_csv('./tag_log_prob_per_word.tsv', sep='\t')
    print(len(tag_dist_words.columns))
    all_tags = tag_dist_words.columns[1:]
    tag_index_map = {all_tags[i]: i for i in range(len(all_tags))}
    return (train_words, train_tags), (test_words, test_tags), tag_dist_words, (all_tags, tag_index_map)


def estimate_transition_probabilities(train_tags, tag_index_map):
    transition_probabilities = np.zeros((len(tag_index_map), len(tag_index_map)))
    for line in train_tags:
        for a, b in zip(line[:-1], line[1:]):
            ind_a = tag_index_map[a]
            ind_b = tag_index_map[b]
            transition_probabilities[ind_a,ind_b] += 1
    return transition_probabilities


def estimate_prior_probabilities(train_tags, tag_index_map):
    prior_probabilities = np.zeros((len(tag_index_map)))
    for line in train_tags:
        for a in line:
            ind = tag_index_map[a]
            prior_probabilities[ind] += 1
    # print(prior_probabilities)
    return prior_probabilities


def viterbi():
    pass


def estimate_best_tag():
    pass


def per_word_accuracy():
    pass


(train_words, train_tags), (test_words, test_tags), tag_dist_words, (all_tags, tag_index_map) = read_data()

estimate_transition_probabilities(train_tags, tag_index_map)
estimate_prior_probabilities(train_tags, tag_index_map)