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
            else:
                train_words.append(line_word_rep)
                train_tags.append(line_tag_rep)
                line_word_rep = []
                line_tag_rep = []

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
            else:
                test_words.append(line_word_rep)
                test_tags.append(line_tag_rep)
                line_word_rep = []
                line_tag_rep = []

    tag_dist_words = pd.read_csv('./tag_logit_per_word.tsv', sep='\t', index_col=0)
    all_tags = tag_dist_words.columns
    tag_index_map = {all_tags[i]: i for i in range(len(all_tags))}
    return (train_words, train_tags), (test_words, test_tags), tag_dist_words, (all_tags, tag_index_map)


def estimate_transition_probabilities(train_tags, tag_index_map):
    transition_probabilities = np.zeros((len(tag_index_map), len(tag_index_map)))
    for line in train_tags:
        for a, b in zip(line[:-1], line[1:]):
            ind_a = tag_index_map[a]
            ind_b = tag_index_map[b]
            transition_probabilities[ind_a, ind_b] += 1
    transition_probabilities /= transition_probabilities.sum(axis=0)
    prep = transition_probabilities - np.diag(np.ones(len(tag_index_map)))
    a = np.row_stack((prep[:-1, :], np.ones(len(tag_index_map))))
    b = np.zeros(len(tag_index_map))
    b[-1] = 1
    prior_probabilities = np.linalg.solve(a, b)

    return transition_probabilities, prior_probabilities


def estimate_prior_probabilities(train_tags, tag_index_map):
    prior_probabilities = np.zeros((len(tag_index_map)))
    for line in train_tags:
        for a in line:
            ind = tag_index_map[a]
            prior_probabilities[ind] += 1
    # print(prior_probabilities)
    return prior_probabilities


def viterbi(transition_probabilities, prior_probabilities, tag_dist_words, test_set, tag_index_map):
    probs = np.zeros((len(test_set), len(tag_index_map)))
    maxes = np.zeros((len(test_set), len(tag_index_map)), dtype=int)
    # print(test_set)
    # print(tag_dist_words)
    # print(prior_probabilities)
    # print(tag_dist_words.loc[test_set[0]].values[0])
    # print(np.exp(tag_dist_words.loc[test_set[0]].values))
    # print(np.exp(tag_dist_words.loc[test_set[0]].values).shape)
    probs[0] = prior_probabilities * np.exp(tag_dist_words.loc[test_set[0]].values)
    # print(probs[0])
    # print(transition_probabilities * probs[0][:, np.newaxis])

    #### one step
    for i in range(1, len(probs)):
        # print(test_set[i])
        # print(np.exp(tag_dist_words.loc[test_set[i]].values).shape)
        # print(transition_probabilities)
        probs[i] = np.max(
            transition_probabilities * probs[i - 1] * np.exp(tag_dist_words.loc[test_set[i]].values)[:, np.newaxis],
            axis=1)
        maxes[i] = np.argmax(transition_probabilities * probs[i - 1], axis=1)
    # print(np.max(probs[-1]))
    index = int(np.argmax(probs[-1]))
    path = []
    for i in range(len(maxes) - 1, -1, -1):
        path.append(index)
        index = maxes[i, index]
    path.reverse()
    return [list(tag_index_map.keys())[i] for i in path]


def estimate_best_tag():
    pass


def per_word_accuracy(predicted,correct):
    i = 0
    for p,c in zip(predicted,correct):
        if p == c:
            i+=1
    return i,len(predicted)


(train_words, train_tags), (test_words, test_tags), tag_dist_words, (all_tags, tag_index_map) = read_data()
# print(tag_index_map)
# print(tag_dist_words.loc['Rockwell']['WRB'])
# tag_dist_words = np.array([[0.2, 0.8], [0.6, 0.4]])
transition_probabilities, prior_probabilities = estimate_transition_probabilities(train_tags, tag_index_map)
# test_set = [1, 1, 0, 0, 0, 1]
# tag_index_map = {'sun': 0, 'rain': 1}
# train_tags = [
#     ['sun', 'sun', 'sun', 'sun', 'rain', 'rain', 'rain', 'sun', 'sun', 'sun', 'sun', 'rain', 'rain', 'sun', 'sun',
#      'sun']]

test_set = test_words[0]
acc, l = 0, 0
for test_set, test_correct in zip(test_words,test_tags):
    predicted = viterbi(transition_probabilities, prior_probabilities, tag_dist_words, test_set, tag_index_map)
    a, b = per_word_accuracy(predicted,test_correct)
    acc += a
    l += b
print(acc/l)
# estimate_prior_probabilities(train_tags, tag_index_map)
