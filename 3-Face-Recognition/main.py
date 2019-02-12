import argparse
import sys

import os
import cv2
import glob
import numpy as np
import pickle as pkl
from dataloader import DataLoader
from model import ModelSiamese
import tensorflow as tf


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    data_loader = DataLoader(config.dataset)
    try:
        test = pkl.load(open("test.p", "rb"))
        print('loaded saved latent test')
    except:
        test = data_loader.get_test(weights=config.weights)
        pkl.dump(test, open("test.p", "wb"))
        sys.exit(-1)

    try:
        latent_space = pkl.load(open("latent_space.p", "rb"))
        print('loaded saved latent space')
    except:
        latent_space = data_loader.get_latent(config.weights)
        pkl.dump(latent_space, open("latent_space.p", "wb"))
        sys.exit(-1)

    X, y = zip(*latent_space)
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    a, p, n = map(list, zip(*test))
    a = np.array(a)
    p = np.array(p)
    n = np.array(n)
    print(a.shape, p.shape, n.shape)

    model = ModelSiamese()
    model.train(X, y, a, p, n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('dataset')
    config = parser.parse_args()
    print(config)
    main(config)
