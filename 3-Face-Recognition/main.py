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
    data_loader.get_test()
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
    model = ModelSiamese()
    model.train(X, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('dataset')
    config = parser.parse_args()
    print(config)
    main(config)
