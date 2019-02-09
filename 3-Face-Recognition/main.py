import argparse
import os
import cv2
import glob
import numpy as np

from dataloader import DataLoader

def main(config):
    data_loader = DataLoader(config.dataset)
    latent_space = data_loader.get_latent(config.weights)
    print(latent_space[0].shape)
    print(latent_space[0])
    print(latent_space.shape)
    np.save("latent_space.npz", latent_space)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('dataset')
    config = parser.parse_args()
    print(config)
    main(config)
