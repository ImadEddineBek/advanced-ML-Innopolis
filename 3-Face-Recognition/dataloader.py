import argparse
import os
import cv2
import glob
import numpy as np
from model import ModelInception
import pandas as pd


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.names = os.listdir(self.dataset)
        # print("nb of people:", len(self.names))
        self.names2label = {name: label for name, label in zip(self.names, range(len(self.names)))}
        # print(self.names2label)

    def get_latent(self, weights):
        model = ModelInception(weights)
        im_size = 299
        load_image = lambda im_path: cv2.resize(cv2.imread(im_path), (im_size, im_size))
        latent_space = []
        print("[INFO] generate latent representations of images")
        progress = 0
        for name in self.names:
            print("progress %d/%d" % (progress, len(self.names)))
            progress += 1
            for path in glob.glob(self.dataset + "/" + name + "/*.jpg"):
                img = load_image(path)
                latent = model.get_latent_space(img)
                latent_space.append((latent.flatten(), self.names2label[name]))
            if progress > 10:
                break
        return latent_space

    def get_test(self):
        test = pd.read_csv("test_set.csv")
        anch = test.Anchor
        pos = test.Positive
        neg = test.Negative
        print(anch, pos, neg)
