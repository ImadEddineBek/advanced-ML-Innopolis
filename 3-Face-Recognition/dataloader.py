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
            # if progress > 10:
            #     break
        return latent_space

    def get_test(self, weights):
        test = pd.read_csv("test_set.csv")
        anch = test.Anchor
        pos = test.Positive
        neg = test.Negative
        model = ModelInception(weights)
        im_size = 299
        load_image = lambda im_path: cv2.resize(cv2.imread(im_path), (im_size, im_size))
        latent_space = []
        print("[INFO] generate latent representations of test")
        progress = 0
        for a, p, n in zip(anch, pos, neg):
            print("progress %d/%d" % (progress, len(n)))
            progress += 1
            for ap, pp, np in zip(glob.glob(self.dataset + "/" + a), glob.glob(self.dataset + "/" + p),
                                  glob.glob(self.dataset + "/" + n)):
                img = load_image(ap)
                latent = model.get_latent_space(img)
                img2 = load_image(pp)
                latent2 = model.get_latent_space(img2)
                img3 = load_image(n)
                latent3 = model.get_latent_space(img3)
                latent_space.append((latent.flatten(), latent2.flatten(), latent3.flatten()))
        return anch, pos, neg
