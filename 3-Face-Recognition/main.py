import argparse
import os
import cv2
import glob
import numpy as np
from model import  Model

def main(config):
    names = os.listdir(config.dataset)
    # print("nb of people:", len(names))
    names2label = {name: label for name, label in zip(names, range(len(names)))}
    # print(names2label)
    model = Model(config.weights)
    im_size = 299
    load_image = lambda im_path: cv2.resize(cv2.imread(im_path), (im_size, im_size))
    latent_space = []
    print("[INFO] generate latent representations of images")
    progress = 0
    for name in names:
        print("progress %d/%d" % (progress, len(names)))
        progress += 1
        for path in glob.glob(config.dataset + "/" + name + "/*.jpg"):
            img = load_image(path)
            latent = model.get_latent_space(img)
            latent_space.append([latent.flatten(), names2label[name]])
        if progress > 10:
            break
    latent_space = np.array(latent_space)
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
