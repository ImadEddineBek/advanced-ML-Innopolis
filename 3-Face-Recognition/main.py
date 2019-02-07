import argparse
import os
import cv2
import glob


def main(config):
    names = os.listdir(config.dataset)
    print("nb of people:", len(names))
    names2label = {name: label for name, label in zip(names, range(len(names)))}
    print(names2label)

    im_size = 299
    load_image = lambda im_path: cv2.resize(cv2.imread(im_path), (im_size, im_size))

    for name in names:
        for path in glob.glob(config.dataset+"/"+name+"/*.jpg"):
            img = load_image(path)
            print(img.shape, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights')
    parser.add_argument('dataset')
    config = parser.parse_args()
    print(config)
    main(config)
