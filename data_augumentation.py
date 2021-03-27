# -*- coding: utf-8 -*-
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
import cv2
from scipy import ndimage


def data_augumentation(input, output, size, ex, ran):
    files = glob.glob(input + '/*.' + ex)
    if os.path.isdir(output) == False:
        os.mkdir(output)

    for i, file in enumerate(files):
        img = load_img(file)
        img = img.resize((size, size))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        datagen = ImageDataGenerator(
            channel_shift_range=50,
            rotation_range=180,
            zoom_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            )

        g = datagen.flow(x, batch_size=1, save_to_dir=output, save_prefix='img', save_format='jpg')
        for i in range(ran):
            batch = g.next()

def main():
    parser = argparse.ArgumentParser(description='output mixed images')
    parser.add_argument('--size', '-s', type=int, default=256, help='size to resize images')
    parser.add_argument('--out', '-o', default='./', help='Path to the folder containing images')
    parser.add_argument('--input', '-i', default='./', help='Path to the folder containing images')
    parser.add_argument('--range', '-r', default=9,type = int, help='data_augumentation range')
    parser.add_argument('--extension', '-e', default='jpg', help='File extension to images')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data_augumentation(args.input, args.out, args.size, args.extension, args.range)


if __name__ == '__main__':
    main()