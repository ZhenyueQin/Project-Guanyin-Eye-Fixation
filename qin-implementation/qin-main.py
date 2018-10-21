from __future__ import division
import os
from config import *
from keras.layers import Input
from keras.models import Model
from models import sam_vgg, sam_resnet


if __name__ == '__main__':
    output_folder = 'predictions/'
    imgs_test_path = '../sample_images'
    print(os.path.dirname(os.path.abspath(__file__)))
    file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    file_names.sort()
    nb_imgs_test = len(file_names)

    phase = 'test'
    x = Input((3, shape_r, shape_c))
    x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

    m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))

    if nb_imgs_test % b_s != 0:
        print(
            "The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        exit()