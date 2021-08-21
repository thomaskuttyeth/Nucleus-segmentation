
import os
import numpy as np
import random
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread, imshow
import matplotlib.pyplot as plt


project_dir = 'C:\\Users\\ASUS\\Desktop\\unets'
os.chdir(project_dir)
os.listdir(project_dir)


class Dataloader:
    def __init__(self, project_dir, img_width, img_height, img_channels, train_path, test_path):
        self.project_dir = project_dir
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.train_path = train_path
        self.test_path = test_path
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def preprocess(self):
        os.chdir(project_dir)
        train_ids = next(os.walk(self.train_path))[1]
        test_ids = next(os.walk(self.test_path))[1]

        self.x_train = np.zeros(
            (len(train_ids), self.img_height, self.img_width, self.img_channels), dtype=np.uint8)
        self.y_train = np.zeros(
            (len(train_ids), self.img_height, self.img_width, 1), dtype=np.bool)

        # preprocessing train images and the masks
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = self.train_path+id_
            img = imread(path+'/images/'+id_+'.png')[:, :, :self.img_channels]
            img = resize(img, (self.img_height, self.img_width),
                         mode='constant', preserve_range=True)
            self.x_train[n] = img

            mask = np.zeros(
                (self.img_height, self.img_width, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path+'/masks/'+mask_file)
                mask_ = np.expand_dims(resize(
                    mask_, (self.img_height, self.img_width), mode='constant', preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            self.y_train[n] = mask

        # test images
        self.x_test = np.zeros((len(test_ids), self.img_height, self.img_width,
                                self.img_channels), dtype=np.uint8)
        sizes_test = []
        print('Resizing test images')
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = self.test_path + id_
            img = imread(path + '/images/' + id_ +
                         '.png')[:, :, :self.img_channels]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (self.img_height, self.img_width),
                         mode='constant', preserve_range=True)
            self.x_test[n] = img
