import numpy as np
import cv2
from random import shuffle
import os
from os import listdir
from random import shuffle
import re

class Dataset:
    def __init__(self, train_list, test_list, val_list, cam, n_classes, shuffleType, seqLength, CNN_type):
        # Load training images (path) and labels
        self.cam = cam
        self.seqLength = seqLength
        with open(train_list) as f:
            lines = f.readlines()
            self.train_image = []
            self.train_label = []
            if shuffleType == 'normal':
                shuffle(lines)
                for l in lines:
                    items = l.split()
                    self.train_image.append(items[0])
                    self.train_label.append(int(items[1]))
            elif shuffleType == 'seq':
                num_seq = len(lines)
                shuffle(lines)
                # ind = np.random.permutation(num_seq)
                for i in range(num_seq):
                    items = lines[i].split()
                    self.train_image.append(items[0])
                    self.train_label.append(int(items[1]))

        with open(test_list) as f:
            lines = f.readlines()
            self.test_image = []
            self.test_label = []
            if shuffleType == 'normal':
                shuffle(lines)
                for l in lines:
                    items = l.split()
                    self.test_image.append(items[0])
                    self.test_label.append(int(items[1]))
            elif shuffleType == 'seq':
                num_seq = len(lines) / seqLength
                # ind = np.random.permutation(num_seq)
                for i in range(num_seq):
                    for jj in range(seqLength):
                        items = lines[i*seqLength + jj].split()
                        self.test_image.append(items[0])
                        self.test_label.append(int(items[1]))

        with open(val_list) as f:
            lines = f.readlines()
            self.val_image = []
            self.val_label = []
            if shuffleType == 'normal':
                shuffle(lines)
                for l in lines:
                    items = l.split()
                    self.val_image.append(items[0])
                    self.val_label.append(int(items[1]))
            elif shuffleType == 'seq':
                num_seq = len(lines) / seqLength
                # ind = np.random.permutation(num_seq)
                for i in range(num_seq):
                    for jj in range(seqLength):
                        items = lines[i * seqLength + jj].split()
                        self.val_image.append(items[0])
                        self.val_label.append(int(items[1]))



        # Load testing images (path) and labels
        #with open(test_list) as f:
        #    lines = f.readlines()
        #    self.test_image = []
        #    self.test_label = []
        #    for l in lines:
        #        items = l.split()
        #        self.test_image.append(items[0])
        #        self.test_label.append(int(items[1]))

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.val_ptr = 0
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.val_size = len(self.val_label)
        if CNN_type == 'vgg':
            self.crop_size = 224
        else:
            self.crop_size = 112
        self.scale_size = 112
        self.mean = np.array([122., 104., 100.])
        # self.mean = np.array([104., 117., 124.])
        self.n_classes = n_classes

    def next_batch_cross(self, batch_size, phase):
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_image[self.train_ptr:self.train_ptr + batch_size]
                target_paths = list()
                for i in xrange(len(paths)):
                    print 'path 111:' + paths[i]
                    target_paths.append("IXMAS_train_video_middle/cam" + str(self.cam) + "/" + paths[i][30:])
                # path_f = paths[10::11]
                # print(path_f)
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                paths = self.train_image[self.train_ptr:] + self.train_image[:new_ptr]
                target_paths = list()
                for i in xrange(len(paths)):
                    target_paths.append("IXMAS_train_video_middle/cam" + str(self.cam) + "/" + paths[i][30:])
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                target_paths = list()
                for i in xrange(len(paths)):
                    target_paths.append("IXMAS_teset_video_middle/cam" + str(self.cam) + "/" + paths[i][30:])
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                target_paths = list()
                for i in xrange(len(paths)):
                    target_paths.append("IXMAS_teset_video_middle/cam" + str(self.cam) + "/" + paths[i][30:])
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        elif phase == 'val':
            if self.val_ptr + batch_size < self.val_size:
                paths = self.val_image[self.val_ptr:self.val_ptr + batch_size]
                target_paths = list()
                for i in xrange(len(paths)):
                    target_paths.append("IXMAS_val_video_middle/cam/" + paths[i][28:])
                labels = self.val_label[self.val_ptr:self.val_ptr + batch_size]
                self.val_ptr += batch_size
            else:
                new_ptr = (self.val_ptr + batch_size) % self.val_size
                paths = self.val_image[self.val_ptr:] + self.val_image[:new_ptr]
                target_paths = list()
                for i in xrange(len(paths)):
                    target_paths.append("IXMAS_val_video_middle/cam" + str(self.cam) + "/" + paths[i][28:])
                labels = self.val_label[self.val_ptr:] + self.val_label[:new_ptr]
                self.val_ptr = new_ptr
        videos = np.ndarray([batch_size, self.seqLength, self.crop_size, self.crop_size, 3])
        for i in xrange(len(paths)):
            image_list = listdir(paths[i])
            j = 0
            for image_nm in image_list:
                img = cv2.imread(os.path.join(paths[i], image_nm))
                h, w, c = img.shape
                assert c == 3

                img = cv2.resize(img, (self.scale_size, self.scale_size))
                img = img.astype(np.float32)
                # img -= self.mean
                img = img / 255
                shift = int((self.scale_size - self.crop_size) / 2)
                img_crop = img[shift:shift + self.crop_size, shift:shift + self.crop_size, :]
                videos[i, j] = img_crop
                j += 1
        videos_target = np.ndarray([batch_size, self.seqLength, self.crop_size, self.crop_size, 3])
        for i in xrange(len(target_paths)):
            # print target_paths[i]
            image_list = listdir(target_paths[i])
            j = 0
            for image_nm in image_list:
                img = cv2.imread(os.path.join(target_paths[i], image_nm))
                h, w, c = img.shape
                assert c == 3

                img = cv2.resize(img, (self.scale_size, self.scale_size))
                img = img.astype(np.float32)
                # img -= self.mean
                img = img / 255
                shift = int((self.scale_size - self.crop_size) / 2)
                img_crop = img[shift:shift + self.crop_size, shift:shift + self.crop_size, :]
                videos_target[i, j] = img_crop
                j += 1

        # Expand labels
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in xrange(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        output_label = np.empty((batch_size, 1), int)
        for i in xrange(len(labels)):
            output_label[i] = labels[i]

        return videos, one_hot_labels, output_label, videos_target

    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_image[self.train_ptr:self.train_ptr + batch_size]
                # path_f = paths[10::11]
                # print(path_f)
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                paths = self.train_image[self.train_ptr:] + self.train_image[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None

        videos = np.ndarray([batch_size, self.seqLength, self.crop_size, self.crop_size, 3])
        for i in xrange(len(paths)):
            image_list = listdir(paths[i])
            j = 0
            for image_nm in image_list:

                img = cv2.imread(os.path.join(paths[i], image_nm))
                h, w, c = img.shape
                assert c == 3

                img = cv2.resize(img, (self.scale_size, self.scale_size))
                img = img.astype(np.float32)
                # img -= self.mean
                img = img / 255
                shift = int((self.scale_size - self.crop_size) / 2)
                img_crop = img[shift:shift + self.crop_size, shift:shift + self.crop_size, :]
                videos[i, j] = img_crop
                j += 1

        # img_count = 0
        # OpticalFlow_video = np.ndarray([batch_size * (self.seqLength - 1), self.crop_size * self.crop_size])
        # for i in xrange(len(paths)):
        #     for _, _, images in os.walk("IXMAS_optical_flow/" + paths[i][12:]):
        #         # images = sorted(images, key=numericalSort)
        #         for img_name in images:
        #             print paths[i]
        #             # print(os.path.join("IXMAS_optical_flow/" + paths[i][12:], img_name))
        #             img = cv2.imread(os.path.join("IXMAS_optical_flow/" + paths[i][12:], img_name))
        #             # print img.shape
        #             # img = img.astype(np.float32) / 255
        #             img_tmp = img[:, :, 0]
        #             OpticalFlow_video[img_count, :] = img_tmp.flatten()
        #             img_count += 1


        # Expand labels
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in xrange(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        output_label = np.zeros((batch_size, 1))
        for i in xrange(len(labels)):
            output_label[i] = labels[i]

        return videos, one_hot_labels, output_label

    @staticmethod
    def test_video(list, batch_size, n_classes):
        seqLength = 20
        crop_size = 112
        scale_size = 112
        mean = np.array([122., 104., 100.])

        with open(list) as f:
            lines = f.readlines()
            train_image = []
            train_label = []

        num_seq = len(lines)
        # ind = np.random.permutation(num_seq)
        for i in range(num_seq):
            items = lines[i].split()
            train_image.append(items[0])
            train_label.append(int(items[1]))

        paths = train_image
        videos = np.ndarray([batch_size, seqLength, crop_size, crop_size, 3])
        for i in xrange(len(paths)):
            image_list = listdir(paths[i])
            j = 0
            for image_nm in image_list:
                img = cv2.imread(os.path.join(paths[i], image_nm))
                h, w, c = img.shape
                assert c == 3

                img = cv2.resize(img, (scale_size, scale_size))
                img = img.astype(np.float32)
                # img -= self.mean
                img = img / 255
                shift = int((scale_size - crop_size) / 2)
                img_crop = img[shift:shift + crop_size, shift:shift + crop_size, :]
                videos[i, j] = img_crop
                j += 1

        one_hot_labels = np.zeros((batch_size, n_classes))
        for i in xrange(len(train_label)):
            one_hot_labels[i][train_label[i]] = 1
        return videos, one_hot_labels


