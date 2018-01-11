import os
from PIL import Image
import numpy as np
import cv2
import re

data_directory="IXMAS_teset_video_middle/cam0/"

des_directory="IXMAS_optical_flow/cam0"

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


for folder_names in os.listdir(data_directory):
    video_name = folder_names
    image_names = os.listdir(os.path.join(data_directory, video_name))
    if not os.path.exists(os.path.join(des_directory, video_name)):
        os.makedirs(os.path.join(des_directory, video_name))

    count = 0
    print video_name

    image_names = sorted(image_names, key=numericalSort)
    for image in image_names:
        print image
        if count == 0:
            pre_img = cv2.imread(os.path.join(data_directory, video_name, image), 0)
            count += 1
        else:
            img = cv2.imread(os.path.join(data_directory, video_name, image))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hsv = np.zeros_like(img)
            hsv[..., 1] = 255

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(pre_img, img, 0.5, 1, 3, 15, 3, 5, 1)


            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            pre_img = img
            # flow = flow.astype(np.uint8)
            cv2.imwrite(os.path.join(des_directory, video_name, image), bgr)
            # flow.save(os.path.join(des_directory, video_name, image))

        # print img
    # for subdir, dirs, images in os.walk(data_directory):




