import os
from PIL import Image
import random
from os import listdir
from shutil import copyfile

camera = "cam4"

data_train_directory="IXMAS_train_video/" + camera
data_test_directory="IXMAS_test_video/" + camera
data_val_directory="IXMAS_val_video/" + camera


des_train_directory="IXMAS_train_video_middle/" + camera
des_test_directory="IXMAS_teset_video_middle/" + camera
des_val_directory="IXMAS_val_video_middle/" + camera


text_train_file = open("video_train_list_middle_" + camera + ".txt", "w")
text_test_file = open("video_test_list_middle_" + camera + ".txt", "w")
text_val_file = open("video_val_list_middle_" + camera + ".txt", "w")

# with open('data.txt') as f:
#     print sum(1 for _ in f)

action_length = 20
video_num = 1
for folder_names in os.listdir(data_train_directory):
    actor_name = folder_names
    print actor_name
    imagesList = listdir(os.path.join(data_train_directory, actor_name))

    if not os.path.exists(os.path.join(des_train_directory, actor_name)):
        os.makedirs(os.path.join(des_train_directory, actor_name))
        text_train_file.write(
            os.path.join(des_train_directory, actor_name)
            + ' ' + str(int(actor_name[-1])) + ' \n')

    for image_name in imagesList[int(len(imagesList)/2)-10:int(len(imagesList)/2)+10]:
        # print (int(len(imagesList)/2))
        copyfile(os.path.join(data_train_directory, actor_name, image_name),
                 os.path.join(des_train_directory, actor_name, image_name))
        print image_name

# text_test_file = open("video_test_list_middle.txt", "w")

# with open('data.txt') as f:
#     print sum(1 for _ in f)

action_length = 20
video_num = 1
for folder_names in os.listdir(data_test_directory):
    actor_name = folder_names
    print actor_name
    imagesList = listdir(os.path.join(data_test_directory, actor_name))

    if not os.path.exists(os.path.join(des_test_directory, actor_name)):
        os.makedirs(os.path.join(des_test_directory, actor_name))
        text_test_file.write(
            os.path.join(des_test_directory, actor_name)
            + ' ' + str(int(actor_name[-1])) + ' \n')

    for image_name in imagesList[int(len(imagesList)/2)-10:int(len(imagesList)/2)+10]:
        # print (int(len(imagesList)/2))
        copyfile(os.path.join(data_test_directory, actor_name, image_name),
                 os.path.join(des_test_directory, actor_name, image_name))
        print image_name


video_num = 1
for folder_names in os.listdir(data_val_directory):
    actor_name = folder_names
    print actor_name
    imagesList = listdir(os.path.join(data_val_directory, actor_name))

    if not os.path.exists(os.path.join(des_val_directory, actor_name)):
        os.makedirs(os.path.join(des_val_directory, actor_name))
        text_val_file.write(
            os.path.join(des_val_directory, actor_name)
            + ' ' + str(int(actor_name[-1])) + ' \n')

    for image_name in imagesList[int(len(imagesList)/2)-10:int(len(imagesList)/2)+10]:
        # print (int(len(imagesList)/2))
        copyfile(os.path.join(data_val_directory, actor_name, image_name),
                 os.path.join(des_val_directory, actor_name, image_name))
        print image_name
text_train_file.close()
text_test_file.close()
text_val_file.close()


