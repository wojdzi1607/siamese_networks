import os
import cv2

mask = cv2.imread('mask_ok.png', 0)
mask = cv2.resize(mask, (160, 120))
mask = cv2.flip(mask, -1)

dataset_path = '/home/legion/PycharmProjects/mgr/NBProgram/dataset_NB_masked'

for section in os.listdir(dataset_path):
    section_path = os.path.join(dataset_path, section)
    data_dict = os.listdir(section_path)
    for each in data_dict:
        each = os.path.join(section_path, each)
        img = cv2.imread(each)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(each, masked_img)
