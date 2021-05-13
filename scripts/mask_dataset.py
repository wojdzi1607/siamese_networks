import os
import cv2

mask = cv2.imread('mask_ok.png', 0)

dataset_path = '/home/legion/PycharmProjects/mgr/NBProgram/data/dataset_640x480_masked/Sekcje'

for section in os.listdir(dataset_path):
    section_path = os.path.join(dataset_path, section)
    data_dict = os.listdir(section_path)
    for each in data_dict:
        each = os.path.join(section_path, each)
        img = cv2.imread(each)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        masked_img = masked_img[:, 52:573]
        masked_img = cv2.resize(masked_img, (int(masked_img.shape[1] * 3 / 4), int(masked_img.shape[0] * 3 / 4)))
        # cv2.imshow(' ', masked_img)
        # cv2.waitKey()
        cv2.imwrite(each, masked_img)

# for section in os.listdir(dataset_path):
#     section_path = os.path.join(dataset_path, section)
#     data_dict = os.listdir(section_path)
#     for each in data_dict:
#         each = os.path.join(section_path, each)
#         img = cv2.imread(each)
#         img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
#         cv2.imwrite(each, img)

