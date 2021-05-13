import cv2
import os
import numpy as np
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import model_from_json
from pathlib import Path

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# session = tf.Session(config=config)


def load_data(dataset_train_path):
    train_dataset = []
    train_dataset_path = []
    for section in os.listdir(dataset_train_path):
        section_path = os.path.join(dataset_train_path, section)
        data_dict = os.listdir(section_path)
        for each in data_dict:
            each = os.path.join(section_path, each)
            train_dataset_path.append(each)
            img = cv2.imread(each, cv2.IMREAD_GRAYSCALE)
            train_dataset.append(img)
    return train_dataset, train_dataset_path


def main():
    # model_name = 'model_NB_masked'
    model_name = 'model_train_NOMASK'

    # Model reconstruction from JSON file
    with open(f'models/{model_name}.json', 'rb') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(f'models/{model_name}.h5')
    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Load train data
    train_dataset, train_dataset_path = load_data('data/split_dataset_NOMASK/train')

    # Test table
    test_table = [np.zeros((len(train_dataset), train_dataset[0].shape[0], train_dataset[0].shape[1], 1)) for i in range(2)]
    for idx, train_img in enumerate(train_dataset):
        test_table[1][idx, :, :, 0] = train_img

    # Load test data
    test_dataset, test_dataset_path = load_data('data/split_dataset_NOMASK/test')

    acc = 0
    n_acc = 0
    mean_time = 0
    for idx, test_image in enumerate(test_dataset):
        n_acc += 1

        test_image_path = test_dataset_path[idx]
        test_table[0][:, :, :, 0] = test_image

        start = time.time()

        pred = model.predict(test_table)

        end = time.time()
        mean_time += (end - start)

        best_idx = np.argmax(pred)
        top_image_path = train_dataset_path[best_idx]

        if str(Path(test_image_path).stem[0:2]) == str(Path(top_image_path).stem[0:2]):
            acc += 1

        print(f"processing: {n_acc}/{len(test_dataset)}, mean time: {mean_time / n_acc}, mean acc: {acc / n_acc}")
    print(f"\nMean accuracy: {acc / n_acc}\nMean time: {mean_time / n_acc}\nTotal time: {mean_time}\n")

    with open('logs/NB_test.txt', 'a') as file:
        file.write(str("Mean accuracy: " + str(acc / n_acc) + ", mean time: " + str(mean_time / n_acc) + ", total time: " + str(mean_time) + '\n'))


if __name__ == '__main__':
    main()
