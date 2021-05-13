import tensorflow as tf
import keras.backend as K

import numpy as np
from keras.models import Model, Sequential
from keras.regularizers import l2
from modified_sgd import Modified_SGD
from keras.optimizers import Adam, SGD
from omniglot_loader import OmniglotLoader
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
import cv2
import time
import keras
start = time.time()
print("hello")
end = time.time()
print(end - start)
normal_rv = tf.Variable(tf.truncated_normal([4,4096],stddev = 0.1))
normal_rv2 = tf.Variable(tf.truncated_normal([4,4096],stddev = 0.1))

# L1 distance
start = time.time()
l1_distance_layer = Lambda(
    lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1, keepdims=True))
l1_distance = l1_distance_layer([normal_rv, normal_rv2])
prediction = Dense(units=1, activation='sigmoid')(l1_distance)
end = time.time()
print(f'Calculate L1 distance with Dense: {end - start}')

# L2 distance
l2_distance_layer = Lambda(
    lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1, keepdims=True)))
l2_distance = l2_distance_layer([normal_rv, normal_rv2])
prediction2 = Dense(units=1, activation='sigmoid')(l2_distance)

# Binaries
start = time.time()
binary_layer = Lambda(
    lambda tensor: K.relu(((tensor - K.min(tensor)) / (K.max(tensor) - K.min(tensor)) - 0.5)*100000, max_value=1.0)
)
binary_normal_rv = binary_layer(normal_rv)
binary_normal_rv2 = binary_layer(normal_rv2)

# Hamming distance
hamming_distance_layer = Lambda(
    lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1, keepdims=True))
hamming_distance = hamming_distance_layer([binary_normal_rv, binary_normal_rv2])
prediction3 = Dense(units=1, activation='sigmoid')(hamming_distance)
end = time.time()
print(f'Calculate binarization and Hamming distance with Dense: {end - start}')

# initialize the variable
init_op = tf.initialize_all_variables()

# run the graph
with tf.Session() as sess:
    sess.run(init_op)   # execute init_op

    normal_rv = sess.run(normal_rv)
    binary_normal_rv = sess.run(binary_normal_rv)
    l1_distance = sess.run(l1_distance)
    prediction2 = sess.run(prediction2)

    l2_distance = sess.run(l2_distance)
    prediction = sess.run(prediction)

    hamming_distance = sess.run(hamming_distance)
    prediction3 = sess.run(prediction3)

# print(normal_rv, len(normal_rv))
# print('\n')
# print(binary_normal_rv, len(binary_normal_rv))

print('\nL1 distance:')
print(l1_distance)
print('\nL1 distance > Dense:')
print(prediction)

# print('\nL2 distance:')
# print(l2_distance)
# print('\nL2 distance > Dense:')
# print(prediction2)

print('\nHamming distance:')
print(hamming_distance)
# print(np.sum(hamming_distance, axis=-1, keepdims=True))
print('\nHamming distance > Dense:')
print(prediction3)
