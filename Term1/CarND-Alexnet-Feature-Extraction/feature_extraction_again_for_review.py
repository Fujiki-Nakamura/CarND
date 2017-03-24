"""Reviewing feature extraction without copying and pasting the former codes."""

import time

import numpy as np
import pandas as pd
from scipy.misc import imread
import tensorflow as tf

from alexnet import AlexNet


# parameters
nb_classes = 43

mean = 0.0
stddev = 1e-2

sign_names = pd.read_csv('signnames.csv')


def AlexNetFeatureExtractor(features):
    fc7 = AlexNet(features, feature_extract=True)

    shape = (fc7.get_shape().as_list()[-1], nb_classes)
    fc8W = tf.Variable(tf.truncated_normal(shape, mean, stddev))
    fc8b = tf.Variable(tf.zeros(nb_classes))
    logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    probs = tf.nn.softmax(logits)

    return probs


def main():
    features = tf.placeholder(tf.float32, (None, 32, 32, 3))
    resized = tf.image.resize_images(features, (227, 227))
    probs = AlexNetFeatureExtractor(resized)
    # init_op = tf.global_variables_initializer()
    init_op = tf.initialize_all_variables()

    # input data
    # NOTE: I copied and pasted these codes.
    im1 = imread("construction.jpg").astype(np.float32)
    im1 = im1 - np.mean(im1)
    im2 = imread("stop.jpg").astype(np.float32)
    im2 = im2 - np.mean(im2)

    t0 = time.time()
    with tf.Session() as sess:
        sess.run(init_op)
        output = sess.run(probs, feed_dict={features: [im1, im2]})

    # check the output
    # NOTE: I copied and pasted these codes.
    for input_im_ind in range(output.shape[0]):
        inds = np.argsort(output)[input_im_ind, :]
        print("Image", input_im_ind)
        for i in range(5):
            print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))

    print('Time {:.3f} seconds.'.format(time.time() - t0))


if __name__ == '__main__':
    main()
