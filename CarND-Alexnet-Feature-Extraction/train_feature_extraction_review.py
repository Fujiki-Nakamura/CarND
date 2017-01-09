"""Reviewing the exercise of training feature extraction without referring to the former codes."""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from alexnet import AlexNet


nb_classes = 43

mean = 0.0
stddev = 1e-2

resize_to = (227, 227)

learning_rate = 1e-3

epochs = 10
batch_size = 128


# TODO: make this function work
def AlexNetFeatureExtractor(features, labels):
    fc7 = AlexNet(features, feature_extract=True)
    fc7 = tf.stop_gradient(fc7)

    shape = (fc7.get_shape().as_list()[-1], nb_classes)
    fc8W = tf.Variable(tf.truncated_normal(shape, mean, stddev))
    fc8b = tf.Variable(tf.zeros(nb_classes))
    logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    # probs = tf.nn.softmax(logits)
    preds = tf.argmax(logits, axis=1)
    trues = tf.argmax(labels, axis=1)
    acc_op = tf.reduce_mean(tf.equal(preds, trues), tf.float32)
    loss_op = tf.softmax_cross_entropy_with_logits(logits, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, var_list=[fc8W, fc8b])

    return train_op, acc_op, loss_op


def load_data():
    with open('train.p', 'rb') as f:
        data = pickle.load(f)

    return data['features'], data['labels']


def main():
    # tensors
    features = tf.placeholder(tf.float32, (None, 32, 32, 3))
    resized = tf.image.resize_images(features, (227, 227))
    labels = tf.placeholder(tf.int32, (None))
    labels_one_hot = tf.one_hot(labels, nb_classes)
    # TODO: make this function work above
    # train_op, acc_op, loss_op = AlexNetFeatureExtractor(resized, labels)

    # feature extraction from AlexNet
    fc7 = AlexNet(resized, feature_extract=True)
    fc7 = tf.stop_gradient(fc7)

    shape = (fc7.get_shape().as_list()[-1], nb_classes)
    fc8W = tf.Variable(tf.truncated_normal(shape, mean, stddev))
    fc8b = tf.Variable(tf.zeros(nb_classes))
    logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    # probs = tf.nn.softmax(logits)
    preds = tf.argmax(logits, axis=1)
    trues = tf.argmax(labels_one_hot, axis=1)
    acc_op = tf.reduce_mean(tf.cast(tf.equal(preds, trues), tf.float32))
    loss_op = tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=[fc8W, fc8b])

    init_op = tf.global_variables_initializer()

    # load the data
    X, y = load_data()
    # split the data into train and val
    X_train, X_val, y_train, y_val = \
    train_test_split(X, y, test_size=0.25, random_state=0)

    def evaluate(sess, X_eval, y_eval):
        nb_samples = len(X)
        total_loss = 0
        total_acc = 0

        for start_i in range(0, nb_samples, batch_size):
            end_i = start_i + batch_size
            X_eval_batch = X_eval[start_i:end_i]
            y_eval_batch = y_eval[start_i:end_i]

            loss, accuracy = \
            sess.run([loss_op, acc_op], feed_dict={features: X_eval_batch, labels: y_eval_batch})
            total_loss += (loss * X_eval_batch.shape[0])
            total_loss += (accuracy * X_eval_batch.shape[0])

        return total_loss / nb_samples, total_acc / nb_samples

    with tf.Session() as sess:
        sess.run(init_op)
        nb_samples = len(X_train)

        # TODO: write more concisely if possible
        for epoch_i in range(epochs):
            print('Epoch {}'.format(epoch_i + 1))
            X_train, y_train_one_hot = shuffle(X_train, y_train)
            t0 = time.time()

            for start_i in range(0, nb_samples, batch_size):
                end_i = start_i + batch_size
                X_batch = X_train[start_i:end_i]
                y_batch = y_train_one_hot[start_i:end_i]
                sess.run(train_op, feed_dict={features: X_batch, labels: y_batch})

            val_loss, val_acc = evaluate(sess, X_val, y_val)
            # show loss and accuracy for each epoch
            print('Time {:.3f} seconds'.format(epoch_i, time.time() - t0))
            print('Validation Loss = ', val_loss)
            print('Validation Accuracy = ', val_acc)


if __name__ == '__main__':
    main()
