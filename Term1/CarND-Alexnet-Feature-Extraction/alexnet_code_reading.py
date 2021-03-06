import numpy as np
import tensorflow as tf

net_data = np.load('bvlc-alexnet.npy', encoding='latin1').item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding='VALID', group=1):

    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        # NOTE
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)

    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list[1:])


def AlexNet(features, feature_extract=False):

    # conv1
    k_h = 11
    k_w = 11
    c_o = 96
    s_h = 4
    s_w = 4
    conv1W = tf.Variable(net_data['conv1'][0])
    conv1b = tf.Variable(net_data['conv1'][1])
    conv1_in = \
    conv(features, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding='SAME', group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    redius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn1 = \
    tf.nn.local_response_normalization(
        conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

    # maxpool1
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool1 = \
    tf.nn.max_pool(
        lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1],
        padding=padding)

    # conv2
    k_h = 5
    k_w = 5
    c_o = 256
    s_h = 1
    s_w = 1
    group = 2
    conv2W = tf.Variable(net_data['conv2'][0])
    conv2b = tf.Variable(net_data['conv2'][1])
    conv2_in = \
    conv(
        maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w,
        padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = \
    tf.nn.local_response_normalization(
        conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

    # maxpool2
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool2 = \
    tf.nn.max_pool(
        lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3
    k_w = 3
    c_o = 384
    s_h = 1
    s_w = 1
    group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
