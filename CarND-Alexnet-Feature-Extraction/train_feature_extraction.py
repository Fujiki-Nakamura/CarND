import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from alexnet import AlexNet


nb_classes = 43
resize_to = (227, 227)

epochs = 10
batch_size = 128

mean = 0.0
stddev = 1e-2

learning_rate = 1e-3

# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
    train = pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = \
train_test_split(
    train['features'], train['labels'],
    test_size=0.25, random_state=0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, resize_to)

y = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(y, nb_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)

fc8W = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=stddev))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

preds = tf.argmax(logits, axis=1)
trues = tf.argmax(one_hot_y, axis=1)
correct_prediction = tf.equal(preds, trues)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


def evaluate(X_data, y_data):
    total_loss = 0
    total_accuracy = 0

    num_examples = len(X_data)
    sess = tf.get_default_session()

    for offset in range(0, num_examples, batch_size):
        X_batch = X_train[offset: offset + batch_size]
        y_batch = y_train[offset: offset + batch_size]
        loss, accuracy = \
            sess.run([loss_operation, accuracy_operation],
                     feed_dict={x: X_batch, y: y_batch})
        total_loss += (loss * len(X_batch))
        total_accuracy += (accuracy * len(X_batch))

    return total_loss / num_examples, total_accuracy / num_examples


init = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)

    for i in range(epochs):
        t0 = time.time()

        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            X_batch, y_batch = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: X_batch, y: y_batch})

        val_loss, val_acc = evaluate(X_val, y_val)
        print("Epoch", i + 1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)

    saver.save(sess, './model/alexnet_feature_extractor')
    print("Model saved")
