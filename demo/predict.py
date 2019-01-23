import os
import argparse
import tensorflow as tf
import numpy as np

LEARNING_RATE = 4e-4
MODEL_DIR = './model/'
MODEL = os.path.join(MODEL_DIR, 'layer5_4e4_3_nomlp.ckpt')


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./img/',
                    help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


meta = {"num_per_image": 4, "width": 200, "height": 70, "label_size": 26, "label_choices": "abcdefghijklmnopqrstuvwxyz"}

LABEL_SIZE = meta['label_size']
NUM_PER_IMAGE = meta['num_per_image']
IMAGE_HEIGHT = meta['height']
IMAGE_WIDTH = meta['width']
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# variable in the graph for input data
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name='x')
    y_ = tf.placeholder(tf.float32, [None, NUM_PER_IMAGE * LABEL_SIZE], name='y_')

    # must be 4-D with shape `[batch_size, height, width, channels]`
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    tf.summary.image('input', x_image, max_outputs=LABEL_SIZE)

# define the model
with tf.name_scope('convolution-layer-1'):
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('convolution-layer-2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('convolution-layer-3'):
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

with tf.name_scope('convolution-layer-4'):
    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

with tf.name_scope('convolution-layer-5'):
    W_conv5 = weight_variable([3, 3, 256, 512])
    b_conv5 = bias_variable([512])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)

with tf.name_scope('densely-connected'):
    size = int(h_pool5.get_shape()[1] * h_pool5.get_shape()[2] * h_pool5.get_shape()[3])
    W_fc1 = weight_variable([size, 4096])
    b_fc1 = bias_variable([4096])

    h_pool5_flat = tf.reshape(h_pool5, [-1, size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
    # To reduce overfitting, we will apply dropout before the readout layer
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('readout'):
    W_fc2 = weight_variable([4096, NUM_PER_IMAGE * LABEL_SIZE])
    b_fc2 = bias_variable([NUM_PER_IMAGE * LABEL_SIZE])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('reshape'):
    y_expect_reshaped = tf.reshape(y_, [-1, NUM_PER_IMAGE, LABEL_SIZE])
    y_got_reshaped = tf.reshape(y_conv, [-1, NUM_PER_IMAGE, LABEL_SIZE])

# Define loss and optimizer
# Returns:
# A 1-D `Tensor` of length `batch_size`
# of the same type as `logits` with the softmax cross entropy loss.
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_expect_reshaped, logits=y_got_reshaped))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    variable_summaries(cross_entropy)

# forword prop
with tf.name_scope('forword-prop'):
    predict = tf.argmax(y_got_reshaped, axis=2, name='predict')
    expect = tf.argmax(y_expect_reshaped, axis=2)

# evaluate accuracy
with tf.name_scope('evaluate_accuracy'):
    correct_prediction = tf.equal(predict, expect)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    variable_summaries(accuracy)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, MODEL)



def getToken(img):
    y = sess.run(predict, feed_dict={x: img.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH)), keep_prob: 1.0})[0]
    res = ''.join([meta['label_choices'][i] for i in y])
    return res
