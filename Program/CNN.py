
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from image_handler import *
from rating_handler import *

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def cnn_model(X, img_height, img_width, img_channels, img_classes):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    # x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps 3 channel RGB image to 96 feature maps of size 7x7.
    # W_conv1 = weight_variable([5, 5, 1, 32])
    W_conv1 = weight_variable([7, 7, 3, 96])
    b_conv1 = bias_variable([96])
    h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)
    print("Size after first downsampling: ", h_pool1.shape)

    # Second convolutional layer -- maps 96 feature maps to 256 of size 5x5.
    W_conv2 = weight_variable([5, 5, 96, 256])
    b_conv2 = bias_variable([256])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)
    print("Size after second downsampling: ", h_pool2.shape)

    # Third convolutional layer -- maps 256 to 384 filters of size 3x3.
    W_conv3 = weight_variable([3, 3, 256, 384])
    b_conv3 = bias_variable([384])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # Third pooling layer.
    h_pool3 = max_pool_2x2(h_conv3)
    print("Size after third downsampling: ", h_pool3.shape)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.

    # 40*30 -> 20*15 -> 10*8 just check sizes after pooling. Learn how to calculate the size sometime.
    W_fc1 = weight_variable([int(h_pool3.get_shape()[1])*int(h_pool3.get_shape()[2])*384, 1024])
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, int(h_pool3.get_shape()[1])*int(h_pool3.get_shape()[2])*384])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, img_classes])
    b_fc2 = bias_variable([img_classes])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # print(y_conv.get_shape())
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def save_model(saver, sess, file_name='model.ckpt'):
    # Prepare for saving the model after training
    current_dir = os.path.dirname(__file__)
    file_path = '../Models/{}'.format(file_name)
    file_rel_path = os.path.join(current_dir, file_path)

    save_path = saver.save(sess, file_rel_path)
    print("Model saved in file: %s" % save_path)

def get_model_path(file_name='model.ckpt'):
    saver = tf.train.Saver()

    current_dir = os.path.dirname(__file__)
    file_path = '../Models/{}'.format(file_name)
    file_rel_path = os.path.join(current_dir, file_path)

    return file_rel_path

def main(_):

    #import data
    all_images = get_all_resized_images(dim1=32,dim2=32)
    all_ratings = get_all_ratings(factor=2)
    one_hot_ratings = one_hot_encode(all_ratings, n_classes=10)

    # 80% training
    training_X = all_images[:400]
    test_X = all_images[400:]

    # 20% testing
    training_Y = one_hot_ratings[:400]
    test_Y = one_hot_ratings[400:]

    # define sizes (used to create matrixes of correct sizes)
    img_height = training_X[0].shape[0]
    img_width = training_X[0].shape[1]
    img_channels = training_X[0].shape[2]
    img_classes = training_Y[0].shape[0]
    print("Dimensions: ", img_height, img_width, img_channels, img_classes)

    # Define input
    x = tf.placeholder(tf.float32, [None, img_height, img_width, img_channels])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, img_classes])

    # Build the graph for the deep net
    y_conv, keep_prob = cnn_model(x, img_height, img_width, img_channels, img_classes)

    # print(y_.get_shape(), y_conv.get_shape())
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predict_operation = tf.argmax(y_conv, axis=1)

    # For saving and restoring the model
    saver = tf.train.Saver()


    batch_size = 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        true_targets = np.argmax(test_Y, axis=1)
        done = False
        for i in range(100):

            if done == True:
                break

            pred_targets = np.array([])
            true_targets = np.argmax(test_Y, axis=1)
            for start, end in zip(range(0, len(test_X), batch_size), range(batch_size, len(test_X) + batch_size, batch_size)):
                batch_pred_targets = sess.run(predict_operation, feed_dict={
                    x: test_X[start:end],
                    keep_prob: 1.0
                })
                pred_targets = np.hstack((pred_targets, np.array(batch_pred_targets)))

            pred_targets = pred_targets.astype(int)
            tot_correct = 0
            for t in range(len(pred_targets)):
                if true_targets[t] == pred_targets[t]:
                    # print("%d -> %d" % (true_targets[t], pred_targets[t]))
                    tot_correct += 1
            test_accuracy = tot_correct/len(true_targets)

            print("\n")
            print("Step %d -> TEST Accuracy: %g" % (i, test_accuracy))

            for start, end in zip(range(0, len(training_X), batch_size), range(batch_size, len(training_X) + batch_size, batch_size)):

                train_step.run(feed_dict={
                    x: training_X[start:end], 
                    y_: training_Y[start:end], 
                    keep_prob: 0.5
                })

                train_accuracy = accuracy.eval(feed_dict={
                    x: training_X[start:end], 
                    y_: training_Y[start:end], 
                    keep_prob: 1.0
                })

                print('\tBatch %d - %d -> TRAINING accuracy %g' % (start, end, train_accuracy))

                if train_accuracy > 0.999:
                    done = True
                    accuracy_score = accuracy.eval(feed_dict={
                        x: test_X, 
                        y_: test_Y, 
                        keep_prob: 1.0
                    })
                    print("Accuracy on test set is: %d" % accuracy_score)
                    save_model(saver, sess)
                    break

        saver.restore(sess, get_model_path())

        # pred_targets = np.array([])
        # for start, end in zip(range(0, len(training_X), batch_size), range(batch_size, len(training_X) + batch_size, batch_size)):
        #     batch_pred_targets = sess.run(predict_operation, feed_dict={
        #         x: test_X,
        #         keep_prob: 1.0
        #     })
        #     print(np.hstack((pred_targets, np.array(batch_pred_targets))))
        #     pred_targets = np.apply_along_axis(int, 0, np.hstack((pred_targets, np.array(batch_pred_targets))))
        # accuracy_score = np.mean(true_targets == pred_targets)
        # for i in range(len(true_targets)):
        #     print(true_targets[i], " - ", pred_targets[i])
        # print("Accuracy on test set is: %d" % accuracy_score)

if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--data_dir', type=str,
#                       default='/tmp/tensorflow/mnist/input_data',
#                       help='Directory for storing input data')
#   FLAGS, unparsed = parser.parse_known_args()
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=main)