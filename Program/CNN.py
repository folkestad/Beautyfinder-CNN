
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
from performance_measures import *

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import random

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
    W_conv1 = weight_variable([3, 3, 3, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.elu(conv2d(X, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Normalize
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    print("Size after first downsampling: ", h_norm1.shape)

    # Second convolutional layer -- maps 96 feature maps to 256 of size 5x5.
    W_conv2 = weight_variable([5, 5, 64, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.elu(conv2d(h_norm1, W_conv2) + b_conv2)

    # Normalize
    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_norm2)
    print("Size after second downsampling: ", h_pool2.shape)

    # Third convolutional layer -- maps 256 to 384 filters of size 3x3.
    W_conv3 = weight_variable([7, 7, 32, 16])
    b_conv3 = bias_variable([16])
    h_conv3 = tf.nn.elu(conv2d(h_pool2, W_conv3) + b_conv3)

    # Normalize
    h_norm3 = tf.nn.lrn(h_conv3, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # Third pooling layer.
    h_pool3 = max_pool_2x2(h_norm3)
    print("Size after third downsampling: ", h_pool3.shape)

    # Fourth conv layer
    W_conv4 = weight_variable([9, 9, 16, 8])
    b_conv4 = bias_variable([8])
    h_conv4 = tf.nn.elu(conv2d(h_pool3, W_conv4) + b_conv4)

    # Fourth pooling layer
    h_pool4 = max_pool_2x2(h_conv4)
    print("Size after third downsampling: ", h_pool4.shape)

    last_pool = h_pool4
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.

    # 40*30 -> 20*15 -> 10*8 just check sizes after pooling. Learn how to calculate the size sometime.
    W_fc1 = weight_variable([int(last_pool.get_shape()[1])*int(last_pool.get_shape()[2])*8, 32])
    b_fc1 = bias_variable([32])

    last_pool_flat = tf.reshape(last_pool, [-1, int(last_pool.get_shape()[1])*int(last_pool.get_shape()[2])*8])
    h_fc1 = tf.nn.elu(tf.matmul(last_pool_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([32, img_classes])
    b_fc2 = bias_variable([img_classes])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # print(y_conv.get_shape())
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


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

def calc_accuracy(pred_targets, true_targets):
    pred_targets = pred_targets.astype(int)
    tot_correct = 0
    for i in range(len(pred_targets)):
        if true_targets[i] == pred_targets[i]:
            # print("%d -> %d - %s -> %s" % (true_targets[i], pred_targets[i], type(true_targets[i]), type(pred_targets[i])))
            tot_correct += 1
    test_accuracy = tot_correct/len(true_targets)
    return test_accuracy

def split_to_training_and_test(data_set=[], label_set=[], n_samples=0):
    index_list = []
    l = len(data_set) - 1
    while len(index_list) < n_samples:
        # print("In")
        index = random.randint(0, l)
        if index not in index_list:
            index_list.append(index)

    test_X = [ data_set[i] for i in range(len(data_set)) if i in index_list ]
    test_Y = [ label_set[i] for i in range(len(label_set)) if i in index_list ]

    training_X = [ data_set[i] for i in range(len(data_set)) if i not in index_list ]
    training_Y = [ label_set[i] for i in range(len(label_set)) if i not in index_list ]
    
    print("Length of sets (training_X, training_Y, test_X, test_Y):", len(training_X), len(training_Y), len(test_X), len(test_Y))
    return training_X, training_Y, test_X, test_Y

def main(_):

    #import data
    all_images = get_all_resized_images(dim1=32,dim2=32, haar=False, dir_name="Processed_Many_datasets")
    print("No. images", len(all_images), "-> Dims e0: ", all_images[0].shape)
    all_ratings = get_all_ratings(file_name='Many_datasets_ratings.txt')
    print("Rating e0: ", all_ratings[0])
    one_hot_ratings = one_hot_encode(all_ratings, n_classes=10)
    print("One Hot Rating e0: ", one_hot_ratings[0])
    size_training_set = int(math.floor(len(all_images)*0.8))
    size_test_set = len(all_images)-size_training_set
    print("Size training set -> {}, Size test set -> {}".format(size_training_set, size_test_set))

    # 80% training
    training_X, training_Y, test_X, test_Y = split_to_training_and_test(
        data_set=all_images, 
        label_set=one_hot_ratings, 
        n_samples=size_test_set
    )

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

    batch_size = 60
    accuracy_treshold = 0.90
    print("\n")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        true_targets = np.argmax(test_Y, axis=1)
        done = False
        test_accuracy = 0
        for i in range(500):

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
            
            test_accuracy = calc_accuracy(pred_targets, true_targets)

            print("\n")
            print("Step %d -> TEST Accuracy: %g" % (i, test_accuracy))

            if test_accuracy > accuracy_treshold:
                print("Finished training by exceeding accuracy treshold {}".format(accuracy_treshold))
                break

            pred_targets = np.array([])
            true_targets = np.argmax(training_Y, axis=1)
            for start, end in zip(range(0, len(training_X), batch_size), range(batch_size, len(training_X) + batch_size, batch_size)):
                train_step.run(feed_dict={
                    x: training_X[start:end], 
                    y_: training_Y[start:end], 
                    keep_prob: 0.8
                })

                batch_pred_targets = sess.run(predict_operation, feed_dict={
                    x: training_X[start:end],
                    keep_prob: 1.0
                })
                batch_true_targets = np.argmax(training_Y[start:end], axis=1)
                print('\tBatch %d - %d -> TRAINING accuracy %g' % (start, end, calc_accuracy(batch_pred_targets, batch_true_targets)))

                pred_targets = np.hstack((pred_targets, np.array(batch_pred_targets)))

            training_accuracy = calc_accuracy(pred_targets, true_targets)
            print('\tAvg Batch -> TRAINING accuracy %g' % (training_accuracy))
            # print(classification_report(pred_targets, true_targets))

            if training_accuracy > accuracy_treshold:
                print("The average accuracy exceeded the accuracy threshold {}".format(accuracy_treshold))
                break

        save_model(saver, sess)
        print("Model saved.")

if __name__ == '__main__':
    tf.app.run(main=main)