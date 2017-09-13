
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np
import random
import math

from image_handler import *
from rating_handler import *
from performance_measures import *

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
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.elu(conv2d(X, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Normalize
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    print("Size after first downsampling: ", h_norm1.shape)

    # Second convolutional layer -- maps 96 feature maps to 256 of size 5x5.
    W_conv2 = weight_variable([5, 5, 32, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.elu(conv2d(h_norm1, W_conv2) + b_conv2)

    # Normalize
    h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    
    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_norm2)
    print("Size after second downsampling: ", h_pool2.shape)

    # Third convolutional layer -- maps 256 to 384 filters of size 3x3.
    W_conv3 = weight_variable([7, 7, 16, 8])
    b_conv3 = bias_variable([8])
    h_conv3 = tf.nn.elu(conv2d(h_pool2, W_conv3) + b_conv3)

    # Normalize
    h_norm3 = tf.nn.lrn(h_conv3, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # Third pooling layer.
    h_pool3 = max_pool_2x2(h_norm3)
    print("Size after third downsampling: ", h_pool3.shape)

    # Fourth conv layer
    W_conv4 = weight_variable([9, 9, 8, 4])
    b_conv4 = bias_variable([4])
    h_conv4 = tf.nn.elu(conv2d(h_pool3, W_conv4) + b_conv4)

    # Normalize
    h_norm4 = tf.nn.lrn(h_conv4, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # Fourth pooling layer
    h_pool4 = max_pool_2x2(h_norm4)
    print("Size after fourth downsampling: ", h_pool4.shape)

    # Fifth conv layer
    W_conv5 = weight_variable([12, 12, 4, 2])
    b_conv5 = bias_variable([2])
    h_conv5 = tf.nn.elu(conv2d(h_pool4, W_conv5) + b_conv5)

    # Normalize
    h_norm5 = tf.nn.lrn(h_conv5, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    # Fourth pooling layer
    h_pool5 = max_pool_2x2(h_norm5)
    print("Size after fourth downsampling: ", h_pool5.shape)

    last_pool = h_pool5
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.

    # 40*30 -> 20*15 -> 10*8 just check sizes after pooling. Learn how to calculate the size sometime.
    W_fc1 = weight_variable([int(last_pool.get_shape()[1])*int(last_pool.get_shape()[2])*2, 64])
    b_fc1 = bias_variable([64])

    last_pool_flat = tf.reshape(last_pool, [-1, int(last_pool.get_shape()[1])*int(last_pool.get_shape()[2])*2])
    h_fc1 = tf.nn.elu(tf.matmul(last_pool_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([64, img_classes])
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
    all_images = get_all_resized_images(dim1=64,dim2=64, haar=False, dir_name="Processed_Combined_Datasets")
    print("No. images", len(all_images), "-> Dims e0: ", all_images[0].shape)
    all_ratings = get_all_ratings(file_name='Processed_Combined_Datasets_Ratings.txt')

    ratings_count = {}
    for r in all_ratings:
        if r not in ratings_count:
            ratings_count[r] = 1
        else:
            ratings_count[r] += 1
    print(ratings_count)
    # sys.exit(0)

    all_ratings = [
        (lambda r: 2)(r)
        if r > 7 else
        (lambda r: 1)(r)
        if r > 4 else
        (lambda r: 0)(r)
        for r in all_ratings
    ]

    ratings_count = {}
    for r in all_ratings:
        if r not in ratings_count:
            ratings_count[r] = 1
        else:
            ratings_count[r] += 1
    print(ratings_count)
    # sys.exit(0)

    print("Rating e0: ", all_ratings[0])
    one_hot_ratings = one_hot_encode(all_ratings, n_classes=3)
    print("One Hot Rating e0: ", one_hot_ratings[0])

    # Generate new data by vertically flipping the images --> creating symmetric copies over the verical axis
    new_data = []
    new_labels = []
    for i in range(len(one_hot_ratings)):
        if one_hot_ratings[i].tolist() == [1.0, 0.0, 0.0] or one_hot_ratings[i].tolist() == [0.0, 0.0, 1.0]:
            vimg=cv2.flip(all_images[i],1)
            new_data.append(vimg)
            new_labels.append(np.array(one_hot_ratings[i], copy=True))
    
    all_images = all_images+new_data
    labels = np.array(new_labels)
    print(one_hot_ratings.shape, labels.shape, type(one_hot_ratings[0]), type(labels[0]))
    one_hot_ratings = np.concatenate((one_hot_ratings, labels), axis=0)

    size_training_set = int(math.floor(len(all_images)*0.9))
    size_test_set = len(all_images)-size_training_set
    print("Size training set -> {}, Size test set -> {}".format(size_training_set, size_test_set))
        
    # 80% training
    training_X, training_Y, test_X, test_Y = split_to_training_and_test(
        data_set=all_images, 
        label_set=one_hot_ratings, 
        n_samples=size_test_set
    )

    # for i in range(len(training_X)):
    #     if type(training_X[i]) != type(training_X[0]) or type(training_Y[i]) != type(training_X[0]):
    #         print(type(training_X[i]), type(training_Y[i]))
    
    # for i in range(len(test_X)):
    #     if type(test_X[i]) != type(training_X[0]) or type(test_Y[i]) != type(training_X[0]):
    #         print(type(test_X[i]), type(test_Y[i]))
    # sys.exit(0)
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
    accuracy_treshold = 0.99
    print("\n")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        true_targets = np.argmax(test_Y, axis=1)
        done = False
        test_accuracy = 0
        for i in range(500):

            pred_targets = np.array([])
            true_targets = np.argmax(test_Y, axis=1)
            for start, end in zip(range(0, len(test_X), batch_size), range(batch_size, len(test_X) + batch_size, batch_size)):
                batch_pred_targets = sess.run(predict_operation, feed_dict={
                    x: test_X[start:end],
                    keep_prob: 1.0
                })
                pred_targets = np.hstack((pred_targets, np.array(batch_pred_targets)))
            print(len(pred_targets), len(true_targets))
            test_accuracy = get_accuracy(pred_targets, true_targets)

            print("\n")
            print("Step %d -> TEST Accuracy: %g" % (i, test_accuracy))
            print("Precision, Recall, F-score, Support: {}".format(get_performance(pred_targets, true_targets)))
            print("{} ----- Classification Report TEST SET -----".format(i+1))
            print(get_classification_report(pred_targets, true_targets))
            
            if test_accuracy > accuracy_treshold:
                print("Finished training by exceeding accuracy treshold {}".format(accuracy_treshold))
                break

            if done == True:
                break

            pred_targets = np.array([])
            true_targets = np.argmax(training_Y, axis=1)
            # print(training_Y[0])
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
                print('\tBatch %d - %d -> TRAINING accuracy %g' % (start, end, get_accuracy(batch_pred_targets, batch_true_targets)))

                pred_targets = np.hstack((pred_targets, np.array(batch_pred_targets)))

            training_accuracy = get_accuracy(pred_targets, true_targets)
            print('\tAvg Batch -> TRAINING accuracy %g' % (training_accuracy))
            print("Precision, Recall, F-score, Support: {}".format(get_performance(pred_targets, true_targets)))
            print("{} ----- Classification Report TRAINING SET -----".format(i+1))
            print(get_classification_report(pred_targets, true_targets))

            if training_accuracy > accuracy_treshold:
                print("The average accuracy exceeded the accuracy threshold {}".format(accuracy_treshold))
                done = True

        save_model(saver, sess)
        print("Model saved.")

if __name__ == '__main__':
    tf.app.run(main=main)