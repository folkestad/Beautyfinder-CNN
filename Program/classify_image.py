import numpy
import os
import cv2
import tensorflow as tf
from CNN import *
from HAAR import *

def classify_image_to_beauty_scale(file_name="test.jpg", dim1=32, dim2=32):

    # img = get_image(file_name=file_name)
    haar_img = haar_cascade(dir_name="Test", file_name=file_name)
    resized_image = resize_image(haar_img, dim1, dim2)

    x = tf.placeholder(tf.float32, [None, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2]])
    y_conv, keep_prob = cnn_model(x, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2], 11)
    predict_operation = tf.argmax(y_conv, axis=1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, get_model_path())
        result = sess.run(predict_operation, feed_dict={
            x: [resized_image],
            keep_prob: 1.0
        })
        print("The person in the image is a {} on the beauty scale.".format(result))
    
    current_dir = os.path.dirname(__file__)
    file_path = '../Data/Test/{}'.format(file_name)
    file_rel_path = os.path.join(current_dir, file_path)
    img = cv2.imread(file_rel_path)
    print(type(img), img.shape)
    show_image(img)



def get_model_path(file_name='model.ckpt'):
    saver = tf.train.Saver()

    current_dir = os.path.dirname(__file__)
    file_path = '../Models/{}'.format(file_name)
    file_rel_path = os.path.join(current_dir, file_path)

    return file_rel_path

def show_image(img, name='Picture'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image(file_name):
    current_dir = os.path.dirname(__file__)
    file_path = '../Data/Test/{}'.format(file_name)
    file_rel_path = os.path.join(current_dir, file_path)
    img = cv2.imread(file_rel_path)
    return img

def resize_image(img, dim1, dim2):
    resized_img = cv2.resize(img, (dim2, dim1)) 
    return resized_img

if __name__ == '__main__':
    classify_image_to_beauty_scale(file_name="b1.jpg", dim1=32, dim2=32)