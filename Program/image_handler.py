import numpy
import os
import cv2 as opencv
from HAAR import *

def get_image(file_name):
    current_dir = os.path.dirname(__file__)
    file_path = '../Data/Data_Collection/{}'.format(file_name)
    file_rel_path = os.path.join(current_dir, file_path)
    img = opencv.imread(file_rel_path)
    # print(img)
    # print("\n")
    return img

def show_image(img, name='Picture'):
    opencv.imshow(name, img)
    opencv.waitKey(0)
    opencv.destroyAllWindows()

def resize_image(img, dim1, dim2):
    resized_img = opencv.resize(img, (dim2, dim1)) 
    return resized_img

def get_all_images():
    all_images = []
    for i in range(1,501):
        img = get_image('SCUT-FBP-{}.jpg'.format(i))
        all_images.append(img)
    return all_images

def get_all_resized_images(dim1=400, dim2=300, haar=False):
    all_resized_images = []
    for i in range(1,501):
        if haar:
            print(i)
            img = haar_cascade('SCUT-FBP-{}.jpg'.format(i))
        else:
            img = get_image('SCUT-FBP-{}.jpg'.format(i))
        resized_img = resize_image(img, dim1, dim2)
        all_resized_images.append(resized_img)
    return all_resized_images


if __name__ == '__main__':
    # img = get_image('SCUT-FBP-2.jpg')
    img = haar_cascade(file_name='SCUT-FBP-2.jpg')
    r_img = resize_image(img, dim1=64, dim2=64)
    # show_image(img, 'original')
    show_image(r_img, 'resized')