import numpy
import os
import cv2 as opencv
from HAAR import *

def get_image(dir_name='Renamed_CFD', file_name='test.jpg'):
    current_dir = os.path.dirname(__file__)
    file_path = '../Data/Datasets/{}/{}'.format(dir_name, file_name)
    file_rel_path = os.path.join(current_dir, file_path)
    img = opencv.imread(file_rel_path)
    return img

def show_image(img, name='Picture'):
    opencv.imshow(name, img)
    opencv.waitKey(0)
    opencv.destroyAllWindows()

def resize_image(img, dim1, dim2):
    resized_img = opencv.resize(img, (dim2, dim1)) 
    return resized_img

def get_all_images(dir_name="Renamed_CFD"):
    all_images = []
    src_dir = '../Data/{}'.format(dir_name)
    files = os.walk(src_dir).next()[2]
    for file_name in files:
        print(dir_name, file_name)
        img = get_image(dir_name=dir_name, file_name=file_name)
        all_images.append(img)
    return all_images

def get_all_resized_images(dim1=400, dim2=300, haar=False, dir_name="Renamed_CFD"):
    all_resized_images = []
    current_dir = os.path.dirname(__file__)
    src_dir = '../Data/Datasets/{}'.format(dir_name)
    dir_rel_path = os.path.join(current_dir, src_dir)
    files = os.walk(dir_rel_path).next()[2]
    for file_name in files:
        if file_name != '.DS_Store':
            if haar:
                img = haar_cascade(dir_name=dir_name, file_name=file_name)
            else:
                img = get_image(dir_name=dir_name, file_name=file_name)
            resized_img = resize_image(img, dim1, dim2)
            all_resized_images.append(resized_img)
    return all_resized_images

def move_images(src_dir='../Data/CFD_Version/Images', dest_dir='../Data/New_Dataset'):
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(src_dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file_name in files:
                if "N.jpg" in file_name:                                                                                        
                    r.append(subdir + "/" + file_name)   
    for i,file_name in enumerate(r):
        os.rename(file_name, dest_dir+"/"+file_name.split("/")[-1].replace("N.jpg", "{}.jpg".format(i)))

def change_names():
    names = []
    src_dir = '../Data/New_Dataset'
    dest_dir = '../Data/Renamed_CFD'
    files = os.walk(src_dir).next()[2]
    for file_name in files:
        names.append(src_dir+"/"+file_name)
    for i,file_name in enumerate(names):
        # os.rename(file_name, dest_dir+'/'+file_name)
        os.rename(file_name, file_name.split("/")[-1].replace("N.jpg", "{}.jpg".format(i+1)))


if __name__ == '__main__':
    # img = get_image('SCUT-FBP-2.jpg')
    # img = haar_cascade(file_name='SCUT-FBP-2.jpg')
    # r_img = resize_image(img, dim1=64, dim2=64)
    # show_image(img, 'original')
    # show_image(r_img, 'resized')
    # move_images()
    change_names()