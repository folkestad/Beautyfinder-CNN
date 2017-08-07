import numpy as np
import os
import cv2

def haar_cascade(dir_name='Renamed_CFD', file_name='test.jpg'):

    current_dir = os.path.dirname(__file__)

    face_rel_path = os.path.join(current_dir, '../Data/opencv/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_rel_path)
    eye_rel_path = os.path.join(current_dir, '../Data/opencv/haarcascade_eye.xml')
    eye_cascade = cv2.CascadeClassifier(eye_rel_path)
    
    # file_path = '../Data/Test/{}'.format(file_name)
    file_path = '../Data/{}/{}'.format(dir_name, file_name)
    file_rel_path = os.path.join(current_dir, file_path)
    img = cv2.imread(file_rel_path)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    # gray = clahe.apply(gray)
    gray = cv2.equalizeHist(gray);

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_img = None
    for (x,y,w,h) in faces:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]
        face_img = img[y:y+h, x:x+w]
        break
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # cv2.imshow('img',roi_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_dest_path = os.path.join(current_dir, '../Data/HAAR_CFD/', file_name)
    print(img_dest_path)
    if face_img is None:
        print("Could not find face in file {}.".format(file_rel_path))
        cv2.imwrite(img_dest_path, img)
        # cv2.waitKey(0)
        return img
    print("Found face in file {}".format(file_rel_path))
    cv2.imwrite(img_dest_path, face_img)
    # cv2.waitKey(0)
    return face_img

if __name__ == '__main__':
    haar_cascade(file_name='CFD-AF-200-228-1.jpg')