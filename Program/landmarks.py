import numpy as np
import cv2
import dlib
import math
import os
from align_face import *

def rate_face(benchmarks, face):
    rating = 0
    print("Eyes-Nose ratio and benchmark: {} and {}".format(abs(benchmarks[0][0]-face[0]), benchmarks[0][1]))
    if abs(benchmarks[0][0]-face[0]) < benchmarks[0][1]*4.5:
        if abs(benchmarks[0][0]-face[0]) < benchmarks[0][1]*2:
            rating += 1
        rating += 1
    print("Nose ratio and benchmark: {} and {}".format(abs(benchmarks[1][0]-face[1]), benchmarks[1][1]))
    if abs(benchmarks[1][0]-face[1]) < benchmarks[1][1]*9.5:
        if abs(benchmarks[1][0]-face[1]) < benchmarks[1][1]*5:
            rating += 1
        rating += 1
    print("Mouth ratio and benchmark: {} and {}".format(abs(benchmarks[2][0]-face[2]), benchmarks[2][1]))
    if abs(benchmarks[2][0]-face[2]) < benchmarks[2][1]*5.5:
        if abs(benchmarks[2][0]-face[2]) < benchmarks[2][1]*4.5:
            rating += 1
        rating += 1
    print("Eyes-Nose angle ratio and benchmark: {} and {}".format(abs(benchmarks[3][0]-face[3]), benchmarks[3][1]))
    if abs(benchmarks[3][0]-face[3]) < benchmarks[3][1]*10.5:
        if abs(benchmarks[3][0]-face[3]) < benchmarks[3][1]*5:
            rating += 1
        rating += 1
    print("Eyes-Mouth ratio and benchmark: {} and {}".format(abs(benchmarks[4][0]-face[4]), benchmarks[4][1]))
    if abs(benchmarks[4][0]-face[4]) < benchmarks[4][1]*6.5:
        if abs(benchmarks[4][0]-face[4]) < benchmarks[4][1]*6:
            rating += 1
        rating += 1
    return rating


def calc_ratios(landmarks):
    landmarks = np.squeeze(np.asarray(landmarks))
    left_eye_w  = landmarks[36]
    left_eye_e  = landmarks[39]
    right_eye_w = landmarks[42]
    right_eye_e = landmarks[45]
    middle_eyes = [
        int(math.floor(left_eye_e[0] + 0.5*(right_eye_w[0]-left_eye_e[0]))), 
        int(math.floor(left_eye_e[1] + 0.5*(right_eye_w[1]-left_eye_e[1])))
    ]
    nose_n      = landmarks[27]
    nose_w      = landmarks[31]
    nose_s      = landmarks[33]
    nose_e      = landmarks[35]
    mouth_n     = landmarks[51]
    mouth_w     = landmarks[48]
    mouth_s     = landmarks[57]
    mouth_e     = landmarks[54]

    # eyes_ratio = abs(
    #     math.sqrt(
    #         math.pow((left_eye_w[0]-left_eye_e[0]), 2) + math.pow((left_eye_w[1]-left_eye_e[1]), 2)
    #     )
    #     -
    #     math.sqrt(
    #         math.pow((right_eye_w[0]-right_eye_e[0]), 2) + math.pow((right_eye_w[1]-right_eye_e[1]), 2)
    #     )
    # )
    eyes_nose_ratio = (
        math.sqrt(
            math.pow((left_eye_e[0]-right_eye_w[0]), 2) + math.pow((left_eye_e[1]-right_eye_w[1]), 2)
        )
        /
        math.sqrt(
            math.pow((middle_eyes[0]-nose_s[0]), 2) + math.pow((middle_eyes[1]-nose_s[1]), 2)
        )
    )

    nose_ratio = (
        math.sqrt(
            math.pow((nose_n[0]-nose_s[0]), 2) + math.pow((nose_n[1]-nose_s[1]), 2)
        )
        /
        math.sqrt(
            math.pow((nose_w[0]-nose_e[0]), 2) + math.pow((nose_w[1]-nose_w[1]), 2)
        )
    )

    mouth_ratio = (
        math.sqrt(
            math.pow((mouth_w[0]-mouth_e[0]), 2) + math.pow((mouth_w[1]-mouth_w[1]), 2)
        )
        /
        math.sqrt(
            math.pow((mouth_n[0]-mouth_s[0]), 2) + math.pow((mouth_n[1]-mouth_s[1]), 2)
        )
    )

    # eyes_nose_ratio = abs(
    #     math.sqrt(
    #         math.pow((left_eye_e[0]-nose_s[0]), 2) + math.pow((left_eye_e[1]-nose_s[1]), 2)
    #     )
    #     -
    #     math.sqrt(
    #         math.pow((right_eye_w[0]-nose_s[0]), 2) + math.pow((right_eye_w[1]-nose_s[1]), 2)
    #     )
    # )

    left_eye_nose = math.sqrt(
        math.pow((left_eye_e[0]-nose_s[0]), 2) + math.pow((left_eye_e[1]-nose_s[1]), 2)
    )
    right_eye_nose = math.sqrt(
        math.pow((right_eye_w[0]-nose_s[0]), 2) + math.pow((right_eye_w[1]-nose_s[1]), 2)
    )
    eyes_nose_angle_ratio = (left_eye_nose/right_eye_nose) if left_eye_nose > right_eye_nose else (right_eye_nose/left_eye_nose)

    # eyes_mouth_ratio = abs(
    #     math.sqrt(
    #         math.pow((left_eye_e[0]-mouth_w[0]), 2) + math.pow((left_eye_e[1]-mouth_w[1]), 2)
    #     )
    #     -
    #     math.sqrt(
    #         math.pow((right_eye_w[0]-mouth_e[0]), 2) + math.pow((right_eye_w[1]-mouth_e[1]), 2)
    #     )
    # )

    left_eye_mouth = math.sqrt(
        math.pow((left_eye_e[0]-mouth_w[0]), 2) + math.pow((left_eye_e[1]-mouth_w[1]), 2)
    )
    right_eye_mouth = math.sqrt(
        math.pow((right_eye_w[0]-mouth_e[0]), 2) + math.pow((right_eye_w[1]-mouth_e[1]), 2)
    )
    eyes_mouth_ratio = (left_eye_mouth/right_eye_mouth) if left_eye_mouth > right_eye_mouth else (right_eye_mouth/left_eye_mouth)

    print "Eyes-Nose ratio:", eyes_nose_ratio
    print "Nose ratio:", nose_ratio
    print "Mouth ratio:", mouth_ratio
    print "Eyes-Nose Angle ratio:", eyes_nose_angle_ratio
    print "Eyes-Mouth ratio:", eyes_mouth_ratio

    return {
        'eyes_nose_ratio': eyes_nose_ratio,
        'nose_ratio': nose_ratio,
        'mouth_ratio': mouth_ratio,
        'eyes_nose_angle_ratio': eyes_nose_angle_ratio,
        'eyes_mouth_ratio': eyes_mouth_ratio
    }

def get_landmarks(dirname="Presentation", dest_dir="Presentation", filename="donald.jpg", new_filename="donald_landmarks.jpg", showimg=True, dim1=100, dim2=100, save_images=True):
    cascade_path = '../Data/opencv/haarcascade_frontalface_default.xml'
    predictor_path = '../Data/dlib/shape_predictor_68_face_landmarks.dat'
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # create the landmark predictor
    predictor = dlib.shape_predictor(predictor_path)

    # Read the image
    # image = align_face(dirname=dirname, filename=filename)
    image = cv2.imread("../Data/Datasets/{}/{}".format(dirname, filename))
    real_image = image

    # cv2.imshow("Aligned", image)
    # cv2.waitKey(0)
    # Resize the image (not necessary)
    # image = cv2.resize(image, (500, 500)) 

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    if len(faces) != 1:
        return Exception


    image_2 = image.copy()
    # image = gray

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = real_image[y:y+h, x:x+w]
        # if save_images:
        #     current_dir = os.path.dirname(__file__)
        #     file_path = '../Data/Datasets/{}/{}'.format(dest_dir, new_filename)
        #     file_rel_path = os.path.join(current_dir, file_path)
        #     cv2.imwrite(file_rel_path, face_img)
        # face_img = opencv.resize(face_img, (dim2, dim1)) 
        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # print dlib_rect

        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
        

        # copying the image so we can see side-by-side
        image_copy = image_2.copy()
        # image_copy = real_image.copy()

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            # annotate the positions
            cv2.putText(image_copy, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(0, 0, 255))

            # draw points on the landmark positions
            cv2.circle(image_copy, pos, 10, thickness=2, color=(0, 0, 0))
        
        for (x, y, w, h) in faces:
            image_copy = image_copy[y:y+h+100, x-50:x+w+50]
        if save_images:
            cv2.imwrite("../Data/Datasets/{}/{}".format(dest_dir, new_filename), image_copy)

    if showimg:
        cv2.imshow("Faces found", image)
        cv2.imshow("Landmarks found", image_copy)
        cv2.waitKey(0)
    return landmarks

if __name__ == '__main__':
    get_landmarks()
