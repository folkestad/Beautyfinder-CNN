import numpy as np
import cv2
import dlib
import math

def calc_ratios(landmarks):
    landmarks = np.squeeze(np.asarray(landmarks))
    left_eye_w  = landmarks[36]
    left_eye_e  = landmarks[39]
    right_eye_w = landmarks[42]
    right_eye_e = landmarks[45]
    nose_n      = landmarks[27]
    nose_w      = landmarks[31]
    nose_s      = landmarks[33]
    nose_e      = landmarks[35]
    mouth_n     = landmarks[51]
    mouth_w     = landmarks[48]
    mouth_s     = landmarks[57]
    mouth_e     = landmarks[54]

    print(type(left_eye_e))

    eye_ratio = abs(
        math.sqrt(
            math.pow((left_eye_w[0]-left_eye_e[0]), 2) + math.pow((left_eye_w[1]-left_eye_e[1]), 2)
        )
        -
        math.sqrt(
            math.pow((right_eye_w[0]-right_eye_e[0]), 2) + math.pow((right_eye_w[1]-right_eye_e[1]), 2)
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

    eyes_nose_ratio = abs(
        math.sqrt(
            math.pow((left_eye_e[0]-nose_s[0]), 2) + math.pow((left_eye_e[1]-nose_s[1]), 2)
        )
        -
        math.sqrt(
            math.pow((right_eye_w[0]-nose_s[0]), 2) + math.pow((right_eye_w[1]-nose_s[1]), 2)
        )
    )

    print "Eye ratio:", eye_ratio
    print "Nose ratio:", nose_ratio
    print "Mouth ratio:", mouth_ratio
    print "Eyes-Nose ratio:", eyes_nose_ratio

def get_landmarks():
    image_path = '../Data/Test/b1.jpg'
    cascade_path = '../Data/opencv/haarcascade_frontalface_default.xml'
    predictor_path = '../Data/dlib/shape_predictor_68_face_landmarks.dat'

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # create the landmark predictor
    predictor = dlib.shape_predictor(predictor_path)

    # Read the image
    image = cv2.imread(image_path)

    # Resize the image (not necessary)
    image = cv2.resize(image, (1600, 1000)) 

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

    image = gray

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        print dlib_rect

        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
        

        # copying the image so we can see side-by-side
        image_copy = image.copy()

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            # annotate the positions
            cv2.putText(image_copy, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))

            # draw points on the landmark positions
            cv2.circle(image_copy, pos, 3, color=(0, 255, 255))

    cv2.imshow("Faces found", image)
    cv2.imshow("Landmarks found", image_copy)
    cv2.waitKey(0)
    return landmarks

if __name__ == '__main__':
    calc_ratios(get_landmarks())
