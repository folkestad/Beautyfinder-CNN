import numpy as np
import cv2
import dlib
import math
import os
from align_face import *
from landmarks import *


def rate_face(benchmarks, face):
    rating = 0
    print("Eyes-Nose ratio and benchmark: {} and {}".format(abs(benchmarks[0][0]-face[0]), benchmarks[0][1]))
    if abs(benchmarks[0][0]-face[0]) < benchmarks[0][1]*4:
        if abs(benchmarks[0][0]-face[0]) < benchmarks[0][1]*1.5:
            rating += 1
        rating += 1
    print("Nose ratio and benchmark: {} and {}".format(abs(benchmarks[1][0]-face[1]), benchmarks[1][1]))
    if abs(benchmarks[1][0]-face[1]) < benchmarks[1][1]*9:
        if abs(benchmarks[1][0]-face[1]) < benchmarks[1][1]*4.5:
            rating += 1
        rating += 1
    print("Mouth ratio and benchmark: {} and {}".format(abs(benchmarks[2][0]-face[2]), benchmarks[2][1]))
    if abs(benchmarks[2][0]-face[2]) < benchmarks[2][1]*5:
        if abs(benchmarks[2][0]-face[2]) < benchmarks[2][1]*4:
            rating += 1
        rating += 1
    print("Eyes-Nose angle ratio and benchmark: {} and {}".format(abs(benchmarks[3][0]-face[3]), benchmarks[3][1]))
    if abs(benchmarks[3][0]-face[3]) < benchmarks[3][1]*10:
        if abs(benchmarks[3][0]-face[3]) < benchmarks[3][1]*4.5:
            rating += 1
        rating += 1
    print("Eyes-Mouth ratio and benchmark: {} and {}".format(abs(benchmarks[4][0]-face[4]), benchmarks[4][1]))
    if abs(benchmarks[4][0]-face[4]) < benchmarks[4][1]*6:
        if abs(benchmarks[4][0]-face[4]) < benchmarks[4][1]*5.5:
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

    left_eye_nose = math.sqrt(
        math.pow((left_eye_e[0]-nose_s[0]), 2) + math.pow((left_eye_e[1]-nose_s[1]), 2)
    )
    right_eye_nose = math.sqrt(
        math.pow((right_eye_w[0]-nose_s[0]), 2) + math.pow((right_eye_w[1]-nose_s[1]), 2)
    )
    eyes_nose_angle_ratio = (left_eye_nose/right_eye_nose) if left_eye_nose > right_eye_nose else (right_eye_nose/left_eye_nose)

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

# def calc_celeb_ratios(landmarks):
#     lefteye = landmarks[0]
#     righteye = landmarks[1]
#     nose = landmarks[2]
#     left_mouth = landmarks[3]
#     right_mouth = landmarks[4]

if __name__ == '__main__':
    people = []
    dirname="Combined_datasets"
    write_to_file = False

    current_dir = os.path.dirname(__file__)
    file_path = '../Data/{}'.format("benchmarks.txt")
    file_rel_path = os.path.join(current_dir, file_path)
    benchmarks_file = open(file_rel_path, 'r')
    benchmarks = [ [float(b.replace("\r\n", "").split(";")[0]), float(b.replace("\r\n", "").split(";")[1])] for b in benchmarks_file ]
    benchmarks_file.close()
    
    src_dir = '../Data/{}'.format(dirname)
    files = os.walk(src_dir).next()[2]
    file_names_ratings = []
    for i,file_name in enumerate(files):
        if not file_name == ".DS_Store":
            try:
                p = calc_ratios(get_landmarks(
                    dirname=dirname,
                    dest_dir="Processed_Combined_datasets",
                    filename="{}".format(file_name), 
                    showimg=False, 
                    dim1=500, 
                    dim2=500, 
                    save_images=True
                ))
                people.append(p)
                rating = rate_face(benchmarks, [
                    p['eyes_nose_ratio'],
                    p['nose_ratio'],
                    p['mouth_ratio'],
                    p['eyes_nose_angle_ratio'],
                    p['eyes_mouth_ratio']
                ])
                file_names_ratings.append((file_name, rating))
                print("{}: {} --> {}".format(i+1, file_name, rating))
            except:
                print("Could not append to people: {}.".format(file_name))
            
    current_dir = os.path.dirname(__file__)
    file_path = '../Data/{}'.format("Combined_datasets_ratings.txt")
    file_rel_path = os.path.join(current_dir, file_path)
    destfile = open(file_rel_path, 'w')
    for fr in file_names_ratings:
        destfile.write("{};{}\r\n".format(fr[0], fr[1]))
    destfile.close()
    print("Done writing to file.")
    # print people
    # print "\n"
    # avg_eyes_nose_ratio = 0
    # avg_nose_ratio = 0
    # avg_mouth_ratio = 0
    # avg_eyes_nose_angle_ratio = 0
    # avg_eyes_mouth_ratio = 0
    # for i, p in enumerate(people):
    #     avg_eyes_nose_ratio += p['eyes_nose_ratio']
    #     avg_nose_ratio += p['nose_ratio']
    #     avg_mouth_ratio += p['mouth_ratio']
    #     avg_eyes_nose_angle_ratio += p['eyes_nose_angle_ratio']
    #     avg_eyes_mouth_ratio += p['eyes_mouth_ratio']
    #     print("Person {} - {}".format(i+1, file_names[i]))
    #     print("Eyes-Nose ratio diff:       {}".format(p['eyes_nose_ratio']-benchmarks[0][0]))
    #     print("Nose ratio diff:            {}".format(p['nose_ratio']-benchmarks[1][0]))
    #     print("Mouth ratio diff:           {}".format(p['mouth_ratio']-benchmarks[2][0]))
    #     print("Eyes-Nose angle ratio diff: {}".format(p['eyes_nose_angle_ratio']-benchmarks[3][0]))
    #     print("Eyes-Mouth ratio diff:      {}".format(p['eyes_mouth_ratio']-benchmarks[4][0]))
    #     print("Rating:                     {}".format(rate_face(benchmarks, [
    #         p['eyes_nose_ratio'],
    #         p['nose_ratio'],
    #         p['mouth_ratio'],
    #         p['eyes_nose_angle_ratio'],
    #         p['eyes_mouth_ratio']
    #     ])))
    #     print("\n")

    # avg_eyes_nose_ratio = avg_eyes_nose_ratio/len(people)
    # avg_nose_ratio = avg_nose_ratio/len(people)
    # avg_mouth_ratio = avg_mouth_ratio/len(people)
    # avg_eyes_nose_angle_ratio = avg_eyes_nose_angle_ratio/len(people)
    # avg_eyes_mouth_ratio = avg_eyes_mouth_ratio/len(people)
    
    # var_eyes_nose_ratio = 0
    # var_nose_ratio = 0
    # var_mouth_ratio = 0
    # var_eyes_nose_angle_ratio = 0
    # var_eyes_mouth_ratio = 0
    # for p in people:
    #     var_eyes_nose_ratio += math.pow(p['eyes_nose_ratio'] - avg_eyes_nose_ratio, 2)
    #     var_nose_ratio += math.pow(p['nose_ratio'] - avg_nose_ratio, 2)
    #     var_mouth_ratio += math.pow(p['mouth_ratio'] - avg_mouth_ratio, 2)
    #     var_eyes_nose_angle_ratio += math.pow(p['eyes_nose_angle_ratio'] - avg_eyes_nose_angle_ratio, 2)
    #     var_eyes_mouth_ratio += math.pow(p['eyes_mouth_ratio'] - avg_eyes_mouth_ratio, 2)

    # var_eyes_nose_ratio = var_eyes_nose_ratio/len(people)
    # var_nose_ratio = var_nose_ratio/len(people)
    # var_mouth_ratio = var_mouth_ratio/len(people)
    # var_eyes_nose_angle_ratio = var_eyes_nose_angle_ratio/len(people)
    # var_eyes_mouth_ratio = var_eyes_mouth_ratio/len(people)

    
    # print "Avg eyes ratio: {}".format(avg_eyes_nose_ratio)
    # print "SD eyes ratio: {}".format(math.sqrt(var_eyes_nose_ratio))
    
    # print("\n")
    # print "Avg nose ratio: {}".format(avg_nose_ratio)
    # print "SD nose ratio: {}".format(math.sqrt(var_nose_ratio))
    
    # print("\n")
    # print "Avg mouth ratio: {}".format(avg_mouth_ratio)
    # print "SD mouth ratio: {}".format(math.sqrt(var_mouth_ratio))
    # print("\n")
    # print "Avg eyes-nose ratio: {}".format(avg_eyes_nose_angle_ratio)
    # print "SD eyes-nose ratio: {}".format(math.sqrt(var_eyes_nose_angle_ratio))
    # print("\n")
    # print "Avg eyes-mouth ratio: {}".format(avg_eyes_mouth_ratio)
    # print "SD eyes-mouth ratio: {}".format(math.sqrt(var_eyes_mouth_ratio))
    # print("\n")

    # if write_to_file:
    #     current_dir = os.path.dirname(__file__)
    #     file_path = '../Data/{}'.format("benchmarks.txt")
    #     file_rel_path = os.path.join(current_dir, file_path)
    #     destfile = open(file_rel_path, 'w')

    #     destfile.write("{};{}\r\n".format(avg_eyes_nose_ratio, math.sqrt(var_eyes_nose_ratio)))
    #     destfile.write("{};{}\r\n".format(avg_nose_ratio, math.sqrt(var_nose_ratio)))
    #     destfile.write("{};{}\r\n".format(avg_mouth_ratio, math.sqrt(var_mouth_ratio)))
    #     destfile.write("{};{}\r\n".format(avg_eyes_nose_angle_ratio, math.sqrt(var_eyes_nose_angle_ratio)))
    #     destfile.write("{};{}\r\n".format(avg_eyes_mouth_ratio, math.sqrt(var_eyes_mouth_ratio)))

    #     destfile.close()
    #     print("Done writing.")