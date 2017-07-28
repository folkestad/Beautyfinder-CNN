import os
import math
import numpy as np

def get_all_ratings(factor=10):
    
    current_dir = os.path.dirname(__file__)
    file_name = 'attractiveness_rating.csv'
    file_path = '../Data/{}'.format(file_name)
    file_rel_path = os.path.join(current_dir, file_path)
    rating_file = open(file_rel_path, 'r')

    all_ratings = []
    
    for i,line in enumerate(rating_file):
        if i != 0:
            all_ratings.append(int(math.floor(factor*(float(line.replace('"', '').split(',')[1])))))
    # print all_ratings[0]
    return all_ratings

def one_hot_encode(labels, n_classes=50):
    array = np.array([ int(l) for l in labels ])
    one_hot = np.zeros((array.size, n_classes))
    one_hot[np.arange(array.size), array] = 1
    return one_hot


if __name__ == '__main__':
    all_ratings = get_all_ratings()
    one_hot_ratings = one_hot_encode(all_ratings, 50)
    for i in one_hot_ratings:
        print i
