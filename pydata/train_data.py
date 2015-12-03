from xml.etree import ElementTree
import math
import numpy as np
import scipy.io as sio


#create function that gives "articulation index" from points
angle_marks = [0, -10*math.pi/180, -20*math.pi/180, -30*math.pi/180, -40*math.pi/180]
def assign_articulation_index(L):
    """Returns an articulation index, starting from 1, that represents an
    arbitrary phase of the walking cycle."""
    leg = L[1] - L[0]
    angle = math.atan2(-leg[1], leg[0]) + math.pi/2
    for i in range(len(angle_marks)):
        if angle_marks[i] <= angle:
            return i + 1
    return len(angle_marks) + 1


def get_train_data(anno_filename):
    """Collects training data from an annotation file. For each image, it
    collects its L and computes its articulation index."""
    annolist = ElementTree.parse(sys.argv[1]).getroot()
    aTrain = []
    LTrain = []
    boxesTrain = []
    imgnamesTrain = []
    for anno in annolist:
        annorect = anno.find('annorect')
        L = get_L(annorect)
        LTrain.append(L)
        imgnamesTrain.append(anno.find('image').find('name').text)
        boxesTrain.append(get_box(annorect))
        aTrain.append(assign_articulation_index(L))
    return (np.array(imgnamesTrain, dtype=np.object), np.atleast_2d(aTrain).T, np.array(LTrain), np.array(boxesTrain))


def get_box(annorect):
    box = np.array([
        int(annorect.find('x1').text),
        int(annorect.find('y1').text),
        int(annorect.find('x2').text),
        int(annorect.find('y2').text),
        int(annorect.find('score').text)
    ])
    return box


def get_L(annorect):
    """Gets L for one image in the annotation file. L is the collection of
    centroids from specific pairs of points."""
    objpos = annorect.find('objpos')
    x0 = np.array([int(objpos.find('x').text), int(objpos.find('y').text)])
    points = dict()
    annopoints = annorect.find('annopoints')
    for point in annopoints:
        id_ = point.find('id').text
        if id_ in points:
            print('duplicate point id: {}'.format(id_))
            exit(1)
        points[id_] = np.array([int(point.find('x').text), int(point.find('y').text)])
    if len(points) != 11:
        print('found {} point(s) instead of 11'.format(len(points)))
        exit(1)
    #(pairs: 1-2, 3-4, 5-6, 7-8, 8-9, 4-9, 9-10, and 11 alone)
    L = [x0]
    L.append((points['1'] + points['2'])/2)
    L.append((points['3'] + points['4'])/2)
    L.append((points['5'] + points['6'])/2)
    L.append((points['7'] + points['8'])/2)
    L.append((points['8'] + points['9'])/2)
    L.append((points['4'] + points['9'])/2)
    L.append((points['9'] + points['10'])/2)
    L.append(points['11'])
    return np.array(L)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print('usage: train_data.py annotation_file mat_filename')
        exit(1)
    imgnamesTrain, aTrain, LTrain, boxesTrain = get_train_data(sys.argv[1])
    sio.savemat(sys.argv[2], {'imgnamesTrain': imgnamesTrain, 'aTrain': aTrain, 'LTrain': LTrain, 'boxesTrain': boxesTrain})
