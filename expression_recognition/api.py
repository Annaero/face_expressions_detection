# -*- coding: utf-8 -*-

import dlib
import numpy as np

from skimage import io
from .detectors import is_mouth_opened, is_smiling

try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
    print("pip install git+https://github.com/annaero/face_recognition_models")
    quit()

face_detector = dlib.get_frontal_face_detector()
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

def detect_open_mouth(landmarks):
    return is_mouth_opened(landmarks)

def detect_smile(landmarks):
    return is_smiling(landmarks)

def read_landmarks(files):
    """Read precalculated landmarks from list of files.
    Each face landmarks should be in separate file.
    function expects that landmaks will be in the second line of file,
    in format x1,x2,...xn,yn.

    Args:
        files (list or tuple): list of paths to files with landmarks

    Returns:
        np.array: readed landmarks

    """

    landmarks = []
    for f in files:
        with open(f, "r") as f:
            lines = f.readlines()
            landmark_line = lines[1]
            coordinates = [float(c) for c in landmark_line.strip().split(" ")]

            xlist = coordinates[::2]
            ylist = coordinates[1::2]

            norm_coordinates = _normalize_landmarks(xlist, ylist)
            landmarks.append(norm_coordinates)
    return landmarks


def compute_landmarks(files, indexes=False):
    """Computes landmarks from list of image files.

    Args:
        files (list or tuple): list of paths to files with landmarks
        indexes (bool): if True, than function will return list with mask array
                        that shows for what files algorithm is unable to detect face.
    Returns:
        np.array: readed landmarks or tuple where first element is readed landmarks 
                  and second is mask

    """

    landmarks = []
    indexes = []

    for f in files:
        img = io.imread(f)
        norm_coordinates = _get_landmarks(img)

        if norm_coordinates is not None:
            indexes.append(1)
            landmarks.append(norm_coordinates)
        else:
            indexes.append(0)

    ret = np.stack(landmarks)

    if not indexes:
        return ret
    return ret, indexes


def _get_landmarks(image):
    """Detect faces on image and compute landmarks on face.
    In current verion find landmarks only for first founded face.

    Args:
        image (str): path to the image file
    Returns:
        np.array: landmarks

    """
    detections = face_detector(image, 1)
    
    if not detections:
        return None
    
    d = detections[0] #to simplify, process only first founded face

    shape = pose_predictor_68_point(image, d)
    xlist = []
    ylist = []
    for i in range(0, 68):
        xlist.append(float(shape.part(i).x))
        ylist.append(float(shape.part(i).y))

    return _normalize_landmarks(xlist, ylist)


def _normalize_landmarks(xlist, ylist):
    """Takes coordinates of landmarks and normalize them
    to place landmarks in rectangle with size form -0.5 to 0.5
    on both coordinates.

    Args:
        xlist: list of x coordinates of landmarks
        ylist: list of y coordinates of landmarks

    Returns:
        np.array: 2d array of normalized landmarks coordinates

    """

    xs = np.array(xlist)
    ys = np.array(ylist)
    
    ys -= ys.min()
    ys = ys / ys.max()
    ys -= ys.mean()
    
    xs -= xs.min()
    xs = xs / xs.max()
    xs -= xs.mean()
     
    coordinates = np.vstack([xs, ys]).T
    return coordinates
