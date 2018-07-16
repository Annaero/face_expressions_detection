# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from warnings import warn

from .feature_utils import flatten, to_dists_dataset

#try:
    #mouth_opened_model = joblib.load("models/mouth_opened_model.sav")
smile_model = joblib.load("models/smile_model.sav")
#except:
#    warn("""Models "models/mouth_opened_model.sav" and "models/smile_model.sav".\n
#            Please prepare models using "experiments/models_selection.ipynb" and rebuild package""")

def is_mouth_opened(landmarks):
    #features = flatten(landmarks)
    #prediction = mouth_opened_model.predict(features)
    prediction = [0 for i in range(landmarks.shape[0])] # mock
    return prediction

def is_smiling(landmarks):
    features = to_dists_dataset(landmarks)
    prediction = smile_model.predict(features)
    return prediction