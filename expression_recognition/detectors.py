# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from warnings import warn
from pkg_resources import resource_filename

from .feature_utils import flatten, to_dists_dataset

try:
    mo_model_file = resource_filename(__name__, "models/mouth_opened_model.sav")
    mouth_opened_model = joblib.load(mo_model_file)
except:
    warn("""Model "expression_recognition/models/mouth_opened_model.sav" not found.
            Please prepare models using "experiments/models_selection.ipynb" and rebuild package""")

try:
    smile_model_file = resource_filename(__name__, "models/smile_model.sav")
    smile_model = joblib.load(smile_model_file)
except:
    warn("""Model "expression_recognition/models/smile_model.sav" not found.
            Please prepare models using "experiments/models_selection.ipynb" and rebuild package""")


def is_mouth_opened(landmarks):
    features = flatten(landmarks)
    prediction = mouth_opened_model.predict(features)
    return prediction

def is_smiling(landmarks):
    features = to_dists_dataset(landmarks)
    prediction = smile_model.predict(features)
    return prediction