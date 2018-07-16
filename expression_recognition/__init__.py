__author__ = """Aleksei Krikunov"""
__email__ = 'alexey.v.krikunov@yandex.ru'
__version__ = '0.1.3'

from .api import compute_landmarks, read_landmarks, detect_open_mouth, detect_smile
from .feature_utils import get_mouthes_only, flatten, to_dists_dataset
