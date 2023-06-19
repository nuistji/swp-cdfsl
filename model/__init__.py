from model.base import BaseModel
from model.byol import BYOL
from model.moco import MoCo
from model.simclr import SimCLR
from model.simsiam import SimSiam
from model.swav import SwAV
_model_class_map = {
    'base': BaseModel,
    'simclr': SimCLR,
    'byol': BYOL,
    'moco': MoCo,
    'simsiam': SimSiam,
    'swav': SwAV
}


def get_model_class(key):
    if key in _model_class_map:
        return _model_class_map[key]
    else:
        raise ValueError('Invalid model: {}'.format(key))

