from .mobilenet_v3 import get_mobilenet_v3_small
from .efficientnet_v2_l import get_efficientnet_v2_l

def get_model(name):
    if name == 'mobilenet_v3':
        return get_mobilenet_v3_small()
    elif name == 'efficientnet_v2_l':
        return get_efficientnet_v2_l()
    else:
        raise ValueError(f"Model {name} not supported")