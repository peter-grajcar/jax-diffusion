import json

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return AttrDict(config)

