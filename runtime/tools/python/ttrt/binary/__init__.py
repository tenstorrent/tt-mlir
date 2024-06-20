from ._C import load_from_path

import json

def as_dict(bin):
    return json.loads(bin.as_json())
