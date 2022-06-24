from argparse import Namespace
from copy import deepcopy


def convert_dict_to_namespace(d):
    d_clone = deepcopy(d)

    for k, v in d.items():
        if isinstance(v, dict):
            d_clone[k] = convert_dict_to_namespace(v)
        else:
            d_clone[k] = v

    return Namespace(**d_clone)


if __name__ == "__main__":
    d = {
        'AAA': {
            'bbb': {
                'gg': 1
            },
            'cc': 3
        },
        'DDDD': [1,2,3]
    }
