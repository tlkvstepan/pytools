# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
from collections import defaultdict

import torch as th


def create_meshgrid(width, height, is_cuda):
    x, y = th.meshgrid([th.arange(0, width), th.arange(0, height)])
    x, y = (x.transpose(0, 1).float(), y.transpose(0, 1).float())
    if is_cuda:
        x = x.cuda()
        y = y.cuda()
    return x, y


def list_of_dictionaries_to_dictionary_of_lists(list_of_dictionaries):
    dictionary_of_lists = defaultdict(lambda: [])
    for dictionary in list_of_dictionaries:
        for item_name, item_value in dictionary.items():
            dictionary_of_lists[item_name].append(item_value)
    return dictionary_of_lists
