import torch
import numpy as np


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


def make_model_name(cfg):
    return cfg['model'] + '_' + cfg['dataset_name']


if __name__ == '__main__':
    print(f'{make_divisible(32)}')
