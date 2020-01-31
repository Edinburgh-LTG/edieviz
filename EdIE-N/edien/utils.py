import subprocess
import numpy as np
from functools import reduce, partial
from itertools import chain
from collections import namedtuple


def deep_unpack(x, axis=0):
    # Unpack elements from nested iterators
    # at depth = axis
    # Assumes nested depth is same for all list
    for i in range(axis):
        x = tuple(chain.from_iterable(x))
    return x


def deep_apply(func, x, axis=0):
    # Apply func(i) elementwise to each element in x
    # at depth = axis
    # Assumes nested depth is same for all list
    assert(axis >= 0)
    num_map = axis + 1
    deep_func = reduce(lambda x, y: lambda z: tuple(partial(y, x)(z)),
                       [map, ] * num_map,
                       func)
    return deep_func(x)


def chain_namedtuples(*args):
    fields = tuple(chain.from_iterable(each._fields for each in args))
    data = tuple(chain.from_iterable(each for each in args))
    return namedtuple('ChainedNamedTuple', fields)(*data)


def get_current_git_hash():
    cli_out = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    # encode to str
    str_git_hash = cli_out.rstrip().decode()
    return str_git_hash
