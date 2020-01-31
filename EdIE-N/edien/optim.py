import math
import importlib
import torch


class TorchOptWrapper(object):
    """Wrap pytorch optimizers that want the params in the constructor."""

    def __init__(self, classname, **args):
        module = importlib.import_module('torch.optim')
        self.cls = getattr(module, classname)
        self.args = args
        for k, v in args.items():
            setattr(self, k, v)

    def __call__(self, params):
        return self.cls(params, **self.args)


class TorchScheduleWrapper(object):
    """Wrap pytorch schedulers."""

    def __init__(self, lib, classname, **args):
        module = importlib.import_module(lib)
        self.cls = getattr(module, classname)
        self.args = args
        for k, v in args.items():
            setattr(self, k, v)

    def __call__(self, *params):
        return self.cls(*params, **self.args)
