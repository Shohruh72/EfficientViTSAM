import torch
import torch.nn as nn
from inspect import signature
from functools import partial
import torch.nn.functional as F


# ---------------------- Network helper functions -------------------------
def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


norm_list = {"bn2d": nn.BatchNorm2d, "ln": nn.LayerNorm, "ln2d": LayerNorm2d}


def build_kwargs_from_config(config, target_func):
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def build_norm(name="bn2d", num_features=None, **kwargs):
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in norm_list:
        norm_cls = norm_list[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


activation_list = {"relu": nn.ReLU, "relu6": nn.ReLU6, "hswish": nn.Hardswish, "silu": nn.SiLU,
                   "gelu": partial(nn.GELU, approximate="tanh")}


def build_act(name: str, **kwargs):
    if name in activation_list:
        act_cls = activation_list[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def resize(x, size=None, scale_factor=None, mode="bicubic", align_corners=False, ):
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size, scale_factor, mode, align_corners)
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size, scale_factor, mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


class UpSampleLayer(nn.Module):
    def __init__(self, mode="bicubic", size=None, factor=2, align_corners=False):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)
