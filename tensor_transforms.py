import random

import torch


def convert_to_coord_format(b, h, w, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    return torch.cat((x_channel, y_channel), dim=1).squeeze(0)

def convert_to_coord_with_t(b, h, w, ts, device='cpu', integer_values=False):
    coords = {}
    for t in ts:
        if integer_values:
            x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
            y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
            t_channel = torch.ones((b, 1, w, h), dtype=torch.float, device=device)*t
        else:
            t_norm = t/max(ts)
            x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
            y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
            t_channel = torch.ones((b, 1, w, h), dtype=torch.float, device=device)*t_norm
        coords[t] = torch.cat((x_channel, y_channel, t_channel), dim=1).squeeze(0)
    return coords

def convert_to_coord_uneven_t(b, h, w, t, unit = 365, sin = False, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
        t_channel = torch.ones((b, 1, w, h), dtype=torch.float, device=device)*t
    else:
#         t_norm = t/unit
        if sin:
            t_norm = torch.sin(torch.tensor(t, dtype=torch.float))
        else:
            t_norm = t/unit
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
        t_channel = torch.ones((b, 1, w, h), dtype=torch.float, device=device)*t_norm
    coords = torch.cat((x_channel, y_channel, t_channel), dim=1).squeeze(0)
    return coords


def random_crop(tensor, size):
    assert tensor.dim() == 4, tensor.shape  # frames x channels x h x w
    h, w = tensor.shape[-2:]
    h_start = random.randint(0, h - size) if h - size > 0 else 0
    w_start = random.randint(0, w - size) if w - size > 0 else 0
    return tensor[:, :, h_start: h_start + size, w_start: w_start + size]

def random_crop_dim3(tensor, size):
    assert tensor.dim() == 3, tensor.shape  # frames x channels x h x w
    h, w = tensor.shape[-2:]
    h_start = random.randint(0, h - size) if h - size > 0 else 0
    w_start = random.randint(0, w - size) if w - size > 0 else 0
    return tensor[:, h_start: h_start + size, w_start: w_start + size], h_start, w_start


def patch_crop_dim3(tensor, h_start, w_start, size):
    assert tensor.dim() == 3, tensor.shape  # frames x channels x h x w
    return tensor[:, h_start: h_start + size, w_start: w_start + size]


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        return random_crop(tensor, self.size)

class RandomCropDim3:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        return random_crop_dim3(tensor, self.size)


def random_horizontal_flip(tensor):
    flip = random.randint(0, 1)
    if flip:
        return tensor.flip(-1)
    else:
        return tensor


def identity(tensor):
    return tensor
