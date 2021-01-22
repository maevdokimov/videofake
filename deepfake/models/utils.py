import torch
import torch.nn.functional as F


def same_pad(x: torch.tensor, kernel_size: int, stride: int):
    in_height, in_width = x[-2:]

    if in_height % stride == 0:
        pad_along_height = max(kernel_size - stride, 0)
    else:
        pad_along_height = max(kernel_size - (in_height % stride), 0)
    if in_width % stride == 0:
        pad_along_width = max(kernel_size - stride, 0)
    else:
        pad_along_width = max(kernel_size - (in_width % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
