import math
import numpy as np
import torch

def xavier_fan_in_(tensor):
    """Initialize the given weight tensor with Xavier fan-in init.

    Unfortunately, :func:`torch.nn.init.xavier_uniform_` doesn't give
    us the choice to use fan-in init (always uses the harmonic mean).
    Therefore, we provide our own implementation.

    Args:
        tensor (torch.Tensor): Weight tensor that will be modified
            (initialized) in-place.
    """
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = 1. / math.sqrt(fan_in)
    # Note, std(Unif(-a, a)) = a / sqrt(3)
    a = math.sqrt(3.0) * std

    torch.nn.init._no_grad_uniform_(tensor, -a, a)

def calc_fan_in_and_out(shapes):
    """Calculate fan-in and fan-out.

    Note:
        This function expects the shapes of an at least 2D tensor.

    Args:
        shapes (list): List of integers.

    Returns:
        (tuple) Tuple containing:

        - **fan_in**
        - **fan_out**
    """
    assert len(shapes) > 1
    
    fan_in = shapes[1]
    fan_out = shapes[0]

    if len(shapes) > 2:
        receptive_field_size = int(np.prod(shapes[2:]))
    else:
        receptive_field_size = 1

    fan_in *= receptive_field_size
    fan_out *= receptive_field_size

    return fan_in, fan_out

if __name__ == '__main__':
    pass


