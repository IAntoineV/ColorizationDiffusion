

import torch




LAB_MATRIX = torch.tensor([[1/3,1/3,1/3], [1,-1,0], [-1/2,-1/2,1]]).T
AB_MATRIX = torch.tensor([[1,-1,0], [-1/2,-1/2,1]]).T
LAB_INVERSE = torch.linalg.inv(LAB_MATRIX)


def rgb_to_lab(x, device=None):
    x = x.permute(0, 2, 3, 1)  # From (batch_size, C, H, W) to (batch_size, H, W, C)

    M = LAB_MATRIX
    if device is not None:
        M = LAB_MATRIX.to(device)
    lab = x @ M

    # Permute back to return to (batch_size, C, H, W)
    lab = lab.permute(0, 3, 1, 2)
    return lab

def normalize_lab(x):
    x[:,0] = 2*x[:,0]-1
    return x

def unnormalize_lab(x):
    x[:,0] = (x[:,0]+1)/2
    return x
def lab_to_rgb(x, device=None):
    x = x.permute(0, 2, 3, 1)  # From (batch_size, C, H, W) to (batch_size, H, W, C)

    M = LAB_INVERSE
    if device is not None:
        M = LAB_INVERSE.to(device)
    rgb = x @ M

    # Permute back to return to (batch_size, C, H, W)
    rgb = rgb.permute(0, 3, 1, 2)
    return rgb
