

import torch




LAB_MATRIX = torch.tensor([[1/3,1/3,1/3], [1,-1,0], [-1/2,-1/2,1]]).T
LAB_INVERSE = torch.linalg.inv(LAB_MATRIX)


def rgb_to_lab(x):
    return x @ LAB_MATRIX

def normalize_lab(x):
    x[:,:,:,0] = 2*x[:,:,:,0]-1
    return x

def unnormalize_lab(x):
    x[:,:,:,0] = (x[:,:,:,0]+1)/2
    return x
def lab_to_rgb(x):
    return x @ LAB_INVERSE
