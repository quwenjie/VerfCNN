"""
Script for running inference of model in C using ctypes
"""
import argparse

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import ctypes
import os
import sys
from torch.utils.data import DataLoader

def load_c_lib(library):
    """
    Load C shared library
    :param library: 
    :return:
    """
    try:
        c_lib = ctypes.CDLL(f"{os.path.dirname(os.path.abspath(__file__))}/{library}")
    except OSError:
        print("Unable to load the requested C library")
        sys.exit()
    return c_lib

def ensure_contiguous(array):
    """
    Ensure that array is contiguous
    :param array:
    :return:
    """
    return np.ascontiguousarray(array) if not array.flags['C_CONTIGUOUS'] else array




def run_convnet(x, c_lib):
    """
    Call 'run_mlp' function from C in Python
    :param x:
    :param c_lib:
    :return:
    """
    N = len(x)
    x = x.flatten()
    x = ensure_contiguous(x.numpy())
    x = x.astype(np.intc)

    class_indices = ensure_contiguous(np.zeros(N, dtype=np.uintc))

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_run_convnet = c_lib.run_convnet
    c_run_convnet.argtypes = (c_int_p, c_uint_p)
    c_run_convnet.restype = None
    c_run_convnet(x.ctypes.data_as(c_int_p), class_indices.ctypes.data_as(c_uint_p))

    return np.ctypeslib.as_array(class_indices, N)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for testing post-training quantization of a pre-trained model in C",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)

    args = parser.parse_args()

    mnist_testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ]))

    print(f'Start verifiable inference:')
    
    test_loader = DataLoader(mnist_testset, batch_size=args.batch_size, num_workers=1, shuffle=False)

    # load c library
    c_lib = load_c_lib(library='convnet.so')
    FXP_VALUE=6
    acc = 0
    cnt=0
    for samples, labels in test_loader:
        samples = (samples * (2 ** FXP_VALUE)).round() 
        preds = run_convnet(samples, c_lib).astype(int)
        acc += (torch.from_numpy(preds) == labels).sum()
        print("Predicted label ",preds, "ground-truth: ",labels.item())
        exit(0)
        cnt+=1
        if cnt%20==0:
            print("now ac: ",acc/cnt)
        #if acc>10:
        #    break
    print(f"Accuracy: {(acc / len(mnist_testset.data)) * 100.0:.2f}%")
