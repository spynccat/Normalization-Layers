import numpy as np
import matplotlib.pyplot as plt

from batchnormalization import *


def print_mean_std(x, axis=0):
    print('--------------------------')
    print('mean: ', x.mean(axis=axis))
    print('std: ', x.std(axis=axis))
    print('--------------------------')



np.random.seed(123)

N, hidden1, hidden2, out = 100, 40, 30, 3
x = np.random.randn(N, hidden1)
w1 = np.random.randn(hidden1, hidden2)
w2 = np.random.randn(hidden2, out)
# Linear -> ReLU -> Linear
a = np.maximum(0, x.dot(w1)).dot(w2)

# Before Batch Normalization
print_mean_std(a, axis=0)

gamma = np.ones((out,))
beta = np.zeros((out,))


a_norm, cache = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
# After Batch Normalization (gamma=1, beta=0)
print_mean_std(a_norm, axis=0)

gamma = np.asarray([1.0, 2.0, 3.0])
beta = np.asarray([11.0, 12.0, 13.0])

a_norm, cache = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
# After Batch Normalization (gamma=gamma, beta=beta)
print_mean_std(a_norm, axis=0)


