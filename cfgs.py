import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import scipy.io as io
import pandas as pd
import pickle
from dataset import *

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0001
milestone = [200, 400, 800]
Epoch = 500

def plot_embedding_2d(X, labels, title=None, file_root='./results'):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(labels[i]),
                color=plt.cm.Set2(int(labels[i])/10.),
                # fontdict={'weight': 'bold',
                #           'size': 9}
                fontsize=9
                )
    if title is not None:
        # plt.title(title)
        plt.savefig(os.path.join(file_root, title.split(' ')[0]))
        # plt.show()
    else:
        plt.title(file_root.split('/')[-2] + '_' + file_root.split('/')[-1])
        plt.savefig(os.path.join(file_root, file_root.split('/')[-2] + '_' + file_root.split('/')[-1]))

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP #熵
    # H = np.log(sumP) - (np.sum(P * np.log(P))) / sumP
    # print('H:', H)
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):

    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    # print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X) #距离矩阵
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # print(i)
        # Print progress
        # if i % 500 == 0:
        #     print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # print('tries =', tries)
            # If not, increase or decrease precision
            if Hdiff > 0:
                # print('11')
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])


            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    return P

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / y_pred.size

def load_data(data_name):
    if data_name == 'mnist':
        return load_mnist()
        
    elif data_name == 'cifar10':
        return load_cifar10()
        
    elif data_name == 'norb':
        return load_norb()
    