import argparse
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from network import Network

import scipy.optimize
import scipy.io
import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling

def trainingdata(N_u, N_f):
    x = np.linspace(-np.pi, np.pi, 100)

    u = 0

    for k in range(1, 6):
        u += np.sin(2 * k * x) / (2 * k)

    X_u_test = tf.reshape(x, [100, 1])

    # Domain bounds
    lb = X_u_test[0]  # [-pi]
    ub = X_u_test[-1]  # [pi]

    u = tf.reshape(u, (100, 1))


    '''Boundary Conditions'''

    # Left egde (x = -π and u = 0)
    leftedge_x = -np.pi
    leftedge_u = 0

    # Right egde (x = -π and u = 0)
    rightedge_x = np.pi
    rightedge_u = 0

    X_u_train = np.vstack([leftedge_x, rightedge_x])  # X_u_train [2,1]
    u_train = np.vstack([leftedge_u, rightedge_u])  # corresponding u [2x1]

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f_train = lb + (ub - lb) * lhs(1, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))  # append training points to collocation points

    return u, ub, lb, X_u_test, X_f_train, X_u_train, u_train


def _loss_pde(network, x_to_train_f):
    g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)

    nu = 0.01 / np.pi

    x_f = g[:, 0:1]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_f)

        z = network.evaluate(x_f)
        u_x = tape.gradient(z, x_f)

    u_xx = tape.gradient(u_x, x_f)

    del tape

    source = 0

    for k in range(1, 6):
        source += (2 * k) * np.sin(2 * k * x_f)

    f = u_xx + source

    loss_f = tf.reduce_mean(tf.square(f))

    return loss_f

def solutionplot(X_u_test, u_pred, timestamp, args):
    fig, ax = plt.subplots()
    plt.plot(X_u_test, u_pred)
    plt.xlabel('x')
    plt.xlabel('u')
    plt.title('Poisson Equation')

    plt.savefig(f'../images/poisson_{timestamp}_{args.iter}.png', dpi=500)