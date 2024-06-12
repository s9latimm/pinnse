import argparse
import os

import tensorflow as tf

from network import NNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from pyDOE import lhs  # Latin Hypercube Sampling

# generates same random numbers each time
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version: {}".format(tf.__version__))

data = scipy.io.loadmat('burgers_shock_mu_01_pi.mat')  # Load data from file
x = data['x']  # 256 points between -1 and 1 [256x1]
t = data['t']  # 100 time points between 0 and 1 [100x1]
usol = data['usol']  # solution of 256x100 grid points

X, T = np.meshgrid(x, t)
''' X_u_test = [X[i],T[i]] [25600,2] for interpolation'''
X_u_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Domain bounds
lb = X_u_test[0]  # [-1. 0.]
ub = X_u_test[-1]  # [1.  0.99]
'''
   Fortran Style ('F') flatten,stacked column wise!
   u = [c1 
        c2
        .
        .
        cn]

   u =  [25600x1] 
'''
u = usol.flatten('F')[:, None]

pbar = None


def trainingdata(N_u, N_f):
    """Boundary Conditions"""

    # Initial Condition -1 =< x =<1 and t = 0
    leftedge_x = np.hstack((X[0, :][:, None], T[0, :][:, None]))  # L1
    leftedge_u = usol[:, 0][:, None]

    # Boundary Condition x = -1 and 0 =< t =<1
    bottomedge_x = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))  # L2
    bottomedge_u = usol[-1, :][:, None]

    # Boundary Condition x = 1 and 0 =< t =<1
    topedge_x = np.hstack((X[:, -1][:, None], T[:, 0][:, None]))  # L3
    topedge_u = usol[0, :][:, None]

    all_X_u_train = np.vstack(
        [leftedge_x, bottomedge_x,
         topedge_x])  # X_u_train [456,2] (456 = 256(L1)+100(L2)+100(L3))
    all_u_train = np.vstack([leftedge_u, bottomedge_u,
                             topedge_u])  # corresponding u [456x1]

    # choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False)

    X_u_train = all_X_u_train[idx, :]  # choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idx, :]  # choose corresponding u
    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack(
        (X_f_train, X_u_train))  # append training points to collocation points

    return X_f_train, X_u_train, u_train


def solutionplot(u_pred, X_u_train, u_train):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred,
                  interpolation='nearest',
                  cmap='rainbow',
                  extent=[T.min(), T.max(), X.min(),
                          X.max()],
                  origin='lower',
                  aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1],
            X_u_train[:, 0],
            'kx',
            label='Data (%d points)' % (u_train.shape[0]),
            markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$u(x,t)$', fontsize=10)
    ''' 
    Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    '''

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, usol.T[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.25s$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, usol.T[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50s$', fontsize=10)
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.35),
              ncol=5,
              frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, usol.T[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75s$', fontsize=10)

    plt.savefig('Burgers.png', dpi=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument(
        '--iter',
        type=int,
        metavar='<iter>',
        default=5000,
    )
    args = parser.parse_args()

    N_u = 100  # Total number of data points for 'u'
    N_f = 10000  # Total number of collocation points

    # Training data
    X_f_train, X_u_train, u_train = trainingdata(N_u, N_f)

    PINN = NNetwork(args.iter, ub, lb, X_u_train, u_train, X_f_train, X_u_test,
                    u)

    init_params = PINN.get_weights().numpy()

    start_time = time.time()

    # train the model with Scipy L-BFGS optimizer
    results = scipy.optimize.minimize(
        fun=PINN.optimize,
        x0=init_params,
        args=(),
        method='L-BFGS-B',
        jac=
        True,  # If jac is True, fun is assumed to return the gradient along with the objective function
        callback=PINN.callback,
        options={
            'disp': None,
            'maxcor': 200,
            'ftol': 1 * np.finfo(
                float
            ).eps,  # The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
            'gtol': 5e-8,
            'maxfun': args.iter * 10,
            'maxiter': args.iter,
            'iprint': -1,
            'maxls': 50
        })

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    print(results)

    PINN._set_weights(results.x)
    ''' Model Accuracy '''
    u_pred = PINN.evaluate(X_u_test)

    error_vec = np.linalg.norm((u - u_pred), 2) / np.linalg.norm(
        u, 2)  # Relative L2 Norm of the error (Vector)
    print('Test Error: %.5f' % (error_vec))

    u_pred = np.reshape(u_pred, (256, 100),
                        order='F')  # Fortran Style ,stacked column wise!
    ''' Solution Plot '''
    solutionplot(u_pred, X_u_train, u_train)
