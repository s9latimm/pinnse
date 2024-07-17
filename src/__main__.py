import argparse
import datetime
import os

import tensorflow as tf

from network import Network
from burgers import trainingdata, _loss_pde, solutionplot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import scipy.optimize
import scipy.io
import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling

# generates same random numbers each time
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version: {}".format(tf.__version__))

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    X, x, T, t, usol, u, ub, lb, X_u_test, X_f_train, X_u_train, u_train = trainingdata(
        N_u, N_f)

    PINN = Network(args.iter, ub, lb, X_u_train, u_train, X_f_train, X_u_test,
                   u, _loss_pde)

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
    solutionplot(u_pred, X_u_train, u_train, timestamp, args, X, x, T, t, usol)
