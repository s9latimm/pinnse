import argparse
import datetime
import typing as tp

import numpy as np
import scipy.optimize
import tensorflow as tf
from multipledispatch import dispatch

from network import Network
from src.plot import Plot


def disc_1d(x: tp.Tuple[float, float], step: float) -> np.ndarray:
    return np.linspace(x[0] + step, x[1] - step,
                       int(abs(x[1] - x[0]) / step) - 1)


def disc_2d(x: tp.Tuple[float, float], y: tp.Tuple[float, float],
            step: float) -> tp.Tuple[tp.Tuple[int], np.ndarray]:
    x, y = np.mgrid[x[0]:x[1] + step:step, y[0]:y[1] + step:step]

    return x.shape, np.hstack([
        x.flatten()[:, None],
        y.flatten()[:, None],
    ])


def span(x: tp.Union[np.ndarray, float], y: tp.Union[np.ndarray, float],
         z: tp.Union[np.ndarray, float]) -> np.ndarray:
    stack = [x, y, z]
    shape = None
    for s in stack:
        if not np.isscalar(s):
            shape = s.shape

    for i, s in enumerate(stack):
        if np.isscalar(s):
            stack[i] = np.full(shape, s)[:, None]
        else:
            stack[i] = stack[i][:, None]

    return np.hstack(stack)


def data(
        sample: int
) -> tp.Tuple[tp.Tuple[int], np.ndarray, np.ndarray, np.ndarray]:
    step: float = .01

    f_border = np.vstack([
        span(disc_1d((-1, 0), step), 0, 1),
        span(disc_1d((0, 1), step), .5, 0),
        span(-1, disc_1d((0, 1), step), 0),
        span(0, disc_1d((0, .5), step), 0),
        span(1, disc_1d((.5, 1), step), 0),
        span(disc_1d((0, 1), step), 0, 0),
        span(1, disc_1d((0, .5), step), 0),
        span(disc_1d((-1, 1), step), 1, .5),
    ])

    shape, x_test = disc_2d((-1, 1), (0, 1), step)

    np.random.seed(42)

    x_train = x_test[
        np.random.choice(len(x_test), int(len(x_test) *
                                          sample), replace=False), :]

    return shape, x_test, x_train, f_border


def _loss_pde2(self, x, y, t):
    lambda_1 = 1
    lambda_2 = 1

    psi_and_p = self.neural_net(tf.concat([x, y, t], 1), self.weights,
                                self.biases)
    psi = psi_and_p[:, 0:1]
    p = psi_and_p[:, 1:2]

    u = tf.gradients(psi, y)[0]
    v = -tf.gradients(psi, x)[0]

    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]

    v_t = tf.gradients(v, t)[0]
    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    v_xx = tf.gradients(v_x, x)[0]
    v_yy = tf.gradients(v_y, y)[0]

    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]

    f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
    f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

    return u, v, p, f_u, f_v


def _loss_pde(network, fun_x_train):

    g = tf.Variable(fun_x_train, dtype='float64', trainable=False)

    x = g[:, 0:1]
    y = g[:, 1:2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)

        g = tf.stack([x[:, 0], y[:, 0]], axis=1)

        z = network.evaluate(g)

        u_x = tape.gradient(z, x)
        u_t = tape.gradient(z, y)
        u_xx = tape.gradient(u_x, x)

    del tape

    u = (network.evaluate(g))

    f = u_t + u * u_x - (0.01 / np.pi) * u_xx

    loss_f = tf.reduce_mean(tf.square(f))

    return loss_f


def refine_grid(x, y, step):
    return x[::step].transpose()[::step].transpose(), y[::step].transpose(
    )[::step].transpose()


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument(
        '--iter',
        type=int,
        metavar='<iter>',
        default=500,
    )
    parser.add_argument(
        '--sample',
        type=float,
        metavar='<sample>',
        default=.25,
    )
    args = parser.parse_args()

    # Training data
    shape, x_test, x_train, f_border = data(args.sample)

    Plot.scatter_3d('training', f_border,
                    np.hstack((x_train, np.zeros(len(x_train))[:, None])))

    x, y = refine_grid(x_test[:, 0:1].reshape(shape),
                       x_test[:, 1:2].reshape(shape), 10)

    Plot.arrows('training', x, y, x, -y)

    PINN = Network(args.iter, f_border, x_train, _loss_pde)

    results = scipy.optimize.minimize(fun=PINN.optimize,
                                      x0=PINN.get_weights().numpy(),
                                      args=(),
                                      method='L-BFGS-B',
                                      jac=True,
                                      callback=PINN.callback,
                                      options={
                                          'disp': None,
                                          'maxcor': 200,
                                          'ftol': 1 * np.finfo(float).eps,
                                          'gtol': 5e-8,
                                          'maxfun': args.iter * 10,
                                          'maxiter': args.iter,
                                          'iprint': -1,
                                          'maxls': 50
                                      })

    print(results)

    PINN._set_weights(results.x)
    ''' Model Accuracy '''
    u_pred = PINN.evaluate(x_test)

    Plot.scatter_3d('prediction', f_border, np.hstack((x_test, u_pred)))

    # solutionplot(u_pred, X_u_train, u_train, timestamp, args, X, x, T, t, usol)
