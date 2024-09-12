import argparse
import datetime
import typing as tp

import numpy as np
import scipy.optimize
import tensorflow as tf

from foam.pitzDaily.get_foam_results import get_normalized_maps
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
    step: float = .02

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


def _loss_pde_2(network, fun_x_train):
    lambda_1 = 1
    lambda_2 = 1

    g = tf.Variable(fun_x_train, dtype='float64', trainable=False)

    x = g[:, 0:1]
    y = g[:, 1:2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)

        psi_and_p = network.evaluate(tf.stack([x[:, 0], y[:, 0]], axis=1))

        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = tape.gradient(psi, y)
        v = -tape.gradient(psi, x)

        u_x = tape.gradient(u, x)
        u_xx = tape.gradient(u_x, x)
        u_y = tape.gradient(u, y)
        u_yy = tape.gradient(u_y, y)

        v_x = tape.gradient(v, x)
        v_xx = tape.gradient(v_x, x)
        v_y = tape.gradient(v, y)
        v_yy = tape.gradient(v_y, y)

        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)

    del tape

    f_u = lambda_1 * (u * u_x + v * u_y) + lambda_2 * p_x - (u_xx + u_yy)
    f_v = lambda_1 * (u * v_x + v * v_y) + lambda_2 * p_x - (v_xx + v_yy)

    loss_f = tf.reduce_mean(tf.square(f_u)) + tf.reduce_mean(tf.square(f_v))

    return loss_f


def _loss_pde(network, fun_x_train):

    g = tf.Variable(fun_x_train, dtype='float64', trainable=False)

    x = g[:, 0:1]
    t = g[:, 1:2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)

        g = tf.stack([x[:, 0], t[:, 0]], axis=1)

        u = network.evaluate(g)

        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
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
        default=10,
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

    ux_map, uy_map, p_map = get_normalized_maps()

    print(ux_map.shape)

    # Plot.scatter_3d('training', f_border,
    #                 np.hstack((x_train, np.zeros(len(x_train))[:, None])))

    PINN = Network(args.iter, f_border, x_train, _loss_pde_2)

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

    g = tf.Variable(x_test, dtype='float64', trainable=False)

    x = g[:, 0:1]
    y = g[:, 1:2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)

        psi_and_p = PINN.evaluate(tf.stack([x[:, 0], y[:, 0]], axis=1))

        psi = psi_and_p[:, 0:1]

    u = tape.gradient(psi, y)
    v = -tape.gradient(psi, x)

    del tape

    Plot.scatter_3d('prediction', f_border, np.hstack((x_test, u)))
