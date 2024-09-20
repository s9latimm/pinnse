import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cpuinfo
import numpy as np
import psutil
import scipy
import tensorflow as tf
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from network import BaseNetwork
from src.plotter import Plotter
from src.timer import CallbackTimer

NU = 0.08
RHO = 1
STEPS = 10000
SAMPLE = 0
SEED = 42
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
DIR = Path('..') / 'output' / 'navier-stokes' / TIMESTAMP


class NavierStokesData:

    def __init__(self, sample: float = None):

        from foam.pitzDaily.get_foam_results import get_foam
        u, v, p = get_foam()

        def refine(x, step):
            return x[::step].transpose()[::step].transpose()

        self.u = refine(u, 10).transpose()
        self.v = refine(v, 10).transpose()
        self.p = refine(p, 10).transpose()

        self.grid = np.dstack(np.mgrid[0:80, 0:20])

        foam = np.hstack([
            self.grid[:, :, 0].flatten()[:, None],
            self.grid[:, :, 1].flatten()[:, None],
            self.u.flatten()[:, None],
            self.v.flatten()[:, None],
            self.p.flatten()[:, None],
        ])

        self.intake = np.hstack([
            self.grid[0, 0:20, 0][:, None],
            self.grid[0, 0:20, 1][:, None],
            np.zeros(20)[:, None],
            np.zeros(20)[:, None],
            np.zeros(20)[:, None],
        ])

        self.outtake = np.hstack([
            self.grid[-1, 0:20, 0][:, None],
            self.grid[-1, 0:20, 1][:, None],
            np.zeros(20)[:, None],
            np.zeros(20)[:, None],
            np.zeros(20)[:, None],
        ])

        border = np.vstack([
            np.hstack([
                self.grid[0, 0:10, 0][:, None],
                self.grid[0, 0:10, 1][:, None],
                np.full(10, 10)[:, None],
                np.zeros(10)[:, None],
                np.zeros(10)[:, None],
            ]),
            np.hstack([
                self.grid[0, 10:20, 0][:, None],
                self.grid[0, 10:20, 1][:, None],
                np.zeros(10)[:, None],
                np.zeros(10)[:, None],
                np.zeros(10)[:, None],
            ]),
            np.hstack([
                self.grid[1:, 0, 0][:, None],
                self.grid[1:, 0, 1][:, None],
                np.zeros(79)[:, None],
                np.zeros(79)[:, None],
                np.zeros(79)[:, None],
            ]),
            np.hstack([
                self.grid[1:, -1, 0][:, None],
                self.grid[1:, -1, 1][:, None],
                np.zeros(79)[:, None],
                np.zeros(79)[:, None],
                np.zeros(79)[:, None],
            ]),
        ])

        # corner
        for i in range(1, 10):
            border = np.vstack([
                border,
                np.hstack([
                    self.grid[i, -10:-1, 0][:, None],
                    self.grid[i, -10:-1, 1][:, None],
                    np.zeros(9)[:, None],
                    np.zeros(9)[:, None],
                    np.zeros(9)[:, None],
                ]),
            ])

        self.border = np.unique(border, axis=0)
        self.foam = np.unique(foam, axis=0)

        self.train = np.array([
            x for x in set(tuple(x) for x in self.foam) -
            set(tuple(x) for x in self.border)
        ])
        # assert (len(self.train) == len(self.foam) - len(self.border))

        if sample > 0:
            self.train = self.train[np.random.choice(
                len(self.train), int(len(self.train) *
                                     sample), replace=False), :]


class NavierStokesNetwork(BaseNetwork):

    def __init__(self, data: NavierStokesData):
        self.error = np.zeros(7)

        self.__nu = NU
        self.__rho = RHO

        self.__data = data

        super().__init__([2, 20, 20, 20, 20, 20, 20, 20, 20, 3], tf.nn.tanh)

    def evaluate(self, frame):
        g = tf.Variable(frame, dtype='float64', trainable=False)

        x = g[:, 0]
        y = g[:, 1]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)

            r = self.forward(tf.stack([x, y], axis=1))
            u = r[:, 0]
            v = -r[:, 1]
            p = r[:, 2]

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

        f = self.__rho * (u * u_x + v * u_y) + p_x - self.__nu * (u_xx + u_yy)
        g = self.__rho * (u * v_x + v * v_y) + p_y - self.__nu * (v_xx + v_yy)

        m = u_x + v_y

        return u, v, p, f, g, m

    def loss(self):
        stack = [
            self.__data.border,
            self.__data.intake,
            self.__data.outtake,
            self.__data.train,
        ]

        off = [0]
        for frame in stack:
            off.append(off[-1] + len(frame))

        u, v, p, f, g, m = self.evaluate(np.vstack(stack))

        error = []

        # border
        u_err = tf.reduce_sum(tf.square(self.__data.border[:, 2] - u[:off[1]]))
        v_err = tf.reduce_sum(
            tf.square(self.__data.border[:, 3] - v[:off[1]])) + tf.reduce_sum(
                tf.square(self.__data.outtake[:, 3] - v[off[2]:off[3]]))

        error.append(u_err)
        error.append(v_err)

        # transport
        t_err = tf.square(
            tf.reduce_sum(u[off[1]:off[2]]) - tf.reduce_sum(u[off[2]:off[3]]))
        s_err = tf.reduce_sum(
            tf.square(u[off[2]:off[3]] - u[off[2]:off[3]][::-1]))

        error.append(t_err)
        error.append(s_err)

        # pde
        f_err = tf.reduce_sum(tf.square(f[off[3]:]))
        g_err = tf.reduce_sum(tf.square(g[off[3]:]))
        m_err = tf.reduce_sum(tf.square(m[off[3]:]))

        error.append(f_err)
        error.append(g_err)
        error.append(m_err)

        self.error = np.vstack([
            self.error,
            [e.numpy() for e in error],
        ])

        return tf.reduce_sum(error)


def main():
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument(
        '--steps',
        type=int,
        metavar='<steps>',
        default=STEPS,
    )
    parser.add_argument(
        '--sample',
        type=float,
        metavar='<sample>',
        default=SAMPLE,
    )
    args = parser.parse_args()

    # from tensorflow.python.client import device_lib
    #
    # logging.info(device_lib.list_local_devices())

    logging.info(f'NU = {NU}')
    logging.info(f'STEPS = {STEPS}')
    logging.info(f'SAMPLE = {SAMPLE}')
    logging.info(f'SEED = {SEED}')
    logging.info(f'TIMESTAMP = {TIMESTAMP}')

    logging.info(f'CPU: {cpuinfo.get_cpu_info()["brand_raw"]} ')
    logging.info(f'LOGICAL: {psutil.cpu_count(logical=True)}')
    logging.info(f'MEMORY: {psutil.virtual_memory().total}')

    data = NavierStokesData(args.sample)

    Plotter.heatmap(
        'foam',
        data.grid[:, :, 0],
        data.grid[:, :, 1],
        [
            ('u', data.u),
            ('v', data.v),
            ('p', data.p),
        ],
        out=DIR / 'foam.png',
    )

    pinn = NavierStokesNetwork(data)

    for variable in pinn.trainable_variables:
        logging.info(f'{variable.name} {variable.shape}')

    def render(step, final=False):
        u, v, p, f, g, m = pinn.evaluate(data.foam)

        u_pred = u.numpy().reshape(data.grid[:, :, 0].shape)
        v_pred = -v.numpy().reshape(data.grid[:, :, 0].shape)
        p_pred = p.numpy().reshape(data.grid[:, :, 0].shape)

        Plotter.heatmap(
            f'pred {step}',
            data.grid[:, :, 0],
            data.grid[:, :, 1],
            [
                ('u', u_pred),
                ('v', v_pred),
                ('p', p_pred),
            ],
            out=DIR / f'pred_{step}.png',
        )

        if final:
            Plotter.heatmap(
                f'diff {step}',
                data.grid[:, :, 0],
                data.grid[:, :, 1],
                [
                    ('u', np.abs(u_pred - data.u)),
                    ('v', np.abs(v_pred - data.v)),
                    ('p', np.abs(p_pred - data.p)),
                ],
                grids=[
                    data.border[:, [0, 1]],
                ],
                out=DIR / f'diff_{step}.png',
            )

        Plotter.error(
            'error',
            [
                ('border', [
                    ('u', pinn.error[1:, 0]),
                    ('v', pinn.error[1:, 1]),
                ]),
                ('PDE', [
                    ('f', pinn.error[1:, 4]),
                    ('g', pinn.error[1:, 5]),
                    ('m', pinn.error[1:, 6]),
                ]),
                ('transport', [
                    ('t', pinn.error[1:, 2]),
                    ('s', pinn.error[1:, 3]),
                ]),
            ],
            out=DIR / f'err.svg',
        )

    with tqdm(total=args.steps, position=0,
              leave=True) as pbar, logging_redirect_tqdm():
        with CallbackTimer(logging.info):

            losses = []

            def callback(*_):
                if pbar.n % 100 == 0:
                    render(pbar.n)
                pbar.update(1)

            results = scipy.optimize.minimize(
                fun=pinn.optimize,
                x0=pinn.get_weights().numpy(),
                args=(),
                method='L-BFGS-B',
                jac=True,
                callback=callback,
                options={
                    'disp': None,
                    'maxcor': 200,
                    'ftol': 1 * np.finfo(float).eps,
                    'gtol': 5e-8,
                    'maxfun': args.steps * 10,
                    'maxiter': args.steps,
                    'iprint': -1,
                    'maxls': 50
                })

    logging.info(results)
    pinn.set_weights(results.x)

    render(args.steps, final=True)


class LogFilter(logging.Filter):

    def __init__(self, level):
        super().__init__()
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


if __name__ == '__main__':
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    DIR.mkdir(parents=True, exist_ok=False)
    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(format='%(message)s',
                        handlers=[
                            logging.FileHandler(DIR / 'log.txt', mode='w'),
                            logging.StreamHandler(sys.stdout)
                        ],
                        encoding='utf-8',
                        level=logging.INFO)
    try:
        main()
    except KeyboardInterrupt:
        pass
