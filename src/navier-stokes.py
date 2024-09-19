import argparse
import datetime

import numpy as np
import scipy
import tensorflow as tf
from tqdm import tqdm

from network import BaseNetwork
from src.plot import Plot

NU = 0.08
ITER = 1000
SAMPLE = .3
SEED = 42


class NavierStokesData:

    def __init__(self, sample: float = None):

        from foam.pitzDaily.get_foam_results import get_foam
        u, v, p = get_foam()

        def refine(x, step):
            return x[::step].transpose()[::step].transpose()

        self.u = refine(u, 10)[::-1].transpose()
        self.v = refine(v, 10)[::-1].transpose()
        self.p = refine(p, 10)[::-1].transpose()

        self.grid = np.dstack(np.mgrid[0:80, 0:20])

        foam = np.hstack([
            self.grid[:, :, 0].flatten()[:, None],
            self.grid[:, :, 1].flatten()[:, None],
            self.u.flatten()[:, None],
            self.v.flatten()[:, None],
            self.p.flatten()[:, None],
        ])

        border = np.vstack([
            np.hstack([
                self.grid[:, 0, 0][:, None],
                self.grid[:, 0, 1][:, None],
                self.u[:, 0][:, None],
                self.v[:, 0][:, None],
                self.p[:, 0][:, None],
            ]),
            np.hstack([
                self.grid[:, -1, 0][:, None],
                self.grid[:, -1, 1][:, None],
                self.u[:, -1][:, None],
                self.v[:, -1][:, None],
                self.p[:, -1][:, None],
            ]),
            np.hstack([
                self.grid[0, :, 0][:, None],
                self.grid[0, :, 1][:, None],
                self.u[0, :][:, None],
                self.v[0, :][:, None],
                self.p[0, :][:, None],
            ]),
            np.hstack([
                self.grid[-1, :, 0][:, None],
                self.grid[-1, :, 1][:, None],
                self.u[-1, :][:, None],
                self.v[-1, :][:, None],
                self.p[-1, :][:, None],
            ]),
        ])

        for i in range(1, 10):
            border = np.vstack([
                border,
                np.hstack([
                    self.grid[i, 1:10, 0][:, None],
                    self.grid[i, 1:10, 1][:, None],
                    self.u[i, 1:10][:, None],
                    self.v[i, 1:10][:, None],
                    self.p[i, 1:10][:, None],
                ]),
            ])

        self.border = np.unique(border, axis=0)
        self.foam = np.unique(foam, axis=0)

        self.train = np.array([
            x for x in set(tuple(x) for x in self.foam) -
            set(tuple(x) for x in self.border)
        ])
        assert (len(self.train) == len(self.foam) - len(self.border))
        print(len(self.train), len(np.unique(self.train, axis=0)))

        if sample is not None:
            self.border = np.vstack([
                self.border,
                self.train[np.random.choice(len(self.train),
                                            int(len(self.train) * sample),
                                            replace=False), :],
            ])
            self.train = np.array([
                x for x in set(tuple(x) for x in self.train) -
                set(tuple(x) for x in self.border)
            ])
        assert (len(self.border) == len(np.unique(self.border, axis=0)))


class NavierStokesNetwork(BaseNetwork):

    def __init__(self, data: NavierStokesData):
        self.u_err = []
        self.v_err = []
        self.f_err = []
        self.g_err = []
        self.m_err = []

        self.__viscosity = NU
        self.__density = 1.0

        self.__border = data.border
        self.__train = data.train

        super().__init__([2, 20, 20, 20, 20, 20, 20, 20, 20, 3], tf.nn.tanh)

    def evaluate(self, x_train):
        g = tf.Variable(x_train, dtype='float64', trainable=False)

        x = g[:, 0:1]
        y = g[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)

            psi = self.forward(tf.stack([x[:, 0], y[:, 0]], axis=1))
            u = psi[:, 0:1]
            v = psi[:, 1:2]
            p = psi[:, 2:3]

            # psi = psi_and_p[:, 0:1]
            # p = psi_and_p[:, 1:2]
            #
            # u = tape.gradient(psi, y)
            # v = -tape.gradient(psi, x)

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

        f = self.__density * (u * u_x +
                              v * u_y) + p_x - self.__viscosity * (u_xx + u_yy)
        g = self.__density * (u * v_x +
                              v * v_y) + p_y - self.__viscosity * (v_xx + v_yy)

        m = v_x - v_y

        return u, v, p, f, g, m

    def __loss_border(self):
        u, v, p, f, g, m_pred = self.evaluate(self.__border)

        u_err = tf.reduce_sum(tf.square(self.__border[:, 2:3] - u))
        v_err = tf.reduce_sum(tf.square(self.__border[:, 3:4] - v))

        self.u_err.append(u_err.numpy())
        self.v_err.append(v_err.numpy())

        return u_err + v_err

    def __loss_pde(self):
        _, _, _, f, g, m = self.evaluate(self.__train)

        f_err = tf.reduce_sum(tf.square(f))
        g_err = tf.reduce_sum(tf.square(g))
        m_err = tf.reduce_sum(tf.square(m))

        self.f_err.append(f_err.numpy())
        self.g_err.append(g_err.numpy())
        self.m_err.append(m_err.numpy())

        return f_err + g_err + m_err

    def loss(self):
        return self.__loss_border() + self.__loss_pde()


def main():
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument(
        '--iter',
        type=int,
        metavar='<iter>',
        default=ITER,
    )
    parser.add_argument(
        '--sample',
        type=float,
        metavar='<sample>',
        default=SAMPLE,
    )
    args = parser.parse_args()

    data = NavierStokesData(args.sample)

    Plot.heatmap(
        'foam',
        data.grid[:, :, 0],
        data.grid[:, :, 1],
        [
            ('u', data.u),
            ('v', data.v),
            ('p', data.p),
        ],
        path='../images/navier-stokes_foam',
    )

    pinn = NavierStokesNetwork(data)

    for v in pinn.trainable_variables:
        print(v.name)

    pbar = tqdm(total=args.iter, position=0, leave=True)

    def callback(loss):
        pbar.update(1)

    results = scipy.optimize.minimize(fun=pinn.optimize,
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
                                          'maxfun': args.iter * 10,
                                          'maxiter': args.iter,
                                          'iprint': -1,
                                          'maxls': 50
                                      })
    pbar.close()
    print(results)
    pinn.set_weights(results.x)

    u, v, p, f, g, m = pinn.evaluate(data.foam)

    u_pred = u.numpy().reshape(data.grid[:, :, 0].shape)
    v_pred = v.numpy().reshape(data.grid[:, :, 0].shape)
    p_pred = p.numpy().reshape(data.grid[:, :, 0].shape)

    Plot.heatmap(
        'pred',
        data.grid[:, :, 0],
        data.grid[:, :, 1],
        [
            ('u', u_pred),
            ('v', v_pred),
            ('p', p_pred),
        ],
        grids=[
            data.border[:, [0, 1]],
            # data.train[:, [0, 1]],
        ],
        path=f'../images/navier-stokes_pred_{args.iter}',
    )

    Plot.heatmap(
        'diff',
        data.grid[:, :, 0],
        data.grid[:, :, 1],
        [
            ('u', np.abs(u_pred - data.u)),
            ('v', np.abs(v_pred - data.v)),
            ('p', np.abs(p_pred - data.p)),
        ],
        grids=[
            data.border[:, [0, 1]],
            # data.train[:, [0, 1]],
        ],
        path=f'../images/navier-stokes_diff_{args.iter}',
    )

    Plot.error(
        'error',
        [
            ('u', pinn.u_err),
            ('v', pinn.v_err),
            ('f', pinn.f_err),
            ('g', pinn.g_err),
            ('m', pinn.m_err),
        ],
        path=f'../images/navier-stokes_err_{args.iter}',
    )


if __name__ == "__main__":
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    main()
