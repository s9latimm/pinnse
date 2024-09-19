import argparse
import datetime
import typing as tp

import numpy as np
import scipy
from tqdm import tqdm

from foam.pitzDaily.get_foam_results import get_foam
from network import Network
from src.plot import Plot


class NavierStokes:

    ITER = 100

    @staticmethod
    def data(
        sample: float
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        u_foam, v_foam, p_foam = get_foam()

        def refine(x, step):
            return x[::step].transpose()[::step].transpose()

        u_foam = refine(u_foam, 10)[::-1].transpose()
        v_foam = refine(v_foam, 10)[::-1].transpose()
        p_foam = refine(p_foam, 10)[::-1].transpose()

        grid = np.dstack(np.mgrid[0:80, 0:20])

        Plot.heatmap(
            'foam',
            grid[:, :, 0],
            grid[:, :, 1],
            [
                ('u', u_foam),
                ('v', v_foam),
                ('p', p_foam),
            ],
            # path='../../images/navier-stokes_foam',
        )

        foam = np.hstack([
            grid[:, :, 0].flatten()[:, None],
            grid[:, :, 1].flatten()[:, None],
            u_foam.flatten()[:, None],
            v_foam.flatten()[:, None],
            p_foam.flatten()[:, None],
        ])

        np.random.seed(42)

        border = np.vstack([
            np.hstack([
                grid[:, 0, 0][:, None],
                grid[:, 0, 1][:, None],
                u_foam[:, 0][:, None],
                v_foam[:, 0][:, None],
                p_foam[:, 0][:, None],
            ]),
            np.hstack([
                grid[:, -1, 0][:, None],
                grid[:, -1, 1][:, None],
                u_foam[:, -1][:, None],
                v_foam[:, -1][:, None],
                p_foam[:, -1][:, None],
            ]),
            np.hstack([
                grid[0, :, 0][:, None],
                grid[0, :, 1][:, None],
                u_foam[0, :][:, None],
                v_foam[0, :][:, None],
                p_foam[0, :][:, None],
            ]),
            np.hstack([
                grid[-1, :, 0][:, None],
                grid[-1, :, 1][:, None],
                u_foam[-1, :][:, None],
                v_foam[-1, :][:, None],
                p_foam[-1, :][:, None],
            ]),
        ])

        for i in range(1, 10):
            border = np.vstack([
                border,
                np.hstack([
                    grid[i, 1:10, 0][:, None],
                    grid[i, 1:10, 1][:, None],
                    u_foam[i, 1:10][:, None],
                    v_foam[i, 1:10][:, None],
                    p_foam[i, 1:10][:, None],
                ]),
            ])

        border = np.unique(border, axis=0)
        foam = np.unique(foam, axis=0)

        train = np.array([
            x for x in set(tuple(x)
                           for x in foam) - set(tuple(x) for x in border)
        ])

        train = train[np.random.choice(
            len(train), int(len(train) * sample), replace=False), :]

        return grid, foam, train, border


if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument(
        '--iter',
        type=int,
        metavar='<iter>',
        default=NavierStokes.ITER,
    )
    parser.add_argument(
        '--sample',
        type=float,
        metavar='<sample>',
        default=1,
    )
    args = parser.parse_args()

    grid, foam, train, border = NavierStokes.data(args.sample)

    pinn = Network([2, 20, 20, 20, 20, 20, 20, 20, 20, 3], args.iter, border,
                   train)

    for v in pinn.trainable_variables:
        print(v.name)

    def callback(loss):
        pbar.update(1)
        pinn.callback(loss)

    pbar = tqdm(total=args.iter, position=0, leave=True)
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

    print(results)
    pinn._set_weights(results.x)

    u, v, p, f, g, m = pinn.evaluate(foam)

    u_pred = u.numpy().reshape(grid[:, :, 0].shape)
    v_pred = v.numpy().reshape(grid[:, :, 0].shape)
    p_pred = p.numpy().reshape(grid[:, :, 0].shape)

    Plot.error(
        'error',
        [
            ('u', pinn.u_err),
            ('v', pinn.v_err),
            ('f', pinn.f_err),
            ('g', pinn.g_err),
            ('m', pinn.m_err),
        ],
        path=f'../../images/navier-stokes_err_{args.iter}',
    )

    Plot.heatmap(
        'pred',
        grid[:, :, 0],
        grid[:, :, 1],
        [
            ('u', u_pred),
            ('v', v_pred),
            ('p', p_pred),
        ],
        path=f'../../images/navier-stokes_pred_{args.iter}',
    )
