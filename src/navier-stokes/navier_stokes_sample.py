import argparse
import datetime
import typing as tp

import numpy as np
import scipy.optimize
import tensorflow as tf

from foam.pitzDaily.get_foam_results import get_normalized_maps
from src.navier_stokes import Network
from src.plot import Plot


def data(sample: float) -> tp.Tuple[tp.Tuple[int], np.ndarray, np.ndarray]:

    u_ref, v_ref, p_ref = get_normalized_maps()

    x, y = np.meshgrid(np.linspace(0, 1, 800), np.linspace(0, 1, 200))

    uvp_ref = np.hstack([
        x.flatten()[:, None],
        y.flatten()[:, None],
        u_ref.flatten()[:, None],
        v_ref.flatten()[:, None],
        p_ref.flatten()[:, None],
    ])

    np.random.seed(42)

    uvp_train = uvp_ref[np.random.choice(
        len(uvp_ref), int(len(uvp_ref) * sample), replace=False), :]

    return p_ref.shape, uvp_ref, uvp_train


# def refine(x, step):
#     return x[::step].transpose()[::step].transpose()

if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument(
        '--iter',
        type=int,
        metavar='<iter>',
        default=100,
    )
    parser.add_argument(
        '--sample',
        type=float,
        metavar='<sample>',
        default=.1,
    )
    args = parser.parse_args()

    ref_shape, uvp_ref, uvp_train = data(args.sample)


    # Plot.scatter_3d('velocity_ref',
    #                 ('u_ref', np.hstack([uvp_ref[:, 0:2], uvp_ref[:, [2]]])),
    #                 ('v_ref', np.hstack([uvp_ref[:, 0:2], uvp_ref[:, [3]]])))
    #
    #
    # Plot.scatter_3d('abs-velocity_ref',
    #                 ('uv_ref',
    #                  np.hstack([
    #                      uvp_ref[:, 0:2],
    #                      np.abs(uvp_ref[:, [2]]) + np.abs(uvp_ref[:, [3]])
    #                  ])))

    pinn = Network(args.iter, uvp_train, uvp_train)

    for v in pinn.trainable_variables:
        print(v.name)

    results = scipy.optimize.minimize(fun=pinn.optimize,
                                      x0=pinn.get_weights().numpy(),
                                      args=(),
                                      method='L-BFGS-B',
                                      jac=True,
                                      callback=pinn.callback,
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

    g = tf.Variable(uvp_ref, dtype='float64', trainable=False)

    x = g[:, 0:1]
    y = g[:, 1:2]

    u_pred, v_pred, p_pred, _, _ = pinn.evaluate(
        tf.stack([x[:, 0], y[:, 0]], axis=1))

    Plot.scatter_3d(f'velocity_pred_{args.iter}',
                    ('u_pred', np.hstack([uvp_ref[:, 0:2], u_pred])),
                    ('v_pred', np.hstack([uvp_ref[:, 0:2], v_pred])))

    # Plot.scatter_3d(
    #     'abs-velocity_pred',
    #     ('uv_pred', np.hstack([uvp_ref[:, 0:2],
    #                           np.abs(u_pred) + np.abs(v_pred)])))

    # step = 3
    #
    # Plot.arrows(
    #     'training',
    #     refine(uvp_ref[:, [0]].reshape(ref_shape) * ref_shape[0], step),
    #     refine(uvp_ref[:, [1]].reshape(ref_shape) * ref_shape[1], step),
    #     refine(uvp_ref[:, [2]].reshape(ref_shape), step),
    #     refine(uvp_ref[:, [3]].reshape(ref_shape), step),
    # )

    # print(uvp_ref)
