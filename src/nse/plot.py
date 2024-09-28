import numpy as np

import src.nse.config as config
from src.base.plot import plot_heatmaps, plot_clouds, plot_losses, plot_arrows, plot_streamlines
from src.nse.geometry import NSEGeometry
from src.nse.model import NSEModel


def decoration(ax):
    ax.hlines(y=.95, xmin=-.05, xmax=.95, colors='black', linestyles='dotted')
    ax.vlines(x=.95, ymin=-.05, ymax=.95, colors='black', linestyles='dotted')


def decoration_hires(ax):
    ax.hlines(y=.995, xmin=-.005, xmax=.995, colors='black', linestyles='dotted')
    ax.vlines(x=.995, ymin=-.005, ymax=.995, colors='black', linestyles='dotted')


def plot_diff(n, geometry: NSEGeometry, model: NSEModel, identifier: str):
    grid = geometry.default_grid
    x, y = grid.x.numpy(), grid.y.numpy()
    u, v, p, *_ = model.predict(grid.flatten())

    u = u.detach().cpu().numpy().reshape(x.shape)
    v = v.detach().cpu().numpy().reshape(x.shape)
    p = p.detach().cpu().numpy().reshape(x.shape)

    plot_heatmaps(
        f'OpenFOAM vs. Prediction [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        [
            ('u', np.abs(u - geometry.u)),
            ('v', np.abs(v - geometry.v)),
            ('p', np.abs(p - geometry.p)),
        ],
        path=config.OUTPUT_DIR / identifier / f'diff_uvp.pdf',
        decoration=decoration,
    )


def plot_hires(n, geometry: NSEGeometry, model: NSEModel, identifier: str):
    grid = geometry.hires_grid
    x, y = grid.x.numpy(), grid.y.numpy()
    u, v, p, *_ = model.predict(grid.flatten())

    u = u.detach().cpu().numpy().reshape(x.shape)
    v = v.detach().cpu().numpy().reshape(x.shape)
    p = p.detach().cpu().numpy().reshape(x.shape)

    p = p - p.min()

    plot_heatmaps(
        f'Prediction HiRes [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        [
            ('u', u),
            ('v', v),
            ('p', p),
        ],
        path=config.OUTPUT_DIR / identifier / f'pred_uvp_hires.pdf',
        decoration=decoration_hires,
    )


def plot_prediction(n, geometry: NSEGeometry, model: NSEModel, identifier: str):
    grid = geometry.default_grid
    x, y = grid.x.numpy(), grid.y.numpy()
    u, v, p, *_ = model.predict(grid.flatten())

    u = u.detach().cpu().numpy().reshape(x.shape)
    v = v.detach().cpu().numpy().reshape(x.shape)
    p = p.detach().cpu().numpy().reshape(x.shape)

    p = p - p.min()

    plot_heatmaps(
        f'Prediction [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        [
            ('u', u),
            ('v', v),
            ('p', p),
        ],
        path=config.OUTPUT_DIR / identifier / f'pred_uvp.pdf',
        decoration=decoration,
    )

    plot_heatmaps(
        f'Prediction [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        [
            ('u', u),
            ('v', v),
            ('p', p),
        ],
        path=config.OUTPUT_DIR / identifier / 'steps' / f'pred_uvp_{n}.pdf',
        decoration=decoration,
    )

    plot_streamlines(
        f'Prediction Streamlines [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        u,
        v,
        path=config.OUTPUT_DIR / identifier / 'steps' / f'pred_str_{n}.pdf',
        decoration=decoration,
    )

    plot_arrows(
        f'Prediction Arrows [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        u,
        v,
        path=config.OUTPUT_DIR / identifier / 'steps' / f'pred_arr_{n}.pdf',
        decoration=decoration,
    )


def plot_history(n, geometry: NSEGeometry, model: NSEModel, identifier: str):
    plot_losses(
        f'Loss [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        [
            ('Border', [
                ('u', model.history[1:, 3]),
                ('v', model.history[1:, 4]),
            ]),
            ('PDE', [
                ('f', model.history[1:, 0]),
                ('g', model.history[1:, 1]),
            ]),
            ('Sum', [
                ('$\Sigma$', model.history[1:, 2]),
            ]),
        ],
        path=config.OUTPUT_DIR / identifier / f'err.pdf',
    )


def plot_foam(geometry: NSEGeometry, identifier: str):
    grid = geometry.foam_grid
    x, y = grid.x.numpy(), grid.y.numpy()

    data = grid.transform(lambda i: geometry.foam_cloud[i])
    u, v, p = data.u.numpy(), data.v.numpy(), data.p.numpy()

    plot_heatmaps(
        'OpenFOAM',
        x,
        y,
        [
            ('u', u),
            ('v', v),
            ('p', p - p.min()),
        ],
        path=config.OUTPUT_DIR / identifier / 'foam' / 'foam_uvp.pdf',
        decoration=decoration,
    )

    plot_streamlines(
        'OpenFOAM Streamlines',
        x,
        y,
        u,
        v,
        path=config.OUTPUT_DIR / identifier / 'foam' / f'foam_str.pdf',
        decoration=decoration,
    )

    plot_arrows(
        'OpenFOAM Arrows',
        x,
        y,
        u,
        v,
        path=config.OUTPUT_DIR / identifier / 'foam' / f'foam_arr.pdf',
        decoration=decoration,
    )


def plot_geometry(geometry: NSEGeometry, identifier: str):

    grid = geometry.rimmed_grid
    x, y = grid.x.numpy(), grid.y.numpy()

    plot_clouds(
        "Geometry",
        x,
        y,
        geometry.rim_cloud,
        ['u', 'v', 'p'],
        grid=geometry.pde_cloud.numpy(),
        path=config.OUTPUT_DIR / identifier / 'model' / 'geometry.pdf',
        decoration=decoration,
    )
