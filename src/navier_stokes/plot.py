import numpy as np

import src.navier_stokes.config as config
from src.navier_stokes.geometry import NavierStokesGeometry
from src.navier_stokes.model import NavierStokesModel
from src.utils.plot import plot_heatmaps, plot_clouds, plot_losses, plot_arrows, plot_streamlines


def plot_diff(n, geometry: NavierStokesGeometry, model: NavierStokesModel, identifier: str):
    x, y = geometry.grid[:, :, 0], geometry.grid[:, :, 1]
    u, v, p, *_ = model.predict(geometry.all)

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
    )


def plot_hires(n, geometry: NavierStokesGeometry, model: NavierStokesModel, identifier: str):
    u, v, p, *_ = model.predict(
        np.hstack([
            geometry.mesh[:, :, 0].flatten()[:, None],
            geometry.mesh[:, :, 1].flatten()[:, None],
        ]))

    u = u.detach().cpu().numpy().reshape(geometry.mesh[:, :, 0].shape)
    v = v.detach().cpu().numpy().reshape(geometry.mesh[:, :, 0].shape)
    p = p.detach().cpu().numpy().reshape(geometry.mesh[:, :, 0].shape)

    p = p - p.min()

    plot_heatmaps(
        f'Prediction HiRes [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        geometry.mesh[:, :, 0],
        geometry.mesh[:, :, 1],
        [
            ('u', u),
            ('v', v),
            ('p', p),
        ],
        path=config.OUTPUT_DIR / identifier / f'pred_uvp_hires.pdf',
    )


def plot_prediction(n, geometry: NavierStokesGeometry, model: NavierStokesModel, identifier: str):
    x, y = geometry.grid[:, :, 0], geometry.grid[:, :, 1]
    u, v, p, *_ = model.predict(geometry.all)

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
    )

    plot_streamlines(
        f'Prediction Streamlines [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        u,
        v,
        path=config.OUTPUT_DIR / identifier / 'steps' / f'pred_str_{n}.pdf',
    )

    plot_arrows(
        f'Prediction Arrows [n={n}, $\\nu$={geometry.nu:.3E}, $\\rho$={geometry.rho:.3E}]',
        x,
        y,
        u,
        v,
        path=config.OUTPUT_DIR / identifier / 'steps' / f'pred_arr_{n}.pdf',
    )


def plot_history(n, geometry: NavierStokesGeometry, model: NavierStokesModel, identifier: str):
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
            ('Mass', [
                ('m', model.history[1:, 2]),
            ]),
        ],
        path=config.OUTPUT_DIR / identifier / f'err.pdf',
    )


def plot_foam(geometry: NavierStokesGeometry, identifier: str):
    x, y = geometry.grid[:, :, 0], geometry.grid[:, :, 1]
    plot_heatmaps(
        'OpenFOAM',
        x,
        y,
        [
            ('u', geometry.u),
            ('v', geometry.v),
            ('p', geometry.p - geometry.p.min()),
        ],
        path=config.OUTPUT_DIR / identifier / 'foam' / 'foam_uvp.pdf',
    )

    plot_streamlines(
        'OpenFOAM Streamlines',
        x,
        y,
        geometry.u.reshape(x.shape),
        geometry.v.reshape(x.shape),
        path=config.OUTPUT_DIR / identifier / 'foam' / f'foam_str.pdf',
    )

    plot_arrows(
        'OpenFOAM Arrows',
        x,
        y,
        geometry.u.reshape(x.shape),
        geometry.v.reshape(x.shape),
        path=config.OUTPUT_DIR / identifier / 'foam' / f'foam_arr.pdf',
    )


def plot_geometry(geometry: NavierStokesGeometry, identifier: str):
    x, y = geometry.grid[:, :, 0], geometry.grid[:, :, 1]
    plot_clouds(
        "Intake",
        x,
        y,
        [
            ('u', geometry.intake[:, [0, 1, 2]]),
            ('v', geometry.intake[:, [0, 1, 3]]),
            ('p', geometry.intake[:, [0, 1, 4]]),
        ],
        path=config.OUTPUT_DIR / identifier / 'model' / 'intake.pdf',
    )

    plot_clouds(
        "Border",
        x,
        y,
        [
            ('u', geometry.border[:, [0, 1, 2]]),
            ('v', geometry.border[:, [0, 1, 3]]),
            ('p', geometry.border[:, [0, 1, 4]]),
        ],
        path=config.OUTPUT_DIR / identifier / 'model' / 'border.pdf',
    )

    plot_clouds(
        "Geometry",
        x,
        y,
        [
            ('u', geometry.geometry[:, [0, 1, 2]]),
            ('v', geometry.geometry[:, [0, 1, 3]]),
            ('p', geometry.geometry[:, [0, 1, 4]]),
        ],
        grid=geometry.train,
        path=config.OUTPUT_DIR / identifier / 'model' / 'geometry.pdf',
    )
