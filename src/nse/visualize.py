import numpy as np

from src import OUTPUT_DIR, HIRES
from src.base.mesh import Grid
from src.base.plot import plot_seismic, plot_history, plot_arrows, plot_stream
from src.nse.experiments.experiment import NSEExperiment
from src.nse.simulation import Simulation


def predict(grid: Grid, model: Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, v, p, psi = model.predict(grid.flatten())

    u = u.detach().cpu().numpy().reshape(grid.x.shape)
    v = v.detach().cpu().numpy().reshape(grid.x.shape)
    p = p.detach().cpu().numpy().reshape(grid.x.shape)
    psi = psi.detach().cpu().numpy().reshape(grid.x.shape)

    return u, v, p, psi


def plot_prediction(n, experiment: NSEExperiment, model: Simulation, identifier: str, hires=False):
    if hires:
        grid = Grid(experiment.x.arrange(.1 / HIRES, True), experiment.y.arrange(.1 / HIRES, True))
    else:
        grid = Grid(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    x, y = grid.x, grid.y
    u, v, p, _ = predict(grid, model)

    p_min = np.infty

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (x[i, j], y[i, j]) not in experiment.obstruction:
                p_min = min(p_min, float(p[i, j]))

    p = p - p_min

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (x[i, j], y[i, j]) in experiment.obstruction:
                u[i, j] = 0
                v[i, j] = 0
                p[i, j] = 0

    if hires:
        plot_seismic(
            f'Prediction HiRes [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            [
                ('u', u),
                ('v', v),
                ('p', p),
            ],
            path=OUTPUT_DIR / identifier / 'pred_uvp_hires.pdf',
            boundary=experiment.boundary,
            figure=experiment.obstruction,
        )

    else:
        plot_seismic(
            f'Prediction [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            [
                ('u', u),
                ('v', v),
                ('p', p),
            ],
            path=OUTPUT_DIR / identifier / 'pred_uvp.pdf',
            boundary=experiment.boundary,
            figure=experiment.obstruction,
        )

        plot_stream(
            f'Prediction Streamlines [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            u,
            v,
            path=OUTPUT_DIR / identifier / 'pred_str.pdf',
            boundary=experiment.boundary,
            figure=experiment.obstruction,
        )

        plot_arrows(
            f'Prediction Arrows [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            u,
            v,
            path=OUTPUT_DIR / identifier / 'pred_arw.pdf',
            boundary=experiment.boundary,
            figure=experiment.obstruction,
        )

        plot_seismic(
            f'Prediction [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            [
                ('u', u),
                ('v', v),
                ('p', p),
            ],
            path=OUTPUT_DIR / identifier / 'steps' / f'pred_uvp_{n}.pdf',
            boundary=experiment.boundary,
            figure=experiment.obstruction,
        )

        plot_stream(
            f'Prediction Streamlines [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            u,
            v,
            path=OUTPUT_DIR / identifier / 'steps' / f'pred_str_{n}.pdf',
            boundary=experiment.boundary,
            figure=experiment.obstruction,
        )

        plot_arrows(
            f'Prediction Arrows [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            u,
            v,
            path=OUTPUT_DIR / identifier / 'steps' / f'pred_arw_{n}.pdf',
            boundary=experiment.boundary,
            figure=experiment.obstruction,
        )


def plot_losses(n, model: Simulation, identifier: str):
    plot_history(
        f'Loss [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        [
            ('Border', [
                ('u', [i[3] for i in model.history[1:]]),
                ('v', [i[4] for i in model.history[1:]]),
            ]),
            ('PDE', [
                ('f', [i[0] for i in model.history[1:]]),
                ('g', [i[1] for i in model.history[1:]]),
            ]),
            ('Sum', [
                ('$\\Sigma$', [i[2] for i in model.history[1:]]),
            ]),
        ],
        path=OUTPUT_DIR / identifier / 'err.pdf',
    )
