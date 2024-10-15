import numpy as np

from src import OUTPUT_DIR, HIRES
from src.base.mesh import Grid
from src.base.plot import plot_heatmaps, plot_cloud, plot_losses, plot_arrows, plot_streamlines
from src.nse.data import NSECloud
from src.nse.experiments.experiment import NSEExperiment
from src.nse.simulation import Simulation


def predict(grid: Grid, model: Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, v, p, psi = model.predict(grid.flatten())

    u = u.detach().cpu().numpy().reshape(grid.x.shape)
    v = v.detach().cpu().numpy().reshape(grid.x.shape)
    p = p.detach().cpu().numpy().reshape(grid.x.shape)
    psi = psi.detach().cpu().numpy().reshape(grid.x.shape)

    return u, v, p, psi


def plot_diff(n, experiment: NSEExperiment, model: Simulation, identifier: str):
    grid = experiment.foam.grid
    x, y = grid.x, grid.y
    u, v, p, _ = predict(grid, model)

    foam = grid.transform(experiment.foam.knowledge)

    plot_heatmaps(
        f'OpenFOAM vs. Prediction [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        x,
        y,
        [
            ('u', np.abs(u - foam.u)),
            ('v', np.abs(v - foam.v)),
            ('p', np.abs(p - foam.p)),
        ],
        path=OUTPUT_DIR / identifier / 'diff_uvp.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )


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
        plot_heatmaps(
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
        plot_heatmaps(
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

        plot_streamlines(
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

        plot_heatmaps(
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

        plot_streamlines(
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


def plot_history(n, model: Simulation, identifier: str):
    plot_losses(
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


def plot_foam(experiment: NSEExperiment, identifier: str):
    grid = experiment.foam.grid
    x, y = grid.x, grid.y

    data = grid.transform(experiment.foam.knowledge)
    u, v, p = data.u, data.v, data.p

    plot_heatmaps(
        'OpenFOAM',
        x,
        y,
        [
            ('u', u),
            ('v', v),
            ('p', p - p.min()),
        ],
        path=OUTPUT_DIR / identifier / 'foam' / 'foam_uvp.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_streamlines(
        'OpenFOAM Streamlines',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'foam' / 'foam_str.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    # plot_arrows(
    #     'OpenFOAM Arrows',
    #     x,
    #     y,
    #     u,
    #     v,
    #     path=OUTPUT_DIR / identifier / 'steps' / 'foam_arw.pdf',
    #     boundary=experiment.boundary,
    #     figure=experiment.obstruction,
    # )


def plot_experiment(experiment: NSEExperiment, identifier: str):
    grid = Grid(experiment.x.arrange(1), experiment.y.arrange(1))
    x, y = grid.x, grid.y

    cloud = NSECloud()

    for k, v in experiment.inlet:
        cloud.insert(k, v)

    for k, v in experiment.knowledge:
        cloud.insert(k, v)

    plot_cloud(
        experiment.name,
        x,
        y,
        cloud,
        ['u', 'v', 'p'],
        marker=experiment.learning.keys(),
        path=OUTPUT_DIR / identifier / 'model' / 'experiment.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )
