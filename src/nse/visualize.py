import numpy as np

from src import OUTPUT_DIR, HIRES
from src.base.mesh import Mesh
from src.base.plot import plot_heatmaps, plot_clouds, plot_losses, plot_arrows, plot_streamlines
from src.nse.experiments.experiment import NSEExperiment
from src.nse.simulation import Simulation


def predict(mesh: Mesh, model: Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, p, *_ = model.predict(mesh.flatten())

    u = u.detach().cpu().numpy().reshape(mesh.x.shape)
    v = v.detach().cpu().numpy().reshape(mesh.x.shape)
    p = p.detach().cpu().numpy().reshape(mesh.x.shape)

    return u, v, p


def plot_diff(n, experiment: NSEExperiment, model: Simulation, identifier: str):
    mesh = experiment.foam.mesh
    x, y = mesh.x, mesh.y
    u, v, p = predict(mesh, model)

    foam = mesh.transform(experiment.foam.knowledge)

    plot_heatmaps(
        f'OpenFOAM vs. Prediction [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        x,
        y,
        [
            ('u', np.abs(u - foam.u)),
            ('v', np.abs(v - foam.v)),
            ('p', np.abs(p - foam.p)),
        ],
        path=OUTPUT_DIR / identifier / f'diff_uvp.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )


def plot_prediction(n, experiment: NSEExperiment, model: Simulation, identifier: str, hires=False):
    if hires:
        mesh = Mesh(experiment.x.arrange(.1 / HIRES, True), experiment.y.arrange(.1 / HIRES, True))
    else:
        mesh = Mesh(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    x, y = mesh.x, mesh.y
    u, v, p = predict(mesh, model)

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
            path=OUTPUT_DIR / identifier / f'pred_uvp_hires.pdf',
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


def plot_history(n, experiment: NSEExperiment, model: Simulation, identifier: str):
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
        path=OUTPUT_DIR / identifier / f'err.pdf',
    )


def plot_foam(experiment: NSEExperiment, identifier: str):
    mesh = experiment.foam.mesh
    x, y = mesh.x, mesh.y

    data = mesh.transform(experiment.foam.knowledge)
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
        path=OUTPUT_DIR / identifier / 'foam' / f'foam_str.pdf',
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


def plot_geometry(experiment: NSEExperiment, identifier: str):
    mesh = Mesh(experiment.x.arrange(1), experiment.y.arrange(1))
    x, y = mesh.x, mesh.y

    plot_clouds(
        "Geometry",
        x,
        y,
        experiment.knowledge,
        ['u', 'v', 'p'],
        marker=experiment.learning.keys(),
        path=OUTPUT_DIR / identifier / 'model' / 'experiment.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )
