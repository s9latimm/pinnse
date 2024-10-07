import numpy as np

from src import OUTPUT_DIR
from src.base.geometry import Mesh
from src.base.plot import plot_heatmaps, plot_clouds, plot_losses, plot_arrows, plot_streamlines
from src.nse.experiments import NSEExperiment
from src.nse.model import NSEModel


def predict(mesh: Mesh, model: NSEModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v, p, *_ = model.predict(mesh.flatten())

    u = u.detach().cpu().numpy().reshape(mesh.x.shape)
    v = v.detach().cpu().numpy().reshape(mesh.x.shape)
    p = p.detach().cpu().numpy().reshape(mesh.x.shape)

    return u, v, p


def plot_diff(n, experiment: NSEExperiment, model: NSEModel, identifier: str):
    mesh = Mesh(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    x, y = mesh.x, mesh.y
    u, v, p = predict(mesh, model)

    plot_heatmaps(
        f'OpenFOAM vs. Prediction [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        x,
        y,
        [
            ('u', np.abs(u - experiment.u)),
            ('v', np.abs(v - experiment.v)),
            ('p', np.abs(p - experiment.p)),
        ],
        path=OUTPUT_DIR / identifier / f'diff_uvp.pdf',
        geometry=experiment.geometry,
    )


def plot_prediction(n, experiment: NSEExperiment, model: NSEModel, identifier: str, hires=False):
    if hires:
        mesh = Mesh(experiment.x.arrange(.01, True), experiment.y.arrange(.01, True))
    else:
        mesh = Mesh(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    x, y = mesh.x, mesh.y
    u, v, p = predict(mesh, model)

    p = p - p.min()

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
            geometry=experiment.geometry,
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
            geometry=experiment.geometry,
        )

        # plot_streamlines(
        #     f'Prediction Streamlines [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        #     x,
        #     y,
        #     u,
        #     v,
        #     path=OUTPUT_DIR / identifier / 'pred_str.pdf',
        #     geometry=experiment.geometry,
        # )

        plot_arrows(
            f'Prediction Arrows [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            u,
            v,
            path=OUTPUT_DIR / identifier / 'pred_arr.pdf',
            geometry=experiment.geometry,
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
            geometry=experiment.geometry,
        )

        # plot_streamlines(
        #     f'Prediction Streamlines [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        #     x,
        #     y,
        #     u,
        #     v,
        #     path=OUTPUT_DIR / identifier / 'steps' / f'pred_str_{n}.pdf',
        #     geometry=experiment.geometry,
        # )

        plot_arrows(
            f'Prediction Arrows [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
            x,
            y,
            u,
            v,
            path=OUTPUT_DIR / identifier / 'steps' / f'pred_arr_{n}.pdf',
            geometry=experiment.geometry,
        )


def plot_history(n, experiment: NSEExperiment, model: NSEModel, identifier: str):
    plot_losses(
        f'Loss [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
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
        path=OUTPUT_DIR / identifier / f'err.pdf',
    )


def plot_foam(experiment: NSEExperiment, identifier: str):
    mesh = Mesh(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    x, y = mesh.x, mesh.y

    data = mesh.transform(experiment.foam_facts)
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
        geometry=experiment.geometry,
    )

    plot_streamlines(
        'OpenFOAM Streamlines',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'foam' / f'foam_str.pdf',
        geometry=experiment.geometry,
    )

    plot_arrows(
        'OpenFOAM Arrows',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'foam' / f'foam_arr.pdf',
        geometry=experiment.geometry,
    )


def plot_geometry(experiment: NSEExperiment, identifier: str):
    mesh = Mesh(experiment.x.arrange(.05), experiment.y.arrange(.05))
    x, y = mesh.x, mesh.y

    plot_clouds(
        "Geometry",
        x,
        y,
        experiment.knowledge,
        ['u', 'v', 'p'],
        marker=experiment.learning.keys() + experiment.knowledge.keys(),
        path=OUTPUT_DIR / identifier / 'model' / 'experiment.pdf',
        geometry=experiment.geometry,
    )
