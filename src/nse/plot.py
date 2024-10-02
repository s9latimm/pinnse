import numpy as np

from src import OUTPUT_DIR
from src.base.plot import plot_heatmaps, plot_clouds, plot_losses, plot_arrows, plot_streamlines
from src.nse.experiments import NSEExperiment
from src.nse.model import NSEModel


def plot_diff(n, experiment: NSEExperiment, model: NSEModel, identifier: str):
    mesh = experiment.finite_elements_mesh
    x, y = mesh.x.numpy(), mesh.y.numpy()
    u, v, p, *_ = model.predict(mesh.flatten())

    u = u.detach().cpu().numpy().reshape(x.shape)
    v = v.detach().cpu().numpy().reshape(x.shape)
    p = p.detach().cpu().numpy().reshape(x.shape)

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


def plot_hires(n, experiment: NSEExperiment, model: NSEModel, identifier: str):
    mesh = experiment.evaluation_mesh
    x, y = mesh.x.numpy(), mesh.y.numpy()
    u, v, p, *_ = model.predict(mesh.flatten())

    u = u.detach().cpu().numpy().reshape(x.shape)
    v = v.detach().cpu().numpy().reshape(x.shape)
    p = p.detach().cpu().numpy().reshape(x.shape)

    p = p - p.min()

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


def plot_prediction(n, experiment: NSEExperiment, model: NSEModel, identifier: str):
    mesh = experiment.finite_elements_mesh
    x, y = mesh.x.numpy(), mesh.y.numpy()
    u, v, p, *_ = model.predict(mesh.flatten())

    u = u.detach().cpu().numpy().reshape(x.shape)
    v = v.detach().cpu().numpy().reshape(x.shape)
    p = p.detach().cpu().numpy().reshape(x.shape)

    p = p - p.min()

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

    plot_streamlines(
        f'Prediction Streamlines [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'pred_str.pdf',
        geometry=experiment.geometry,
    )

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

    plot_streamlines(
        f'Prediction Streamlines [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'steps' / f'pred_str_{n}.pdf',
        geometry=experiment.geometry,
    )

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
    mesh = experiment.finite_elements_mesh
    x, y = mesh.x.numpy(), mesh.y.numpy()

    data = mesh.map(lambda i: experiment.foam_facts[i])
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

    mesh = experiment.training_mesh
    x, y = mesh.x.numpy(), mesh.y.numpy()

    plot_clouds(
        "Geometry",
        x,
        y,
        experiment.rim_facts,
        ['u', 'v', 'p'],
        overlay=experiment.pde_facts.numpy(),
        path=OUTPUT_DIR / identifier / 'model' / 'experiment.pdf',
        geometry=experiment.geometry,
    )
