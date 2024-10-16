import logging

import numpy as np

from src import OUTPUT_DIR
from src.base.model.mesh import Grid, Mesh
from src.base.view.plot import plot_seismic, plot_stream, plot_arrows, plot_mesh
from src.nse.controller.simulation import Simulation
from src.nse.model.experiments.experiment import Experiment
from src.nse.model.record import Record


def plot_setup(experiment: Experiment, identifier: str):
    grid = Grid(experiment.x.arrange(1), experiment.y.arrange(1))
    x, y = grid.x, grid.y

    mesh = Mesh[Record]()

    for k, v in experiment.inlet:
        mesh.insert(k, v)

    for k, v in experiment.knowledge:
        mesh.insert(k, v)

    plot_mesh(
        experiment.name,
        x,
        y,
        mesh,
        ['u', 'v', 'p'],
        marker=experiment.learning.keys(),
        path=OUTPUT_DIR / identifier / 'grading' / f'{experiment.name.lower()}.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )


def plot_foam(experiment: Experiment, identifier: str):
    grid = experiment.foam.knowledge.grid()
    x, y = grid.x, grid.y

    data = grid.transform(experiment.foam.knowledge)
    u, v, p = data.u, data.v, data.p

    plot_seismic(
        'OpenFOAM',
        x,
        y,
        [
            ('u', u),
            ('v', v),
            ('p', p - p.min()),
        ],
        path=OUTPUT_DIR / identifier / 'grading' / 'foam_uvp.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_stream(
        'OpenFOAM Streamlines',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'grading' / 'foam_str.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_arrows(
        'OpenFOAM Arrows',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'grading' / 'foam_arw.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )


def plot_diff(n, experiment: Experiment, model: Simulation, identifier: str):
    grid = experiment.foam.knowledge.grid()
    x, y = grid.x, grid.y

    foam = grid.transform(experiment.foam.knowledge)

    prediction = grid.transform(model.predict(grid.mesh()))
    u, v, p = prediction.u, prediction.v, prediction.p

    u_err, v_err, p_err, norm = 0, 0, 0, 0

    p_min = np.infty
    f_min = np.infty
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (x[i, j], y[i, j]) in experiment.obstruction:
                u[i, j] = 0
                v[i, j] = 0
            else:
                u[i, j] = np.abs(u[i, j] - foam.u[i, j])
                v[i, j] = np.abs(v[i, j] - foam.v[i, j])
                u_err += u[i, j]
                v_err += v[i, j]
                p_min = min(p_min, float(p[i, j]))
                f_min = min(f_min, float(foam.p[i, j]))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (x[i, j], y[i, j]) in experiment.obstruction:
                p[i, j] = 0
            else:
                norm += 1
                p[i, j] = np.abs(p[i, j] - p_min - foam.p[i, j] + f_min)
                p_err += p[i, j]

    logging.info(f'ERROR: u:{u_err / norm:.3E}, v:{v_err / norm:.3E}, p:{p_err / norm:.3E}')

    plot_seismic(
        f'OpenFOAM vs. Prediction [n={n}, $\\nu$={model.nu:.3E}, $\\rho$={model.rho:.3E}]',
        x,
        y,
        [
            ('u', u),
            ('v', v),
            ('p', p),
        ],
        path=OUTPUT_DIR / identifier / 'grading' / 'diff_uvp.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )
