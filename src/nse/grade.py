import numpy as np

from src import OUTPUT_DIR
from src.base.plot import plot_seismic, plot_stream, plot_arrows
from src.nse.experiments.experiment import Experiment
from src.nse.simulation import Simulation


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
        path=OUTPUT_DIR / identifier / 'foam' / 'foam_uvp.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_stream(
        'OpenFOAM Streamlines',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'foam' / 'foam_str.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_arrows(
        'OpenFOAM Arrows',
        x,
        y,
        u,
        v,
        path=OUTPUT_DIR / identifier / 'foam' / 'foam_arw.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )


def plot_diff(n, experiment: Experiment, model: Simulation, identifier: str):
    grid = experiment.foam.knowledge.grid()
    x, y = grid.x, grid.y

    foam = grid.transform(experiment.foam.knowledge)

    prediction = grid.transform(model.predict(grid.mesh()))
    u, v, p = prediction.u, prediction.v, prediction.p

    plot_seismic(
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
