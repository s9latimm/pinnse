from pathlib import Path

import numpy as np

from src import OUTPUT_DIR
from src.base.model.algebra import Integer
from src.base.model.mesh import Grid, Mesh
from src.base.view.plot import plot_seismic, plot_stream, plot_arrows
from src.nse.controller.simulation import Simulation
from src.nse.model.experiments.experiment import Experiment
from src.nse.model.record import Record
from src.utils.timer import Stopwatch


def plot_setup(experiment: Experiment, identifier: str):
    grid = Grid(experiment.x.arrange(1), experiment.y.arrange(1))
    x, y = grid.x, grid.y

    mesh = Mesh[Record]()

    for k, v in experiment.inlet:
        mesh.insert(k, v)

    for k, v in experiment.knowledge:
        mesh.insert(k, v)

    data = grid.transform(mesh)

    plot_seismic(
        experiment.name,
        x,
        y,
        [('u', data.u), ('v', data.u), ('p', data.u)],
        path=OUTPUT_DIR / identifier / 'setup.pdf',
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


def export(timer: Stopwatch, experiment: Experiment, model: Simulation, identifier: str, suffix: str):
    path = OUTPUT_DIR / identifier / f'{experiment.foam.name}_{suffix}'

    mesh = Mesh(Record)
    for k, v in experiment.foam.knowledge:
        if k not in experiment.knowledge and k not in experiment.obstruction:
            mesh.insert(k, v)
    knowledge = experiment.knowledge + experiment.inlet

    boundary = model.predict(knowledge)
    prediction = model.predict(mesh)

    model.save(path / 'loss.csv')
    knowledge.save(path / 'boundary_init.csv')
    boundary.save(path / 'boundary_pred.csv')
    mesh.save(path / 'mesh_init.csv')
    prediction.save(path / 'mesh_pred.csv')
    timer.save(path / 'time.csv')


def grade(experiment: Experiment, identifier: str, suffix: str):
    path = OUTPUT_DIR / identifier / f'{experiment.foam.name}_{suffix}'

    boundary_init = Mesh(Record).load(path / 'boundary_init.csv')
    boundary_pred = Mesh(Record).load(path / 'boundary_pred.csv')
    boundary_diff = boundary_pred - boundary_init
    boundary_diff.save(path / 'boundary_diff.csv')

    mesh_init = Mesh(Record).load(path / 'mesh_init.csv')
    mesh_pred = Mesh(Record).load(path / 'mesh_pred.csv')
    mesh_diff = mesh_pred - mesh_init
    mesh_diff.save(path / 'mesh_diff.csv')

    reverse = 0
    for _, v in mesh_pred:
        if v.u < 0:
            reverse += 1
    Integer(reverse).save(path / 'reverse.csv')

    u_diff = 0
    v_diff = 0
    p_diff = 0
    for _, v in boundary_diff:
        u_diff += v.u
        v_diff += v.v
    u_diff = u_diff / len(boundary_diff)
    v_diff = v_diff / len(boundary_diff)
    Record(u_diff, v_diff, p_diff).save(path / 'boundary_mean.csv')

    u_diff = 0
    v_diff = 0
    p_diff = 0
    for _, v in mesh_diff:
        u_diff += v.u
        v_diff += v.v
        p_diff += v.p
    u_diff = u_diff / len(mesh_diff)
    v_diff = v_diff / len(mesh_diff)
    p_diff = p_diff / len(mesh_diff)
    Record(u_diff, v_diff, p_diff).save(path / 'mesh_mean.csv')

    plot_mesh(mesh_pred, experiment, OUTPUT_DIR / identifier, 'pred')
    plot_mesh(mesh_diff, experiment, OUTPUT_DIR / identifier, 'diff', r'$\Delta$')


def plot_mesh(mesh: Mesh[Record], experiment: Experiment, path: Path, suffix: str, label: str = ''):
    grid = mesh.grid()
    x, y = grid.x, grid.y

    data = grid.transform(mesh)

    plot_seismic(
        '',
        x,
        y,
        [(f'{label}u', data.u)],
        path=path / 'images' / f'u_{suffix}.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_seismic(
        '',
        x,
        y,
        [(f'{label}v', data.v)],
        path=path / 'images' / f'v_{suffix}.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    p = data.p
    plot_seismic(
        '',
        x,
        y,
        [(f'{label}p', p - np.nanmin(p))],
        path=path / 'images' / f'p_{suffix}.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )
