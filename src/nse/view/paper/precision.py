import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from base.model.algebra import Real
from nse.model.experiments import Block, Cylinder
from src import RESULT_DIR, OUTPUT_DIR
from src.base.model.mesh import Mesh, Grid, arrange
from src.base.view import SCALE, PHI, COLORS
from src.base.view.plot import save_fig, plot_seismic, plot_stream
from src.nse.model.loss import Losses
from src.nse.model.record import Record


def __plot_uvp(mesh: Mesh) -> None:
    experiment = Block(.01, 1, 1)
    grid = Grid(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    d = grid.transform(mesh)

    plot_seismic(
        '',
        grid.x,
        grid.y,
        [
            (r'$\Delta u$', d.u),
        ],
        path=OUTPUT_DIR / 'paper' / 'u_pred.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_seismic(
        '',
        grid.x,
        grid.y,
        [
            (r'$\Delta v$', d.v),
        ],
        path=OUTPUT_DIR / 'paper' / 'v_pred.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )

    plot_seismic(
        '',
        grid.x,
        grid.y,
        [
            ('p', d.p - np.nanmin(d.p)),
        ],
        path=OUTPUT_DIR / 'paper' / 'p_pred.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )


def __plot_stream(mesh: Mesh) -> None:
    experiment = Cylinder(.01, 1, 1)
    grid = Grid(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    d = grid.transform(mesh)

    plot_stream(
        '',
        grid.x,
        grid.y,
        d.u,
        d.v,
        path=OUTPUT_DIR / 'paper' / 'stream.pdf',
        boundary=experiment.boundary,
        figure=experiment.obstruction,
    )


def __plot_loss(losses: Losses) -> None:
    fig, ax = plt.subplots(figsize=(PHI * SCALE, SCALE))

    s = []
    f = []
    g = []
    u = []
    v = []
    for loss in losses:
        s.append(loss.loss)
        f.append(loss.f)
        g.append(loss.g)
        u.append(loss.u)
        v.append(loss.v)

    s = s[:1000]
    f = f[:1000]
    g = g[:1000]
    u = u[:1000]
    v = v[:1000]

    ax.plot(arrange(1, len(s) - 1, 1), u[1:], label=r'$f$', color=COLORS[0])
    ax.plot(arrange(1, len(s) - 1, 1), g[1:], label=r'$g$', color=COLORS[1])
    ax.plot(arrange(1, len(s) - 1, 1), u[1:], label=r'$\hat{u}$', color=COLORS[2])
    ax.plot(arrange(1, len(s) - 1, 1), v[1:], label=r'$\hat{v}$', color=COLORS[3])
    ax.plot(arrange(1, len(s) - 1, 1), s[1:], label=r'$\mathcal{L}$', color=COLORS[4])

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('log$_{10}$(Loss)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_yscale('log')
    ax.minorticks_off()

    ax.set_xlim([1, 1100])
    ax.set_xticks([1, 500, 1000])
    ax.set_xticklabels(['1', '500', '1000'])

    ax.set_ylim([1e-5, 5e-1])
    ax.set_yticks([1e-5, 1e-3, 1e-1])
    ax.set_yticklabels(['1e-5', '1e-3', '1e-1'])
    ax.plot(1, 5e-1, "^k", clip_on=False)
    ax.plot(1100, 1e-5, ">k", clip_on=False)

    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fancybox=False).get_frame().set_edgecolor('k')

    save_fig(fig, OUTPUT_DIR / 'paper' / 'loss_1000.pdf')

    plt.clf()
    plt.close()


def plot_table(paths: list[Path]) -> None:
    for path in paths:
        losses = Losses.load(path / 'loss.csv')
        mean = Record.load(path / 'mesh_mean.csv')
        n = len(losses) - 1
        t = datetime.timedelta(seconds=int(float(Real.load(path / 'time.csv'))))
        print(f'& {n:6d} & {t} & {mean.u:.3e} & {mean.v:.3e}'.replace('e-0', 'e-'))


if __name__ == '__main__':

    def main():
        paths = [
            # RESULT_DIR / 'precision' / 'empty__0_100__0_020__0_500.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_020__0_500.cuda__150__3',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_010__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_010__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_020__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_020__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_020__2_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_020__2_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_040__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'empty__0_100__0_040__1_000.cuda__150__3',
            #
            # RESULT_DIR / 'precision' / 'step__0_100__0_010__0_500.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'step__0_100__0_010__0_500.cuda__150__3',
            # RESULT_DIR / 'precision' / 'step__0_100__0_005__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'step__0_100__0_005__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'step__0_100__0_010__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'step__0_100__0_010__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'step__0_100__0_010__2_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'step__0_100__0_010__2_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'step__0_100__0_020__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'step__0_100__0_020__1_000.cuda__150__3',
            #
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_010__0_500.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_010__0_500.cuda__150__3',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_005__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_005__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_010__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_010__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_010__2_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_010__2_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_020__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'slalom__0_100__0_020__1_000.cuda__150__3',
            #
            # RESULT_DIR / 'precision' / 'block__0_100__0_020__0_500.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'block__0_100__0_020__0_500.cuda__150__3',
            # RESULT_DIR / 'precision' / 'block__0_100__0_010__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'block__0_100__0_010__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'block__0_100__0_020__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'block__0_100__0_020__1_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'block__0_100__0_020__2_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'block__0_100__0_020__2_000.cuda__150__3',
            # RESULT_DIR / 'precision' / 'block__0_100__0_040__1_000.cuda__150__3__1000',
            # RESULT_DIR / 'precision' / 'block__0_100__0_040__1_000.cuda__150__3',
        ]

        # plot_table(paths)

        path = RESULT_DIR / 'precision' / 'cylinder__0_100__0_020__1_000.cuda__150__3'
        losses = Losses.load(path / 'loss.csv')
        mesh = Mesh(Record).load(path / 'mesh_pred.csv')
        #
        __plot_loss(losses)
        __plot_uvp(mesh)
        __plot_stream(mesh)

    main()
