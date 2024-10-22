import numpy as np
from matplotlib import pyplot as plt

from base.model.mesh import Mesh, Grid, arrange
from nse.model.experiments import Step
from nse.model.loss import Losses
from nse.model.record import Record
from src import RESULT_DIR, OUTPUT_DIR
from src.base.view import SCALE, PHI, COLORS
from src.base.view.plot import save_fig, plot_seismic, plot_stream


def __plot_uvp(mesh: Mesh) -> None:
    experiment = Step(.01, 1, 1)
    grid = Grid(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    d = grid.transform(mesh)

    plot_seismic(
        '',
        grid.x,
        grid.y,
        [
            (r'$\Delta u$', d.u),
        ],
        path=OUTPUT_DIR / 'paper' / 'u_diff.pdf',
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
        path=OUTPUT_DIR / 'paper' / 'v_diff.pdf',
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
    experiment = Step(.01, 1, 1)
    grid = Grid(experiment.x.arrange(.1, True), experiment.y.arrange(.1, True))
    d = grid.transform(mesh)

    plot_stream(
        '',
        grid.x,
        grid.y,
        d.u,
        d.v,
        path=OUTPUT_DIR / 'paper' / 'stream_foam.pdf',
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

    ax.plot(arrange(1, len(losses) - 1, 1), u[1:], label=r'$f$', color=COLORS[0])
    ax.plot(arrange(1, len(losses) - 1, 1), g[1:], label=r'$g$', color=COLORS[1])
    ax.plot(arrange(1, len(losses) - 1, 1), u[1:], label=r'$\hat{u}$', color=COLORS[2])
    ax.plot(arrange(1, len(losses) - 1, 1), v[1:], label=r'$\hat{v}$', color=COLORS[3])
    ax.plot(arrange(1, len(losses) - 1, 1), s[1:], label=r'$\mathcal{L}$', color=COLORS[4])

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('log$_{10}$(Loss)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_yscale('log')
    ax.minorticks_off()

    ax.set_xlim([1, 2.2e4])
    ax.set_xticks([1, 1e4, 2e4])
    ax.set_xticklabels(['1', '1e4', '2e4'])

    ax.set_ylim([1e-7, 5e-1])
    ax.set_yticks([1e-7, 1e-4, 1e-1])
    ax.set_yticklabels(['1e-7', '1e-4', '1e-1'])
    ax.plot(1, 5e-1, "^k", clip_on=False)
    ax.plot(2.2e4, 1e-7, ">k", clip_on=False)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fancybox=False).get_frame().set_edgecolor('k')

    save_fig(fig, OUTPUT_DIR / 'paper' / 'loss.pdf')

    plt.clf()
    plt.close()


if __name__ == '__main__':

    def main():
        path = RESULT_DIR / 'precision' / 'step-0_100-0_010-01_cuda_150-03_030000'
        losses = Losses.load(path / 'loss.csv')
        mesh = Mesh(Record).load(path / 'mesh_diff.csv')

        __plot_loss(losses)
        __plot_uvp(mesh)
        __plot_stream(mesh)

    main()
