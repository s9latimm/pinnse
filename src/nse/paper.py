import matplotlib.pyplot as plt

from src import OUTPUT_DIR
from src.base.mesh import arrange
from src.base.plot import save_fig, draw_shape
from src.nse.experiments import EXPERIMENTS
from src.nse.experiments.experiment import NSEExperiment

SCALE: float = 2.5


def plot_inlets(experiments: list[NSEExperiment]):
    fig = plt.Figure(figsize=(SCALE, len(experiments) * SCALE))
    for i, experiment in enumerate(experiments):

        ax = fig.add_subplot(len(experiments), 1, i + 1)

        x = []
        y = []
        for k, v in experiment.inlet.detach():
            x.append(k.y)
            y.append(v.u)

        ax.plot(x, y, color='k')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        ax.set_xlabel('y')
        ax.set_xticks([0, 1, 2])
        ax.set_xlim([0, 2.2])

        ax.set_ylabel('u')
        ax.set_yticks([0, 1, 2])
        ax.set_ylim([0, 2.2])

        ax.set_title(experiment.name)

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR / 'paper' / 'inlets.pdf')

    plt.clf()
    plt.close()


def plot_experiments(experiments: list[NSEExperiment]):
    fig = plt.Figure(figsize=(10 / 2.2 * SCALE, len(experiments) * SCALE))
    for i, experiment in enumerate(experiments):

        ax = fig.add_subplot(len(experiments), 1, i + 1)

        for figure in experiment.obstruction:
            draw_shape(ax, figure, style='-', width=3)

        for figure in experiment.boundary:
            draw_shape(ax, figure, style='-', width=3)

        s = 0
        for k, v in experiment.inlet:
            if 2 > k.y > 0 == k.y % .25:
                u = v.u
                s += u
                ax.arrow(
                    k.x,
                    k.y,
                    u,
                    0,
                    width=.05,
                    head_width=.13,
                    head_length=.1,
                    length_includes_head=True,
                    color='k',
                )

        u = s / 7
        for y in arrange(0.25, 1.75, .25):
            ax.arrow(
                10 - u,
                y,
                u,
                0,
                width=.05,
                head_width=.13,
                head_length=.1,
                length_includes_head=True,
                color='k',
            )

        ax.set_axis_off()

        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_xlim([0, 10])

        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_ylim([-.1, 2.1])

        ax.set_title(str(experiment.inlet_f))

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR / 'paper' / 'experiments.pdf')

    plt.clf()
    plt.close()


if __name__ == '__main__':
    plot_experiments(list(i() for i in EXPERIMENTS.values()))
    plot_inlets(list(i() for i in EXPERIMENTS.values()))
