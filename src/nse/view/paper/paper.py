import matplotlib.pyplot as plt

from src import OUTPUT_DIR
from src.base.view.plot import save_fig, draw_shape, COLORS
from src.nse.model.experiments import Step, EXPERIMENTS
from src.nse.model.experiments.experiment import Experiment

SCALE: float = 2.5


def plot_inlets(experiments: list[Experiment]):
    fig = plt.Figure(figsize=(SCALE, len(experiments) * SCALE))
    for i, experiment in enumerate(experiments):

        ax = fig.add_subplot(len(experiments), 1, i + 1)

        x = []
        y = []
        for k, v in experiment.inlet.detach():
            x.append(k.y)
            y.append(v.u)

        ax.plot(x, y, color=COLORS[1], label=experiment.inlet_f)

        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axes.set_aspect('equal')

        ax.set_xlabel('x', fontname='cmmi10')
        ax.set_xticks([0, 1, 2])
        ax.set_xlim([0, 2.2])

        ax.set_ylabel('u', fontname='cmmi10')
        ax.set_yticks([0, 1, 2])
        ax.set_ylim([0, 2.2])

        # ax.set_title(str(experiment.inlet_f))

    save_fig(fig, OUTPUT_DIR / 'paper' / 'inlets.pdf')

    plt.clf()
    plt.close()


def plot_grid(experiment: Experiment):
    fig = plt.Figure(figsize=(10 / 2 * SCALE, SCALE))

    ax = fig.add_subplot(1, 1, 1)

    for figure in experiment.obstruction:
        draw_shape(ax, figure, style='-', width=.5)

    for figure in experiment.boundary:
        draw_shape(ax, figure, style='-', width=.5)

    xs = []
    ys = []
    for x in experiment.x.arrange(.1):
        for y in experiment.y.arrange(.1):
            if experiment.x.start < x and experiment.y.start < y < experiment.y.stop:
                if (x, y) not in experiment.obstruction:
                    xs.append(x)
                    ys.append(y)

    ax.scatter(xs, ys, color=COLORS[1], marker='+', s=20, linewidth=.5)
    ax.scatter(xs, ys, color=COLORS[1], marker='o', s=2.5, linewidth=.5)

    xs = []
    ys = []
    for x in experiment.x.arrange(.05):
        if x > 1.01:
            xs.append(x)
            ys.append(0)
        else:
            xs.append(x)
            ys.append(1)
        xs.append(x)
        ys.append(2)

    for y in experiment.y.arrange(.05):
        if y < 1:
            xs.append(1)
            ys.append(y)

    ax.scatter(xs, ys, color='k', marker='D', s=2.5, linewidth=.5, zorder=9999)

    xs = []
    ys = []
    for y in experiment.y.arrange(.05):
        if 1.01 < y < 1.99:
            xs.append(0)
            ys.append(y)

    ax.scatter(xs, ys, color=COLORS[2], marker='>', s=2.5, linewidth=.5, zorder=999)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.axes.set_aspect('equal')

    ax.set_xlabel('x', fontname='cmmi10')
    ax.set_xticks([0, 1, 5, 10])
    ax.set_xlim([-.05, 10.05])

    ax.set_ylabel('y', fontname='cmmi10')
    ax.set_yticks([0, 1, 2])
    ax.set_ylim([-.05, 2.05])

    save_fig(fig, OUTPUT_DIR / 'paper' / 'grid.pdf')

    plt.clf()
    plt.close()


def plot_experiments(experiments: list[Experiment]):
    fig = plt.Figure(figsize=(10 / 2 * SCALE, len(experiments) * SCALE))
    for i, experiment in enumerate(experiments):

        ax = fig.add_subplot(len(experiments), 1, i + 1)

        for figure in experiment.obstruction:
            draw_shape(ax, figure, style='-', width=3)

        for figure in experiment.boundary:
            draw_shape(ax, figure, style='-', width=3)

        s = 0
        n = 0
        for k, v in experiment.inlet:
            if k.y % .25 == 0 and k not in experiment.obstruction and k not in experiment.boundary:
                u = v.u
                s += u
                n += 1
                ax.arrow(
                    k.x,
                    k.y,
                    u,
                    0,
                    width=.05,
                    head_width=.13,
                    head_length=.1,
                    length_includes_head=True,
                    color=COLORS[1],
                )

        u = s / n * len(experiment.inlet) / len(experiment.outlet)
        for k, v in experiment.outlet:
            if k.y % .25 == 0 and k not in experiment.obstruction and k not in experiment.boundary:
                ax.arrow(
                    10 - u,
                    k.y,
                    u,
                    0,
                    width=.05,
                    head_width=.13,
                    head_length=.1,
                    length_includes_head=True,
                    color=COLORS[1],
                )

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axes.set_aspect('equal')

        ax.set_xlabel('x', fontname='cmmi10')
        ax.set_xticks([0, 1, 5, 10])
        ax.set_xlim([-.05, 10.05])

        ax.set_ylabel('u', fontname='cmmi10')
        ax.set_yticks([0, 1, 2])
        ax.set_ylim([-.05, 2.05])

        ax.set_title(experiment.name, fontname='cmss10')

    save_fig(fig, OUTPUT_DIR / 'paper' / 'experiments.pdf')

    plt.clf()
    plt.close()


if __name__ == '__main__':
    plot_experiments(list(i() for i in EXPERIMENTS.values()))
    plot_inlets(list(i() for i in EXPERIMENTS.values()))
    plot_grid(Step())
