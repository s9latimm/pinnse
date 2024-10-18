from matplotlib import pyplot as plt

from base.model.mesh import arrange
from nse.model.loss import Losses
from src import RESULT_DIR, OUTPUT_DIR
from src.base.model.algebra import Real
from src.base.view import SCALE, DEFAULT_COLOR, PHI, COLORS
from src.base.view.plot import save_fig
from src.nse.model.record import Record


def plot_layer_series():
    series = [
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_020-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_040-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_060-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_080-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_100-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_120-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_140-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_160-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_180-04_010000',
        RESULT_DIR / 'step-0_100-0_010-1_000_cuda_200-04_010000',
    ]

    time = []
    mesh_mean = []
    boundary_mean = []
    losses = []
    for path in series:
        time.append(Real.load(path / 'time.csv'))
        mesh_mean.append(Record.load(path / 'mesh_mean.csv'))
        boundary_mean.append(Record.load(path / 'boundary_mean.csv'))
        losses.append(Losses.load(path / 'loss.csv'))

    plot_time(time)
    plot_mesh_mean(mesh_mean)
    plot_boundary_mean(boundary_mean)
    plot_losses([losses[0], losses[4], losses[9]])


def plot_losses(losses: list[Losses]):
    fig = plt.Figure(figsize=(PHI * SCALE, SCALE))
    ax = fig.add_subplot(1, 1, 1)

    labels = ['20', '100', '200']
    for i, loss in enumerate(losses):
        ax.plot([float(j) for j in loss][1:], color=COLORS[i], label=labels[i])

    # ax.set_xlim([0, 10000])

    x = arrange(1000, 10000, 1000)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i:.0f}' if i > 1 else '1' for i in x], rotation='vertical')
    ax.set_xlim([0, 10500])
    ax.set_xlabel('steps')

    ax.set_yscale('log')
    ax.set_ylim([1e-6, .5e-1])
    y = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    ax.set_yticks(y)
    ax.set_yticklabels([f'{i:.0e}'.replace('-0', r'-') for i in y])
    ax.set_ylabel(r'$\Sigma$ loss')

    ax.plot(10500, 1e-6, ">k", clip_on=False)
    ax.plot(0, .5e-1, "^k", clip_on=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(frameon=False)

    ax.set_title('losses', fontname='cmss10')

    save_fig(fig, OUTPUT_DIR / 'paper' / 'loss.pdf')

    plt.clf()
    plt.close()


def plot_boundary_mean(means: list[Record]):
    fig = plt.Figure(figsize=(PHI * SCALE, SCALE))
    ax = fig.add_subplot(1, 1, 1)

    x = arrange(20, 200, 20)
    y = arrange(0, .002, .001)
    u = [float(i.u) for i in means]
    v = [float(i.v) for i in means]

    ax.plot(x, u, color=COLORS[0], label=r'$\Delta$u')
    ax.plot(x, v, color=COLORS[1], label=r'$\Delta$v')

    ax.set_xticks(x)
    ax.set_xlim([15, 210])
    ax.set_xlabel('layer size')

    ax.set_yticks(y)
    ax.set_yticklabels([f'{i:.0e}'.replace('-0', r'-') if i != 0 else '0' for i in y])
    ax.set_ylim([0, .0025])

    ax.plot(210, 0, ">k", clip_on=False)
    ax.plot(15, .0025, "^k", clip_on=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    import matplotlib.font_manager as font_manager
    ax.legend(frameon=False, prop=font_manager.FontProperties(family='cmmi10'))

    ax.set_title('boundary mean', fontname='cmss10')

    save_fig(fig, OUTPUT_DIR / 'paper' / 'boundary_mean.pdf')

    plt.clf()
    plt.close()


def plot_mesh_mean(means: list[Record]):
    fig = plt.Figure(figsize=(PHI * SCALE, SCALE))
    ax = fig.add_subplot(1, 1, 1)

    x = arrange(20, 200, 20)
    y = arrange(0, 0.04, .01)
    u = [float(i.u) for i in means]
    v = [float(i.v) for i in means]

    ax.plot(x, u, color=COLORS[0], label=r'$\Delta$u')
    ax.plot(x, v, color=COLORS[1], label=r'$\Delta$v')

    ax.set_xticks(x)
    ax.set_xlim([15, 210])
    ax.set_xlabel('layer size')

    ax.set_yticks(y)
    ax.set_yticklabels([f'{i:.0e}'.replace('-0', r'-') if i != 0 else '0' for i in y])
    ax.set_ylim([0, .05])

    ax.plot(210, 0, ">k", clip_on=False)
    ax.plot(15, .05, "^k", clip_on=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    import matplotlib.font_manager as font_manager

    ax.legend(frameon=False, prop=font_manager.FontProperties(family='cmmi10'))

    ax.set_title('mesh mean', fontname='cmss10')

    save_fig(fig, OUTPUT_DIR / 'paper' / 'mesh_mean.pdf')

    plt.clf()
    plt.close()


def plot_time(time: list[Real]):
    fig = plt.Figure(figsize=(PHI * SCALE, SCALE))
    ax = fig.add_subplot(1, 1, 1)

    x = arrange(20, 200, 20)
    y = [float(i / 60) for i in time]

    ax.plot(x, y, color=DEFAULT_COLOR)

    ax.set_xticks(x)
    # ax.set_xticklabels([f'{int(i):d}' for i in x], rotation='vertical')
    ax.set_xlim([15, 210])
    ax.set_xlabel('layer size')

    ax.set_yticks(arrange(0, 25, 5))
    ax.set_ylim([0, 27.5])
    ax.set_ylabel('minutes')

    ax.plot(210, 0, ">k", clip_on=False)
    ax.plot(15, 27.5, "^k", clip_on=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('time', fontname='cmss10')

    save_fig(fig, OUTPUT_DIR / 'paper' / 'time.pdf')

    plt.clf()
    plt.close()


if __name__ == '__main__':
    plot_layer_series()
