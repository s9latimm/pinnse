from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# plt.rcParams['text.usetex'] = True
# pprint(sorted(matplotlib.font_manager.get_font_names()))
plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.switch_backend('Agg')

COLORS = ['black'] + list(colors.TABLEAU_COLORS.keys())[1:]

DPI = 1000
SCALE = 5


def save_fig(fig, path: Path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], transparent=False, dpi=DPI / SCALE)


def plot_losses(title: str, plots, path: Path = None):
    fig = plt.figure(figsize=(5 * SCALE, SCALE * len(plots)))

    for i, plot in enumerate(plots):
        label, lines = plot

        ax = fig.add_subplot(len(plots), 1, i + 1)

        for j, line in enumerate(lines):
            l, y = line

            ax.plot(np.arange(1, len(y) + 1, 1), y, label=l, color=COLORS[j])

        ax.set_xlabel('iter')
        ax.set_ylabel('err')
        ax.set_title(label)

        ax.axhline(y=1, color='black', linestyle='--')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_yscale('log')
        ax.set_xticks([1, len(lines[0][1])])
        # ax.set_yticks([0., .5, 1])

        ax.legend(loc='upper right')

    fig.suptitle(title)

    fig.tight_layout()
    save_fig(fig, path)

    plt.clf()
    plt.close()


seismic = colors.LinearSegmentedColormap.from_list('seismic', plt.get_cmap('seismic')(np.linspace(0, 1., 100)))
seismic_neg = colors.LinearSegmentedColormap.from_list('seismic_neg', plt.get_cmap('seismic')(np.linspace(0., .5, 50)))
seismic_pos = colors.LinearSegmentedColormap.from_list('seismic_pos', plt.get_cmap('seismic')(np.linspace(.5, 1., 50)))
seismic_dis = colors.LinearSegmentedColormap.from_list('seismic_dis', plt.get_cmap('seismic')(np.linspace(.55, 1., 45)))


def plot_heatmaps(title: str, x, y, plots, grid=None, masks=None, path: Path = None):
    fig = plt.figure(figsize=(5 * SCALE, len(plots) * SCALE))

    for i, plot in enumerate(plots):
        label, z = plot

        ax = fig.add_subplot(len(plots), 1, i + 1)

        if z.min() < 0 < z.max():
            slope = colors.TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())
            # slope = colors.Normalize(vmin=min(z.min(), -1),
            #                          vmax=max(z.max(), 1))
            # m = max(-min(z.min(), -1), max(z.max(), 1))
            # start = .5 - -min(z.min(), -1) / m * .5
            # stop = .5 + max(z.max(), 1) / m * .5
            cmap = seismic
        elif z.min() == z.max():
            slope = colors.Normalize(vmin=z.min(), vmax=z.max())
            cmap = seismic
        elif z.max() < 0:
            slope = colors.Normalize(vmin=z.min(), vmax=0)
            cmap = seismic_neg
        else:
            slope = colors.Normalize(vmin=0, vmax=z.max())
            cmap = seismic_pos

        img = ax.pcolormesh(
            x,
            y,
            z,
            cmap=cmap,
            shading='nearest',
            norm=slope,
            zorder=1,
            rasterized=True,
            antialiased=True,
        )

        if masks:
            cmap = plt.get_cmap('Grays')
            cmap.set_under('k', alpha=0)
            ax.pcolormesh(
                x,
                y,
                masks[i],
                cmap=cmap,
                shading='nearest',
                zorder=2,
                rasterized=True,
                antialiased=True,
                vmin=0,
                vmax=1,
            )

        if grid is not None:
            ax.scatter(
                grid[:, 0],
                grid[:, 1],
                marker='.',
                c=COLORS[0],
                zorder=3,
            )

        ax.set_xlabel('x')
        ax.set_xticks([0, 1])

        ax.set_ylabel('y')
        ax.set_yticks([0, 1])

        ax.set_title(label)

        # ax.hlines(10, xmin=0, xmax=9, colors='black', linestyles='dotted')
        # ax.vlines(9, ymin=10, ymax=19, colors='black', linestyles='dotted')

        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation='vertical',
            fraction=0.046,
            pad=0.04,
            format=FuncFormatter(lambda x, pos: f'{x:.1f}'),
        )

        ticks = list()
        if z.min() < 0:
            ticks.append(z.min())
        ticks.append(0)
        if z.max() > 0:
            ticks.append(z.max())
        cbar.set_ticks(ticks)

    fig.suptitle(title)

    fig.tight_layout()
    save_fig(fig, path)

    plt.clf()
    plt.close()


def plot_clouds(title: str, x, y, clouds, grid=None, path: Path = None):
    plots = []
    masks = []
    for label, cloud in clouds:
        plot = np.zeros(x.shape)
        mask = np.full(x.shape, .5)
        for c in cloud:
            if not np.isnan(c[2]):
                plot[int(c[0] * 10)][int(c[1] * 10)] += c[2]
                mask[int(c[0] * 10)][int(c[1] * 10)] = -1
        plots.append((label, plot))
        masks.append(mask)

    plot_heatmaps(title, x, y, plots, grid=grid, masks=masks, path=path)


def plot_streamlines(title: str, x, y, u, v, path: Path = None):
    fig = plt.figure(figsize=(5 * SCALE, SCALE))

    ax = fig.add_subplot()

    ax.set_xlabel('x')
    ax.set_xticks([0, 1])

    ax.set_ylabel('y')
    ax.set_yticks([0, 1])

    speed = np.sqrt(np.square(u) + np.square(v))
    speed = 1 + 5 * speed / speed.max()

    ax.streamplot(
        x.transpose(),
        y.transpose(),
        u.transpose(),
        v.transpose(),
        broken_streamlines=False,
        arrowsize=2,
        color=COLORS[0],
        density=.5,
        linewidth=speed.transpose(),
    )

    mask = np.zeros(x.shape, dtype=bool)
    mask[0:10, 10:20] = True

    ax.pcolormesh(
        x,
        y,
        ~mask,
        shading='nearest',
        alpha=0.1,
        cmap='gray',
        zorder=1,
        rasterized=True,
        antialiased=True,
    )

    ax.set_title(title)

    fig.tight_layout()
    save_fig(fig, path)

    plt.clf()
    plt.close()


def plot_arrows(title: str, x, y, u, v, path: Path = None):
    fig = plt.figure(figsize=(5 * SCALE, SCALE))

    ax = fig.add_subplot()

    ax.set_xlabel('x')
    ax.set_xticks([0, 1])

    ax.set_ylabel('y')
    ax.set_yticks([0, 1])

    mask = np.zeros(x.shape, dtype=bool)
    mask[0:10, 10:20] = True

    ax.pcolormesh(
        x,
        y,
        ~mask,
        shading='nearest',
        alpha=0.1,
        cmap='gray',
        zorder=1,
        rasterized=True,
        antialiased=True,
    )

    for i in range(100):
        for j in range(20):
            s = np.sqrt(np.square(u[i, j]) + np.square(v[i, j])) + 1e-6
            ax.arrow(
                x[i, j],
                y[i, j],
                u[i, j] / s * .05,
                v[i, j] / s * .05,
                width=.001,
                head_width=.015,
                head_length=.02,
                length_includes_head=True,
                color=COLORS[0],
            )

    ax.set_title(title)

    fig.tight_layout()
    save_fig(fig, path)

    plt.clf()
    plt.close()
