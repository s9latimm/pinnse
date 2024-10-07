import typing as tp
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.base.geometry import Coordinate, arrange, Cloud
from src.base.shapes import Shape

# plt.rcParams['text.usetex'] = True
# pprint(sorted(matplotlib.font_manager.get_font_names()))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = .5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = .5
# plt.switch_backend('Agg')

COLORS = ['black'] + list(colors.TABLEAU_COLORS.keys())[1:]

DPI = 1000
SCALE = 5


def decorate(ax: plt.Axes, decorations: tp.Sequence[Shape]) -> None:
    for decoration in decorations:
        shape = decoration[.05]
        ax.plot(shape.x, shape.y, color='k', linestyle='--', linewidth=1, zorder=999)
        # ax.scatter([i.x for i in shape], [i.y for i in shape], color='k', marker='x', zorder=998)


def save_fig(fig, path: Path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], transparent=False, dpi=DPI / SCALE)


def plot_losses(
        title: str,
        plots: tp.Sequence[tuple[str, tp.Sequence[tuple[str, np.ndarray]]]],
        path: Path = None,
        geometry=(),
):
    fig = plt.figure(figsize=(5 * SCALE, SCALE * len(plots)))

    for i, plot in enumerate(plots):
        label, lines = plot

        ax = fig.add_subplot(len(plots), 1, i + 1)

        m = np.infty
        for j, line in enumerate(lines):
            l, y = line

            ax.plot(np.arange(1, len(y) + 1, 1), y, label=l, color=COLORS[j])
            ax.axhline(y=y[-1], color=COLORS[j], linestyle='--')
            m = min(m, y.min())

        ax.set_xlabel('iter')
        ax.set_ylabel('err')
        ax.set_title(label)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_yscale('log')
        ax.set_xticks([1] + list(arrange(1000, len(lines[0][1]), 1000)))
        ax.set_yticks([10**np.floor(np.log10(m)), 10**np.ceil(np.log10(m))])

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


def plot_heatmaps(
        title: str,
        x: np.ndarray,
        y: np.ndarray,
        plots: tp.Sequence[tuple[str, np.ndarray]],
        masks: tp.Sequence[np.ndarray] = None,
        path: Path = None,
        marker: tp.Sequence[Coordinate] = (),
        geometry: tp.Sequence[Shape] = (),
):
    fig = plt.figure(figsize=(5 * SCALE, len(plots) * SCALE))

    for i, plot in enumerate(plots):
        label, z = plot

        ax = fig.add_subplot(len(plots), 1, i + 1)

        if z.min() < 0 < z.max():
            slope = colors.TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())
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

        ax.scatter(
            [i.x for i in marker],
            [i.y for i in marker],
            marker='.',
            c=COLORS[0],
            zorder=3,
        )

        ax.set_xlabel('x')
        ax.set_xticks([0, 1, int(np.round(x.max()))])
        ax.set_xlim([0, int(np.round(x.max()))])

        ax.set_ylabel('y')
        ax.set_yticks([0, 1, int(np.round(y.max()))])
        ax.set_ylim([0, int(np.round(y.max()))])

        ax.set_title(label)

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

        decorate(ax, geometry)

    fig.suptitle(title)

    fig.tight_layout()
    save_fig(fig, path)

    plt.clf()
    plt.close()


def plot_clouds(
        title: str,
        x: np.ndarray,
        y: np.ndarray,
        cloud: Cloud,
        labels=(),
        path: Path = None,
        marker: tp.Sequence[Coordinate] = (),
        geometry: tp.Sequence[Shape] = (),
):
    plots = []
    masks = []

    for p, label, in enumerate(labels):
        plot = np.zeros(x.shape)
        mask = np.full(x.shape, .5)
        for k, v in cloud:
            if not np.isnan(v[p]):
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if Coordinate(x[i][j], y[i][j]) == k:
                            plot[i][j] += v[p]
                            mask[i][j] = -1
        plots.append((label, plot))
        masks.append(mask)

    plot_heatmaps(title, x, y, plots, marker=marker, masks=masks, path=path, geometry=geometry)


def plot_streamlines(
        title: str,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        path: Path = None,
        geometry: tp.Sequence[Shape] = (),
):
    fig = plt.figure(figsize=(5 * SCALE, SCALE))

    ax = fig.add_subplot()

    ax.set_xlabel('x')
    ax.set_xticks([0, 1])

    ax.set_ylabel('y')
    ax.set_yticks([0, 1])
    ax.pcolormesh(
        x,
        y,
        np.zeros(x.shape, dtype=bool),
        shading='nearest',
        alpha=0.1,
        cmap='binary',
        zorder=1,
        rasterized=True,
        antialiased=True,
    )
    speed = np.sqrt(np.square(u) + np.square(v))
    speed = 1 + 4 * speed / speed.max()

    ax.streamplot(
        x.transpose(),
        y.transpose(),
        u.transpose(),
        v.transpose(),
        broken_streamlines=True,
        arrowsize=2,
        color=COLORS[0],
        density=.5,
        linewidth=speed.transpose(),
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(title)

    decorate(ax, geometry)

    fig.tight_layout()
    save_fig(fig, path)

    plt.clf()
    plt.close()


def plot_arrows(
        title: str,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        path: Path = None,
        geometry: tp.Sequence[Shape] = (),
):
    fig = plt.figure(figsize=(5 * SCALE, SCALE))

    ax = fig.add_subplot()

    ax.set_xlabel('x')
    ax.set_xticks([0, 1])

    ax.set_ylabel('y')
    ax.set_yticks([0, 1])

    ax.pcolormesh(
        x,
        y,
        np.zeros(x.shape, dtype=bool),
        shading='nearest',
        alpha=0.1,
        cmap='binary',
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

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(title)

    decorate(ax, geometry)

    fig.tight_layout()
    save_fig(fig, path)

    plt.clf()
    plt.close()
