from __future__ import annotations

import typing as tp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

from src.base.model.mesh import Coordinate
from src.base.model.shape import Shape, Figure
from src.base.view import SCALE, DPI, COLORS, SEISMIC, SEISMIC_NEGATIVE, SEISMIC_POSITIVE


def draw_shape(ax: plt.Axes, shape: Shape, style: str = '-', width: float = 2.5) -> None:
    polygon = shape[::.05]
    ax.add_patch(
        Polygon([(c.x, c.y) for c in polygon],
                hatch='//',
                facecolor='gray',
                edgecolor='black',
                linestyle=style,
                linewidth=width,
                zorder=999))


def save_fig(fig: plt.Figure, path: Path) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=path.suffix[1:], bbox_inches='tight', transparent=True, dpi=DPI / SCALE)


def plot_history(
    title: str,
    plots: tp.Sequence[tuple[str, tp.Sequence[tuple[str, tp.Sequence[float]]]]],
    path: Path = None,
) -> None:
    fig = plt.figure(figsize=(5 * SCALE, SCALE * len(plots)))

    for i, plot in enumerate(plots):
        label, lines = plot

        ax = fig.add_subplot(len(plots), 1, i + 1)

        t_min, t_max = np.infty, -np.infty
        for j, line in enumerate(lines):
            l, y = line

            ax.plot(np.arange(1, len(y) + 1, 1), y, label=l, color=COLORS[j])
            ax.axhline(y=float(y[-1]), color=COLORS[j], linestyle='--')
            y_min, y_max = np.nanmin(y), np.nanmax(y)
            t_min, t_max = min(t_min, y_min), max(t_max, y_max)

        ax.set_xlabel('iter')
        ax.set_ylabel('err')
        ax.set_title(label)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ax.set_xscale('log')
        ax.set_xlim([1, len(lines[0][1])])
        ax.set_xticks([1, 10**np.floor(np.log10(len(lines[0][1])))])
        ax.set_yscale('log')
        ax.set_yticks([10**np.floor(np.log10(t_max)), 10**np.floor(np.log10(t_min)), 10**np.ceil(np.log10(t_min))])

        ax.legend(loc='upper right')

    fig.suptitle(title)

    save_fig(fig, path)

    plt.clf()
    plt.close()


class Plot(plt.Figure):

    def __init__(self, x: np.ndarray, y: np.ndarray, path: Path = None, n: int = 1) -> None:
        self.__x = x
        self.__y = y
        self.__path = path
        q = (int(np.round(np.max(x))) - int(np.round(np.min(x)))) / (int(np.round(np.max(y))) -
                                                                     int(np.round(np.min(y))))
        super().__init__(figsize=(q * SCALE, n * SCALE))

    def setup(self, ax: plt.Axes, title: str, boundary: Figure = None, figure: Figure = None) -> None:
        ax.pcolormesh(
            self.__x,
            self.__y,
            np.full(self.__x.shape, fill_value=False),
            shading='nearest',
            alpha=0.1,
            cmap='binary',
            zorder=-1,
            rasterized=True,
            antialiased=True,
        )

        ax.axes.set_aspect('equal')

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('x', fontname='cmmi10')
        ax.set_xticks([0, 1, int(np.round(self.__x.max())) // 2, int(np.round(self.__x.max()))])
        ax.set_xlim([int(np.round(self.__x.min())) - .05, int(np.round(self.__x.max())) + .05])

        ax.set_ylabel('y', fontname='cmmi10')
        ax.set_yticks([0, 1, int(np.round(self.__y.max())) // 2, int(np.round(self.__y.max()))])
        ax.set_ylim([int(np.round(self.__y.min())) - .05, int(np.round(self.__y.max())) + .05])

        ax.tick_params(axis='y', labelrotation=90)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

        if boundary is not None:
            for shape in boundary:
                draw_shape(ax, shape)
        if figure is not None:
            for shape in figure:
                draw_shape(ax, shape)

        ax.set_title(title, fontname='cmmi10')

    def __enter__(self, *_) -> Plot:
        return self

    def __exit__(self, *_) -> None:
        if self.__path is not None:
            self.tight_layout()
            save_fig(self, self.__path)

        plt.clf()
        plt.close()


def plot_seismic(
        title: str,
        x: np.ndarray,
        y: np.ndarray,
        plots: tp.Sequence[tuple[str, np.ndarray]],
        masks: tp.Sequence[np.ndarray] = None,
        path: Path = None,
        marker: tp.Sequence[Coordinate] = (),
        boundary: Figure = None,
        figure: Figure = None,
        mesh=None,
) -> None:
    if masks is not None:
        assert len(plots) == len(masks)

    with Plot(x, y, path, len(plots)) as fig:
        fig.suptitle(title, fontname='cmss10')

        for i, plot in enumerate(plots):
            label, z = plot

            ax = fig.add_subplot(len(plots), 1, i + 1)
            fig.setup(ax, '', boundary, figure)

            if np.nanmin(z) < 0 < np.nanmax(z):
                slope = colors.TwoSlopeNorm(vmin=np.nanmin(z), vcenter=0, vmax=np.nanmax(z))
                cmap = SEISMIC
            elif np.nanmin(z) == np.nanmax(z):
                slope = colors.Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
                cmap = SEISMIC
            elif np.nanmax(z) < 0:
                slope = colors.Normalize(vmin=np.nanmin(z), vmax=0)
                cmap = SEISMIC_NEGATIVE
            else:
                slope = colors.Normalize(vmin=0, vmax=np.nanmax(z))
                cmap = SEISMIC_POSITIVE

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

            if masks is not None:
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
                [j.x for j in marker],
                [j.y for j in marker],
                marker='.',
                color=COLORS[0],
                zorder=999999,
                clip_on=False,
            )

            if mesh is not None:
                ps = []
                vs = []
                for k, v in mesh:
                    ps.append(k)
                    vs.append(list(v)[i])
                ax.scatter(
                    [j.x for j in ps],
                    [j.y for j in ps],
                    marker='D',
                    edgecolor='k',
                    c=vs,
                    cmap=SEISMIC_POSITIVE,
                    norm=colors.Normalize(vmin=0, vmax=max(vs)),
                    zorder=99999,
                    clip_on=False,
                )

            cbar = fig.colorbar(
                img,
                ax=ax,
                orientation='vertical',
                fraction=0.046,
                pad=0.04,
                format=FuncFormatter(lambda j, pos: f'{j:.3f}' if j != 0 else '0'),
            )

            ticks = []
            if np.nanmin(z) < 0:
                ticks.append(np.nanmin(z))
            ticks.append(0)
            if np.nanmax(z) > 0:
                ticks.append(np.nanmax(z))
            cbar.set_ticks(ticks)
            cbar.set_label(label, rotation=0, fontname='cmmi10')


def plot_stream(
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    path: Path = None,
    boundary: Figure = None,
    figure: Figure = None,
) -> None:
    with Plot(x, y, path) as fig:
        ax = fig.add_subplot()
        fig.setup(ax, title, boundary, figure)

        # speed = np.sqrt(np.square(u) + np.square(v))
        # speed = 4 * speed / np.nanmax(speed)

        ax.streamplot(
            x.transpose(),
            y.transpose(),
            u.transpose(),
            v.transpose(),
            broken_streamlines=False,
            arrowsize=.5,
            color=COLORS[1],
            density=.4,
            linewidth=1,
        )


def plot_arrows(
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    path: Path = None,
    boundary: Figure = None,
    figure: Figure = None,
) -> None:
    with Plot(x, y, path) as fig:
        ax = fig.add_subplot()
        fig.setup(ax, title, boundary, figure)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s = np.sqrt(np.square(u[i, j]) + np.square(v[i, j])) + 1e-6
                ax.arrow(
                    float(x[i, j]),
                    float(y[i, j]),
                    u[i, j] / s * .05,
                    v[i, j] / s * .05,
                    width=.001,
                    head_width=.015,
                    head_length=.02,
                    length_includes_head=True,
                    color=COLORS[1],
                )
