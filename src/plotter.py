from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# plt.rcParams['text.usetex'] = True


class Plotter:
    COLORS = ['black'] + list(colors.TABLEAU_COLORS.keys())[1:]

    DPI = 96

    @staticmethod
    def error(title: str, plots, out: Path = None):
        fig = plt.figure(figsize=(1024 / Plotter.DPI, 512 / Plotter.DPI),
                         dpi=Plotter.DPI)

        ax = fig.add_subplot()

        for i, d in enumerate(plots):
            l, y = d

            ax.plot(np.arange(1,
                              len(y) + 1, 1),
                    y,
                    label=l,
                    color=Plotter.COLORS[i])

        ax.set_xlabel('iter')
        ax.set_ylabel('err')
        ax.set_title(title)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_yscale('log')
        ax.set_xticks([1, len(plots[0][1])])
        # ax.set_yticks([0., .5, 1])

        ax.legend(loc='upper right')

        fig.tight_layout()
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out,
                        format=out.suffix[1:],
                        transparent=False,
                        dpi=Plotter.DPI)
        plt.show()

    @staticmethod
    def heatmap(title: str, x, y, plots, grids=None, out: Path = None):
        if grids is None:
            grids = list()
        fig = plt.figure(figsize=(.8 * 2048 / Plotter.DPI,
                                  3 * .2 * 2048 / Plotter.DPI),
                         dpi=Plotter.DPI)

        for i, plot in enumerate(plots):
            label, z = plot

            ax = fig.add_subplot(len(plots), 1, i + 1)

            cmap = 'seismic'

            if z.min() < 0 < z.max():
                slope = colors.Normalize(vmin=min(z.min(), -1),
                                         vmax=max(z.max(), 1))
                m = max(-min(z.min(), -1), max(z.max(), 1))
                start = .5 - -min(z.min(), -1) / m * .5
                stop = .5 + max(z.max(), 1) / m * .5
                cmap = colors.LinearSegmentedColormap.from_list(
                    f'seismic_{i}',
                    plt.get_cmap(cmap)(np.linspace(start, stop, 100)))

            elif z.max() < 0:
                slope = colors.Normalize(vmin=min(z.min(), -1), vmax=0)
                cmap = colors.LinearSegmentedColormap.from_list(
                    f'seismic_{i}',
                    plt.get_cmap(cmap)(np.linspace(0., .5, 100)))
            else:
                slope = colors.Normalize(vmin=0, vmax=max(z.max(), 1))
                cmap = colors.LinearSegmentedColormap.from_list(
                    f'seismic_{i}',
                    plt.get_cmap(cmap)(np.linspace(.5, 1., 100)))

            img = ax.pcolormesh(
                x,
                y,
                z,
                cmap=cmap,
                shading='nearest',
                norm=slope,
                zorder=1,
            )

            if grids is not None:
                for j, grid in enumerate(grids):
                    ax.scatter(
                        grid[:, 0],
                        grid[:, 1],
                        marker='.',
                        c=Plotter.COLORS[j],
                        zorder=2,
                    )

            ax.set_xlabel('')
            ax.set_xticks([])

            ax.set_ylabel('')
            ax.set_yticks([])
            ax.yaxis.set_inverted(True)

            ax.set_title(label)

            cbar = fig.colorbar(img,
                                ax=ax,
                                orientation='vertical',
                                fraction=0.046,
                                pad=0.04,
                                format=FuncFormatter(lambda x, pos: f'{x:.0f}'))

            ticks = list()
            if z.min() < 0:
                ticks.append(min(z.min(), -1))
            ticks.append(0)
            if z.max() > 0:
                ticks.append(max(z.max(), 1))
            cbar.set_ticks(ticks)

        fig.suptitle(title)

        fig.tight_layout()
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out,
                        format=out.suffix[1:],
                        transparent=False,
                        dpi=Plotter.DPI)
        plt.show()
