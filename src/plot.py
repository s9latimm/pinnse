import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# plt.rcParams['text.usetex'] = True


class Plot:
    COLORS = ['black'] + list(mcolors.TABLEAU_COLORS.keys())[1:]

    DPI = 96

    @staticmethod
    def error(title, plots, path=None):
        fig = plt.figure(figsize=(1024 / Plot.DPI, 512 / Plot.DPI),
                         dpi=Plot.DPI)

        ax = fig.add_subplot()

        for i, d in enumerate(plots):
            l, y = d

            ax.plot(np.arange(1,
                              len(y) + 1, 1),
                    y,
                    label=l,
                    color=Plot.COLORS[i])

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
        if path is not None:
            fig.savefig(f'{path}.png',
                        format='png',
                        transparent=False,
                        dpi=Plot.DPI)
        plt.show()

    @staticmethod
    def heatmap(title, x, y, plots, grids=None, path=None):
        if grids is None:
            grids = list()
        fig = plt.figure(figsize=(.8 * 2048 / Plot.DPI,
                                  3 * .2 * 2048 / Plot.DPI),
                         dpi=Plot.DPI)

        for i, d in enumerate(plots):
            l, z = d

            ax = fig.add_subplot(len(plots), 1, i + 1)

            if z.min() < 0 < z.max():
                slope = mcolors.TwoSlopeNorm(vmin=np.floor(z.min()),
                                             vcenter=0.,
                                             vmax=np.ceil(z.max()))
                cmap = 'seismic'
            elif z.max() < 0:
                slope = mcolors.Normalize(vmin=np.floor(z.min()), vmax=0)
                cmap = 'Reds'
            else:
                slope = mcolors.Normalize(vmin=0, vmax=np.ceil(z.max()))
                cmap = 'Reds'

            img = ax.pcolormesh(
                x,
                y,
                z,
                cmap=cmap,
                # shading='gouraud',
                norm=slope,
                zorder=1,
            )

            if grids is not None:
                for j, grid in enumerate(grids):
                    ax.scatter(
                        grid[:, 0],
                        grid[:, 1],
                        marker='+',
                        c=Plot.COLORS[j],
                        zorder=2,
                    )

            ax.set_xlabel('')
            ax.set_ylabel('')

            ax.set_title(l)

            ax.set_xticks([])
            ax.set_yticks([])

            cbar = fig.colorbar(img,
                                ax=ax,
                                orientation='vertical',
                                fraction=0.046,
                                pad=0.04,
                                format=FuncFormatter(lambda x, pos: f'{x:.0f}'))

            if z.min() < 0 < z.max():
                cbar.set_ticks([np.floor(z.min()), 0., np.ceil(z.max())])
            elif z.max() < 0:
                cbar.set_ticks([np.floor(z.min()), 0.])
            else:
                cbar.set_ticks([0., np.ceil(z.max())])

        fig.suptitle(title)

        fig.tight_layout()
        if path is not None:
            fig.savefig(f'{path}.png',
                        format='png',
                        transparent=False,
                        dpi=Plot.DPI)
        plt.show()
