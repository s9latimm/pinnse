import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# plt.rcParams['text.usetex'] = True


class Plot:
    COLORS = list(mcolors.TABLEAU_COLORS.keys())

    DPI = 96

    @staticmethod
    def error(title, plots, path=None):
        fig = plt.figure(figsize=(1024 / Plot.DPI, 512 / Plot.DPI),
                         dpi=Plot.DPI)

        ax = fig.add_subplot()

        for i, d in enumerate(plots):
            l, y = d

            # y = (y - np.min(y)) / (np.max(y) - np.min(y))

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
                        transparent=True,
                        dpi=Plot.DPI)
        plt.show()

    @staticmethod
    def heatmap(title, x, y, plots, path=None):
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
            elif z.max() < 0:
                slope = mcolors.TwoSlopeNorm(vmin=np.floor(z.min()),
                                             vcenter=0.,
                                             vmax=1)
            else:
                slope = mcolors.TwoSlopeNorm(vmin=-1,
                                             vcenter=0.,
                                             vmax=np.ceil(z.max()))
            img = ax.pcolormesh(
                x,
                y,
                z,
                cmap='bwr',
                # shading='gouraud',
                norm=slope,
                zorder=1,
            )

            ax.set_xlabel('x')
            ax.set_ylabel('y')

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
                        transparent=True,
                        dpi=Plot.DPI)
        plt.show()

    @staticmethod
    def arrows(title, x, y, dx, dy):
        fig = plt.figure(figsize=(2048 / Plot.DPI, 2048 / Plot.DPI),
                         dpi=Plot.DPI)
        ax = fig.add_subplot()

        ax.quiver(x, y, dx, dy)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)

        fig.tight_layout()
        # fig.savefig(f'{title}.png', format='png', transparent=True, dpi=DPI)
        plt.show()

    @staticmethod
    def scatter_3d(title, plots):

        fig = plt.figure(figsize=(2048 / Plot.DPI * len(plots),
                                  2048 / Plot.DPI),
                         dpi=Plot.DPI)

        for idx, plot in enumerate(plots):
            label, data = plot

            ax = fig.add_subplot(1,
                                 len(plots),
                                 idx + 1,
                                 projection='3d',
                                 computed_zorder=False)
            ax.view_init(elev=45, azim=45, roll=0)

            x, y, z = zip(*data)
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)
            ax.scatter(x,
                       y,
                       z,
                       marker='.',
                       c=Plot.COLORS[idx],
                       zorder=-idx,
                       depthshade=False)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(label)
            ax.set_title(label)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(f'{title}.png',
                    format='png',
                    transparent=False,
                    dpi=Plot.DPI)
        plt.show()
