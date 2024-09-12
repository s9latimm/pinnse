import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class Plot:
    COLORS = list(mcolors.TABLEAU_COLORS.keys())

    DPI = 96

    @staticmethod
    def heatmap(ax, x, y, z):
        ax.pcolormesh(x,
                      y,
                      z,
                      cmap='Oranges',
                      vmin=z.min(),
                      vmax=z.max(),
                      zorder=1)
        ax.scatter(x, y, marker='+', c='black', zorder=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(y.min() - 1, y.max() + 1)

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
    def scatter_3d(title, *args):

        fig = plt.figure(figsize=(4096 / Plot.DPI * len(args), 4096 / Plot.DPI),
                         dpi=Plot.DPI)

        for idx, plot in enumerate(args):
            label, data = plot

            ax = fig.add_subplot(1,
                                 len(args),
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
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(label)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(f'../images/{title}.png',
                    format='png',
                    transparent=False,
                    dpi=Plot.DPI)
        plt.show()


if __name__ == "__main__":
    Y, X = np.mgrid[-10:10:11j, -10:10:11j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2

    # plot_border(X, Y, U)
    #
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=400)
    #
    # ax[0].streamplot(X, Y, U, V, broken_streamlines=False, zorder=1)
    # ax[0].scatter(x=X, y=Y, marker='+', c='black', zorder=2)
    # ax[0].set_xlim(X.min() - 1, X.max() + 1)
    # ax[0].set_ylim(Y.min() - 1, Y.max() + 1)
    # ax[0].axis('off')
    # ax[0].set_title('Streamlines')
    # ax[0].set(aspect='equal')
    #
    # heatmap(ax[1], X, Y, U)
    # ax[1].axis('off')
    # ax[1].set_title('u')
    # ax[1].set(aspect='equal')
    #
    # heatmap(ax[2], X, Y, V)
    # ax[2].axis('off')
    # ax[2].set_title('v')
    # ax[2].set(aspect='equal')
    #
    # print(np.array_str(U, precision=1))
    # print(np.array_str(X, precision=1))
    # print(np.array_str(Y, precision=1))
    #
    # # fig.tight_layout()
    # fig.savefig('streamlines.pdf', format='pdf', transparent=True)
    # plt.show()
