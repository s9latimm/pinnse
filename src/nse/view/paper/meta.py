from matplotlib import pyplot as plt

from base.model.mesh import arrange
from nse.model.loss import Losses
from src import RESULT_DIR, OUTPUT_DIR
from src.base.model.algebra import Real, Integer
from src.base.view import SCALE, PHI, COLORS
from src.base.view.plot import save_fig
from src.nse.model.record import Record


def __plot_parameters_mean(data: list[list[tuple[Integer, Record]]]):
    fig1 = plt.Figure(figsize=(PHI * SCALE, SCALE))
    ax1 = fig1.add_subplot(1, 1, 1)

    xy = []

    for i, series in enumerate(data):
        xy += [(int(j), r.u) for j, r in series]

    xy = sorted(xy, key=lambda x: x[0])

    ax1.plot([i for i, _ in xy], [i for _, i in xy], color=COLORS[i], label=f'{i + 2}')

    # ax1.set_xticks(x)
    # ax1.set_xlim([25, 155])
    ax1.set_xlabel('Network Parameters')

    ax1.tick_params(axis='y', labelrotation=90)
    ax1.set_ylabel(r'$\Delta u$')
    # ax1.set_yticks([0, 1e4, 2e4, 3e4])
    # ax1.set_yticklabels(['0', '1e4', '2e4', '3e4'])
    # ax1.set_ylim([0, 3.2e4])
    for label in ax1.get_yticklabels():
        label.set_verticalalignment('center')

    # ax1.plot(155, 0, ">k", clip_on=False)
    # ax1.plot(25, 3.2e4, "^k", clip_on=False)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    import matplotlib.font_manager as font_manager
    ax1.legend(frameon=False, prop=font_manager.FontProperties(family='cmmi10'))

    save_fig(fig1, OUTPUT_DIR / 'paper' / 'meta_parameters.pdf')

    plt.clf()
    plt.close()


def __plot_time_loss(data: list[list[tuple[Real, int, Record, Record, Record, Integer]]]):
    fig1, ax1 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig2, ax2 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig3, ax3 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig4, ax4 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig5, ax5 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig6, ax6 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig7, ax7 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig8, ax8 = plt.subplots(figsize=(PHI * SCALE, SCALE))
    fig9, ax9 = plt.subplots(figsize=(PHI * SCALE, SCALE))

    x = arrange(25, 150, 25)

    for i, series in enumerate(data):
        y = [float(i / 60) for i, *_ in series]
        z = [i for _, i, *_ in series]
        u = [i.u for _, _, i, _, _, _ in series]
        v = [i.v for _, _, i, _, _, _ in series]
        bu = [i.u for _, _, _, i, _, _ in series]
        bv = [i.v for _, _, _, i, _, _ in series]
        r = [i.u for _, _, _, _, i, _ in series]
        p = [int(i) for _, _, _, _, _, i in series]

        ax1.plot(x, z, color=COLORS[i], label=f'{i + 2}')
        ax2.plot(x, y, color=COLORS[i], label=f'{i + 2}')
        ax3.plot(x, u, color=COLORS[i], label=f'{i + 2}')
        ax4.plot(x, v, color=COLORS[i], label=f'{i + 2}')
        ax5.plot(x, bu, color=COLORS[i], label=f'{i + 2}')
        ax6.plot(x, bv, color=COLORS[i], label=f'{i + 2}')
        ax7.plot(x, r, color=COLORS[i], label=f'{i + 2}')
        ax8.scatter(y, u, color=COLORS[i], label=f'{i + 2}', zorder=i, s=[j // 250 for j in p])
        ax9.scatter(y, v, color=COLORS[i], label=f'{i + 2}', zorder=i, s=[j // 250 for j in p])

        ax7.axhline(6.670006642199995)

        for j, _ in enumerate(u):
            ax8.annotate(f'{(j + 1) * 25}', (y[j], u[j]), fontsize=5, ha='center', va='center', zorder=i)
            ax9.annotate(f'{(j + 1) * 25}', (y[j], v[j]), fontsize=5, ha='center', va='center', zorder=i)

    ax1.set_ylabel('Training Steps')
    ax1.set_ylim([0, 3.3e4])
    ax1.plot(25, 3.3e4, "^k", clip_on=False)
    ax1.plot(162.5, 0, ">k", clip_on=False)
    ax1.set_yticks([0, 1e4, 2e4, 3e4])
    ax1.set_yticklabels(['0', '1e4', '2e4', '3e4'])

    ax2.set_ylabel('Training Time [min]')
    ax2.set_ylim([0, 35])
    ax2.plot(25, 35, "^k", clip_on=False)
    ax2.plot(162.5, 0, ">k", clip_on=False)
    ax2.set_yticks([0, 15, 30])

    ax3.set_ylabel(r'$\Delta u$')
    ax3.set_ylim([0, 4.4e-2])
    ax3.plot(25, 4.4e-2, "^k", clip_on=False)
    ax3.plot(162.5, 0, ">k", clip_on=False)
    ax3.set_yticks([0, 2e-2, 4e-2])
    ax3.set_yticklabels(['0', '2e-2', '4e-2'])

    ax4.set_ylabel(r'$\Delta v$')
    ax4.set_ylim([0, 1.1e-2])
    ax4.plot(25, 1.1e-2, "^k", clip_on=False)
    ax4.plot(162.5, 0, ">k", clip_on=False)
    ax4.set_yticks([0, 5e-3, 1e-2])
    ax4.set_yticklabels(['0', '5e-3', '1e-2'])

    ax5.set_ylabel(r'$\Delta \hat{u}$')
    ax5.set_ylim([0, 8.8e-4])
    ax5.plot(25, 8.8e-4, "^k", clip_on=False)
    ax5.plot(162.5, 0, ">k", clip_on=False)
    ax5.set_yticks([0, 4e-4, 8e-4])
    ax5.set_yticklabels(['0', '4e-4', '8e-4'])

    ax6.set_ylabel(r'$\Delta \hat{v}$')
    ax6.set_ylim([0, 8.8e-4])
    ax6.plot(25, 8.8e-4, "^k", clip_on=False)
    ax6.plot(162.5, 0, ">k", clip_on=False)
    ax6.set_yticks([0, 4e-4, 8e-4])
    ax6.set_yticklabels(['0', '4e-4', '8e-4'])

    ax7.set_ylabel(r'$u_{\text{rev}}$')
    ax7.set_ylim([7, 11])
    ax7.plot(25, 11, "^k", clip_on=False)
    ax7.plot(162.5, 7, ">k", clip_on=False)
    ax7.set_yticks([7, 8, 9, 10])
    ax7.set_yticklabels(['0', '8', '9', '10'])
    ax7.plot(
        [25, 25],
        [7.4, 7.5],
        marker=[(-1, -.5), (1, .5)],
        markersize=8,
        linestyle="none",
        color='k',
        mec='k',
        mew=1,
        clip_on=False,
    )

    ax8.set_xlabel('Training Time [min]')
    ax8.set_xlim([0, 35])
    ax8.set_xticks([0, 15, 30])
    ax8.set_ylabel(r'$\Delta u$')
    ax8.set_ylim([1.5e-2, 4.4e-2])
    ax8.set_yticks([1.5e-2, 2e-2, 3e-2, 4e-2])
    ax8.set_yticklabels(['0', '2e-2', '3e-2', '4e-2'])
    ax8.plot(
        [0, 0],
        [1.71e-2, 1.75e-2],
        marker=[(-1, -.5), (1, .5)],
        markersize=8,
        linestyle="none",
        color='k',
        mec='k',
        mew=1,
        clip_on=False,
    )
    ax8.tick_params(axis='y', labelrotation=90)
    for t in ax8.get_yticklabels():
        t.set_verticalalignment('center')
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.plot(0, 4.4e-2, "^k", clip_on=False)
    ax8.plot(35, 1.5e-2, ">k", clip_on=False)

    ax9.set_xlabel('Training Time [min]')
    ax9.set_xlim([0, 35])
    ax9.set_xticks([0, 15, 30])
    ax9.set_ylabel(r'$\Delta v$')
    ax9.set_ylim([3e-3, 1.1e-2])
    ax9.set_yticks([3e-3, 4e-3, 7e-3, 1e-2])
    ax9.set_yticklabels(['0', '4e-3', '7e-3', '1e-2'])
    ax9.plot(
        [0, 0],
        [3.38e-3, 3.5e-3],
        marker=[(-1, -.5), (1, .5)],
        markersize=8,
        linestyle="none",
        color='k',
        mec='k',
        mew=1,
        clip_on=False,
    )
    ax9.tick_params(axis='y', labelrotation=90)
    for t in ax9.get_yticklabels():
        t.set_verticalalignment('center')
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.plot(0, 1.1e-2, "^k", clip_on=False)
    ax9.plot(35, 3e-3, ">k", clip_on=False)
    ax9.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fancybox=False).get_frame().set_edgecolor('k')

    def __default(ax: plt.Axes, legend: bool):
        ax.set_xticks(x)
        ax.set_xlim([25, 163])
        ax.set_xlabel('Layer Size')
        ax.tick_params(axis='y', labelrotation=90)
        for tick in ax.get_yticklabels():
            tick.set_verticalalignment('center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True,
                      fancybox=False).get_frame().set_edgecolor('k')

    __default(ax1, False)
    __default(ax2, True)
    __default(ax3, False)
    __default(ax4, True)
    __default(ax5, False)
    __default(ax6, True)
    __default(ax7, True)

    save_fig(fig1, OUTPUT_DIR / 'paper' / 'meta_steps.pdf')
    save_fig(fig2, OUTPUT_DIR / 'paper' / 'meta_time.pdf')
    save_fig(fig3, OUTPUT_DIR / 'paper' / 'meta_mesh_error_u.pdf')
    save_fig(fig4, OUTPUT_DIR / 'paper' / 'meta_mesh_error_v.pdf')
    save_fig(fig5, OUTPUT_DIR / 'paper' / 'meta_boundary_error_u.pdf')
    save_fig(fig6, OUTPUT_DIR / 'paper' / 'meta_boundary_error_v.pdf')
    # save_fig(fig7, OUTPUT_DIR / 'paper' / 'meta_reverse.pdf')
    save_fig(fig8, OUTPUT_DIR / 'paper' / 'meta_error_time_u.pdf')
    save_fig(fig9, OUTPUT_DIR / 'paper' / 'meta_error_time_v.pdf')

    plt.clf()
    plt.close()

    # cbar.set_label('$u$', rotation=0)


if __name__ == '__main__':

    def main():
        data = [
            [
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_025-02_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_050-02_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_075-02_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_100-02_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_125-02_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_150-02_030000',
            ],
            [
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_025-03_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_050-03_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_075-03_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_100-03_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_125-03_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_150-03_030000',
            ],
            [
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_025-04_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_050-04_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_075-04_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_100-04_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_125-04_030000',
                RESULT_DIR / 'step-0_100-0_010-1_000_cuda_150-04_030000',
            ],
        ]

        time_loss = []
        parameters_mean = []
        for series in data:
            time_loss.append([])
            parameters_mean.append([])
            for path in series:
                mesh_mean = Record.load(path / 'mesh_mean.csv')
                boundary_mean = Record.load(path / 'boundary_mean.csv')
                reverse = Record.load(path / 'reverse.csv')
                parameter = Integer.load(path / 'parameter.csv')
                time_loss[-1].append((
                    Real.load(path / 'time.csv'),
                    len(list(Losses.load(path / 'loss.csv'))),
                    mesh_mean,
                    boundary_mean,
                    reverse,
                    parameter,
                ))
                parameters_mean[-1].append((parameter, mesh_mean))

        __plot_time_loss(time_loss)
        __plot_parameters_mean(parameters_mean)

    '''
    
    '''

    main()
