import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, font_manager

COLORS: list[tuple[float, float, float, float]] = [plt.get_cmap('tab10')(i) for i in range(10)]

DPI: int = 1000
SCALE: float = 2.5

SEISMIC: plt.Colormap = colors.LinearSegmentedColormap.from_list('seismic',
                                                                 plt.get_cmap('seismic')(np.linspace(0, 1., 100)))
SEISMIC_NEGATIVE: plt.Colormap = colors.LinearSegmentedColormap.from_list(
    'seismic_neg',
    plt.get_cmap('seismic')(np.linspace(0., .5, 50)))
SEISMIC_POSITIVE: plt.Colormap = colors.LinearSegmentedColormap.from_list(
    'seismic_pos',
    plt.get_cmap('seismic')(np.linspace(.5, 1., 50)))

PHI = (1. + np.sqrt(5.)) / 2.

plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = .5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = .5

if __name__ == '__main__':
    print(sorted(font_manager.get_font_names()))
