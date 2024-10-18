import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

COLORS: list[str] = ['black'] + list(colors.TABLEAU_COLORS.keys())[1:]
DEFAULT_COLOR: str = COLORS[1]

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

# print(sorted(mpl.font_manager.get_font_names()))
plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = .5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = .5
