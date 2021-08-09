import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from data import get_sensitivities_gt
from data import get_lambda_grid


def plot_sens(wavelengths_points_numbers: dict, sens: dict, pattern='-', show=False):
    sensitivities_df = pd.read_excel('canon600d.xlsx')
    sensitivities_gt = get_sensitivities_gt(sensitivities_df['wavelength'], sensitivities_df)
    sens_max = max([el for arr in sens.values() for el in arr])

    for i,c in enumerate('rgb'):
        sensitivity = sens[i]
        sensitivity /= sens_max
        wavelengths = get_lambda_grid(400, 721, wavelengths_points_numbers[i])
        plt.plot(wavelengths, sensitivity, pattern, c=c)
        plt.plot(sensitivities_df['wavelength'], sensitivities_gt[:,i], '--', c=c)
    if show: plt.show()


def plot_spectra(wavelengths: list, spectras: dict, show=False):
    for i in range(24):
        plt.plot(wavelengths, spectras[i], '--')
    if show: plt.show()


def draw_colorchecker(stimuli: dict, show=False):
    carray = np.asarray([stimuli[i] for i in range(24)])
    carray = carray.reshape((6, 4, 3))
    carray = carray / carray.max()
    plt.imshow(carray)
    if show: plt.show()
    return carray


def visualization(nslices: int, tips: list):
    """
    This function builds heatmap to visualize accuracy usings learning and test samples
    Args:
        nslices (int): number of bars in colorchecker
        tips (list): list of sensitivities
    """    
    a = np.array(tips)
    tips1 = a.reshape((nslices, -1))
    value_max = max(tips1, key=lambda item: item[1])[1]
    value_min = min(tips1, key=lambda item: item[1])[1]
    sns.set_theme()
    sns.heatmap(tips1, annot = True, vmin=value_min, vmax=value_max, center= (value_min+value_max)//2, fmt='.3g', cmap= 'coolwarm')

