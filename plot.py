import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from data import get_sensitivities_gt
from main_experimental_version import get_lambda_grid

global channels, alphabet, colors_RGB, illuminants_number, patches_number, choosed_patches_number, wavelengths

def plot_sens(wavelengths_points_numbers, sens, pattern='-', show=False):
    sensitivities_df = pd.read_excel('canon600d.xlsx')
    sensitivities_gt = get_sensitivities_gt(sensitivities_df['wavelength'], sensitivities_df)
    sens /= sens.max()

    for i,c in enumerate('rgb'):
        wavelengths = get_lambda_grid(400, 721, wavelengths_points_numbers[i])
        plt.plot(wavelengths, sens[:,i], pattern, c=c)
        plt.plot(sensitivities_df['wavelength'], sensitivities_gt[:,i], '--', c=c)
    if show: plt.show()
    
def plot_spectra(wavelengths, spectras, show=False):
    for i in range(spectras.shape[-1]):
        plt.plot(wavelengths, spectras[:, i], '--')
    if show: plt.show()

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


def draw_colorchecker(stimuli, patches_number, show=False):
    carray = np.asarray([stimuli[i] for i in range(patches_number)])
    carray = carray.reshape((6, 4, 3))
    carray = carray / carray.max()
    plt.imshow(carray)
    if show: plt.show()
    return carray


def draw_compared_colorcheckers(C, P, sensitivities, sensitivities_given, E_df, R, R_babelcolor):
    draw_colorchecker(C @ sensitivities)
    a_carray = draw_colorchecker(spectras_matrix(E_df, R) @ sensitivities_given)
    b_carray = draw_colorchecker(spectras_matrix(E_df, R_babelcolor) @ sensitivities_given)
    carray = draw_colorchecker(P)
    tmp  = np.hstack([carray, np.zeros((6, 1, 3)), b_carray, np.zeros((6, 1, 3)), a_carray])
    plt.imshow(tmp / tmp.max())
    plt.show()
