from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

global channels, alphabet, colors_RGB, illuminants_number, patches_number, choosed_patches_number, wavelengths
wavelengths = list(range(400, 721, 20))

#def plot_sens(sens, pattern='-', show=False):
    #for i,c in enumerate('rgb'):
     #   plt.plot(wavelengths, sens[:, i], pattern, c=c)
    #if show: plt.show()

def plot_sens(wavelengths:Tuple, sens:np.ndarray, sensitivities_gt:np.ndarray, pattern='-', show=False) -> None:
    """[summary]

    Args:
        wavelengths (Tuple): 
            [description]
        sens (np.ndarray): 
            [description]
        sensitivities_gt (np.ndarray): 
            [description]
        pattern (str, optional): 
            Defaults to '-'.
        show (bool, optional): 
            Defaults to False.
    """    
    sens /= sens.max()
    for i,c in enumerate('rgb'):
        plt.plot(wavelengths, sens[:, i], pattern, c=c)
        plt.plot(wavelengths, sensitivities_gt[:,i], '--', c=c)
    if show: plt.show()
     
    
def plot_spectra(spectras, show=True):
    for i in range(spectras.shape[-1]):
        plt.plot(wavelengths, spectras[:, i], '--')
    if show: plt.show()

def plot_chart(workbook, worksheet, title, x_axis, y_axis, categories_coord, values_coord, chart_coord, data_series, colors):
    """This function is auxiliary for plotting graphs

    """    
    chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
    for plot in data_series:
        chart.add_series({
            'name': str(plot),
            'line':   {'width': 1.25, 'color': colors[plot]},
            'categories': categories_coord,
            'values': values_coord[data_series.index(plot)],
        })

    chart.set_title({'name': title})
    chart.set_x_axis(x_axis)
    chart.set_y_axis(y_axis)

    chart.set_style(15)
    worksheet.insert_chart(chart_coord, chart, {'x_offset': 50, 'y_offset': 50, 'x_scale': 1.5, 'y_scale': 1.5})

def error_heatmap(nslices: int, tips: list) -> None:
    """
    This function builds heatmap to visualize accuracy usings learning and test samples
    Args:
        nslices (int): 
            number of bars in colorchecker
        tips (list): 
            list of sensitivities
    """    
    a = np.array(tips)
    tips1 = a.reshape((nslices, -1))
    value_max = max(tips1, key=lambda item: item[1])[1]
    value_min = min(tips1, key=lambda item: item[1])[1]
    sns.set_theme()
    sns.heatmap(tips1, annot = True, vmin=value_min, vmax=value_max, center= (value_min+value_max)//2, fmt='.3g', cmap= 'coolwarm')

def draw_colorchecker(stimuli, shape, show=False):
    rows = shape[0]
    columns = shape[1]
    carray = np.asarray([stimuli[i] for i in range(rows * columns)])
    carray = carray.reshape((rows, columns, 3))
    carray = carray / carray.max()
    plt.imshow(carray)
    if show: plt.show()
    return carray
