from ntpath import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import inv
import string
import json
import cv2                      # pip install opencv-python
from skimage import draw
from pathlib import Path
from raw_prc_pipeline.pipeline import PipelineExecutor, RawProcessingPipelineDemo
from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata
import math
import random
import statistics
import seaborn as sns
from scipy import interpolate

global channels, alphabet, colors_RGB, illuminants_number, patches_number, choosed_patches_number, wavelengths

def do_nothing(img, meta): return img
 

class DNGProcessingDemo():
    def __init__(self, tone_mapping=False, denoising_flg=False):
        self.pipeline_demo = RawProcessingPipelineDemo(
            denoise_flg=denoising_flg, tone_mapping=tone_mapping)
        self.pipeline_demo.tone_mapping = do_nothing
        self.pipeline_demo.autocontrast = lambda img, meta: img/np.max(img)

 
    def __call__(self, img_path: Path):
        raw_image = get_visible_raw_image(str(img_path))
        
        metadata = get_metadata(str(img_path))
        metadata['cfa_pattern'] = [1,2,0,1]
 
        pipeline_exec = PipelineExecutor(
                raw_image, metadata, self.pipeline_demo, last_stage='demosaic')
                # raw_image, metadata, self.pipeline_demo, last_stage='white_balance')
        
        return pipeline_exec()

    
def calc_mean_color(img, points):
    '''
    Args:
        img(np.array): img with int values
        points(list): list of regions coords
    '''
    points = np.array(points)
    y, x = draw.polygon(points[:,1], points[:,0], shape=img.shape)

    # img_tmp = np.copy(img)
    # img_tmp /= np.quantile(img_tmp, 0.99)
    # img_tmp[y, x] = [1, 0, 0]
    # plt.imshow(img_tmp)
    # plt.show()
    region_color = np.mean(img[y, x], axis=0)

    img[y, x] = region_color
    return region_color


def process_markup(json_path, img):
    with open(json_path, 'r') as file:
        markup_json = json.load(file)
    color_per_region = {}
    for object in markup_json['objects']:
        color_per_region[object['tags'][0]] = calc_mean_color(img, object['data'])
    return color_per_region


def choose_learning_sample(valid, achromatic_single, ratio=0.8):
    chromatic_learning_sample = []
    achromatic = [i * patches_number + single for i in range(illuminants_number) for single in achromatic_single]
    all_chromatic_potential = [patch for patch in valid if patch not in achromatic]
    chromatic_learning_number = int(ratio * choosed_patches_number - len(achromatic_single))

    for i in range(illuminants_number):
        potential = [patch for patch in all_chromatic_potential if i * patches_number <= patch < i * patches_number + patches_number]
        chromatic_learning_sample += sorted(random.sample(potential, k=chromatic_learning_number))

    patches = {patch: 1 if patch in chromatic_learning_sample or patch in achromatic else 0 for patch in valid}
    learning_sample = [patch for patch, flag in patches.items() if flag == 1]
    return learning_sample, patches


def check_accuracy(patches_number, stimulus_predicted, stimulus_genuine):
    angles = []
    norms = []

    for i in range(patches_number):
        predicted_stimulus = stimulus_predicted[i]
        unit_predicted_stimulus = predicted_stimulus / np.linalg.norm(predicted_stimulus)
        genuine_stimulus = stimulus_genuine[i]
        print(predicted_stimulus, genuine_stimulus)
        unit_genuine_stimulus = genuine_stimulus / np.linalg.norm(genuine_stimulus)
        dot_product = np.dot(unit_predicted_stimulus, unit_genuine_stimulus)
        angles.append(np.arccos(dot_product) * 180 / 3.1415)
        norms.append(np.linalg.norm(predicted_stimulus - genuine_stimulus, 2))


    mean_angle = sum(angles) / patches_number
    variance_angles = statistics.variance(angles)
    angles_fig = sns.histplot(angles).get_figure()

    mean_norm = np.mean(norms)
    variance_norms = statistics.variance(norms)
    norms_fig = sns.histplot(norms).get_figure()

    return mean_angle, variance_angles, angles_fig, mean_norm, variance_norms, norms_fig


def draw_chart(workbook, worksheet, title, x_axis, y_axis, categories_coord, values_coord, chart_coord, data_series, colors):
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


def write_to_excel(file_path, sensitivities, learning_sample):
    R_learning = []
    for illuminant_index in range(illuminants_number):
            R_learning += [R[patch % patches_number] for patch in learning_sample 
                        if illuminant_index * patches_number <= patch < illuminant_index * patches_number + patches_number]
    R_learning = np.transpose(np.array(R_learning))

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    workbook = writer.book
    pd.DataFrame(sensitivities).to_excel(writer, sheet_name='Sheet1',
                            index=False, header=channels, startrow=1, startcol=1)
    pd.DataFrame(R_learning).to_excel(writer, sheet_name='Sheet2',
                            index=False, header=learning_sample, startrow=1, startcol=1)
    worksheet_1 = writer.sheets['Sheet1']
    worksheet_2 = writer.sheets['Sheet2']
    bold = workbook.add_format({'bold': 1})

    for row_num, data in enumerate(wavelengths):
        worksheet_1.write(row_num + 2, 0, data)
        worksheet_2.write(row_num + 2, 0, data)
    worksheet_1.write('A2', 'wavelegths', bold)
    worksheet_1.write('C1', 'Sensitivities', bold)
    worksheet_2.write('A2', 'wavelegths', bold)
    worksheet_2.write('B1', "Patches' reflectances", bold)

    sensitivities_x_axis = {'name': 'Wavelengths, nm', 'min': wavelengths[0], 'max': wavelengths[-1]}
    sensitivities_y_axis = {'name': 'Sensitivities Function'}
    sensitivities_values_coord = []  
    for channel in range(3):
        value_letter = alphabet[alphabet.index('B') + channel]
        sensitivities_values_coord.append('=Sheet1!$' + value_letter + '$3:$' + value_letter + '$109') 
    draw_chart(workbook, worksheet_1, 'Sensitivities', sensitivities_x_axis, sensitivities_y_axis, \
        '=Sheet1!$A$3:$A$109', sensitivities_values_coord, 'F1', channels, colors_RGB)

    patches_x_axis = {'name': 'Wavelengths, nm', 'min': wavelengths[0], 'max': wavelengths[-1]}
    patches_y_axis = {'name': 'Reflectance spectra'}
    patches_values_coord = []  
    for patch in range(len(R_learning[0])):
        value_letter = alphabet[alphabet.index('B') + patch]
        patches_values_coord.append('=Sheet2!$' + value_letter + '$3:$' + value_letter + '$109')
    # cmap = plt.cm.get_cmap('viridis')
    # colors_patches = [cmap(i) for i in range(cmap.N)] 
    colors_patches = {i:'blue' for i in learning_sample}
    
    draw_chart(workbook, worksheet_1, "Patches' reflectance", patches_x_axis, patches_y_axis, \
        '=Sheet2!$A$3:$A$109', patches_values_coord, 'F26', learning_sample, colors_patches)
    
    workbook.close()


def spectras_matrix(E_df, R):
    C = np.zeros(shape=(len(learning_sample), len(wavelengths)))
    C_current_index = 0

    x = E_df['Lambda grid']
    for illuminant_index in range(illuminants_number):
        y = E_df[str(1 + illuminant_index) + 'Norm']
        E_interpolated=interpolate.interp1d(x, y)
        E = np.diag(E_interpolated(wavelengths))
        R_learning = [R[patch % 24] for patch in learning_sample if illuminant_index * 24 <= patch < illuminant_index * 24 + 24]
        C[C_current_index:C_current_index + len(R_learning)] = np.transpose(E @ np.transpose(R_learning))
        C_current_index += len(R_learning)
    
    return C


def reflectances_matrix(R_df):
    R = np.zeros(shape=(patches_number, len(wavelengths)))
    x = R_df['Lambda grid']
    for patch in range(patches_number):
        y = R_df[str(patch + 1) + 'Avg']
        R_interpolated = interpolate.interp1d(x, y)
        R[patch] = R_interpolated(wavelengths)
    R /= R.max(axis=0)
    return R


def R_babelcolor_matrix(R_df):
    R = np.zeros(shape=(patches_number, len(wavelengths)))
    x = R_df['wavelengths']
    for patch in range(patches_number):
        y = R_df[patch]
        R_interpolated = interpolate.interp1d(x, y)
        R[patch] = R_interpolated(wavelengths)
    R /= R.max(axis=0)
    return R


def get_lambda_grid(start, stop, points_number):
    step = (stop - start) / points_number
    return [start + point * step for point in range(points_number)]


def measure_stimuli():
    P = np.zeros(shape=(patches_number * illuminants_number, 3))
    process  = DNGProcessingDemo()
    illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC'][:illuminants_number]

    for illuminant in illumination_types:
        illuminant_index = illumination_types.index(illuminant)  
        img_path = join(r"C:\Users\adm\Documents\IITP\dng", str(illuminant_index + 1) + '_' + illuminant + ".dng")
        json_path = join(r'C:\Users\adm\Documents\IITP\png_targed', str(illuminant_index + 1) + '_' + illuminant +'.jpg.json')

        img = process(img_path).astype(np.float32)
        img_max = np.quantile(img, 0.99)
        
        color_per_region = process_markup(json_path, img)
        cc_keys = [str(i) for i in range(1, 25)]
        return np.asarray([color_per_region[key] for key in cc_keys])
        carray = carray.reshape((6, 4, 3))
        
        # plt.imshow(img / img_max)
        # plt.show()
        # return carray / img_max
        plt.imshow(carray / img_max)
        plt.show()

        P[patches_number * illuminant_index:patches_number * illuminant_index + patches_number] = \
                        [color_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    return P


def draw_colorchecker(stimuli, show=False):
    carray = np.asarray([stimuli[i] for i in range(patches_number)])
    carray = carray.reshape((6, 4, 3))
    # carray = carray / carray.max()
    plt.imshow(carray)
    if show: plt.show()
    return carray


def plot_spectra(spectras, show=False):
    for i in range(spectras.shape[-1]):
        plt.plot(wavelengths, spectras[:, i], '--')
    if show: plt.show()


def plot_sens(sens, pattern='-', show=False):
    for i,c in enumerate('rgb'):
        plt.plot(wavelengths, sens[:, i], pattern, c=c)
    if show: plt.show()


def get_sensitivities_gt(sensitivities_df):
    sensitivities_given = np.zeros(shape=(len(wavelengths), 3))
    x = sensitivities_df['wavelength']
    for i, channel in enumerate(['red', 'green', 'blue']):
        y = sensitivities_df[channel]
        sensitivities_interpolated = interpolate.interp1d(x, y)
        sensitivities_given[:, i] = sensitivities_interpolated(wavelengths)
    return sensitivities_given


def draw_compared_colorcheckers(C, sensitivities, E_df, R, R_babelcolor):
    draw_colorchecker(C @ sensitivities)
    a_carray = draw_colorchecker(spectras_matrix(E_df, R) @ sensitivities_given)
    b_carray = draw_colorchecker(spectras_matrix(E_df, R_babelcolor) @ sensitivities_given)
    carray = draw_colorchecker(P)
    tmp  = np.hstack([carray, np.zeros((6, 1, 3)), b_carray, np.zeros((6, 1, 3)), a_carray])
    plt.imshow(tmp / tmp.max())
    plt.show()  


def plot_pictures(C, learning_sample, sensitivities_gt, simulated=False):
    if simulated:
        P = C @ sensitivities_gt
    else:
        P = measure_stimuli()
    P_learning = np.array([P[patch] for patch in learning_sample])
    sensitivities = inv((C.T @ C).astype(float)) @ C.T @ P_learning

    plot_sens(sensitivities, show=True)
    plot_spectra(C.T, show=True)
    
##########################

wavelengths = get_lambda_grid(400, 721, 25)
illuminants_number = 1
patches_number = 24                                      # in colorchecker
choosed_patches_number = patches_number                  # how many patches to use 
alphabet_st = list(string.ascii_uppercase)
alphabet = list(string.ascii_uppercase)
for letter1 in alphabet_st:
    alphabet += [letter1 + letter2 for letter2 in alphabet_st]

colors_RGB = {'blue': '#0066CC', 'green': '#339966', 'red': '#993300'}
exceptions = set([])           # patches bringing in large error
achromatic_single = []
valid = set(range(patches_number * illuminants_number)) - exceptions
#########################
E_df = pd.read_excel('LampSpectra.xls', sheet_name='LampsSpectra', skiprows=2)
R_df = pd.read_excel('CCC_Reflectance_1.xls', sheet_name=1, skiprows=4, header=0)
R_internet = R_babelcolor_matrix(pd.read_excel('24_spectras.xlsx'))
sensitivities_df = pd.read_excel('canon600d.xlsx', sheet_name='Worksheet')
channels = list((sensitivities_df.drop(columns='wavelength')).columns)
sensitivities_gt = get_sensitivities_gt(sensitivities_df)

learning_sample, patches = choose_learning_sample(valid, achromatic_single, ratio=1.)

R = reflectances_matrix(R_df)
spectras_Alexander = spectras_matrix(E_df, R)
spectras_internet = spectras_matrix(E_df, R_internet)
P_measured = measure_stimuli()
P_gt = spectras_Alexander @ sensitivities_gt

plot_pictures(spectras_Alexander, learning_sample, sensitivities_gt, simulated=True)

P_learning = np.array([P_measured[patch] for patch in learning_sample])
sensitivities = inv((spectras_Alexander.T @ spectras_Alexander).astype(float)) @ spectras_Alexander.T @ P_learning
write_to_excel('Sensitivities.xlsx', sensitivities, learning_sample)
