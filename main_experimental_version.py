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
from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata, normalize, simple_demosaic
import math
import random
import statistics
import seaborn as sns
from scipy import interpolate

global channels, alphabet, colors_RGB, illuminants_number, patches_number, choosed_patches_number, wavelengths


class SimpleRawProcessing:
    # Linearization not handled.
    def linearize_raw(self, raw_img, img_meta):
        return raw_img

    def normalize(self, linearized_raw, img_meta):
        norm = normalize(linearized_raw, img_meta['black_level'], img_meta['white_level'])
        return np.clip(norm , 0, 1)

    def demosaic(self, normalized, img_meta):
        return simple_demosaic(normalized, img_meta['cfa_pattern'])



class DNGProcessingDemo():
    def __init__(self):
        self.pipeline_demo = SimpleRawProcessing()
 
    def __call__(self, img_path: Path):
        raw_image = get_visible_raw_image(str(img_path))
        
        metadata = get_metadata(str(img_path))
        metadata['cfa_pattern'] = [1,2,0,1]
 
        pipeline_exec = PipelineExecutor(
                raw_image, metadata, self.pipeline_demo)
        
        return pipeline_exec()

    
def calc_variance(img, points):
    '''
    Args:
        img(np.array): img with int values
        points(list): list of regions coords
    '''
    points = np.array(points)
    y, x = draw.polygon(points[:,1], points[:,0], shape=img.shape)
    img1 = img[y, x]
    region_variance = []
    for channel in range(3):
        region_variance.append(np.std(img1[:,channel]))
    
    return region_variance


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

    return region_color


def process_markup(json_path, img):
    with open(json_path, 'r') as file:
        markup_json = json.load(file)
    color_per_region = {}
    variance_per_region = {}
    for object in markup_json['objects']:
        color_per_region[object['tags'][0]] = calc_mean_color(img, object['data'])
        variance_per_region[object['tags'][0]] = calc_variance(img, object['data'])
    return color_per_region,  variance_per_region


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


def R_internet_matrix(R_df):
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
    exposures_number = 6
    P = np.zeros(shape=(patches_number * illuminants_number, 3, exposures_number))
    variances = np.zeros(shape=(patches_number * illuminants_number, 3, exposures_number))

    process  = DNGProcessingDemo()
    illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC'][:illuminants_number]

    for illuminant in illumination_types:
        illuminant_index = illumination_types.index(illuminant)
        
        for exp in range(exposures_number):
            img_path = join(r"C:\Users\adm\Documents\IITP\dng_D50", "img_" + str(8332 + exp) + ".dng")
            json_path = join(r"C:\Users\adm\Documents\IITP\D50_targed", "img_" + str(8332 + exp) + ".jpg.json")
            img = process(img_path).astype(np.float32)
            
            color_per_region, variance_per_region = process_markup(json_path, img)


            # img_max = np.quantile(img, 0.99)
            # cc_keys = [str(i) for i in range(1, 25)]
            # carray = np.asarray([color_per_region[key] for key in cc_keys])
            # carray = carray.reshape((6, 4, 3))
            # plt.imshow(img / img_max)
            # plt.show()
            # plt.imshow(carray / img_max)
            # plt.show()
            
            P[patches_number * illuminant_index:patches_number * illuminant_index + patches_number, : , exp] = \
                [color_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
            variances[patches_number * illuminant_index:patches_number * illuminant_index + patches_number, : , exp] = \
                [variance_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    return P, variances


def choose_best_stimuls(P, variances):
    variances_procent = np.zeros(shape=(patches_number * illuminants_number, 3, 6))
    for channel in range(3):
        for stimul in range(len(P)): 
            for exposure in range(6):
                variances_procent[stimul, channel, exposure] = \
                    variances[stimul, channel, exposure] / P[stimul, channel, exposure] * 100
   
    P_best = np.full(shape=(patches_number * illuminants_number, 3), fill_value=1.)
    variances_best = np.full(shape=(patches_number * illuminants_number, 3), fill_value=101.)

    for channel in range(3):
        for stimul in range(len(P)): 
            for exposure in range(6):
                if P[stimul, channel, exposure] <= 0.99 and variances_procent[stimul, channel, exposure] < variances_best[stimul, channel]:
                     P_best[stimul, channel] = P[stimul, channel, exposure]
                     variances_best[stimul, channel] = variances_procent[stimul, channel, exposure]
    return P_best, variances_best


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


def plot_sens(sens, sensitivities_gt, pattern='-', show=False):
    sens /= sens.max()
    for i,c in enumerate('rgb'):
        plt.plot(wavelengths, sens[:, i], pattern, c=c)
        plt.plot(wavelengths, sensitivities_gt[:,i], '--', c=c)
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
        P_learning = C @ sensitivities_gt
        # noise = np.random.normal(0,.0001, P_learning.shape)
        # P_learning += noise
    else:
        P, variances = measure_stimuli()
        # P, variances = choose_best_stimuls(P, variances)
        P = P[:,:, 3]
        P_learning = np.array([P[patch] for patch in learning_sample])
    
    sensitivities = inv((C.T @ C).astype(float)) @ C.T @ P_learning

    plot_sens(sensitivities, sensitivities_gt, show=True)
    # plot_spectra(C.T, show=True)


def check_stimuls_accuracy(P, variances):
    # P /= P.max()
    # for channel in range(3): 
    #     mean_stimul = np.mean(P[:, channel])
    #     variance_stimuls = statistics.variance(P[:, channel])
    #     print(channel, mean_stimul, variance_stimuls)
    #     sns.histplot(P[:, channel], kde=True).get_figure()
    #     plt.show()

    # P /= P.max()

    # for channel in range(3):
    #     for stimul in range(len(P)): 
    #         for exposure in range(6):
    #             print(f'p: {stimul}, ch: {channel}, exp: {exposure}, std(%): \
    #                 {variances[stimul, channel, exposure] / P[stimul, channel, exposure] * 100}')
    #         print()

    for channel in range(3):
        for stimul in range(len(P)): 
            print(f'p: {stimul}, ch: {channel}, std(%): \
                    {variances[stimul, channel]}')
            

def regularization(C, P_learning, reg_start = {"red": 0.05, "green": 0.05, "blue": 0.05}, reg_stop = {"red": 5, "green": 5, "blue": 5}):
    def menger(p1, p2, p3):
        residual1, solution1 = p1
        residual2, solution2 = p2
        residual3, solution3 = p3
        p1p2 = (residual2 - residual1)**2 + (solution2 - solution1)**2
        p2p3 = (residual3 - residual2)**2 + (solution3 - solution2)**2
        p3p1 = (residual1 - residual3)**2 + (solution1 - solution3)**2
        numerator = residual1 * solution2 + residual2 * solution3 + residual3 * solution1 - \
                    residual1 * solution3 - residual3 * solution2 - residual2 * solution1
        return (2 * numerator) / (math.sqrt(p1p2 * p2p3 * p3p1))


    def l_curve_P(reg_parameter, channel):
        C_T = np.transpose(C)
        sensitivities[:,channel] = inv((C_T @ C).astype(float) + np.identity(len(wavelengths)) * reg_parameter) @ C_T @ P_learning[:,channel]
        solution = (np.linalg.norm(sensitivities[:,channel], 2) ** 2) 
        residual_vector = C @ sensitivities[:, channel] - P_learning[:,channel]
        residual = (np.linalg.norm(residual_vector, 2) ** 2) 
        return residual, solution


    def find_optimal_parameter():
        optimal_parameter = [0 for _ in range(3)]
        for channel in range(3):
            p = {}
            reg_parameter = {}
            ch_letter = channels[channel]
            reg_parameter[1], reg_parameter[4] = reg_start[ch_letter], reg_stop[ch_letter]   # search extremes
            epsilon = 0.00005                                                                # termination threshold
            phi = (1 + math.sqrt(5)) / 2                                                     # golden section
            
            reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
            reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))

            for i in range(1, 5):
                p[i] = l_curve_P(reg_parameter[i], channel)
            
            while ((reg_parameter[4] - reg_parameter[1]) / reg_parameter[4]) >= epsilon:
                C2 = menger(p[1], p[2], p[3])
                C3 = menger(p[2], p[3], p[4])
                
                while C3 <= 0:
                    reg_parameter[4] = reg_parameter[3]
                    reg_parameter[3] = reg_parameter[2]
                    reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                    p[4] = p[3]
                    p[3] = p[2]
                    p[2] = l_curve_P(reg_parameter[2], channel)
                    C3 = menger(p[2], p[3], p[4])
                
                if C2 > C3:
                    optimal_parameter[channel] = reg_parameter[2]
                    reg_parameter[4] = reg_parameter[3]
                    reg_parameter[3] = reg_parameter[2]
                    reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                    p[4] = p[3]
                    p[3] = p[2]
                    p[2] = l_curve_P(reg_parameter[2], channel)
                else:
                    optimal_parameter[channel] = reg_parameter[3]
                    reg_parameter[1] = reg_parameter[2]
                    reg_parameter[2] = reg_parameter[3]
                    reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))
                    p[1] = p[2]
                    p[2] = p[3]
                    p[3] = l_curve_P(reg_parameter[3], channel)
            
        return optimal_parameter


    optimal_parameter = find_optimal_parameter()
    print(optimal_parameter)
    
    reg_sensitivities = np.zeros(shape=(len(wavelengths), 3))
    for channel in range(3):                
        reg_sensitivities[:,channel] = inv((C.T @ C).astype(float) + np.identity(len(wavelengths)) *\
             optimal_parameter[channel]) @ C.T @ P_learning[:,channel]

    return reg_sensitivities


def easy_regularization(C, P, optimal_parameter):
    reg_sensitivities = np.zeros(shape=(len(wavelengths), 3))
    for channel in range(3):
        reg_sensitivities[:,channel] = inv((C.T @ C).astype(float) + np.identity(len(wavelengths)) \
            * optimal_parameter[channel]) @ C.T @ P[:,channel]
    return reg_sensitivities

##########################

wavelengths = get_lambda_grid(400, 721, 20)
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



#########################
E_df = pd.read_excel('LampSpectra.xls', sheet_name='LampsSpectra', skiprows=2)
R_df = pd.read_excel('CCC_Reflectance_1.xls', sheet_name=1, skiprows=4, header=0)
R_internet = R_internet_matrix(pd.read_excel('24_spectras.xlsx'))
sensitivities_df = pd.read_excel('canon600d.xlsx', sheet_name='Worksheet')
channels = list((sensitivities_df.drop(columns='wavelength')).columns)
sensitivities_gt = get_sensitivities_gt(sensitivities_df)

learning_sample, patches = choose_learning_sample(valid, achromatic_single, ratio=1.)
# print(len(learning_sample))

R = reflectances_matrix(R_df)
spectras_Alexander = spectras_matrix(E_df, R)
spectras_internet = spectras_matrix(E_df, R_internet)
P_measured, variances = measure_stimuli()
P_gt = spectras_Alexander @ sensitivities_gt

norm_val = np.max(P_measured[-1], axis=0)
P_measured /= norm_val
variances /= norm_val



# for i in range(6):
#     print(P_measured[:,:, i])
#     draw_colorchecker(P_measured[:,:, i], show=True)

# P_measured, variances = choose_best_stimuls(P_measured, variances)
# print(P_measured)
# draw_colorchecker(P_measured, show=True)

# check_stimuls_accuracy(P_measured, variances)

P_measured = P_measured[:,:, 3]
P_learning = np.array([P_measured[patch] for patch in learning_sample])
sensitivities = inv((spectras_Alexander.T @ spectras_Alexander).astype(float)) @ spectras_Alexander.T @ P_learning

reg_sensitivities = regularization(spectras_Alexander, P_learning)

plot_pictures(spectras_Alexander, learning_sample, sensitivities_gt, simulated=False)
plot_sens(reg_sensitivities, sensitivities_gt, show=True)


###############################

# write_to_excel('Sensitivities.xlsx', sensitivities, learning_sample)

# optimal_parameter = [0.5067055579111499, 0.6430519533349813, 0.4257159707254087]
# reg_sensitivities = easy_regularization(spectras_Alexander, P_measured, optimal_parameter)


# The number of points in lambda grid shouldn't exceed the length of the learning sample.
