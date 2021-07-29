from ntpath import join
import pandas as pd
import numpy as np
from numpy.linalg import inv
import string
import json
import cv2                      # pip install opencv-python
from skimage import draw
from pathlib import Path
from raw_prc_pipeline.pipeline import PipelineExecutor, RawProcessingPipelineDemo
from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata, ratios2floats, autocontrast
import math
import random
import statistics
import seaborn as sns

global channels, illumination_types, alphabet

def do_nothing(img, meta): return img
 

class DNGProcessingDemo():
    def __init__(self, tone_mapping=1, denoising_flg=1):
        self.pipeline_demo = RawProcessingPipelineDemo(
            denoise_flg=denoising_flg, tone_mapping=tone_mapping)
        self.pipeline_demo.tone_mapping = do_nothing
        self.pipeline_demo.autocontrast = lambda img, meta: img/np.max(img)

 
    def __call__(self, img_path: Path):
        raw_image = get_visible_raw_image(str(img_path))
        
        metadata = get_metadata(str(img_path))
 
        pipeline_exec = PipelineExecutor(
                raw_image, metadata, self.pipeline_demo, last_stage='demosaic')
        
        return pipeline_exec()

    
def calc_mean_color(img, points):
    '''
    Args:
        img(np.array): img with int values
        points(list): list of regions coords
    '''
    points = np.array(points)
    mask = np.zeros([img.shape[0], img.shape[1]], dtype=int)
    y, x = draw.polygon(points[:,1], points[:,0], shape=img.shape)    # Generate coordinates of pixels within polygon
    mask[y, x] = 1
    pixels = img[mask == 1]
    region_color = pixels.mean(axis=0).astype(img.dtype)

    return region_color.tolist()


def process_markup(json_path, img):
    with open(json_path, 'r') as file:
        markup_json = json.load(file)

    color_per_region = {}
    for object in markup_json['objects']:
        color_per_region[object['tags'][0]] = calc_mean_color(img, object['data'])
    return color_per_region


def choose_learning_sample(valid):
    chromatic_learning_sample = []
    achromatic = [patch for patch in valid if patch % 4 == 3]
    all_chromatic_potential = [patch for patch in valid if patch not in achromatic]

    for i in range(6):
        potential = [patch for patch in all_chromatic_potential if i * 24 <= patch < i * 24 + 24]
        chromatic_learning_sample += sorted(random.sample(potential, k=14))

    patches = {patch: 1 if patch in chromatic_learning_sample or patch in achromatic else 0 for patch in valid}
    learning_sample = [patch for patch, flag in patches.items() if flag == 1]
    return learning_sample, patches


def check_accuracy(C, sensitivities, sample):
    P_predicted = C @ sensitivities
    P_genuine = np.array([P[patch] for patch in sample])
    angles = []
    norms = []

    for i in range(len(sample)):
        predicted_stimulus = P_predicted[i]
        unit_predicted_stimulus = predicted_stimulus / np.linalg.norm(predicted_stimulus)
        genuine_stimulus = P_genuine[i]
        unit_genuine_stimulus = genuine_stimulus / np.linalg.norm(genuine_stimulus)
        dot_product = np.dot(unit_predicted_stimulus, unit_genuine_stimulus)
        angles.append(np.arccos(dot_product) * 180 / 3.1415)
        norms.append(np.linalg.norm(predicted_stimulus - genuine_stimulus, 2))

    mean_angle = sum(angles) / len(sample)
    variance_angles = statistics.variance(angles)
    angles_fig = sns.histplot(angles).get_figure()

    mean_norm = np.mean(norms)
    variance_norms = statistics.variance(norms)
    norms_fig = sns.histplot(norms).get_figure()

    return mean_angle, variance_angles, angles_fig, mean_norm, variance_norms, norms_fig


def regularization(reg_start, reg_stop, reg_step, sensitivities, C, P_learning):
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
        sensitivities[:,channel] = inv((C_T @ C).astype(float) + np.identity(107) * reg_parameter) @ C_T @ P_learning[:,channel]
    #     solution = math.log10((np.linalg.norm(sensitivities[:,channel], 2)))
        solution = ((np.linalg.norm(sensitivities[:,channel], 2))) ** 2
        residual_vector = C @ sensitivities[:,channel] - P_learning[:,channel]
    #     residual = math.log10((np.linalg.norm(residual_vector, 2)))
        residual = ((np.linalg.norm(residual_vector, 2))) ** 2
        return residual, solution


    def find_optimal_parameter():
        optimal_parameter = [0 for _ in range(3)]
        for channel in range(3):
            p = {}
            reg_parameter = {}
            ch_letter = channels[channel]
            reg_parameter[1], reg_parameter[4] = reg_start[ch_letter], reg_stop[ch_letter]   # search extremes
            epsilon = 0.00005                                                                  # termination threshold
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
    reg_sensitivities = np.zeros(shape=(107, 3))
    solution_norms = [{} for _ in range(3)]
    residual_norms = [{} for _ in range(3)]

    for channel in range(3):
        ch_letter = channels[channel]
        
        for reg_parameter in np.arange(reg_start[ch_letter], reg_stop[ch_letter], reg_step[ch_letter]):
            residual, solution = l_curve_P(reg_parameter, channel)
            solution_norms[channel][solution] = reg_parameter
            residual_norms[channel][residual] = reg_parameter
                
        reg_sensitivities[:,channel] = inv((C_T @ C).astype(float) + np.identity(107) * optimal_parameter[channel]) @ C_T @ P_learning[:,channel]

    #### plot reg_sensitivities and l-curves ####
    writer = pd.ExcelWriter('Sensitivities_2.xlsx', engine='xlsxwriter')
    workbook = writer.book

    reg_sensitivities_df = pd.DataFrame(sensitivities)
    reg_sensitivities_df.to_excel(writer, sheet_name='Sheet1', index=False, header=channels, startrow=1, startcol=1)

    worksheet = writer.sheets['Sheet1']

    bold = workbook.add_format({'bold': 1})
    worksheet.write('C1', 'Sensitivities', bold)
    worksheet.write('A2', 'Lambda grid', bold)
    for row_num, data in enumerate(E_df['Lambda grid']):
        worksheet.write(row_num + 2, 0, data)

    chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
    start_letter_index = 1
    for channel in channels:
        channel_index = channels.index(channel)
        current_letter = alphabet[start_letter_index + channel_index]
        chart.add_series({
            'name': channel,
            'line':   {'width': 1.25, 'color': colors[channel_index]},
            'categories': '=Sheet1!$A$3:$A$109',
            'values': '=Sheet1!$' + current_letter + '$3:$' + current_letter + '$109',
        })

    chart.set_title({'name': 'Sensitivity'})
    chart.set_x_axis({'name': 'Wavelengths, nm', 'min': 360, 'max': 740})
    chart.set_y_axis({'name': 'Spectral Response Function'})

    chart.set_style(15)
    worksheet.insert_chart('E1', chart, {'x_offset': 50, 'y_offset': 50, 'x_scale': 1.5, 'y_scale': 1.5})

    start_letter = alphabet.index("R")

    for channel in range(3):
        ch_letter = channels[channel]
        parameter_letter = alphabet[start_letter + channel * 16]
        solution_letter = alphabet[start_letter + 1 + channel * 16]
        residual_letter = alphabet[start_letter + 2 + channel * 16]
        end_row = 2 + math.ceil((reg_stop[ch_letter] - reg_start[ch_letter])/reg_step[ch_letter])
        chart_letter = alphabet[start_letter + 3 + channel * 16]
        
        worksheet.write(solution_letter + '1', ch_letter, bold)
        worksheet.write(parameter_letter + '2', "Regularization parameter")
        worksheet.write(solution_letter + '2', "Solution's norms")
        worksheet.write(residual_letter + '2', "Residual's norms")
        
        for row_num, data in enumerate(np.arange(reg_start[ch_letter], reg_stop[ch_letter], reg_step[ch_letter])):
            worksheet.write(row_num + 2, start_letter + channel * 16, data)
            
        for row_num, data in enumerate(solution_norms[channel].keys()):
            worksheet.write(row_num + 2, start_letter + 1 + channel * 16, data)
        
        for row_num, data in enumerate(residual_norms[channel].keys()):
            worksheet.write(row_num + 2, start_letter + 2 + channel * 16, data)
            
        chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        
        chart.add_series({
            'line':   {'width': 1.25, 'color': colors[channel]},
            'categories': '=Sheet1!$' + residual_letter + '$3:$' + residual_letter + '$'+ str(end_row),
            'values': '=Sheet1!$' + solution_letter + '$3:$' + solution_letter + '$' + str(end_row),
        })

        chart.set_title({'name': str(ch_letter) + ': L-curve'})
        chart.set_x_axis({'name': "Residual's norm", 'min': min(residual_norms[channel].keys()) - reg_step[ch_letter]})
        chart.set_y_axis({'name': "Solution's norm", 'min': min(solution_norms[channel].keys()) - reg_step[ch_letter]})
        chart.set_legend({'none': True})
        chart.set_style(15)
        worksheet.insert_chart(chart_letter + '1', chart, {'x_offset': 50, 'y_offset': 50, 'x_scale': 1.5, 'y_scale': 1.5})

    workbook.close()

    return reg_sensitivities

##########################

illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC']
channels = ['R', 'G', 'B']
alphabet_0 = list(string.ascii_uppercase)
alphabet = alphabet_0 + ['A' + letter for letter in alphabet_0] + ['B' + letter for letter in alphabet_0]
colors = ['#993300', '#339966', '#0066CC']
exceptions = set([3, 27, 51, 75, 99, 123, 7, 9, 33, 57, 81, 105, 129, 14, 38, 62, 86, 110, \
                134, 20, 44, 68, 92, 116, 22, 46, 70, 94, 118])
#  print(0.8 * (144 - len(exceptions)))
valid = set(range(144)) - exceptions

learning_sample, patches = choose_learning_sample(valid)

################################
# Read colors from ColorChecker:

P = np.zeros(shape=(24 * 6, 3))
process  = DNGProcessingDemo()
    
for illuminant in illumination_types:
    illuminant_index = illumination_types.index(illuminant)  
    img_path = join(r"C:\Users\adm\Documents\IITP\dng", str(illuminant_index + 1) + '_' + illuminant + ".dng")
    json_path = join(r'C:\Users\adm\Documents\IITP\png_targed', str(illuminant_index + 1) + '_' + illuminant +'.jpg.json')

    img = process(img_path)       
    color_per_region = process_markup(json_path, img)
    P[24 * illuminant_index:24 * illuminant_index + 24] = \
                      [color_per_region[str(patch_index + 1)] for patch_index in range(24)]

###################################

# Read info about reflectances and illuminance and count sensitivities:

E_df = pd.read_excel('LampSpectra.xls', sheet_name='LampsSpectra', skiprows=2)
R_df = pd.read_excel('CCC_Reflectance_1.xls', sheet_name=1, skiprows=4, header=0)
R = np.array([R_df[str(patch + 1) + 'Avg'] for patch in range(24)])
R /= R.max(axis=0)
C = np.zeros(shape=(len(learning_sample), 107))
C_current_index = 0

for illuminant_index in range(6):
    E = np.diag(E_df[str(1 + illuminant_index) + 'Norm'])
    R_learning = [R[patch % 24] for patch in learning_sample if illuminant_index * 24 <= patch < illuminant_index * 24 + 24]
    C[C_current_index:C_current_index + len(R_learning)] = np.transpose(np.matmul(E, np.transpose(R_learning)))
    C_current_index += len(R_learning)

C_T = np.transpose(C)
P_learning = np.array([P[patch] for patch in learning_sample])
sensitivities = inv((C_T @ C).astype(float)) @ C_T @ P_learning

####################################
# Check accurancy usings learning and test samples: 
learn_mean_angle, learn_variance_angles, learn_angles_fig, learn_mean_norm, learn_variance_norms, learn_norms_fig = \
    check_accuracy(C, sensitivities, learning_sample)

test_sample = sorted(list(valid - set(learning_sample)))
# C_test = np.zeros(shape=(len(test_sample), 107))
# C_current_index = 0
# for illuminant_index in range(6):
#     E = np.diag(E_df[str(1 + illuminant_index) + 'Norm'])
#     R_test = [R[patch % 24] for patch in test_sample if illuminant_index * 24 <= patch < illuminant_index * 24 + 24]
#     C_test[C_current_index:C_current_index + len(R_test)] = np.transpose(np.matmul(E, np.transpose(R_test)))
#     C_current_index += len(R_test)

# test_mean_angle, test_variance_angles, test_angles_fig, test_mean_norm, test_variance_norms, test_norms_fig = \
#     check_accuracy(C_test, sensitivities, test_sample)

###########################################

# writing into excel file
writer = pd.ExcelWriter('Sensitivities.xlsx', engine='xlsxwriter')
workbook = writer.book
sensitivities_df = pd.DataFrame(sensitivities)
sensitivities_df.to_excel(writer, sheet_name='Sheet1',
                          index=False, header=channels, startrow=1, startcol=1)
worksheet = writer.sheets['Sheet1']
bold = workbook.add_format({'bold': 1})
for row_num, data in enumerate(E_df['Lambda grid']):
    worksheet.write(row_num + 2, 0, data)

worksheet.write('A1', 'Lambda grid', bold)

#adding the plot of sensitivities 
chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
start_letter_index = 1
for channel in channels:
    channel_index = channels.index(channel)
    current_letter = alphabet[start_letter_index + channel_index]
    chart.add_series({
        'name': channel,
        'line':   {'width': 1.25, 'color': colors[channel_index]},
        'categories': '=Sheet1!$A$3:$A$109',
        'values': '=Sheet1!$' + current_letter + '$3:$' + current_letter + '$109',
    })

chart.set_title({'name': 'Sensitivity'})
chart.set_x_axis({'name': 'Wavelengths, nm', 'min': 360, 'max': 740})
chart.set_y_axis({'name': 'Spectral Response Function'})

chart.set_style(15)
worksheet.insert_chart('F1', chart, {'x_offset': 50, 'y_offset': 50, 'x_scale': 1.5, 'y_scale': 1.5})

workbook.close()

reg_start = {"R": 0.05, "G": 0.05, "B": 0.05}
reg_stop = {"R": 5, "G": 5, "B": 5}
reg_step = {"R": 0.005, "G": 0.005, "B": 0.005}
# reg_sensitivities = regularization(reg_start, reg_stop, reg_step, sensitivities, C, P_learning)