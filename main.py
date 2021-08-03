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
from sim import simulate_stimuls, estimate_sensitivities
from plot import plot_sens, plot_spectra
from sim import change_wavelengths
from data import load_illums, load_refl, load_sens

# def choose_learning_sample(valid, achromatic_single, ratio=0.8):
#     chromatic_learning_sample = []
#     achromatic = [i * patches_number + single for i in range(illuminants_number) for single in achromatic_single]
#     all_chromatic_potential = [patch for patch in valid if patch not in achromatic]
#     chromatic_learning_number = int(ratio * choosed_patches_number - len(achromatic_single))

#     for i in range(illuminants_number):
#         potential = [patch for patch in all_chromatic_potential if i * patches_number <= patch < i * patches_number + patches_number]
#         chromatic_learning_sample += sorted(random.sample(potential, k=chromatic_learning_number))

#     patches = {patch: 1 if patch in chromatic_learning_sample or patch in achromatic else 0 for patch in valid}
#     learning_sample = [patch for patch, flag in patches.items() if flag == 1]
#     return learning_sample, patches


def C_matrix(sample, E, R, patches_number):
    """"
    This function calculates matrix C
    Args:
        sample (list): choosed pathes for learning
        E (np.ndarray): s x k, s-number of illuminations
        R (np.ndarray): n x k, k-number of wavelengths
        patches_number (int): actually, number of patches = n

    Returns:
        [np.ndarray]: (sn) x k
    """    
    C = np.zeros(shape=(len(sample), E.shape[1]))
    C_current_index = 0
    E = np.transpose(E)
    for illuminant_index in range(E.shape[-1]):
        e_diag = np.diag(E[:,illuminant_index])
        R_current = np.array([R[patch % patches_number] for patch in sample 
                    if illuminant_index * patches_number <= patch < illuminant_index * patches_number + patches_number])
        C[C_current_index:C_current_index + len(R_current)] = np.transpose(np.matmul(e_diag, np.transpose(R_current)))
        C_current_index += len(R_current)
    return C

def check_accuracy(patches_number, stimulus_predicted, stimulus_genuine):
    """

    Args:
        patches_number (int): actually, number of patches
        stimulus_predicted (two-dimensional list): total number of patches x 3
        stimulus_genuine (two-dimensional list): total number of patches x 3

    Returns:
        mean_angle[float]: average value of angles between vectors
        variance_angles[float]: variance of angles
        angles_fig[]: histogram for angles
        mean_norm[float]: average value of vector lengths
        variance_norms[float]: variance of vector lengths
        norms_fig[]: histogram for vector lengths
    """    

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

def write_to_excel(file_path, sensitivities, R_learning, learning_sample, channels):
    """This function builds graphs in Excel

    """    
    alphabet_st = list(string.ascii_uppercase)
    alphabet = alphabet_st + ['A' + letter for letter in alphabet_st] + ['B' + letter for letter in alphabet_st]
    colors_RGB = {'blue': '#0066CC', 'green': '#339966', 'red': '#993300'}
    wavelengths = np.arange(400, 721, 10)

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
    print(colors_patches)
    draw_chart(workbook, worksheet_1, "Patches' reflectance", patches_x_axis, patches_y_axis, \
        '=Sheet2!$A$3:$A$109', patches_values_coord, 'F26', learning_sample, colors_patches)
    
    workbook.close()

def get_lambda_grid(start, stop, points_number):
    """ This function creates list of wavelengths

    Args:
        start (int)
        stop (int)
        points_number (int)

    Returns:
        [list]: chosen wavelengths 
    """    
    step = (stop - start) / points_number
    return [start + point * step for point in range(points_number)]


if __name__=='__main__':    
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
    
    def process_markup(json_path, img):
        with open(json_path, 'r') as file:
            markup_json = json.load(file)

        color_per_region = {}
        for object in markup_json['objects']:
            color_per_region[object['tags'][0]] = calc_mean_color(img, object['data'])
        return color_per_region

    def measure_stimuli(patches_number, illuminants_number):
        """ To do getting colors from ColorChecker

         Args:
            patches_number (int)
            illuminants_number (int)

        Returns:
            P (np.ndarray): colors for all patches, patches_number x 3
        """    
        P = np.zeros(shape=(patches_number * illuminants_number, 3))
        process  = DNGProcessingDemo()
        illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC'][:illuminants_number]

        for illuminant in illumination_types:
            illuminant_index = illumination_types.index(illuminant)  
            img_path = join(r"C:\Users\adm\Documents\IITP\dng", str(illuminant_index + 1) + '_' + illuminant + ".dng")
            json_path = join(r'C:\Users\adm\Documents\IITP\png_targed', str(illuminant_index + 1) + '_' + illuminant +'.jpg.json')

            img = process(img_path)       
            color_per_region = process_markup(json_path, img)
            P[patches_number * illuminant_index:patches_number * illuminant_index + patches_number] = \
                            [color_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
        return P
    
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

    class SensEstimator:
        def __init__(self, reg_start, reg_stop, threshold_stop, reg_step):
            self.reg_start = reg_start
            self.reg_stop = reg_stop
            self.threshold_stop = threshold_stop
            self.reg_step = reg_step

        def menger(self, p1, p2, p3):
            residual1, solution1 = p1
            residual2, solution2 = p2
            residual3, solution3 = p3
            p1p2 = (residual2 - residual1)**2 + (solution2 - solution1)**2
            p2p3 = (residual3 - residual2)**2 + (solution3 - solution2)**2
            p3p1 = (residual1 - residual3)**2 + (solution1 - solution3)**2
            numerator = residual1 * solution2 + residual2 * solution3 + residual3 * solution1 - \
                        residual1 * solution3 - residual3 * solution2 - residual2 * solution1
            return (2 * numerator) / (math.sqrt(p1p2 * p2p3 * p3p1))

        def l_curve_P(self, reg_parameter, channel, C, sensitivities, P_learning):
            C_T = np.transpose(C)
            sensitivities[:,channel] = inv((C_T @ C).astype(float) + np.identity(107) * reg_parameter) @ C_T @ P_learning[:,channel]
            solution = ((np.linalg.norm(sensitivities[:,channel], 2))) ** 2
            residual_vector = C @ sensitivities[:,channel] - P_learning[:,channel]
            residual = ((np.linalg.norm(residual_vector, 2))) ** 2
            return residual, solution

        def find_optimal_parameter(self, channels, C, sensitivities, P_learning):
            optimal_parameter = [0 for _ in range(3)]
            for channel in range(3):
                p = {}
                reg_parameter = {}
                ch_letter = channels[channel]
                reg_parameter[1], reg_parameter[4] = self.reg_start[ch_letter], self.reg_stop[ch_letter]   # search extremes
                epsilon = self.threshold_stop                                                             # termination threshold
                phi = (1 + math.sqrt(5)) / 2                                                     # golden section
            
                reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))

                for i in range(1, 5):
                    p[i] = self.l_curve_P(reg_parameter[i], channel, C, sensitivities, P_learning)
            
                while ((reg_parameter[4] - reg_parameter[1]) / reg_parameter[4]) >= epsilon:
                    C2 = self.menger(p[1], p[2], p[3])
                    C3 = self.menger(p[2], p[3], p[4])
                
                    while C3 <= 0:
                        reg_parameter[4] = reg_parameter[3]
                        reg_parameter[3] = reg_parameter[2]
                        reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                        p[4] = p[3]
                        p[3] = p[2]
                        p[2] = self.l_curve_P(reg_parameter[2], channel, C, sensitivities, P_learning)
                        C3 = self.menger(p[2], p[3], p[4])
                
                    if C2 > C3:
                        optimal_parameter[channel] = reg_parameter[2]
                        reg_parameter[4] = reg_parameter[3]
                        reg_parameter[3] = reg_parameter[2]
                        reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                        p[4] = p[3]
                        p[3] = p[2]
                        p[2] = self.l_curve_P(reg_parameter[2], channel, C, sensitivities, P_learning)
                    else:
                        optimal_parameter[channel] = reg_parameter[3]
                        reg_parameter[1] = reg_parameter[2]
                        reg_parameter[2] = reg_parameter[3]
                        reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))
                        p[1] = p[2]
                        p[2] = p[3]
                        p[3] = self.l_curve_P(reg_parameter[3], channel, C, sensitivities, P_learning)
            
            return optimal_parameter


        def estimate(self, sensitivities, C, P_learning, channels):
            C_T = np.transpose(C)
            optimal_parameter = self.find_optimal_parameter()
            reg_sensitivities = np.zeros(shape=(107, 3))
            solution_norms = [{} for _ in range(3)]
            residual_norms = [{} for _ in range(3)]
            for channel in range(3):
                ch_letter = channels[channel]
        
                for reg_parameter in np.arange(self.reg_start[ch_letter], self.reg_stop[ch_letter], self.reg_step[ch_letter]):
                    residual, solution = self.l_curve_P(reg_parameter, channel, C, sensitivities, P_learning)
                    solution_norms[channel][solution] = reg_parameter
                    residual_norms[channel][residual] = reg_parameter
                
                reg_sensitivities[:,channel] = inv((C_T @ C).astype(float) + np.identity(107) * optimal_parameter[channel]) @ C_T @ P_learning[:,channel]
            return reg_sensitivities
     
        def write_to_excel(self, E, sensitivities, channels, solution_norms, residual_norms):
            alphabet_st = list(string.ascii_uppercase)
            alphabet = alphabet_st + ['A' + letter for letter in alphabet_st] + ['B' + letter for letter in alphabet_st]
            colors_RGB = {'blue': '#0066CC', 'green': '#339966', 'red': '#993300'}
            writer = pd.ExcelWriter('Sensitivities_2.xlsx', engine='xlsxwriter')
            workbook = writer.book

            reg_sensitivities_df = pd.DataFrame(sensitivities)
            reg_sensitivities_df.to_excel(writer, sheet_name='Sheet1', index=False, header=channels, startrow=1, startcol=1)

            worksheet = writer.sheets['Sheet1']

            bold = workbook.add_format({'bold': 1})
            worksheet.write('C1', 'Sensitivities', bold)
            worksheet.write('A2', 'Lambda grid', bold)
            for row_num, data in enumerate(E['Lambda grid']):
                worksheet.write(row_num + 2, 0, data)

            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
            start_letter_index = 1
            for channel in channels:
                channel_index = channels.index(channel)
                current_letter = alphabet[start_letter_index + channel_index]
                chart.add_series({
                    'name': channel,
                    'line':   {'width': 1.25, 'color': colors_RGB[channel_index]},
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
                end_row = 2 + math.ceil((self.reg_stop[ch_letter] - self.reg_start[ch_letter])/self.reg_step[ch_letter])
                chart_letter = alphabet[start_letter + 3 + channel * 16]
        
                worksheet.write(solution_letter + '1', ch_letter, bold)
                worksheet.write(parameter_letter + '2', "Regularization parameter")
                worksheet.write(solution_letter + '2', "Solution's norms")
                worksheet.write(residual_letter + '2', "Residual's norms")
        
                for row_num, data in enumerate(np.arange(self.reg_start[ch_letter], self.reg_stop[ch_letter], self.reg_step[ch_letter])):
                    worksheet.write(row_num + 2, start_letter + channel * 16, data)
            
                for row_num, data in enumerate(solution_norms[channel].keys()):
                    worksheet.write(row_num + 2, start_letter + 1 + channel * 16, data)
        
                for row_num, data in enumerate(residual_norms[channel].keys()):
                    worksheet.write(row_num + 2, start_letter + 2 + channel * 16, data)
            
                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        
                chart.add_series({
                    'line':   {'width': 1.25, 'color': colors_RGB[channel]},
                    'categories': '=Sheet1!$' + residual_letter + '$3:$' + residual_letter + '$'+ str(end_row),
                    'values': '=Sheet1!$' + solution_letter + '$3:$' + solution_letter + '$' + str(end_row),
                })

                chart.set_title({'name': str(ch_letter) + ': L-curve'})
                chart.set_x_axis({'name': "Residual's norm", 'min': min(residual_norms[channel].keys()) - self.reg_step[ch_letter]})
                chart.set_y_axis({'name': "Solution's norm", 'min': min(solution_norms[channel].keys()) - self.reg_step[ch_letter]})
                chart.set_legend({'none': True})
                chart.set_style(15)
                worksheet.insert_chart(chart_letter + '1', chart, {'x_offset': 50, 'y_offset': 50, 'x_scale': 1.5, 'y_scale': 1.5})

            workbook.close()


    
    def main():
        exceptions = set([])
        wavelengths = np.arange(400, 721, 10)

        E_dict, E_wavelengths = load_illums()
        R_dict, R_wavelengths = load_refl()
        
        E = np.asarray(list(E_dict.values()))
        E = change_wavelengths(E, E_wavelengths, wavelengths)

        R = np.asarray(list(R_dict.values()))
        R = change_wavelengths(R, R_wavelengths, wavelengths)
        R /= np.max(R, axis=0)

        illuminants_number = len(E_dict)
        patches_number = len(R_dict)
        
        choosed_patches_number = patches_number                  # how many patches to use 
        valid = set(range(patches_number * illuminants_number)) - exceptions
        achromatic_single = list(range(14)) + [patch for patch in range(14, 126) if patch % 14 == 0 or patch % 14 == 13] + list(range(126, 140)) + \
                list(range(60, 66)) + list(range(74, 80))
        # learning_sample, patches = choose_learning_sample(valid, achromatic_single, ratio=1.)
        learning_sample = [i for i in range(patches_number * illuminants_number)]
        #print(len(learning_sample))     
        # C = np.zeros(shape=(len(learning_sample), len(E)))
        # C_current_index = 0
        # for illuminant_index in range(6):
        #     E = np.diag(E_df[str(1 + illuminant_index) + 'Norm'])
        #     R_learning = [R[patch % 24] for patch in learning_sample if illuminant_index * 24 <= patch < illuminant_index * 24 + 24]
        #     C[C_current_index:C_current_index + len(R_learning)] = np.transpose(np.matmul(E, np.transpose(R_learning)))
        #     C_current_index += len(R_learning)

        # E = np.ones((33, 1))
        
        C = C_matrix(learning_sample, E, R, patches_number)
        C_T = np.transpose(C)
    
        sensitivities_given_dict, sens_wavelengths = load_sens()
        sensitivities_given = np.asarray(list(sensitivities_given_dict.values()))
        sensitivities_given = change_wavelengths(sensitivities_given, sens_wavelengths, wavelengths)
        channels = sensitivities_given.shape[0]

        stimulus_learning = simulate_stimuls(sensitivities_given, C_T)

        stops = list(i for i in range(1, 60, 5))

        for stop in stops:
            print(stop)
            sensitivities = estimate_sensitivities(C_T[:, :stop], stimulus_learning[:stop])
            plot_sens(sensitivities, '--')
            plot_sens(sensitivities_given, '-')
            plt.show()

            plot_spectra(C_T[:,:stop], True)
            
        # print(np.concatenate((sensitivities_given, sensitivities), axis=1))
        R_learning = []
        for illuminant_index in range(illuminants_number):
                R_learning += [R[patch % patches_number] for patch in learning_sample 
                            if illuminant_index * patches_number <= patch < illuminant_index * patches_number + patches_number]
        R_learning = np.transpose(np.array(R_learning))
        write_to_excel('Sensitivities.xlsx', sensitivities, R_learning, learning_sample, channels)

        # ####################################
        # Check accuracy usings learning and test samples: 
        # stimulus_predicted = C @ sensitivities
        # patches_number = len(learning_sample)
        # learn_mean_angle, learn_variance_angles, learn_angles_fig, learn_mean_norm, learn_variance_norms, learn_norms_fig = \
        #     check_accuracy(patches_number, stimulus_predicted, stimulus_learning)

        # test_sample = sorted(list(valid - set(learning_sample)))
        # C_test = C_matrix(test_sample, E, R, illuminants_number)
        # patches_number = len(test_sample)
        # stimulus_predicted_test = C_test @ sensitivities
        # stimulus_test = C_test @ sensitivities_given
        # test_mean_angle, test_variance_angles, test_angles_fig, test_mean_norm, test_variance_norms, test_norms_fig = \
        #     check_accuracy(patches_number, stimulus_predicted_test, stimulus_test)

        ######################################

    main()