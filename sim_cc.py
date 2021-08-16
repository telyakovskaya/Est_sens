# from main_experimental_version import E_wavelengths, R_wavelengths
import numpy as np
import pandas as pd
import statistics
import seaborn.apionly as sns
from data import get_lambda_grid, load_sens, load_illums, load_refl
from model import cals_radiances, change_wavelengths, cals_radiances, simulate_stimuls
from plot import draw_colorchecker, error_heatmap
import matplotlib.pyplot as plt
from pathlib import Path
from img_io import imsave
from data2 import get_sensitivities_gt, reflectances_matrix

#def simulate_noise(cc, alpha, beta):
    


def estimate_cc(sens, sens_wavelengths,
                cc_refl, refl_wavelengths,
                illum, illum_wavelengths, 
                wavelengths_number):
    """
    Estimate colors of color target
    """
    wavelengths_int = get_lambda_grid(401, 720, wavelengths_number)

    sens_int = change_wavelengths(sens, sens_wavelengths, wavelengths_int)
    refl_int = change_wavelengths(cc_refl, refl_wavelengths, wavelengths_int)
    illum_int = change_wavelengths(illum, illum_wavelengths, wavelengths_int)
    

    radiances = cals_radiances(refl_int, illum_int)
    stimuli = simulate_stimuls(sens_int, radiances)
    return stimuli

shape = (6, 4)
number_of_patches = shape[0] * shape[1]

E_dict, E_wavelengths = load_illums()
R_dict, R_wavelengths = load_refl()
R_dict = {key: val for key, val in R_dict.items() if 'Avg' in key}



E = np.asarray(list(E_dict.values()))
R = np.asarray(list(R_dict.values()))
    
sensitivities_given_dict, sens_wavelengths = load_sens()
sensitivities_given = np.asarray([sensitivities_given_dict[key] for key in ['red', 'green', 'blue']])



list_of_numbers = (5, 20, 32, 50, 100, 300, 500, 1000)

illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC']

for wavelengths_number in list_of_numbers:
    cc = estimate_cc(sensitivities_given, sens_wavelengths,
                 R, R_wavelengths,
                 E, E_wavelengths,
                 wavelengths_number)
    print(cc.shape)
    exit()

    #outpath_jpg = Path(r'C:\Users\Пользователь\Documents\imgs\cc_all\babelcolor' + str(wavelengths_number) + '.jpg')
    #outpath_tiff = Path(r'C:\Users\Пользователь\Documents\imgs\cc_all\babelcolor' + str(wavelengths_number) + '.tiff')
    for i in range(len(illumination_types)):
        cc_array = np.asarray([cc[j] for j in range(i * number_of_patches, (i + 1) * number_of_patches)])
        #wp = cc_array[0]
        #wp /= wp[1]
        #cc_array_tmp = cc_array /wp 
        outpath_jpg = Path(r'C:\Users\Пользователь\Documents\imgs\cc_24\babelcolor' + str(wavelengths_number) + illumination_types[i] + '.jpg')
        outpath_tiff = Path(r'C:\Users\Пользователь\Documents\imgs\cc_24\babelcolor' + str(wavelengths_number) + illumination_types[i] + '.tiff')
        img = draw_colorchecker(cc_array, shape, show=True)
        plt.imshow(img)
        plt.savefig(outpath_jpg)
        plt.close()
        imsave(outpath_tiff, img, unchanged=False, out_dtype=np.float32, gamma_correction=False)

exit()
list_of_numbers = list(reversed(list_of_numbers))
last_number = list_of_numbers[0]
for wavelengths_number in list_of_numbers:
    cc = estimate_cc(sensitivities_given, sens_wavelengths,
                 R, R_wavelengths,
                 E, E_wavelengths,
                 wavelengths_number)
    if wavelengths_number == last_number:
        cc1 = cc
    outpath_error = Path(r'C:\Users\Пользователь\Documents\imgs\cc_all\babelcolor_error' + str(wavelengths_number) + '.jpg')
    if wavelengths_number != last_number:
        angles = []
        for i in range(shape[0] * shape[1]):
            current_cc = cc[i]
            unit_current_cc = current_cc / np.linalg.norm(current_cc)
            previous_cc = cc1[i]
            unit_previous_cc = previous_cc / np.linalg.norm(previous_cc)
            dot_product = np.dot(unit_current_cc, unit_previous_cc)
            if dot_product > 1:
                dot_product = 1
            if dot_product < -1:
                dot_product = -1
            angles.append(np.arccos(dot_product) * 57.3)
        mean_angle = sum(angles) / (shape[0] * shape[1])
        variance = statistics.variance(angles)
        print(mean_angle, variance) 
        a = np.array(angles)
        a1 = a.reshape((shape[0], -1))
        value_max = max(a1, key=lambda item: item[1])[1]
        value_min = min(a1, key=lambda item: item[1])[1]
        sns.set_theme()
        img_error= sns.heatmap(a1, annot = False, vmin=value_min, vmax=value_max, center= (value_min+value_max)//2, fmt='.3g', cmap= 'coolwarm')
        #plt.savefig('seaborn-on' + str(wavelengths_number) + '.jpg')
        #plt.clf()
        #sns.reset_orig()
        plt.show()
        plt.savefig(outpath_error)
        plt.close()
       