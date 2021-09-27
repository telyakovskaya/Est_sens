from pathlib import Path
import pandas as pd
import numpy as np
from numpy.linalg import inv
from plot2 import plot_sens, plot_spectra, draw_colorchecker
from data2 import choose_learning_sample
from regularization import regularization_Tikhonov, regularization_derivatives
import cv2
from data import get_lambda_grid, load_refl, load_illums, load_sens
from img_io import imread
from model import change_wavelengths, cals_radiances
from tqdm import tqdm


def check_stimuls_accuracy(P, variances):
    for channel in range(3):
        for stimul in range(len(P)): 
            print(f'p: {stimul}, ch: {channel}, std(%): \
                    {variances[stimul, channel] / P[stimul, channel] * 100}')


def filter_stimuli(patches_colors_measured, errors, patches_number, illuminants_number=1):
    patches_colors_filtered = {}
    valid = {}
    for channel in range(3):
        patches_colors_filtered[channel] = {}
        exceptions = set()
        for patch in range(patches_number):
            if errors[patch, channel] > 4.:
                exceptions.add(patch)
            else:
                patches_colors_filtered[channel][patch] = patches_colors_measured[patch, channel]
        valid[channel] = set(range(patches_number * illuminants_number)) - exceptions
    return patches_colors_filtered, valid


def check_optimal(radiances, sensitivities, patches_channel, min_norm_difference_before_reg, optimal_wavelength_number_before_reg):
        colors_check = radiances[channel] @ sensitivities[channel]
        current_norm_difference = np.linalg.norm((colors_check - patches_channel), 2)
        if current_norm_difference < min_norm_difference_before_reg[channel]:
            min_norm_difference_before_reg[channel] = current_norm_difference
            optimal_wavelength_number_before_reg[channel] = wavelengths_number
        return min_norm_difference_before_reg[channel], optimal_wavelength_number_before_reg[channel]


def get_radiances(optimal_wavelength_number, E_arr, E_wavelengths, R_arr, R_wavelengths, patches_number):
    for channel in range(3):
        wavelengths = get_lambda_grid(400, 721, optimal_wavelength_number[channel])
        E = change_wavelengths(E_arr, E_wavelengths, wavelengths)
        R = change_wavelengths(R_arr, R_wavelengths, wavelengths)
        radiances[channel] = cals_radiances(R, E)
        radiances[channel] = np.asarray([radiances[channel][j] for j in range(0, patches_number)])
    return radiances
##########################

illuminants_number = 1
# patches_number = 24                              # in colorchecker
# choosed_patches_number = patches_number          # how many patches to use 
radiances = {}
sensitivities, reg_sensitivities = {}, {}
# wavelengths_number = 5
# wavelengths_points_numbers = {0: wavelengths_number, 1: wavelengths_number, 2: wavelengths_number}

optimal_wavelength_number_before_reg = {ch: 0 for ch in range(3)}
optimal_wavelength_number_reg_Tikhonov = {ch: 0 for ch in range(3)}
optimal_wavelength_number_reg_derivatives = {ch: 0 for ch in range(3)}
min_norm_dif_before_reg = {ch: 9999999999999999 for ch in range(3)}
min_norm_dif_reg_Tikhonov = {ch: 9999999999999999 for ch in range(3)}
min_norm_dif_reg_derivatives = {ch: 9999999999999999 for ch in range(3)}

#########################

E_dict, E_wavelengths = load_illums()
E_dict = {key: val for key, val in E_dict.items() if 'Avg' in key}
R_dict, R_wavelengths = load_refl()
R_dict = {key: val for key, val in R_dict.items() if 'Avg' in key}
E_arr = np.asarray(list(E_dict.values()))
R_arr = np.asarray(list(R_dict.values()))

# errors = imread(Path(r"errors.tiff"), out_dtype=np.float32, linearize_srgb=False)
# errors = np.reshape(errors, (patches_number, 3))
# cc = imread(Path(r"means.tiff"), out_dtype=np.float32, linearize_srgb=False)
# cc = imread(Path(r"babelcolor1000D50_24.tiff"), out_dtype=np.float32, linearize_srgb=False)
cc = imread(Path(r"babelcolor1000D50_140.tiff"), out_dtype=np.float32, linearize_srgb=False)
patches_number = 140 

cc = np.reshape(cc, (patches_number, 3))
patches_colors_measured = np.asarray([cc[j] for j in range(0, patches_number)])
errors = np.zeros(shape=(patches_number, 3))

patches_colors_filtered, valid = filter_stimuli(patches_colors_measured, errors, patches_number)
learning_sample = choose_learning_sample(valid, ratio=1., patches_number=patches_number, illuminants_number=1, choosed_patches_number=patches_number)

for wavelengths_number in tqdm(range(2, 100)):
    wavelengths = get_lambda_grid(400, 721, wavelengths_number)
    E = change_wavelengths(E_arr, E_wavelengths, wavelengths)
    R = change_wavelengths(R_arr, R_wavelengths, wavelengths)

    for channel in range(3):
        # print('\nthe number of patches used: ', len(learning_sample[channel]))
        radiances[channel] = cals_radiances(R, E)
        # choose radinaces only for the first illumination   
        radiances[channel] = np.asarray([radiances[channel][j] for j in range(0, patches_number)])
        
        # choose patches colors used in the learning sample
        patches_channel = np.array([patches_colors_filtered[channel][patch] for patch in learning_sample[channel]])
        
        #lsq optimization
        sensitivities[channel] = inv((radiances[channel].T @ radiances[channel]).astype(float)) @ \
            radiances[channel].T @ patches_channel
        sensitivities[channel][sensitivities[channel] < 0] = 0
        # colors_check = radiances[channel] @ sensitivities[channel]
        # print('norm difference before regularization: ', np.linalg.norm((colors_check - patches_channel), 2))
        min_norm_dif_before_reg[channel], optimal_wavelength_number_before_reg[channel] = \
            check_optimal(radiances, sensitivities, patches_channel, min_norm_dif_before_reg, optimal_wavelength_number_before_reg)

        # reg optimization
        reg_sensitivities[channel] = regularization_Tikhonov(channel, wavelengths, radiances[channel], patches_channel)
        reg_sensitivities[channel][reg_sensitivities[channel] < 0] = 0
        min_norm_dif_reg_Tikhonov[channel], optimal_wavelength_number_reg_Tikhonov[channel] = \
            check_optimal(radiances, reg_sensitivities, patches_channel, min_norm_dif_reg_Tikhonov, optimal_wavelength_number_reg_Tikhonov)


        reg_sensitivities[channel] = regularization_derivatives(channel, wavelengths, radiances[channel], patches_channel)
        reg_sensitivities[channel][reg_sensitivities[channel] < 0] = 0
        min_norm_dif_reg_derivatives[channel], optimal_wavelength_number_reg_derivatives[channel] = \
            check_optimal(radiances, reg_sensitivities, patches_channel, min_norm_dif_reg_derivatives, optimal_wavelength_number_reg_derivatives)
        # reg_colors_check = radiances[channel] @ reg_sensitivities[channel]
        # print('norm difference after regularization: ',np.linalg.norm((reg_colors_check - patches_channel), 2))

print(optimal_wavelength_number_before_reg, min_norm_dif_before_reg)
print(optimal_wavelength_number_reg_Tikhonov, min_norm_dif_reg_Tikhonov)
print(optimal_wavelength_number_reg_derivatives, min_norm_dif_reg_derivatives)
# draw_colorchecker(patches_colors_measured, show=True)


radiances = get_radiances(optimal_wavelength_number_before_reg, E_arr, E_wavelengths, R_arr, R_wavelengths, patches_number)
for channel in range(3):
    patches_channel = np.array([patches_colors_filtered[channel][patch] for patch in learning_sample[channel]])
    sensitivities[channel] = inv((radiances[channel].T @ radiances[channel]).astype(float)) @ \
            radiances[channel].T @ patches_channel
    sensitivities[channel][sensitivities[channel] < 0] = 0
plot_sens(optimal_wavelength_number_before_reg, sensitivities, show=True)

radiances = get_radiances(optimal_wavelength_number_reg_Tikhonov, E_arr, E_wavelengths, R_arr, R_wavelengths, patches_number)
for channel in range(3):
    wavelengths = get_lambda_grid(400, 721, optimal_wavelength_number_reg_Tikhonov[channel])
    patches_channel = np.array([patches_colors_filtered[channel][patch] for patch in learning_sample[channel]])
    reg_sensitivities[channel] = regularization_Tikhonov(channel, wavelengths, radiances[channel], patches_channel)
    reg_sensitivities[channel][reg_sensitivities[channel] < 0] = 0
plot_sens(optimal_wavelength_number_reg_Tikhonov, reg_sensitivities, show=True)

radiances = get_radiances(optimal_wavelength_number_reg_derivatives, E_arr, E_wavelengths, R_arr, R_wavelengths, patches_number)
for channel in range(3):
    wavelengths = get_lambda_grid(400, 721, optimal_wavelength_number_reg_derivatives[channel])
    patches_channel = np.array([patches_colors_filtered[channel][patch] for patch in learning_sample[channel]])
    reg_sensitivities[channel] = regularization_derivatives(channel, wavelengths, radiances[channel], patches_channel)
    reg_sensitivities[channel][reg_sensitivities[channel] < 0] = 0
plot_sens(optimal_wavelength_number_reg_derivatives, reg_sensitivities, show=True)