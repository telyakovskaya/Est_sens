from pathlib import Path
import pandas as pd
import numpy as np
from numpy.linalg import inv
from plot2 import plot_sens, plot_spectra, draw_colorchecker
from data2 import choose_learning_sample
from regularization import regularization_Tikhonov, regularization_derivatives
from measuring import measure_stimuli
import cv2
from data import get_lambda_grid, load_refl, load_illums, load_sens
from img_io import imread
from model import change_wavelengths, cals_radiances


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
            if errors[patch, channel] > 2.8:
                exceptions.add(patch)
            else:
                patches_colors_filtered[channel][patch] = patches_colors_measured[patch, channel]
        valid[channel] = set(range(patches_number * illuminants_number)) - exceptions
    return patches_colors_filtered, valid


##########################

illuminants_number = 1
patches_number = 140                            # in colorchecker
choosed_patches_number = patches_number                  # how many patches to use 
radiances = {}
sensitivities, reg_sensitivities = {}, {}
wavelengths_number = 500
wavelengths_points_numbers = {0: wavelengths_number, 1: wavelengths_number, 2: wavelengths_number}

#########################

#patches_colors_measured = cv2.imread('means.tiff',  cv2.IMREAD_UNCHANGED)
#errors = cv2.imread('errors.tiff',  cv2.IMREAD_UNCHANGED)
E_dict, E_wavelengths = load_illums()
R_dict, R_wavelengths = load_refl()



E = np.asarray(list(E_dict.values()))
R = np.asarray(list(R_dict.values()))


sensitivities_given_dict, sens_wavelengths = load_sens()
sensitivities_given = np.asarray([sensitivities_given_dict[key] for key in ['red', 'green', 'blue']])
cc = imread(Path(r"babelcolor1000D50_140.tiff"), out_dtype=np.float32, linearize_srgb=False)
cc = np.reshape(cc, (patches_number, 3))

patches_colors_measured = np.asarray([cc[j] for j in range(0, patches_number)])
errors = np.zeros(shape=(patches_number, 3))

patches_colors_filtered, valid = filter_stimuli(patches_colors_measured, errors, patches_number)
learning_sample = choose_learning_sample(valid, ratio=1., patches_number=patches_number, illuminants_number=1, choosed_patches_number=140)

wavelengths = get_lambda_grid(400, 721, wavelengths_number)
E = change_wavelengths(E, E_wavelengths, wavelengths)
R = change_wavelengths(R, R_wavelengths, wavelengths)

for channel in range(3):
    # calculating radinaces
    # wavelengths_points_numbers[channel] = len(learning_sample[channel])
    print('\nthe number of patches used: ', len(learning_sample[channel]))
    radiances[channel] = cals_radiances(R, E)
    # choose radinaces only for first illumination   
    radiances[channel] = np.asarray([radiances[channel][j] for j in range(0, patches_number)])
    
    # choose patches colors for only 
    patches_channel = np.array([patches_colors_filtered[channel][patch] for patch in learning_sample[channel]])
    
    #lsq optimization
    sensitivities[channel] = inv((radiances[channel].T @ radiances[channel]).astype(float)) @ \
        radiances[channel].T @ patches_channel
    sensitivities[channel][sensitivities[channel] < 0] = 0
    colors_check = radiances[channel] @ sensitivities[channel]
    print('norm difference before regularization: ', np.linalg.norm((colors_check - patches_channel), 2))

    # reg optimization
    # reg_sensitivities[channel] = regularization_Tikhonov(channel, wavelengths, radiances[channel], patches_channel)
    reg_sensitivities[channel] = regularization_derivatives(channel, wavelengths, radiances[channel], patches_channel)
    reg_sensitivities[channel][reg_sensitivities[channel] < 0] = 0
    reg_colors_check = radiances[channel] @ reg_sensitivities[channel]
    print('norm difference after regularization: ',np.linalg.norm((reg_colors_check - patches_channel), 2))


# draw_colorchecker(patches_colors_measured, show=True)

# plot_sens(wavelengths_points_numbers, sensitivities, show=True)
# plot_sens(wavelengths_points_numbers, reg_sensitivities, show=True)