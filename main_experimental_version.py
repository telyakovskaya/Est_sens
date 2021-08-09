import pandas as pd
import numpy as np
from numpy.linalg import inv
from plot import plot_sens, plot_spectra, draw_colorchecker
from data import R_internet_matrix, reflectances_matrix, choose_learning_sample, get_sensitivities_gt, spectras_matrix
from regularization import regularization
from measuring import measure_stimuli
import cv2
from data import get_lambda_grid
import pprint


def check_stimuls_accuracy(P, variances):
    for channel in range(3):
        for stimul in range(len(P)): 
            print(f'p: {stimul}, ch: {channel}, std(%): \
                    {variances[stimul, channel] / P[stimul, channel] * 100}')


def filter_stimuli(P_measured, errors, patches_number=24, illuminants_number=1):
    P_filtered = {}
    valid = {}
    for channel in range(3):
        P_filtered[channel] = {}
        exceptions = set()
        for patch in range(patches_number):
            if errors[patch, channel] > 2.8:
                exceptions.add(patch)
            else:
                P_filtered[channel][patch] = P_measured[patch, channel]
        valid[channel] = set(range(patches_number * illuminants_number)) - exceptions
    return P_filtered, valid


##########################

illuminants_number = 1
patches_number = 24                                      # in colorchecker
choosed_patches_number = patches_number                  # how many patches to use 
spectras_Alexander = {}
sensitivities, reg_sensitivities = {}, {}
wavelengths_points_numbers = {0: 17, 1: 21, 2: 19}

#########################

E = pd.read_excel('LampSpectra.xls', sheet_name='LampsSpectra', skiprows=2)
R = pd.read_excel('CCC_Reflectance_1.xls', sheet_name=1, skiprows=4)
P_measured = cv2.imread('means.tiff',  cv2.IMREAD_UNCHANGED)
errors = cv2.imread('errors.tiff',  cv2.IMREAD_UNCHANGED)

P_filtered, valid = filter_stimuli(P_measured, errors)
learning_sample = choose_learning_sample(valid, ratio=1.)

for channel in range(3):
    wavelengths = get_lambda_grid(400, 721, wavelengths_points_numbers[channel])
    # wavelengths_points_numbers[channel] = len(learning_sample[channel])
    print('\nthe number of patches used: ', len(learning_sample[channel]))

    spectras_Alexander[channel] = spectras_matrix(learning_sample[channel], wavelengths, E, R) 
    P_learning = np.array([P_filtered[channel][patch] for patch in learning_sample[channel]])
    
    sensitivities[channel] = inv((spectras_Alexander[channel].T @ spectras_Alexander[channel]).astype(float)) @ \
        spectras_Alexander[channel].T @ P_learning
    colors_check = spectras_Alexander[channel] @ sensitivities[channel]
    print('norm difference before regularization: ', np.linalg.norm((colors_check - P_learning), 2))

    reg_sensitivities[channel] = regularization(channel, wavelengths, spectras_Alexander[channel], P_learning)
    reg_colors_check = spectras_Alexander[channel] @ reg_sensitivities[channel]
    print('norm difference after regularization: ',np.linalg.norm((reg_colors_check - P_learning), 2))


# draw_colorchecker(P_measured, show=True)

plot_sens(wavelengths_points_numbers, sensitivities, show=True)
plot_sens(wavelengths_points_numbers, reg_sensitivities, show=True)
