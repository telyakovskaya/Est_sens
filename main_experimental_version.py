import pandas as pd
import numpy as np
from numpy.linalg import inv
import string
import statistics
import seaborn as sns
from plot import plot_sens, plot_spectra, draw_colorchecker
from data import R_internet_matrix, reflectances_matrix, choose_learning_sample, get_sensitivities_gt, spectras_matrix
from regularization import regularization
from measuring import measure_stimuli
import cv2


global illuminants_number, patches_number, choosed_patches_number, wavelengths


def check_accuracy_angles(patches_number, stimulus_predicted, stimulus_genuine):
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


def get_lambda_grid(start, stop, points_number):
    step = (stop - start) / points_number
    return [start + point * step for point in range(points_number)]
 

def plot_pictures(C, learning_sample, sensitivities_df, P):
    sensitivities = np.zeros(shape=(len(wavelengths), 3))
    for channel in range(3):
        P_learning = np.array([P[channel][patch] for patch in learning_sample[channel]])
        sensitivities[:, channel] = inv((C[channel].T @ C[channel]).astype(float)) @ C[channel].T @ P_learning

    plot_sens(wavelengths, sensitivities, sensitivities_df, show=True)
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
                    {variances[stimul, channel] / P[stimul, channel] * 100}')


def filter_stimuli(P_measured, errors, channel, patches_number=24, illuminants_number=1):
    P_filtered = {}
    exceptions = set()
    for patch in range(patches_number):
        if errors[patch, channel] > 3.:
            exceptions.add(patch)
        else:
            P_filtered[patch] = P_measured[patch, channel]
    valid = set(range(patches_number * illuminants_number)) - exceptions
    return P_filtered, valid
##########################

# wavelengths = get_lambda_grid(400, 721, 20)
illuminants_number = 1
patches_number = 24                                      # in colorchecker
choosed_patches_number = patches_number                  # how many patches to use 
colors_RGB = {'blue': '#0066CC', 'green': '#339966', 'red': '#993300'}
spectras_Alexander = {}
P_learning = {}
learning_sample = {}
P_filtered = {}
sensitivities = {}
wavelengths_points_numbers = {0: 19, 1: 22, 2: 19}

#########################

E = pd.read_excel('LampSpectra.xls', sheet_name='LampsSpectra', skiprows=2)
R = pd.read_excel('CCC_Reflectance_1.xls', sheet_name=1, skiprows=4, header=0)
P_measured = cv2.imread('means.tiff',  cv2.IMREAD_UNCHANGED)
errors = cv2.imread('errors.tiff',  cv2.IMREAD_UNCHANGED)

for channel in range(3):
    wavelengths = get_lambda_grid(400, 721, wavelengths_points_numbers[channel])
    P_filtered[channel], valid = filter_stimuli(P_measured, errors, channel)
    learning_sample[channel] = choose_learning_sample(valid, ratio=1.)
    
    spectras_Alexander[channel] = spectras_matrix(learning_sample[channel], wavelengths, E, R) 
    P_learning = np.array([P_filtered[channel][patch] for patch in learning_sample[channel]])
    
    sensitivities[channel] = inv((spectras_Alexander[channel].T @ spectras_Alexander[channel]).astype(float)) @ \
        spectras_Alexander[channel].T @ P_learning

    colors_check = spectras_Alexander[channel] @ sensitivities[channel]
    print(np.linalg.norm((colors_check - P_learning), 2))

    reg_sensitivities = regularization(channel, wavelengths, spectras_Alexander[channel], P_learning)
    colors_check = spectras_Alexander[channel] @ reg_sensitivities
    print(np.linalg.norm((colors_check - P_learning), 2))


# draw_colorchecker(P_measured, patches_number, show=True)

# plot_pictures(spectras_Alexander, learning_sample, P_filtered)
plot_sens(wavelengths_points_numbers, sensitivities, show=True)
plot_sens(wavelengths_points_numbers, reg_sensitivities, show=True)



###############################

# The number of points in lambda grid shouldn't exceed the length of the learning sample.
