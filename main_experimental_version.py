from ntpath import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import inv
import string
import statistics
import seaborn as sns
from plot import plot_sens, plot_spectra, draw_colorchecker
from data import R_internet_matrix, reflectances_matrix, choose_learning_sample, get_sensitivities_gt, spectras_matrix
from regularization import regularization, easy_regularization
from measuring import DNGProcessingDemo, process_markup
import cv2


global illuminants_number, patches_number, choosed_patches_number # wavelengths


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


def get_lambda_grid(start, stop, points_number):
    step = (stop - start) / points_number
    return [start + point * step for point in range(points_number)]


def measure_stimuli():
    P = np.zeros(shape=(patches_number * illuminants_number, 3))
    variances = np.zeros(shape=(patches_number * illuminants_number, 3))
    process  = DNGProcessingDemo()
    # illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC'][:illuminants_number]
    illumination_types = ['LUM']

    for illuminant in illumination_types:
        illuminant_index = illumination_types.index(illuminant)  
        img_path = join(r"C:\Users\adm\Documents\IITP\dng", str(illuminant_index + 1) + '_' + illuminant + ".dng")
        json_path = join(r'C:\Users\adm\Documents\IITP\png_targed', str(illuminant_index + 1) + '_' + illuminant +'.jpg.json')

        img = process(img_path).astype(np.float32)
        # img_max = np.quantile(img, 0.99)
        
        color_per_region, variance_per_region = process_markup(json_path, img)
        # cc_keys = [str(i) for i in range(1, 25)]
        # return np.asarray([color_per_region[key] for key in cc_keys])
        # carray = carray.reshape((6, 4, 3))
        
        # plt.imshow(img / img_max)
        # plt.show()
        # return carray / img_max
        # plt.imshow(carray / img_max)
        # plt.show()

        P[patches_number * illuminant_index:patches_number * illuminant_index + patches_number] = \
                        [color_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
        variances[patches_number * illuminant_index:patches_number * illuminant_index + patches_number] = \
                        [variance_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    return P, variances

    # exposures_number = 6
    # P = np.zeros(shape=(patches_number * illuminants_number, 3, exposures_number))
    # variances = np.zeros(shape=(patches_number * illuminants_number, 3, exposures_number))

    # process  = DNGProcessingDemo()
    # illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC'][:illuminants_number]

    # for illuminant in illumination_types:
    #     illuminant_index = illumination_types.index(illuminant)
        
    #     for exp in range(exposures_number):
    #         img_path = join(r"C:\Users\adm\Documents\IITP\dng_D50", "img_" + str(8332 + exp) + ".dng")
    #         json_path = join(r"C:\Users\adm\Documents\IITP\D50_targed", "img_" + str(8332 + exp) + ".jpg.json")
    #         img = process(img_path).astype(np.float32)
            
    #         color_per_region, variance_per_region = process_markup(json_path, img)


    #         # img_max = np.quantile(img, 0.99)
    #         # cc_keys = [str(i) for i in range(1, 25)]
    #         # carray = np.asarray([color_per_region[key] for key in cc_keys])
    #         # carray = carray.reshape((6, 4, 3))
    #         # plt.imshow(img / img_max)
    #         # plt.show()
    #         # plt.imshow(carray / img_max)
    #         # plt.show()
            
    #         P[patches_number * illuminant_index:patches_number * illuminant_index + patches_number, : , exp] = \
    #             [color_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    #         variances[patches_number * illuminant_index:patches_number * illuminant_index + patches_number, : , exp] = \
    #             [variance_per_region[str(patch_index + 1)] for patch_index in range(patches_number)]
    # return P, variances


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
  

def plot_pictures(C, learning_sample, sensitivities_gt, simulated=False):
    if simulated:
        P_learning = C @ sensitivities_gt
        # noise = np.random.normal(0,.0001, P_learning.shape)
        # P_learning += noise
    else:
        P = cv2.imread('means.tiff',  cv2.IMREAD_UNCHANGED)
        # P, variances = choose_best_stimuls(P, variances)
        # P = P[:,:, 3]
        P_learning = np.array([P[patch] for patch in learning_sample])
    
    sensitivities = inv((C.T @ C).astype(float)) @ C.T @ P_learning

    plot_sens(wavelengths, sensitivities, sensitivities_gt, show=True)
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
            

# def imread(path, out_dtype: np.floating = None, linearize_srgb=False) -> np.ndarray:
#     """
#     Return image data from file as np.ndarray.

#     Parameters
#     ----------
#     path : Union[PathLike, str]
#         Path to image file.
#     out_dtype : np.floating, optional
#         The np.dtype of the returned data.
#         It is assumed to be np.floating.
#         Returned image dtype will be changed to out_dtype using `uint2real` function.
#         If out_dtype is None or image file is tiff file, data returns as is.
#         By default None.
#     linearize_srgb : bool, optional
#         If out_dtype is not None and linearize_srgb is True, the linearization of the pixel value will be performed (see `srgb2lin_rgb` function).
#         By default False.

#     Returns
#     -------
#     np.ndarray
#         Returned image data.

#     Raises
#     ------
#     RuntimeError
#         If image file does not exist.
#     NotImplemented
#         If image file format is one of the raw formats (see `raw_suffixes`).
#     TypeError
#         If image file is not tiff or raw file and image dtype is not unsignedinteger.
#     TypeError
#         If image file is not tiff or raw file and out_dtype is not None.
#     """
#     if not path.is_file():
#         raise RuntimeError(f'File {path} does not exist!')
    
#     if path.suffix.lower() in raw_suffixes:
#         raise NotImplemented(
#             f'Reading raw formats {raw_suffixes} is not implemented!')
#     elif path.suffix.lower() in tiff_suffixes:
#         return tifffile.imread(str(path))
#     else:
#         img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#         if len(img.shape) >= 3:
#             img = swap_rb(img)

#         if out_dtype is None:
#             return img

#         if not np.issubdtype(img.dtype, np.unsignedinteger):
#             raise TypeError(
#                 f'Unsupported img.dtype={img.dtype}, dtype should be unsignedinteger!')

#         # integer to real image
#         if not np.issubdtype(out_dtype, np.floating):
#             raise TypeError(
#                 f'Unsupported out_dtype={out_dtype}, dtype should be floating!')

#         img = uint2real(img, out_dtype=out_dtype)
#         if linearize_srgb:
#             # tranform from sRGB to linear sRGB
#             img = srgb2lin_rgb(img)
#         return img


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

E_df = pd.read_excel('LampSpectra.xls', sheet_name='LampsSpectra', skiprows=2)
R_df = pd.read_excel('CCC_Reflectance_1.xls', sheet_name=1, skiprows=4, header=0)
R_internet = R_internet_matrix(pd.read_excel('24_spectras.xlsx'), patches_number, wavelengths)
sensitivities_df = pd.read_excel('canon600d.xlsx', sheet_name='Worksheet')
channels = list((sensitivities_df.drop(columns='wavelength')).columns)
sensitivities_gt = get_sensitivities_gt(wavelengths, sensitivities_df)

learning_sample, patches = choose_learning_sample(patches_number, choosed_patches_number, illuminants_number, \
    valid, achromatic_single, ratio=1.)
#print(len(learning_sample))

R = reflectances_matrix(R_df, patches_number, wavelengths)
spectras_Alexander = spectras_matrix(learning_sample, wavelengths, illuminants_number, E_df, R)
spectras_internet = spectras_matrix(learning_sample, wavelengths, illuminants_number, E_df, R_internet)

# P_measured, variances = measure_stimuli()
P_gt = spectras_Alexander @ sensitivities_gt
P_measured = cv2.imread('means.tiff',  cv2.IMREAD_UNCHANGED)
variances = cv2.imread('errors.tiff',  cv2.IMREAD_UNCHANGED)
norm_val = np.max(P_measured[-1], axis=0)
# P_measured /= norm_val
variances /= norm_val
#for i in range(6):
    #print(P_measured[:,:, i])
    #draw_colorchecker(P_measured[:,:, i], 24,show=True)

# P_measured, variances = choose_best_stimuls(P_measured, variances)

draw_colorchecker(P_measured, patches_number, show=True)

# check_stimuls_accuracy(P_measured, variances)

P_learning = np.array([P_measured[patch] for patch in learning_sample])
sensitivities = inv((spectras_Alexander.T @ spectras_Alexander).astype(float)) @ spectras_Alexander.T @ P_learning

reg_sensitivities = regularization(wavelengths, channels, sensitivities, spectras_Alexander, P_learning)

plot_pictures(spectras_Alexander, learning_sample, sensitivities_gt, simulated=True)
plot_sens(wavelengths, reg_sensitivities, sensitivities_gt, show=True)


###############################

# write_to_excel('Sensitivities.xlsx', sensitivities, learning_sample)

# optimal_parameter = [0.5067055579111499, 0.6430519533349813, 0.4257159707254087]
# reg_sensitivities = easy_regularization(spectras_Alexander, P_measured, optimal_parameter)


# The number of points in lambda grid shouldn't exceed the length of the learning sample.