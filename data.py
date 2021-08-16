from typing import Dict, Tuple
import pandas as pd
import numpy as np
from scipy import interpolate
import random
### The following functions load data from Excel files

#to delete after

def get_lambda_grid(start, stop, points_number):
    # step = (stop - start) / points_number
    # return [start + point * step for point in range(points_number)]
    return np.linspace(start, stop, num=points_number, endpoint=True)

def R_internet_matrix(R_df, patches_number, wavelengths):
    R = np.zeros(shape=(patches_number, len(wavelengths)))
    x = R_df['wavelengths']
    for patch in range(patches_number):
        y = R_df[patch]
        R_interpolated = interpolate.interp1d(x, y)
        R[patch] = R_interpolated(wavelengths)
    R /= R.max(axis=0)
    return R


def get_sensitivities_gt(wavelengths, sensitivities_df):
    wavelengths = get_lambda_grid(400, 721, 10)
    sensitivities_given = np.zeros(shape=(len(wavelengths), 3))
    x = sensitivities_df['wavelength']
    for i, channel in enumerate(['red', 'green', 'blue']):
        y = sensitivities_df[channel]
        sensitivities_interpolated = interpolate.interp1d(x, y)
        sensitivities_given[:, i] = sensitivities_interpolated(wavelengths)
    return sensitivities_given

def spectras_matrix(learning_sample, wavelengths, illuminants_number, E_df, R):
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


def choose_learning_sample(patches_number, choosed_patches_number, illuminants_number, valid, achromatic_single, ratio=0.8):
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



def reflectances_matrix(R_df, patches_number, wavelengths):
    R = np.zeros(shape=(patches_number, len(wavelengths)))
    x = R_df['Lambda grid']
    for patch in range(patches_number):
        y = R_df[str(patch + 1) + 'Avg']
        R_interpolated = interpolate.interp1d(x, y)
        R[patch] = R_interpolated(wavelengths)
    R /= R.max(axis=0)
    return R
    ####to delete


def excl_read(excl_fname: str, **kwargs):
    return pd.read_excel(excl_fname, **kwargs)

def load_spectral_data(excl_fname: str, wavelengths_name='wavelength', sheet_name=None, skiprows=None, header=0)  -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    spectra_df = excl_read(excl_fname, sheet_name=sheet_name, skiprows=skiprows, header=header)
    wavelengths = spectra_df[wavelengths_name]
    spectras_dict = {key: np.asarray(list(data.values())) for key, data in spectra_df.drop(columns=wavelengths_name).to_dict().items()}
    return spectras_dict, wavelengths

def load_illums(excl_fname: str = 'LampSpectra.xls', wavelengths_name='Lambda grid', sheet_name='LampsSpectra', skiprows=2):
    return load_spectral_data(excl_fname, wavelengths_name, sheet_name=sheet_name, skiprows=skiprows)

def load_refl(excl_fname: str = 'CCC_Reflectance_1.xls', wavelengths_name='Lambda grid', sheet_name=1, skiprows=4, header=0):
    return load_spectral_data(excl_fname, wavelengths_name, sheet_name=sheet_name, skiprows=skiprows, header=header)

def load_sens(excl_fname: str = 'canon600d.xlsx', wavelengths_name='wavelength', sheet_name='Worksheet'):
    return load_spectral_data(excl_fname, wavelengths_name, sheet_name=sheet_name)




if __name__=='__main__':
    def load_example():
        illums, wls = load_illums()
        print(wls)
        print(illums)

        refl, wls = load_refl()
        print(wls)
        print(refl)

        sens, wls = load_sens()
        print(wls)
        print(sens)

    load_example()