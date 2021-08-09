from typing import Dict, Tuple
from PIL.Image import LIBIMAGEQUANT
import pandas as pd
import numpy as np
from scipy import interpolate
import random

### The following functions load data from Excel files


def get_lambda_grid(start, stop, points_number):
    step = (stop - start) / points_number
    return [start + point * step for point in range(points_number)]


def choose_learning_sample(valid, ratio=0.8, patches_number=24, illuminants_number=1, choosed_patches_number=24):
    learning_sample = {}
    for channel in range(3):
        all_potential = [patch for patch in valid[channel]]
        if choosed_patches_number > len(valid[channel]):
            learning_number = int(ratio * len(valid[channel]))
        else:
            learning_number = int(ratio * choosed_patches_number)

        for i in range(illuminants_number):
            potential = [patch for patch in all_potential if i * patches_number <= patch < i * patches_number + patches_number]
            learning_sample[channel] = sorted(random.sample(potential, k=learning_number))

    return learning_sample


def get_sensitivities_gt(wavelengths, sensitivities_df):
    sensitivities_given = np.zeros(shape=(len(wavelengths), 3))
    x = sensitivities_df['wavelength']
    for i, channel in enumerate(['red', 'green', 'blue']):
        y = sensitivities_df[channel]
        sensitivities_interpolated = interpolate.interp1d(x, y)
        sensitivities_given[:, i] = sensitivities_interpolated(wavelengths)
    return sensitivities_given



def spectras_matrix(learning_sample, wavelengths, E_df, R_df, patches_number=24, illuminants_number=1):
    R = reflectances_matrix(R_df, patches_number, wavelengths)
    C = np.zeros(shape=(len(learning_sample), len(wavelengths)))
    C_current_index = 0

    x = E_df['Lambda grid']
    for illuminant_index in range(illuminants_number):
        y = E_df[str(1 + illuminant_index) + 'Norm']
        E_interpolated=interpolate.interp1d(x, y)
        E = np.diag(E_interpolated(wavelengths))
        R_learning = [R[patch % patches_number] for patch in learning_sample if illuminant_index * patches_number \
            <= patch < illuminant_index * patches_number + patches_number]
        C[C_current_index:C_current_index + len(R_learning)] = np.transpose(E @ np.transpose(R_learning))
        C_current_index += len(R_learning)
    
    return C



def reflectances_matrix(R_df, patches_number, wavelengths):
    R = np.zeros(shape=(patches_number, len(wavelengths)))
    x = R_df['Lambda grid']
    for patch in range(patches_number):
        y = R_df[str(patch + 1) + 'Avg']
        R_interpolated = interpolate.interp1d(x, y)
        R[patch] = R_interpolated(wavelengths)
    R /= R.max(axis=0)
    return R


def R_internet_matrix(R_df, patches_number, wavelengths):
    R = np.zeros(shape=(patches_number, len(wavelengths)))
    x = R_df['wavelengths']
    for patch in range(patches_number):
        y = R_df[patch]
        R_interpolated = interpolate.interp1d(x, y)
        R[patch] = R_interpolated(wavelengths)
    R /= R.max(axis=0)
    return R


def excl_read(excl_fname: str, **kwargs):
    return pd.read_excel(excl_fname, **kwargs)

def load_spectral_data(excl_fname: str, wavelengths_name='wavelength')  -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    spectra_df = excl_read(excl_fname, sheet_name='Worksheet')
    wavelengths = spectra_df[wavelengths_name]
    spectras_dict = {key: np.asarray(list(data.values())) for key, data in spectra_df.drop(columns=wavelengths_name).to_dict().items()}
    return spectras_dict, wavelengths

def load_illums(excl_fname: str = 'illuminances_std.xlsx', wavelengths_name='wavelength'):
    return load_spectral_data(excl_fname, wavelengths_name)

def load_refl(excl_fname: str = 'babelcolor.xlsx', wavelengths_name='wavelength'):
    return load_spectral_data(excl_fname, wavelengths_name)

def load_sens(excl_fname: str = 'canon600d.xlsx', wavelengths_name='wavelength'):
    return load_spectral_data(excl_fname, wavelengths_name)


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