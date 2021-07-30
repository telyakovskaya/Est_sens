from typing import Dict, Tuple
import pandas as pd
import numpy as np


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