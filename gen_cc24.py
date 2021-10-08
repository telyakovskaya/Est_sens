import numpy as np
from data import get_lambda_grid, load_sens, load_illums, load_refl
from model import cals_radiances, change_wavelengths, cals_radiances, simulate_stimuls
from pathlib import Path
from img_io import imsave
    

def estimate_cc(sens, sens_wavelengths,
                cc_refl, refl_wavelengths,
                illum, illum_wavelengths, 
                wavelengths_number):
    """
    Estimate colors of color target
    """
    wavelengths_int = get_lambda_grid(400, 720, wavelengths_number)

    sens_int = change_wavelengths(sens, sens_wavelengths, wavelengths_int)
    refl_int = change_wavelengths(cc_refl, refl_wavelengths, wavelengths_int)
    illum_int = change_wavelengths(illum, illum_wavelengths, wavelengths_int)
    radiances = cals_radiances(refl_int, illum_int)[:24]
    stimuli = simulate_stimuls(sens_int, radiances)
    return stimuli


E_dict, E_wavelengths = load_illums()
E_dict = {key: val for key, val in E_dict.items() if 'Avg' in key}
R_dict, R_wavelengths = load_refl()
R_dict = {key: val for key, val in R_dict.items() if 'Avg' in key}
E = np.asarray(list(E_dict.values()))
R = np.asarray(list(R_dict.values()))
    

sensitivities_given_dict, sens_wavelengths = load_sens()
sensitivities_given = np.asarray([sensitivities_given_dict[key] for key in ['red', 'green', 'blue']])

list_of_numbers = (1000, )

for wavelengths_number in list_of_numbers:
    cc = estimate_cc(sensitivities_given, sens_wavelengths,
                 R, R_wavelengths,
                 E, E_wavelengths,
                 wavelengths_number)
    
    cc /= cc.max()
    print(cc.shape)
    imsave(Path(r"babelcolor1000D50_24.tiff"), cc, unchanged=False, out_dtype=np.float32, gamma_correction=False)


    