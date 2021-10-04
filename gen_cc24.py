# import numpy as np
# import pandas as pd
# import statistics
# from math import sqrt
# from data import get_lambda_grid, load_sens, load_illums, load_refl
# from model import cals_radiances, change_wavelengths, cals_radiances, simulate_stimuls
# from plot import draw_colorchecker, error_heatmap
# import matplotlib.pyplot as plt
# from pathlib import Path
# from img_io import imsave
# from data2 import get_sensitivities_gt, reflectances_matrix
 


# def estimate_cc(sens, sens_wavelengths, cc_refl, refl_wavelengths,
#                 illum, illum_wavelengths, 
#                 wavelengths_number):
#     wavelengths_int = get_lambda_grid(401, 720, wavelengths_number)
#     sens_int = change_wavelengths(sens, sens_wavelengths, wavelengths_int)
#     refl_int = change_wavelengths(cc_refl, refl_wavelengths, wavelengths_int)
#     illum_int = change_wavelengths(illum, illum_wavelengths, wavelengths_int)
    

#     radiances = cals_radiances(refl_int, illum_int)
#     stimuli = simulate_stimuls(sens_int, radiances)
#     return stimuli
# # shape = (4,6)
# # number_of_patches = shape[0] * shape[1]

# E_dict, E_wavelengths = load_illums()
# E_dict = {key: val for key, val in E_dict.items() if 'Avg' in key}
# R_dict, R_wavelengths = load_refl()
# R_dict = {key: val for key, val in R_dict.items() if 'Avg' in key}

# E = np.asarray(list(E_dict.values()))
# R = np.asarray(list(R_dict.values()))

# sensitivities_given_dict, sens_wavelengths = load_sens()
# sensitivities_given = np.asarray([sensitivities_given_dict[key] for key in ['red', 'green', 'blue']])



# # list_of_numbers = (5, 20, 32, 50, 100, 300, 500, 1000)
# list_of_numbers = (1000, )

# illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC']

# for wavelengths_number in list_of_numbers:
#     cc = estimate_cc(sensitivities_given, sens_wavelengths,
#                  R, R_wavelengths,
#                  E, E_wavelengths,
#                  wavelengths_number)
    
    
#     cc /= cc.max()
#     print(cc.shape)
#     outpath_tiff = Path('babelcolor' + str(wavelengths_number) + 'D50_24_new' '.tiff')
#     imsave(outpath_tiff, cc, unchanged=False, out_dtype=np.float32, gamma_correction=False)


import numpy as np
import pandas as pd
import statistics
import seaborn.apionly as sns
from math import sqrt
from data import get_lambda_grid, load_sens, load_illums, load_refl
from model import cals_radiances, change_wavelengths, cals_radiances, simulate_stimuls
from plot import draw_colorchecker, error_heatmap
import matplotlib.pyplot as plt
from pathlib import Path
from img_io import imsave
from data2 import get_sensitivities_gt, reflectances_matrix

#def simulate_noise(cc, alpha, beta):
def simulate_noise(cc, alpha, beta, number_of_repetitions):
    res_list =[]
    for k in range(number_of_repetitions):
        cc_std = np.sqrt(alpha * cc + beta)
        cc_new = cc + np.random.normal(0, cc_std) 
        res_list.append(cc_new)
    res_array = np.array(res_list)
    return res_array
    


def estimate_cc(sens, sens_wavelengths, cc_refl, refl_wavelengths,
                illum, illum_wavelengths, 
                wavelengths_number):
    wavelengths_int = get_lambda_grid(401, 720, wavelengths_number)
    sens_int = change_wavelengths(sens, sens_wavelengths, wavelengths_int)
    refl_int = change_wavelengths(cc_refl, refl_wavelengths, wavelengths_int)
    illum_int = change_wavelengths(illum, illum_wavelengths, wavelengths_int)
    

    radiances = cals_radiances(refl_int, illum_int)
    stimuli = simulate_stimuls(sens_int, radiances)
    return stimuli
shape = (10, 14)
number_of_patches = shape[0] * shape[1]

E_dict, E_wavelengths = load_illums()
E_dict = {key: val for key, val in E_dict.items() if 'Avg' in key}
R_dict, R_wavelengths = load_refl()
R_dict = {key: val for key, val in R_dict.items() if 'Avg' in key}

E = np.asarray(list(E_dict.values()))
R = np.asarray(list(R_dict.values()))

sensitivities_given_dict, sens_wavelengths = load_sens()
sensitivities_given = np.asarray([sensitivities_given_dict[key] for key in ['red', 'green', 'blue']])



# list_of_numbers = (5, 20, 32, 50, 100, 300, 500, 1000)
list_of_numbers = (1000, )

illumination_types = ['D50', 'D50+CC1', 'D50+OC6', 'LED', 'LUM', 'INC']

alpha = 0.0005
beta = 0.0
number_of_repetitions = 10000

# outdir = Path(r'C:\Users\Пользователь\Documents\imgs') / f'no_noise_{number_of_repetitions}'
# outdir.mkdir(parents=True, exist_ok=True)

for wavelengths_number in list_of_numbers:
    cc = estimate_cc(sensitivities_given, sens_wavelengths,
                 R, R_wavelengths,
                 E, E_wavelengths,
                 wavelengths_number)
    
    
    cc /= cc.max()
    # for c in range(3):
    #     plt.hist(cc[..., c].reshape(-1))
    # plt.show()
    # plt.close()
    print(cc.shape)
    outpath_tiff = Path('babelcolor_24_' + str(wavelengths_number) + '.tiff')
    #outpath_tiff = outdir / '.'.join(['img', str(wavelengths_number), 'tiff'])
    imsave(outpath_tiff, cc, unchanged=False, out_dtype=np.float32, gamma_correction=False)

