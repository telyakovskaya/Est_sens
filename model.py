import numpy as np
from scipy import interpolate

def change_wavelengths(spectras, wl_in, wl_out):
    """ Interpolation of values to some set of wavelengths

    Args:
        spectras (np.ndarray): 
            s x k 
        wl_in (list): 
            all wavelengths, k
        wl_out (list): 
            desired wavelengths, m

    Returns:
        (np.ndarray): 
            s x m
    """    
    interp = interpolate.interp1d(wl_in, spectras)
    return interp(wl_out)

def simulate_stimuls(sensitivities_given: np.ndarray, spectras: np.ndarray) -> np.ndarray:
    """ This function stimulates learning stimulus

    Args:
        sensitivities_given (np.ndarray):
            Camera sensitivities  n x 3 
        spectras (np.ndarray):
            Spectral functions n x k

    Returns (np.ndarray): 
        Resulted colors k x 3
    """
    
    assert len(spectras.shape) == len(sensitivities_given.shape) == 2
    assert spectras.shape[0] == sensitivities_given.shape[0] 
    stimulus_learning = np.transpose(spectras) @ sensitivities_given
    assert stimulus_learning.shape == (spectras.shape[-1], stimulus_learning.shape[-1]), \
        f'{stimulus_learning.shape} != {(spectras.shape[-1], stimulus_learning.shape[-1])}'
    return stimulus_learning

# add asserts
def radiance_matrix(sample, illums, refl, patches_number) -> np.ndarray:
    """"
    This function calculates radiance matrix 
    Args:
        sample (list): 
            choosed pathes for learning
        illums (np.ndarray): 
            s x k, s-number of illuminations
        refl (np.ndarray): 
            n x k, k-number of wavelengths
        patches_number (int): 
            number of patches = n

    Returns:
        (np.ndarray): 
            (sn) x k
    """    
    radiance = np.zeros(shape=(len(sample), illums.shape[1]))
    radiance_current_index = 0
    illums = np.transpose(illums)
    for illuminant_index in range(illums.shape[-1]):
        e_diag = np.diag(illums[:,illuminant_index])
        refl_current = np.array([refl[patch % patches_number] for patch in sample 
                    if illuminant_index * patches_number <= patch < illuminant_index * patches_number + patches_number])
        radiance[radiance_current_index:radiance_current_index + len(refl_current)] = np.transpose(np.matmul(e_diag, np.transpose(refl_current)))
        radiance_current_index += len(refl_current)
    return radiance