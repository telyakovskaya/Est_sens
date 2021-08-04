import numpy as np
from numpy.linalg import inv
from scipy import interpolate

def change_wavelengths(spectras, wl_in, wl_out):
    """ Interpolation of values to some set of wavelengths

    Args:
        spectras (np.ndarray): s x k 
        wl_in (list): all wavelengths, k
        wl_out (list): desired wavelengths, m

    Returns:
        (np.ndarray): s x m
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

    Returns:
        np.ndarray: 
        Resulted colors k x 3
    """
    
    assert len(spectras.shape) == len(sensitivities_given.shape) == 2
    assert spectras.shape[0] == sensitivities_given.shape[0] 
    stimulus_learning = np.transpose(spectras) @ sensitivities_given
    assert stimulus_learning.shape == (spectras.shape[-1], stimulus_learning.shape[-1]), \
        f'{stimulus_learning.shape} != {(spectras.shape[-1], stimulus_learning.shape[-1])}'
    return stimulus_learning

#rename C_matrix and add asserts
def C_matrix(sample, E, R, patches_number):
    """"
    This function calculates matrix C
    Args:
        sample (list): choosed pathes for learning
        E (np.ndarray): s x k, s-number of illuminations
        R (np.ndarray): n x k, k-number of wavelengths
        patches_number (int): actually, number of patches = n

    Returns:
        [np.ndarray]: (sn) x k
    """    
    C = np.zeros(shape=(len(sample), E.shape[1]))
    C_current_index = 0
    E = np.transpose(E)
    for illuminant_index in range(E.shape[-1]):
        e_diag = np.diag(E[:,illuminant_index])
        R_current = np.array([R[patch % patches_number] for patch in sample 
                    if illuminant_index * patches_number <= patch < illuminant_index * patches_number + patches_number])
        C[C_current_index:C_current_index + len(R_current)] = np.transpose(np.matmul(e_diag, np.transpose(R_current)))
        C_current_index += len(R_current)
    #return radiance
    return C