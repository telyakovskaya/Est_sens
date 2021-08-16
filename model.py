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


def cals_radiances(refl, illums) -> np.ndarray:
    """"
    This function calculates radiance matrix from illuminances ans reflectances 
    Args:
        refls (np.ndarray): 
            refls.shape = (n, k), where n - number of reflectances
        illum (np.ndarray): 
            illums.shape = (k,), where k - a number of wavelengths 
        
    Returns:
        radiances (np.ndarray): 
            radiances.shape = (n, k) 
    """
    assert illums.shape[-1] == refl.shape[-1]
    assert len(illums.shape) == 2
    assert len(refl.shape) == 2
    radiances = np.zeros(shape=(illums.shape[0], refl.shape[0], refl.shape[-1]))
    for i in range(illums.shape[0]):
        radiances[i] = refl * illums[i]
    radiances = radiances.reshape((illums.shape[0] * refl.shape[0], refl.shape[-1]))
    return radiances    


def simulate_stimuls(sensitivities: np.ndarray, radiances: np.ndarray) -> np.ndarray:
    """ This function stimulates stimulus

    Args:
        sensitivities (np.ndarray):
            Camera sensitivities  n x 3
        radiances (np.ndarray):
            Spectral functions n x k

    Returns (np.ndarray): 
        Resulted colors k x 3
    """
    assert len(radiances.shape) == len(sensitivities.shape) == 2
    assert radiances.shape[-1] == sensitivities.shape[-1] 
    stimulus = radiances @ np.transpose(sensitivities)
    assert stimulus.shape == (radiances.shape[0], stimulus.shape[-1]), \
        f'{stimulus.shape} != {(radiances.shape[-1], stimulus.shape[-1])}'
    return stimulus
