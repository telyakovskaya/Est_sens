import numpy as np
from numpy.linalg import inv

def simulate_stimuls(sensitivities_given: np.ndarray, spectras: np.ndarray) -> np.ndarray:
    """[summary]

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
    # print(spectras.shape)
    # print(sensitivities_given.shape)
    assert stimulus_learning.shape == (spectras.shape[-1], stimulus_learning.shape[-1]), \
        f'{stimulus_learning.shape} != {(spectras.shape[-1], stimulus_learning.shape[-1])}'
    return stimulus_learning

def estimate_sensitivities(spectras: np.ndarray, stimulus: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        spectras (np.ndarray): 
            n x k
        stimulus (np.ndarray): 
            k x 3

    Returns:
        np.ndarray: 
            n x 3
    """
    H = inv((spectras @ spectras.T).astype(float)) @ spectras
    assert H.shape == (spectras.shape[0], stimulus.shape[0])
    sensitivities =  H @ stimulus
    return sensitivities