from typing import Dict, Tuple
import numpy as np
import seaborn as sns

def unit(stimulus):
    return stimulus / np.linalg.norm(stimulus)

def calculate_errors(value, patches_number):
    mean_value = sum(value) / patches_number
    variance_value = np.var(value)
    return mean_value, variance_value

def make_angle(value):
    return (np.arccos(value) * 180 / np.pi)

def make_hitsplot(some_list):
    list_fig = sns.histplot(some_list).get_figure()
    return list_fig

def estimate_error_statistics(patches_number: int, stimulus: Tuple, ground_thruth: tuple) -> Tuple[Dict[float, float, Tuple, any]]:
    """This fuction calculates errors between angles and vector lengths

    Args:
        patches_number (int): actually, number of patches
        stimulus (Tuple):  total number of patches x 3
        ground_thruth (tuple):  total number of patches x 3

    Returns:
        Tuple[Dict[float, float, Tuple, any]]: information of errors between angles and vector lengths
        including average value, variance and histogram
    """    
    angles = []
    norms = []

    for i in range(patches_number):
        predicted_stimulus = stimulus[i]
        genuine_stimulus = ground_thruth[i]
        dot_product = np.dot(unit(predicted_stimulus), unit(genuine_stimulus))
        angles.append(make_angle(dot_product))
        norms.append(np.linalg.norm(predicted_stimulus - genuine_stimulus, 2))
    
    mean_angle, variance_angles = calculate_errors(angles, patches_number)
    mean_norm, variance_norms = calculate_errors(angles, patches_number)
    
    angles_dict = {'mean_value': mean_angle, 'variance': variance_angles, 'list_of_values': angles, 'fig': make_hitsplot(angles)}
    norms_dict = {'mean_value': mean_norm, 'variance': variance_norms, 'list_of_values': norms, 'fig': make_hitsplot(norms)}

    return angles_dict, norms_dict
