import numpy as np
import math
from numpy.linalg import inv

def regularization_Tikhonov(channel, wavelengths, C, P_learning, reg_start = \
     {"red": 0.05, "green": 0.05, "blue": 0.05}, reg_stop = {"red": 5, "green": 5, "blue": 5}):
    def menger(p1, p2, p3):
        residual1, solution1 = p1
        residual2, solution2 = p2
        residual3, solution3 = p3
        p1p2 = (residual2 - residual1)**2 + (solution2 - solution1)**2
        p2p3 = (residual3 - residual2)**2 + (solution3 - solution2)**2
        p3p1 = (residual1 - residual3)**2 + (solution1 - solution3)**2
        numerator = residual1 * solution2 + residual2 * solution3 + residual3 * solution1 - \
                    residual1 * solution3 - residual3 * solution2 - residual2 * solution1
        return (2 * numerator) / (math.sqrt(p1p2 * p2p3 * p3p1))


    def l_curve_P(reg_parameter):
        sensitivities = inv((C.T @ C).astype(float) + np.identity(len(wavelengths)) * reg_parameter) @ C.T @ P_learning
        solution = (np.linalg.norm(sensitivities, 2) ** 2) 
        residual_vector = C @ sensitivities - P_learning
        residual = (np.linalg.norm(residual_vector, 2) ** 2) 
        return residual, solution


    def find_optimal_parameter(channel):
        p = {}
        reg_parameter = {}
        ch_letter = ['red', 'green', 'blue'][channel]
        reg_parameter[1], reg_parameter[4] = reg_start[ch_letter], reg_stop[ch_letter]   # search extremes
        epsilon = 0.00005                                                                # termination threshold
        phi = (1 + math.sqrt(5)) / 2                                                     # golden section
        
        reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
        reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))

        for i in range(1, 5):
            p[i] = l_curve_P(reg_parameter[i])
        
        while ((reg_parameter[4] - reg_parameter[1]) / reg_parameter[4]) >= epsilon:
            C2 = menger(p[1], p[2], p[3])
            C3 = menger(p[2], p[3], p[4])
            
            while C3 <= 0:
                reg_parameter[4] = reg_parameter[3]
                reg_parameter[3] = reg_parameter[2]
                reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                p[4] = p[3]
                p[3] = p[2]
                p[2] = l_curve_P(reg_parameter[2])
                C3 = menger(p[2], p[3], p[4])
            
            if C2 > C3:
                optimal_parameter = reg_parameter[2]
                reg_parameter[4] = reg_parameter[3]
                reg_parameter[3] = reg_parameter[2]
                reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                p[4] = p[3]
                p[3] = p[2]
                p[2] = l_curve_P(reg_parameter[2])
            else:
                optimal_parameter = reg_parameter[3]
                reg_parameter[1] = reg_parameter[2]
                reg_parameter[2] = reg_parameter[3]
                reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))
                p[1] = p[2]
                p[2] = p[3]
                p[3] = l_curve_P(reg_parameter[3])
        
        return optimal_parameter


    optimal_parameter = find_optimal_parameter(channel)
    reg_sensitivities = inv((C.T @ C).astype(float) + np.identity(len(wavelengths)) * optimal_parameter) @ C.T @ P_learning
    return reg_sensitivities.T


def regularization_derivatives(channel, wavelengths, C, P_learning, reg_start = \
     {"red": 0.05, "green": 0.05, "blue": 0.05}, reg_stop = {"red": 5, "green": 5, "blue": 5}):
    T = np.zeros(shape=(len(wavelengths),len(wavelengths)))
    for i in range(len(wavelengths)):
        for j in range(len(wavelengths)):
            if i == j:
                if i == 0 or i == len(wavelengths) - 1:
                    T[i][j] = 1
                else:
                    T[i][j] = 2
            elif i - j == 1 or j - i == 1:
                T[i][j] = -1

    def menger(p1, p2, p3):
        residual1, solution1 = p1
        residual2, solution2 = p2
        residual3, solution3 = p3
        p1p2 = (residual2 - residual1)**2 + (solution2 - solution1)**2
        p2p3 = (residual3 - residual2)**2 + (solution3 - solution2)**2
        p3p1 = (residual1 - residual3)**2 + (solution1 - solution3)**2
        numerator = residual1 * solution2 + residual2 * solution3 + residual3 * solution1 - \
                    residual1 * solution3 - residual3 * solution2 - residual2 * solution1
        return (2 * numerator) / (math.sqrt(p1p2 * p2p3 * p3p1))


    def l_curve_P(reg_parameter):
        sensitivities = inv((C.T @ C).astype(float) + (np.transpose(T) @ T) * reg_parameter) @ C.T @ P_learning
        solution = (np.linalg.norm(sensitivities, 2) ** 2) 
        residual_vector = C @ sensitivities - P_learning
        residual = (np.linalg.norm(residual_vector, 2) ** 2) 
        return residual, solution


    def find_optimal_parameter(channel):
        p = {}
        reg_parameter = {}
        ch_letter = ['red', 'green', 'blue'][channel]
        reg_parameter[1], reg_parameter[4] = reg_start[ch_letter], reg_stop[ch_letter]   # search extremes
        epsilon = 0.00005                                                                # termination threshold
        phi = (1 + math.sqrt(5)) / 2                                                     # golden section
        
        reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
        reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))

        for i in range(1, 5):
            p[i] = l_curve_P(reg_parameter[i])
        
        while ((reg_parameter[4] - reg_parameter[1]) / reg_parameter[4]) >= epsilon:
            C2 = menger(p[1], p[2], p[3])
            C3 = menger(p[2], p[3], p[4])
            
            while C3 <= 0:
                reg_parameter[4] = reg_parameter[3]
                reg_parameter[3] = reg_parameter[2]
                reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                p[4] = p[3]
                p[3] = p[2]
                p[2] = l_curve_P(reg_parameter[2])
                C3 = menger(p[2], p[3], p[4])
            
            if C2 > C3:
                optimal_parameter = reg_parameter[2]
                reg_parameter[4] = reg_parameter[3]
                reg_parameter[3] = reg_parameter[2]
                reg_parameter[2] = 10 ** ((math.log10(reg_parameter[4]) + phi * math.log10(reg_parameter[1])) / (1 + phi))
                p[4] = p[3]
                p[3] = p[2]
                p[2] = l_curve_P(reg_parameter[2])
            else:
                optimal_parameter = reg_parameter[3]
                reg_parameter[1] = reg_parameter[2]
                reg_parameter[2] = reg_parameter[3]
                reg_parameter[3] = 10 ** (math.log10(reg_parameter[1]) + math.log10(reg_parameter[4]) - math.log10(reg_parameter[2]))
                p[1] = p[2]
                p[2] = p[3]
                p[3] = l_curve_P(reg_parameter[3])
        
        return optimal_parameter


    optimal_parameter = find_optimal_parameter(channel)
    reg_sensitivities = inv((C.T @ C).astype(float) + (np.transpose(T) @ T) * optimal_parameter) @ C.T @ P_learning
    return reg_sensitivities.T



def easy_regularization(wavelengths, C, P, optimal_parameter):
    reg_sensitivities = np.zeros(shape=(len(wavelengths), 3))
    reg_sensitivities = inv((C.T @ C).astype(float) + np.identity(len(wavelengths)) \
            * optimal_parameter) @ C.T @ P
    return reg_sensitivities