import math
import numpy as np
import pandas as pd

def gaussian(value,mean, variance):
    exp = np.exp(-(((value - mean) ** 2) / (2 * variance)))
    div = np.sqrt(2 * np.pi * variance)
    return (1 / div) * exp

# ==============================================================================
# def gaussian_multivariate(covariance,mean,value)
#     exp = np.exp(-1/2(value-mean))
#     div = np.sqrt(2 * np.pi * (covariance ** 1/2)
#     return (1 / div) * exp
#
# ==============================================================================

def normalizationBySd(matrix): 
    for column in matrix:
        mean = np.mean(column)
        median = np.median(column)
        for item in column:
            item = item-median/mean
            
    return matrix


def normalizationByMaxMin(matrix, max, min):
    for column in matrix:
        for item in column:
            item = item - min / (max-min)

    return matrix
        
        
        
 