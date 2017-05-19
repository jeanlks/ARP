import math
import numpy as np
import pandas as pd
import scipy.linalg as linalg


def normalizationBySd(matrix):
    result = [[]]
    for column in matrix:
        line = []
        mean = np.mean(column)
        median = np.median(column)
        for item in column:
            val = (item - median/mean)
            line.append(val)
            valResult = np.vstack([line])
        result.append(valResult)
    return (result)



def normalizationByMaxMin(matrix, max, min):
    result =[]
    for column in matrix:
        line = []
        for item in column:
            val = (item - min) / (max-min)
            line.append(val)
    result.append(line)
    return result
        
    




def predict(testVector,meansVector,covarianceMatrix):
    # expTerm = (1.0 / 2.0) * np.dot(np.dot(np.transpose((testVector - meansVector)), linalg.inv(covarianceMatrix)),(testVector - meansVector))
    # priorProb = 1.0 / 5.0
    # pdf = priorProb / (2 * math.pi * math.sqrt(linalg.det(covarianceMatrix))) * math.exp(-expTerm)
    pdf = np.subtract(testVector,meansVector)
    return pdf
