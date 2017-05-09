import math
import numpy as np
import pandas as pd
import scipy.linalg as linalg


def normalizationBySd(matrix):
    result = []
    for i in range(len(matrix)):
        line = []
        mean = np.mean(matrix[i])
        median = np.median(matrix[i])
        for j in range(len(matrix[i])):
            val = (matrix[i][j] - median/mean);
            line.append(val)
    result = np.column_stack(line)
            
    return result



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
    expTerm = (1.0 / 2.0) * np.dot(np.dot(np.transpose((testVector - meansVector)), linalg.inv(covarianceMatrix)),(testVector - meansVector))
    priorProb = 1.0 / 5.0
    pdf = priorProb / (2 * math.pi * math.sqrt(linalg.det(covarianceMatrix))) * math.exp(-expTerm)
    return pdf
