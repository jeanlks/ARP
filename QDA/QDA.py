import math
import numpy as np
import pandas as pd


def normalizationBySd(matrix):
    result = []
    for vector in matrix:
        line = []
        mean = np.mean(vector)
        median = np.median(vector)
        for item in vector:
            val = (item - median / mean);
            line.append(val)
    result.append(line)

    return result


def normalizationByMaxMin(matrix, max, min):
    result = []
    for column in matrix:
        line = []
        for item in column:
            val = (item - min) / (max - min)
            line.append(val)
    result.append(line)
    return result


def predict(testVector, meansVector, covarianceMatrix):
    inverseMatrix = np.linalg.inv(covarianceMatrix)
    features_sub = np.subtract(testVector, meansVector)
    partial_result =  np.dot(np.dot(features_sub, inverseMatrix), np.transpose(features_sub))
    return 1/2 * np.log(covarianceMatrix) + partial_result

