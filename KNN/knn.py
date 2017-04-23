import math
import numpy as np
import pandas as pd


def euclideanDist(a,b,length):
  d = 0
  for x in range(length):
   d += pow(a[x]-b[x],2)
  d = np.sqrt(d)
  return(d)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDist(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors