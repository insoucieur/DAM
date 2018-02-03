import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('smoking.txt')
type(data)
import math
from scipy.stats import t

# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return a tuple containing average FEV1 of smokers and nonsmokers 
def meanFEV1(data):
	age = data[:,0]
	FEV1 = data[:,1]
	height = data[:,2]
	gender = data[:,3]
	smoking_status = data[:,4]
	weight = data[:,5]
	nonsmoker_indices = [i for i, n in enumerate(list(smoking_status)) if n == 0]
	smoker_indices = [i for i, n in enumerate(list(smoking_status)) if n == 1]
	nonsmoker = data[nonsmoker_indices,:]
	smoker = data[smoker_indices,:]
	avg_FEV1_nonsmoker = np.mean(nonsmoker[:,1])
	avg_FEV1_smoker = np.mean(smoker[:,1])
	return (avg_FEV1_nonsmoker, avg_FEV1_smoker)

meanFEV1(data)

