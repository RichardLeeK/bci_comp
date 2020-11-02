# Common Spatial Pattern implementation in Python, used to build spatial filters for identifying task-related activity.
import numpy as np
import scipy.linalg as la

# CSP takes any number of arguments, but each argument must be a collection of trials associated with a task
# That is, for N tasks, N arrays are passed to CSP each with dimensionality (# of trials of task N) x (feature vector)
# Trials may be of any dimension, provided that each trial for each task has the same dimensionality,
# otherwise there can be no spatial filtering since the trials cannot be compared
def CSP(tasks):
    filters = ()
    iterator = range(0, len(tasks))
    for x in iterator:
        rx = covarianceMatrix(tasks[x][0])
        for t in range(1, len(tasks[x])):
            rx += covarianceMatrix(tasks[x][t])
        rx = rx / len(tasks[x])
        count = 0
        not_rx = rx * 0
        for not_x in [element for element in iterator if element != x]:
            for t in range(0, len(tasks[not_x])):
                not_rx = covarianceMatrix(tasks[not_x][t])
                count += 1
        not_rx = not_rx / count
        sfx = spatialFilter(rx, not_rx)
        filters += (sfx,)
        
        if len(tasks) == 2:
            filters += (spatialFilter(not_rx, rx),)
            break
    return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	E,U = la.eig(R)

	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
	ord = np.argsort(E)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	E = E[ord]
	U = U[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

	# Find and sort the generalized eigenvalues and eigenvector
	E1,U1 = la.eig(Sa,Sb)
	ord1 = np.argsort(E1)
	ord1 = ord1[::-1]
	E1 = E1[ord1]
	U1 = U1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(np.transpose(U1),P)
	return SFa.astype(np.float32)