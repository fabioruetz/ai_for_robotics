# IPC algorithem
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import copy
from numpy import linalg as LA
from scipy.spatial.distance import cdist

def centroid(X):
    return np.mean(X,axis=0)

def getNearestNeighbors(source, target):
    dist_matrix = cdist(source,target, metric='euclidean')
    indices = dist_matrix.argmin(axis=1)
    distance = dist_matrix[np.arange(dist_matrix.shape[0]),indices]
    mean_error = np.sum(distance) / distance.size
    return indices, mean_error

def computeBestTransformation(source, target):
    source_bar = centroid(source)
    target_bar = centroid(target)
    R_hat = computeBestRotation(source, source_bar, target, target_bar)
    t_hat = computeBestTranslation(source_bar, target_bar, R_hat)
    return getTransformationMatrix(R_hat, t_hat)

def getTransformationMatrix(R, t):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def computeBestTranslation(source_bar, target_bar, R):
    t_opt = target_bar.T - np.dot(R,source_bar.T)
    return t_opt

def computeBestRotation(source, source_bar, target, target_bar):
    R = np.eye(3)
    source_term = source - source_bar
    target_term = target - target_bar
    H = np.dot(source_term.T,target_term )
    [u,s,vt] = LA.svd(H)
    R = np.dot(vt.T, u.T)

    # Reflection Case
    if np.linalg.det(R) < 0:
        vt[2, :] *= -1
        R = np.dot(vt.T,u.T)
    return R


def icp(data_source, data_target, convergence_tolerance = 1.0e-16 , max_iter = 100):
    # Input: data_source/target nx3 or 3xn matrix, where n is the number of points in the pointclound.

    # Reformating data, Check size
    if data_source.shape[1] is 3 and data_source.shape[0] >= 4 :
        source = np.ones((4, data_source.shape[0]))
        source[0:3,:] = np.copy(data_source.T)

    elif data_source.shape[0] is 3 and data_source.shape[1] >= 4:
        source = np.ones((4, data_source.shape[1]))
        source[0:3, :] = np.copy(data_source)
    else:
        assert (False), 'Source data does not have the rigth input format'

    if data_target.shape[1] is 3 and data_target.shape[0] >= 4 :
        target = np.ones((4, data_target.shape[0]))
        target[0:3,:] = np.copy(data_target.T)

    elif data_target.shape[0] is 3 and data_target.shape[1] >= 4:
        target = np.ones((4, data_source.shape[1]))
        target[0:3, :] = np.copy(data_source)
    else:
        assert (False), 'Target data does not have the rigth input format'

    # Initialisation for initial  and final comp
    source_org = source
    previous_mean_error = 1.0e12

    for iter in range(0, max_iter):

        # Get correspondences.
        target_indices, current_mean_error = getNearestNeighbors(source[0:3,:].T, target[0:3,:].T)

        # Compute best transformation.
        T = computeBestTransformation(source[0:3,:].T,target[0:3,target_indices].T)

        # Transform the source pointcloud.
        source = np.dot(T, source)

        # Check convergence.
        if abs(previous_mean_error - current_mean_error) < convergence_tolerance:
            print ("Converged at iteration: ", iter)
            break
        else:
            previous_mean_error = current_mean_error

    T_final = computeBestTransformation(source_org[0:3, :].T, source[0:3, :].T)
    return T_final, source