'''
utils.py

Utility funcions used by various components

Created on Aug 14, 2017

@author: duolu
'''

import numpy as np
import math

import fmkit_cutils




def normalize_warping_path(a1start, a1end):

    n = a1start.shape[0]
    
    wp = (a1start + a1end) / 2
    
    xp = np.arange(n)
    
    x_100 = np.linspace(0, n - 1, 100)
    
    
    yp = np.interp(x_100, xp, wp)
    
    yp_100 = yp / yp[-1] * 99


    m = a1end[-1]

    a1start_100 = np.interp(x_100, xp, a1start)
    a1start_100 = a1start_100 / m * 99
    
    a1end_100 = np.interp(x_100, xp, a1end)
    a1end_100 = a1end_100 / m * 99

    return yp_100, a1start_100.astype(np.int32), a1end_100.astype(np.int32)

def warping_path_to_index_sequences(a1start_100, a1end_100):
    
    xs = []
    ys = []
    
    end_last = 0
    
    for i, (start, end) in enumerate(zip(a1start_100, a1end_100)):
        
        for j in range(end_last, start):

            xs.append(i)
            ys.append(j)
            
        end_last = end
        
        for j in range(start, end + 1):
            
            xs.append(i)
            ys.append(j)
    
            #print(i, j, start, end)
    
    return xs, ys

def bin(start, step, value):
    
    return ((value - start) / step).astype(np.int32)

def search_eer(frrs, fars):
    
    eer = -1
    th = -1
    
    seq = frrs > fars
    
    for i in range(len(seq) - 1):
        
        if seq[i] == True and seq[i + 1] == False:
            
            eer = (max(fars[i], frrs[i + 1]) + min(frrs[i], fars[i + 1])) / 2
            th = i
            break

    return eer, th

def search_farxk(frrs, fars, x):
    
    n = len(frrs)
    farxk = -1
    th = -1
    
    for i in range(1, n):
        
        if fars[i] > x:
            
            ratio = (x - fars[i - 1]) / (fars[i] - fars[i - 1])
            
            farxk = frrs[i - 1] - (frrs[i - 1] - frrs[i]) * ratio
            th = i
            break

    return farxk, th

def calculate_auc(frrs, fars):
    
    auc_frr = 0
    
    for i in range(len(frrs) - 1):
        
        frr = frrs[i]
        frr_next = frrs[i + 1]
        far = fars[i]
        
        # NOTE: FRR is decreasing
        auc_frr += (frr - frr_next) * far


    auc_far = 0
    
    for i in range(len(fars) - 1):
        
        far = fars[i]
        far_next = fars[i + 1]
        frr = frrs[i]
        
        # NOTE: FAR is increasing
        auc_far += (far_next - far) * frr

    return 1 - auc_frr, 1 - auc_far



def stretch_1d(array, l_new):

    l = array.shape[0]
    
    xp = np.linspace(0, l - 1, l)
    fp = array
    
    x = np.linspace(0, l - 1, l_new)
    
    array_new = np.interp(x, xp, fp)
    
    return array_new

def stretch_1ds(array, l_new):

    l = array.shape[0]
    d = array.shape[1]
    
    xp = np.linspace(0, l - 1, l)

    x = np.linspace(0, l - 1, l_new)

    array_new = np.zeros((l_new, d), np.float32)

    for i in range(d):
        
        column = array[:, i]
        column_new = np.interp(x, xp, column)
        array_new[:, i] = column_new
    
#     delta_new = l / l_new 
#     
#     for i in range(l_new):
#         
#         it = i * delta_new
#         ii = int(math.floor(it))
#         dt = it - ii
#         
#         #print(i, delta_new, it, ii)
#         
#         for j in range(d):
#             rate = array[ii + 1, j] - array[ii, j] if ii + 1 < l else array[ii, j] - array[ii - 1, j]
#             array_new[i, j] = array[ii, j] + rate * dt

    return array_new

def calculate_bayes_decision_point_gaussian(m1, m2, std1, std2):

    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)

    xs = np.roots([a, b, c])


    return xs


def calculate_estimated_bayes_error_gaussian(m1, m2, std1, std2, bdp):
    
    pass



def check_rotation_matrix(R):
    '''
    Check whether R is a valid rotation matrix.
    '''

    eps = 1e-6

    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]
    
    assert np.dot(x, y) < eps
    assert np.dot(y, z) < eps
    assert np.dot(z, x) < eps

    assert np.linalg.norm(x) - 1 < eps
    assert np.linalg.norm(y) - 1 < eps
    assert np.linalg.norm(z) - 1 < eps

    assert np.linalg.det(R) - 1 < eps




if __name__ == '__main__':
    
    
    pass















