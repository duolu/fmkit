'''
utils.py

Utility funcions used by various components

Created on Aug 14, 2017

@author: duolu
'''

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math

import fmcode_cutils





def euclidean_distance(point1, point2):
    
    return np.linalg.norm(point1 - point2)
    
    

def dtw_c(data1, data2, l1, l2, window = -1, penalty = 0):
    
    '''
    Dynamic Time warping implemented in C (see ../lib/fmcode_cuitls.c).
    '''
    
    #return fmcode_cutils.dtw_c(data1, data2, data1.shape[1], l1, l2, window, penalty)

    dists = np.zeros((l1 + 1, l2 + 1), np.float32)
    direction = np.zeros((l1 + 1, l2 + 1), np.int32)
     
    # to align data2 to data1
    a1start = np.zeros(l1, np.int32)
    a1end = np.zeros(l1, np.int32)
 
    # to align data1 to data2
    a2start = np.zeros(l2, np.int32)
    a2end = np.zeros(l2, np.int32)
 
    data2_aligned = np.zeros(data1.shape, np.float32)
 
    
    dist = fmcode_cutils.dtw_c(data1, data2, data1.shape[1], l1, l2, window, penalty,
        dists, direction, a1start, a1end, a2start, a2end, data2_aligned)
    
    return dist, dists, direction, a1start, a1end, a2start, a2end, data2_aligned

def dtw(data1, data2, l1, l2, window = -1, penalty = 0, distance = euclidean_distance):

    ''' 
    Dynamic Time Warping
    '''

    dists = np.zeros((l1 + 1, l2 + 1), np.float32)
    direction = np.zeros((l1 + 1, l2 + 1), np.int32)
    
    # to align data2 to data1
    a1start = np.zeros(l1, np.int32)
    a1end = np.zeros(l1, np.int32)

    # to align data1 to data2
    a2start = np.zeros(l2, np.int32)
    a2end = np.zeros(l2, np.int32)

    # initialization
    dists.fill(1e6)
    
    dists[0][0] = 0
    direction[0][0] = 0
    
    if window == -1:
        window = l2 * 2
    
    # find the warping path
    for i in range(1, l1 + 1):
        
        jj = int(float(l2) / l1 * i)
        start = jj - window if jj - window > 1 else 1
        end = jj + window if jj + window < l2 + 1 else l2 + 1
        
        for j in range(start, end):
            
            # CAUTION: data1[0] and data2[0] mapps to dists[1][1],
            #          and i, j here are indexing dists instead of data1 or data2,
            #          i.e., dists[i][j] is comparing data1[i - 1] and data2[j - 1]
            cost = distance(data1[i - 1], data2[j - 1])

            min_dist = dists[i - 1, j - 1]
            direction[i][j] = 1 # 1 stands for diagonal
            
            if dists[i - 1, j] + penalty < min_dist:
                
                min_dist = dists[i - 1, j] + penalty
                direction[i][j] = 2 # 2 stands for the i direction
            
            if dists[i][j - 1] + penalty < min_dist:
                
                min_dist = dists[i][j - 1] + penalty
                direction[i][j] = 4 # 4 stands for the j direction
                
            dists[i][j] = cost + min_dist

    # trace back the warping path to find element-wise mapping

    #print('warping path done')

    a1start[l1 - 1] = l2 - 1
    a1end[l1 - 1] = l2 - 1
    a2start[l2 - 1] = l1 - 1
    a2end[l2 - 1] = l1 - 1

    i = l1
    j = l2
    while True:

        if direction[i][j] == 2: # the i direction

            i -= 1
                                
            a1start[i - 1] = j - 1
            a1end[i - 1] = j - 1
            a2start[j - 1] = i - 1

        elif direction[i][j] == 4: # the j direction

            j -= 1
                
            a2start[j - 1] = i - 1
            a2end[j - 1] = i - 1
            a1start[i - 1] = j - 1

        elif direction[i][j] == 1: # the diagonal direction

            i -= 1
            j -= 1
            if i == 0 and j == 0:
                break
            
            a1start[i - 1] = j - 1
            a1end[i - 1] = j - 1
            a2start[j - 1] = i - 1
            a2end[j - 1] = i - 1
        
        else: # direction[i][j] == 0, the corner
            break


    d = data1.shape[1]
    data_new = np.zeros((l1, d), data1.dtype)
    
    for i, jj, kk in zip(range(l1), a1start, a1end):
        
        if jj == kk:
            
            data_new[i] = data2[jj]
            
        else:
            
            data_new[i] = np.mean(data2[jj:kk + 1,:], axis = 0)



    return dists[l1][l2], dists, direction, a1start, a1end, a2start, a2end, data_new



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



if __name__ == '__main__':
    
    
    m1 = 0.3
    std1 = 1.0
    m2 = 0.7
    std2 = 1.0
    
    result = calculate_bayes_decision_point_gaussian(m1, m2, std1, std2)
    
    print(result)
    
    x = np.linspace(-5,9,1000)
    plot1=plt.plot(x,mlab.normpdf(x,m1,std1))
    plot2=plt.plot(x,mlab.normpdf(x,m2,std2))
    plot3=plt.plot(result,mlab.normpdf(result,m1,std1),'o') 

    plt.show()
















