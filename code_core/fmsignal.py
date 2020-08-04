'''
This file is part of the fmkit framework, which is designed to facilitate
researches on in-air-handwriting related research.

Author: Duo Lu <duolu.cs@gmail.com>

Version: 0.1
License: GPLv3

Updated on Feb. 7, 2020, version 0.1

Created on Aug 14, 2017, draft



* leap, fix posture normalization, p0, p1, p2
* leap, fix z up and z down
* leap and glove, fix amplitude normalization





'''


import time
import csv
import math

import numpy as np
import scipy.stats
import scipy.signal

from pyrotation import Quaternion
from pyrotation import euler_zyx_to_rotation_matrix
from pyrotation import rotation_matrix_to_euler_angles_zyx
from pyrotation import rotation_matrix_to_angle_axis
from pyrotation import normalize_rotation_matrix

try:
    import fmkit_cutils
    DTW_method = 'c'
    #print('fmkit_cutils installed')
except ImportError:
    DTW_method = 'py'
    #print('fmkit_cutils not installed')



def dtw(data1, data2, l1, l2, window = 0, penalty = 0):

    '''Dynamic Time Warping (DTW) on two arraies.
    
    @param data1: the first array, l1-by-d numpy ndarray.
    @param data2: the second array, l2-by-d numpy ndarray.
    @param window: window constraint, by default no constraint.
    @param penalty: misalign penalty, by default zero.
    
    CAUTION: This is the python implementation. It iteratively accesses each
    element of the array, which is rather slow.
    
    '''

    assert len(data1.shape) == 2 and len(data2.shape) == 2
    assert data1.shape[1] == data2.shape[1]

    dists = np.zeros((l1 + 1, l2 + 1), np.float32)
    direction = np.zeros((l1 + 1, l2 + 1), np.int32)
    
    # index mapping to align data2 to data1
    a1start = np.zeros(l1, np.int32)
    a1end = np.zeros(l1, np.int32)

    # index mapping to align data1 to data2
    a2start = np.zeros(l2, np.int32)
    a2end = np.zeros(l2, np.int32)

    # initialization
    dists.fill(1e6)
    
    dists[0, 0] = 0
    direction[0, 0] = 0
    
    if window <= 0:
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
            cost = np.linalg.norm(data1[i - 1] - data2[j - 1])

            min_dist = dists[i - 1, j - 1]
            direction[i, j] = 1 # 1 stands for diagonal
            
            if dists[i - 1, j] + penalty < min_dist:
                
                min_dist = dists[i - 1, j] + penalty
                direction[i, j] = 2 # 2 stands for the i direction
            
            if dists[i][j - 1] + penalty < min_dist:
                
                min_dist = dists[i][j - 1] + penalty
                direction[i, j] = 4 # 4 stands for the j direction
                
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

        if direction[i, j] == 2: # the i direction

            i -= 1
                                
            a1start[i - 1] = j - 1
            a1end[i - 1] = j - 1
            a2start[j - 1] = i - 1

        elif direction[i, j] == 4: # the j direction

            j -= 1
                
            a2start[j - 1] = i - 1
            a2end[j - 1] = i - 1
            a1start[i - 1] = j - 1

        elif direction[i, j] == 1: # the diagonal direction

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



    return dists[l1, l2], dists, direction, a1start, a1end, a2start, a2end, data_new

def dtw_c(data1, data2, l1, l2, window = -1, penalty = 0):
    
    '''Python wrapper of the Dynamic Time Warping (DTW) implemented in C.
    
    @param data1: the first array, l1-by-d numpy ndarray.
    @param data2: the second array, l2-by-d numpy ndarray.
    @param window: window constraint, by default no constraint.
    @param penalty: misalign penalty, by default zero.

    NOTE: This is a Python wrapper of the C implementation, which is relative
    fast. See the "libfmcutils" package in the FMKit project.
    
    '''
    
    #print('begin dtw_c()')
    
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
 
    
    dist = fmkit_cutils.dtw_c(data1, data2, data1.shape[1], l1, l2, window, penalty,
        dists, direction, a1start, a1end, a2start, a2end, data2_aligned)
    
    #print('end dtw_c()')
    
    return dist, dists, direction, a1start, a1end, a2start, a2end, data2_aligned

class FMSignalDesc(object):
    '''A descriptor of a batch of signals (multiple repetitions).
    
    One descriptor object corresponds to one line in the meta file.
    
    Currently, a descriptor contains the following information about a small
    batch of signals generated by the same user writing the same content in 
    a few repetitions.
    
    unique_id:      A unique ID that orders all descriptors in a collections.
                    Usually it is the line number of the descriptor in the meta
                    file.
                    
    user_label:     A label indicating which user generates the signals.
    
    id_label:       A label indicating the unique content of the signals.
                    Note that for spoofing attacks, i.e., different users are
                    asked to write the same content, the signals have the same
                    id_label. This label is used as the account ID for user
                    identification and authentication purpose, and hence it
                    gets the name "id_label".
                    
    device:         This indicates which type of device is used to obtain the
                    signals. Currently it is either "glove" or "leap".
                    
    start:          The start repetition number (inclusive).
    
    end:            The end repetition number (inclusive). The start and end
                    repetition numbers allow the database to only load a
                    specified section of the data, which is very useful when
                    spliting the dataset into training and testing set or 
                    dealing with data augmentation.
                    
    fn_prefix:      File name prefixe, typically it is user_label + "_" + 
                    id_label. The full file name is fn_prefix + "_" + seq,
                    where the seq indicate which repetition.
                
    content:        The actual content that is written. Usually we do not care
                    since we are not going to recognize it.
                    
     
    
    '''
    
    
    def __init__(self, unique_id, user_label, id_label, device, start, end,
                fn_prefix, content):
        '''
        Default constructor.
        '''
        
        self.unique_id = unique_id
        self.user_label = user_label
        self.id_label = id_label
        self.device = device
        
        self.start = start
        self.end = end
        
        self.fn_prefix = fn_prefix
        self.content = content
        
        
        
    
    
    @classmethod
    def construct_from_meta_file(cls, meta_fn):
        '''
        Factory method to build list of FMSignalDesc from meta file
        
        
        Note: The meta file has the following structure (columns are seperated
        by comma):
        
           column    type      content
           --------+---------+--------------
                0    int       id
                1    string    user_label
                2    string    id_label
                3    string    device type
                4    int       start signal index
                5    int       end signal index
                6    string    file name prefix
                7    string    actual content
        
        Currently, the file name prefix field is just "user_label_id_label"
        
        '''
        
        descs = []
        
        with open(meta_fn, 'r') as meta_fd:
            reader = csv.reader(meta_fd)
            
            for row in reader:

                if len(row) == 0:
                    
                    continue

                if row[0].startswith('#'):
                    
                    continue

                strs = []
                for column in row:
                    strs.append(column.lstrip().rstrip())

                desc = cls(int(strs[0]), strs[1], strs[2], strs[3],
                           int(strs[4]), int(strs[5]), strs[6], strs[7])

                descs.append(desc)


        
        return descs

    def __str__(self):
        '''Convert to a convenient human readable representation.
        
        NOTE: Currently it is just "user_label   id_label   device".
        
        '''
        return '%8d\t%20s\t%20s\t%8s\t' \
            % (self.unique_id, self.user_label, self.id_label, self.device)  


class FMSignal(object):
    '''Base class for a 3D finger motion signal.

    This is the data structure for the finger motion signal (collected by the
    glove or the leap motion controller). The finger motion signal is a time 
    series containing samples of physical states of one point on a hand.

    Attributes:

        len:     length of the time series (i.e., number of samples).
        dim:     dimension of each sample (i.e., number of sensor axes).
        ts:      timestamps of each sample, as a len dimension vector.
        data:    the actual time series data samples, as a len * dim matrix.
        
        user_label:  the user that create this signal.
        id_label:    the unique id indicating the content of the signal.
        seq:         the sequence id in a batch when loaded from dataset.

    NOTE: user_label, id_label, and seq are only used for printing
    information for debugging. Use FMSignalDesc in dataset.py for obtaining
    the meta data of the signal. Typically, the signal file is named as
    "user_label_id_label_seq.txt" (or .csv, .npy). 
    
    For example, "duolu_duolu_01.txt" means that the user_label is "duolu",
    the id_label is also "duolu", which indicating the content is this string,
    and the seq is 1.
    
    Usually, for privacy issues, the user_label and the id_label are anonymous
    strings since they only need to be distinctive instead of meaningful.
    For example, "user00_id00_01.txt" means the user_label is "user00",
    the id label is "id00", and the seq is 1.
    
    NOTE: This is the abstract class that only implements the common
    attributs and methods. To construct the actual signal, use derived classes
    such as FMSignalLeap or FMSignalGlove instead.
    
    There are only three ways to construct a signal object.
    
        (1) Construct from a file using the class method "construct_from_file"
            in the derived class.
        (2) Clone from a signal object that is already constructed.
        (3) Align to a template or another signal to obtain an aligned signal
            object.
    
    All three ways are supported by the derivd classes.

    Typically, the signal data has the following 18 sensor axes, i.e., dim = 18
    and the "data" field has the shape of (len, 18).
    
        0-2:    position in x-y-z
        3-5:    speed in x-y-z, currently just the derivative of the position
        6-8:    acceleration in x-y-z, currently just the derivative of speed
    
        9-11:    orientation, i.e., the x, y, z components of the  quaternion
        12-14:    angular speed, currently just the derivative of the orientation
        15-17:    angular acceleration, just the derivative of the angular speed

    NOTE: Timestamp is always in ms and frequency is always in Hz.

    Optional Attributes:
    
        len_origin:    The length before alignment (only for aligned signals).
        dist:          DTW alignment distant (only for aligned signals).
        a1start:       Alignment index (start) (only for aligned signals).
        a1end:         Alignment index (end) (only for aligned signals).
        stat:          Statistical features (only after the method
                       extract_stat_feature() is called or after it is loaded
                       from a file).
        handgeo:       Hand geometry features (only after the method
                       extract_handgeo_features() is called or after it is 
                       loaded from a file).
        
    
    CAUTION: All floating point numbers in the signal data are in numpy.float32.
    

    See "The FMKit Dataset Format" document for more information.

    '''

    def __init__(self, length, dimension, ts, data,
                 user_label='', id_label='', seq=0):
        '''
        Default constructor, which essentially constructs nothing.
        
        '''

        self.len = length
        self.dim = dimension
        self.ts = ts
        self.data = data

        self.user_label = user_label
        self.id_label = id_label
        self.seq = seq

        self.len_origin = self.len

    # ---------------------------- input / output ---------------------------



    def load_from_file_pp_csv(self, fn):
        '''Load the preprocessed signal from a comma separated value (CSV) file.
        
        '''

        fn += '.csv'
        array = np.loadtxt(fn, dtype=np.float32, delimiter=',')

        ts = array[:, 0:1].flatten()
        data = array[:, 1:]

        l = data.shape[0]
        d = data.shape[1]

        self.len = l
        self.dim = d
        self.ts = ts
        self.data = data

    def load_from_file_pp_binary(self, fn):
        '''Load the preprocessed signal from a Numpy binary file (.npy).
        
        '''

        fn += '.npy'
        array = np.load(fn)

        assert array.dtype == np.float32

        ts = array[:, 0:1].flatten()
        data = array[:, 1:]

        l = data.shape[0]
        d = data.shape[1]

        self.len = l
        self.dim = d
        self.ts = ts
        self.data = data

    def save_to_file_pp_csv(self, fn):
        '''Save the preprocessed signal to a comma separated value (CSV) file.
        
        NOTE: Only six digits after the decimal point are kept when floating
        point numbers are converted to CSV strings.
        
        '''

        l = self.len

        array = np.concatenate((self.ts.reshape((l, 1)), self.data), axis=1)
        fn += '.csv'
        np.savetxt(fn, array, fmt='%.6f', delimiter=', ')

    def save_to_file_pp_binary(self, fn):
        '''Save the preprocessed signal to a Numpy binary file (.npy).
        
        '''

        l = self.len
        array = np.concatenate((self.ts.reshape((l, 1)), self.data), axis=1)
        
        assert array.dtype == np.float32
        
        # CAUTION: Numpy add the .npy for us!
        np.save(fn, array)


    # ---------------------------- statistical features ---------------------------

    def extract_stat_feature(self, segs=1, low_cut=3, sample_freq=50):
        '''Extract the statistical features.

        Currently the statistical features contain the following:

        * Mean of of each sensor axis.
        * Variance of each sensor axis.
        * Correlation between pairs of adjacent sensor axex of the same type.
        * Magnitude (i.e., sum of amplitude) of each sensor axis.
        * Portion of energy of low frequency components.

        Statistical features can be extracted both from the whole signal or
        from just a segment. The segs argument specifies how many equal-sized
        segments should be used for feature extraction.

        Note that statistical feature extraction is also separated into two
        parts. This method is the first part, which extract the first four
        types of statistical features (i.e., except entropy).

        NOTE: This method should be called before amplitude normalization.

        '''
        # CAUTION: Do statistical feature extraction before amplitude
        # normalization

        l = self.len
        d = self.dim
        sl = l // segs
        
        # we have 5 types of statistical features in total
        dimension_ss = 5
        stat = np.zeros((segs, dimension_ss, d), np.float32)

        for i in range(segs):
            start = i * sl
            end = (i + 1) * sl
            data_seg = self.data[start:end, :]

            # mean
            mean_seg = np.mean(data_seg, axis=0)
            stat[i, 0] = mean_seg

            # std
            std_seg = np.std(data_seg, axis=0)
            stat[i, 1] = std_seg

            # axis correlation, 3 axes a group
            # CAUTION: we assume that dim is always a multiple of 3
            for j in range(d // 3):

                axis0 = j * 3 + 0
                axis1 = j * 3 + 1
                axis2 = j * 3 + 2

                co01 = np.corrcoef(
                    data_seg[:, axis0], data_seg[:, axis1])[0, 1]
                co12 = np.corrcoef(
                    data_seg[:, axis1], data_seg[:, axis2])[0, 1]
                co20 = np.corrcoef(
                    data_seg[:, axis2], data_seg[:, axis0])[0, 1]

                stat[i, 2, axis0] = co01
                stat[i, 2, axis1] = co12
                stat[i, 2, axis2] = co20

            # magnitude
            data_seg_abs = np.absolute(data_seg)
            mag_seg = np.sum(data_seg_abs, axis=0)
            stat[i, 3] = mag_seg

            # low frequency components
            cut_l = int(low_cut * sl / sample_freq)
            dft_co = np.fft.fft(data_seg, sl, axis=0)
            dft_co = dft_co[0:cut_l]
            low_energy = np.mean(np.abs(dft_co), axis=0)
            stat[i, 4] = low_energy

        self.stat = stat

#     def stat_feature_extract_post(self, segs=3):
#         '''
#         Extract the statistical features (second part).
# 
#         This method is the second part of the statistical feature extraction,
#         which extracts signal entropy of each sensor axis.
# 
#         Note that this method should be called after amplitude normalization.
#         '''
# 
#         l = self.len
#         d = self.dim
#         sl = l // segs
# 
#         stat = self.stat
# 
#         for i in range(segs):
#             start = i * sl
#             end = (i + 1) * sl
#             data_seg = self.data[start:end, :]
# 
#             # entropy
#             for j in range(d):
# 
#                 column = data_seg[:, j]
#                 # print(column.shape)
#                 # NOTE: we use 20 bins and (-3 * sigma, + 3 * sigma) as range.
#                 # CAUTION: assume amplitude normalization is done first.
#                 hist_j, _ = np.histogram(column, bins=20,
#                                          range=(-3, 3), density=True)
#                 # print(hist_j)
#                 entropy_j = scipy.stats.entropy(hist_j)
# 
#                 stat[i, 4, j] = entropy_j


    def load_stat_features_from_file_csv(self, fn):
        '''Load statistical features from a comma separated value (CSV) file.
        
        '''

        fn += '.csv'
        array = np.loadtxt(fn, dtype=np.float32, delimiter=',')
        shape = array.shape
        self.stat = array.reshape((shape[0] // 5, 5, self.dim))

    def load_stat_features_from_file_binary(self, fn):
        '''Load the statistical features from a Numpy binary file (.npy).
        
        '''

        fn += '.npy'
        self.stat = np.load(fn)

    def save_stat_features_to_file_csv(self, fn):
        '''Save statistical features to a comma separated value (CSV) file.
        
        An exception is raised if it does not have such features extracted.
        
        '''

        assert (self.stat is not None), \
            'The signal does not have stat features.'
        shape = self.stat.shape
        array = self.stat.reshape((shape[0] * shape[1], shape[2]))
        fn += '.csv'
        np.savetxt(fn, array, fmt='%.6f', delimiter=', ')


    def save_stat_features_to_file_binary(self, fn):
        '''Save the statistical features to a Numpy binary file (.npy).
        
        '''

        assert (self.stat is not None), \
            'The signal does not have any statistical feature.'

        # CAUTION: Numpy add the .npy for us!
        np.save(fn, self.stat)




    # ---------------------------- preprocessing ---------------------------

    def filter(self, sample_freq, cut_freq):
        '''Low-pass filtering on the signal. 
        
        NOTE: It is assumed that a hand can not move in very high frequency,
        so the high frequency components of the signal is noise.
        
        NOTE: This method use numpy FFT and IFFT.
        '''

        l = self.len
        data = self.data

        cut_l = int(cut_freq * l / sample_freq)

        dft_co = np.fft.fft(data, l, axis=0)

        for i in range(cut_l, l - cut_l):

            dft_co[i] = 0 + 0j

        ifft_c = np.fft.ifft(dft_co, l, axis=0)

        ifft = ifft_c.astype(np.float32)

        for i in range(l):
            self.data[i] = ifft[i]




#     def amplitude_normalize(self):
#         '''Normalize the amplitude of each sensor axis.
#         
#         NOTE: After normalization, the data on each sensor axis has zero mean 
#         and unit standard deviation.
#         '''
# 
#         data = self.data
# 
#         mean = np.mean(data, axis=0)
#         std = np.std(data, axis=0)
# 
# 
#         # This causes a divide warning...
#         #data = np.divide(data - mean, std)
# 
# 
#         for j in range(self.dim):
#             data[:, j] = np.divide(data[:, j] - mean[j], std[j])



    # ---------------------------- data augmentation ----------------------


    def amplitude_pertube(self, t, axis, width, amplitude, sigma):
        '''Pertube the signal at the specified time in the specified axis.

        This method is used in data augmentation.
        
        '''

        pertubation = scipy.signal.gaussian(width, sigma) * amplitude

        start_data = t - width // 2
        start_pertubation = 0
        if start_data < 0:
            start_pertubation = -start_data
            start_data = 0

        end_data = start_data + width
        end_pertubation = width
        if end_data > self.len:
            end_pertubation -= (end_data - self.len)
            end_data = self.len

        # width_data = end_data - start_data
        # width_pertubation = end_pertubation - start_pertubation

        # print(self.len, t, width, start_data, end_data,
        #    start_pertubation, end_pertubation, width_data, width_pertubation)

        self.data[start_data:end_data, axis] \
            += pertubation[start_pertubation:end_pertubation]


    def scale_pertube(self, t, axis, width, smooth_width, scale, sigma):

        kernel_size = width + smooth_width * 2
        kernel = np.ones(kernel_size, dtype=np.float32) * scale
        all_one = np.ones(kernel_size, dtype=np.float32)

        smooth_edges = scipy.signal.gaussian(smooth_width * 2, sigma)
        se_left = smooth_edges[:smooth_width]
        se_right = smooth_edges[smooth_width:]

        kernel[:smooth_width] = np.multiply(kernel[:smooth_width], se_left) \
                                + np.multiply(all_one[:smooth_width], 1 - se_left)

        kernel[-smooth_width:] = np.multiply(kernel[-smooth_width:], se_right) \
                                + np.multiply(all_one[-smooth_width:], 1 - se_right)
        

        l1 = t - smooth_width
        l_kernel = 0
        if l1 < 0:
            l_kernel = -l1
            l1 = 0
        elif l1 > self.len:
            return

        r1 = t + width + smooth_width
        r_kernel = kernel_size
        if r1 >= self.len:
            r_kernel -= (r1 - self.len)
            r1 = self.len
        elif r1 <=0:
            return

        self.data[l1:r1, axis] \
            *= kernel[l_kernel:r_kernel]
        


    def swap_segment(self, other, seg_start, seg_end, soft_margin):
        '''Swap a segment with another signal. 
        
        This method is used in data augmentation.
        
        '''
        a = self.clone()
        b = other.clone()

        data_a = a.data
        ts_a = a.ts
        data_b = b.data
        ts_b = b.ts

        as1 = data_a[:seg_start, :]
        as2 = data_a[seg_start:seg_end, :]
        as3 = data_a[seg_end:, :]

        bs1 = data_b[:seg_start, :]
        bs2 = data_b[seg_start:seg_end, :]
        bs3 = data_b[seg_end:, :]

        ats1 = ts_a[:seg_start]
        ats2 = ts_a[seg_start:seg_end]
        ats3 = ts_a[seg_end:]

        bts1 = ts_b[:seg_start]
        bts2 = ts_b[seg_start:seg_end]
        bts3 = ts_b[seg_end:]

        data_a_new = np.concatenate([as1, bs2, as3], axis=0)
        ts_a_new = np.concatenate([ats1, bts2, ats3], axis=0)
        l_a = data_a_new.shape[0]

        data_b_new = np.concatenate([bs1, as2, bs3], axis=0)
        ts_b_new = np.concatenate([bts1, ats2, bts3], axis=0)
        l_b = data_b_new.shape[0]

        # smooth the segment edges
        if soft_margin != 0:

            margin1_start = max(0, seg_start - soft_margin)
            margin1_end = min(seg_start + soft_margin, min(l_a, l_b))
            margin1_factor = np.linspace(1, 0, margin1_end - margin1_start,
                                         endpoint=True, dtype=np.float32)

            # print(margin1_start, margin1_end)

            for i, ii in \
                zip(range(margin1_start, margin1_end),
                        range(margin1_end - margin1_start)):

                data_a_new[i] = margin1_factor[ii] * data_a[i] \
                    + (1 - margin1_factor[ii]) * data_b[i]
                data_b_new[i] = margin1_factor[ii] * data_b[i] \
                    + (1 - margin1_factor[ii]) * data_a[i]

            margin2_start = max(0, seg_end - soft_margin)
            margin2_end = min(seg_end + soft_margin, min(l_a, l_b))
            margin2_factor = np.linspace(1, 0, margin2_end - margin2_start,
                                         endpoint=True, dtype=np.float32)

            # print(margin2_start, margin2_end)

            for i, ii in \
                zip(range(margin2_start, margin2_end),
                        range(margin2_end - margin2_start)):

                data_a_new[i] = margin2_factor[ii] * data_b[i] \
                    + (1 - margin2_factor[ii]) * data_a[i]
                data_b_new[i] = margin2_factor[ii] * data_a[i] \
                    + (1 - margin2_factor[ii]) * data_b[i]

        a.data = data_a_new
        a.ts = ts_a_new
        a.len = l_a
        b.data = data_b_new
        b.ts = ts_b_new
        b.len = l_b

        return a, b

    def stretch(self, l_new):
        '''Stretch the signal to the specified new length.
        
        NOTE: This method stretch the signal in time by linear interpolation.

        This method is used in temporal normalization for deep learning models.
        
        '''
        # CAUTION: delta_new may not be an integer any more!!!

        l = self.len
        d = self.dim

        data_new = np.zeros((l_new, d), np.float32)
        ts_new = np.zeros((l_new), np.float32)

        data = self.data
        ts = self.ts

        xp = np.linspace(0, l - 1, num=l)
        x = np.linspace(0, l - 1, num=l_new)

        for j in range(d):
           
            data_new[:, j] = np.interp(x, xp, data[:, j])

        ts_new[:] = np.interp(x, xp, ts)

        self.len = l_new
        self.dim = d
        self.ts = ts_new
        self.data = data_new

    def stretch_segment(self, seg_start, seg_end, seg_l_new, sample_period=20):
        '''Stretch a segment of the signal and keep other parts untouched.

        This method is used in data augmentation.
        
        TODO: rewrite this with array splice and linear interpolation
        
        '''

        l = self.len
        d = self.dim
        seg_l = seg_end - seg_start
        l_new = self.len - seg_l + seg_l_new
        delta_new = seg_l / seg_l_new

        seg_start_new = seg_start
        seg_end_new = seg_start_new + seg_l_new

        data_new = np.zeros((l_new, d), np.float32)
        ts_new = np.zeros((l_new, 1), np.float32)

        data = self.data

        data_new[0:seg_start] = data[0:seg_start]
        data_new[seg_end_new:l_new] = data[seg_end:l]

        for i in range(seg_l_new):

            it = seg_start + i * delta_new
            ii_old = int(math.floor(it))
            ii_new = seg_start + i
            dt = it - ii_old

            # print(i, delta_new, it, ii)

            for v in range(d):
                rate = data[ii_old][v] - data[ii_old][v] \
                    if ii_old + 1 < l else data[ii_old][v] - data[ii_old - 1][v]
                data_new[ii_new][v] = data[ii_old][v] + rate * dt

        for i in range(l_new):

            ts_new[i] = sample_period * i

        self.len = l_new
        self.dim = d
        self.ts = ts_new
        self.data = data_new

    def offset(self, offset):
        ''' Add a small offset to the start time of the signal.
        
        NOTE: The offset can be a floating point number since offset is
        implemented as linear interpolation.
        
        This method is used in data augmentation.
        
        '''

        data = self.data
        l = self.len
        d = self.dim
        
        xp = np.arange(0, l, 1, dtype=np.float32)
        x = xp + offset
        
        data_offset = np.zeros((l, d), dtype=np.float32)
        
        for j in range(d):
            
            data_offset[:, j] = np.interp(x, xp, data[:, j])
        
        self.data = data_offset

    # ---------------------------- operations on a signal ---------------------------


    def align_signal_to(self, template, window, penalty, method):
        '''Align the signal self to a template signal.
        
        NOTE: This method uses the Dynamic Time Warping (DTW) algorithm.
        
        '''


        l_new = template.len

        if method == 'py':
        
            tup = dtw(template.data, self.data, template.len, self.len,
                      window=window, penalty=penalty)
            
        elif method == 'c':
            
            tup = dtw_c(template.data, self.data, template.len, self.len,
                        window=window, penalty=penalty)
            
        else:
            
            raise ValueError('No such method (%s)!' % method)

        (dist, _dists, _dir, a1start, a1end, _a2start, _a2end, data_new) = tup

        #print(dist, a1start.dtype)

        ts_new = template.ts.copy()

        return (ts_new, data_new, l_new, dist, a1start, a1end)

    def save_alignment_index(self, fn):
        '''Save the alignment index to a Numpy binary file (.npy).

        The alignment index are the "a1start" and "a1end" attributes.

        '''
        
        l = self.len

        array = np.concatenate(
            (self.a1start.reshape((l, 1)), self.a1end.reshape((l, 1))), 
            axis=1)
        np.save(fn, array)
        
    def load_alignment_index(self, fn):
        '''Load the alignment index from a Numpy binary file (.npy).
        
        The alignment index are the "a1start" and "a1end" attributes.

        '''
        
        array = np.load(fn + '.npy')
        
        self.a1start = array[:, 0]
        self.a1end = array[:, 1]
        
        self.len_origin = self.a1end[-1]
        
    def distance_to(self, other):
        '''Calculate the element-wise absolute distance of two signals.
        
        CAUTION: other must be already aligned to self.
        
        '''

        assert self.len == other.len

        return np.absolute(self.data - other.data)


    def __str__(self, *args, **kwargs):
        '''Convert the meta information of the signal to a human readable string.
        
        The string is "user_label_id_label_seq".
        '''

        return '%s_%s_%02d' % (self.user_label, self.id_label, self.seq)











class FMSignalLeap(FMSignal):
    '''Finger motion signal collected by the Leap Motion controller.

    This class is derived from the abstract base class FMSignal by adding a few
    addtional fields described below.

    Additional Attributes:

        data_aux:   position of each joint (not just the finger tip).
                    It is only available if the data is loaded from a raw file.
                    It is currently a len * 5 * 5 * 3 tensor, i.e., 5 fingers,
                    5 joints on each finger, and 3 coordinates for each joint.
        confs:      confidence value of each sample (one dimension vector).
        valids:     whether the sample is valid (one dimension vector).
    
        handgeo:        hand geometry features, a 22 components vector.
        handgeo_std:    the standard deviation of each component of handgeo, 
                        also a 22 component vector.

    The "data_aux", "confs", and "valids" attributes are only available if the
    signal is loaded from a file with raw data.

    NOTE: For a signal loaded from a raw text file, only the position axes
    have valid data and other axes are all zero. After preprocessing, all axes
    will have valid data.

    NOTE: The signal can be derived either from the center of the hand or the
    tip of the index finger.

    See "The FMKit Dataset Format" document for more information.
    '''

    def __init__(self, length, dimension, ts, data,
                 user_label='', id_label='', seq=0,
                 data_aux=None, confs=None, valids=None):
        '''Default constructor.
        
        '''

        FMSignal.__init__(self, length, dimension, ts, data,
                          user_label, id_label, seq)

        self.data_aux = data_aux
        self.confs = confs
        self.valids = valids

        self.stat = None
        self.handgeo = None
        self.handgeo_std = None

    # ---------------------------- input / output ---------------------------


    def load_from_file(self, fn, mode, point='tip', use_stat=False, use_handgeo=False,
                       user_label='', id_label='', seq=0):
        '''General interface to load the signal object from files.
        
        '''

        if mode == 'raw_internal':
            self.load_from_file_raw_internal(fn, point)
        elif mode == 'raw_csv':
            self.load_from_file_raw_csv(fn, point)
        elif mode == 'raw_binary':
            self.load_from_file_raw_binary(fn, point)
        elif mode == 'pp_csv':
            self.load_from_file_pp_csv(fn)
        elif mode == 'pp_binary':
            self.load_from_file_pp_binary(fn)
        else:
            raise ValueError('Unknown file mode %s!' % mode)

        self.user_label = user_label
        self.id_label = id_label
        self.seq = seq

        if use_stat and mode == 'pp_csv':
            self.load_stat_features_from_file_csv(fn + "_ss")
        if use_stat and mode == 'pp_binary':
            self.load_stat_features_from_file_binary(fn + "_ss")

        if use_handgeo and mode == 'pp_csv':
            self.load_handgeo_features_from_file_csv(fn + "_hg")
        if use_handgeo and mode == 'pp_binary':
            self.load_handgeo_features_from_file_binary(fn + "_hg")

    def save_to_file(self, fn, mode, use_stat, use_handgeo):
        '''General interface to dump the signal object to files.
        
        '''

        if mode == 'raw_csv':
            self.save_to_file_raw_csv(fn)
        elif mode == 'raw_binary':
            self.save_to_file_raw_binary(fn)
        elif mode == 'pp_csv':
            self.save_to_file_pp_csv(fn)
        elif mode == 'pp_binary':
            self.save_to_file_pp_binary(fn)
        else:
            raise ValueError('Unknown file mode %s!' % mode)

        if use_stat and mode == 'pp_csv':
            self.save_stat_features_to_file_csv(fn + "_ss")
        if use_stat and mode == 'pp_binary':
            self.save_stat_features_to_file_binary(fn + "_ss")

        if use_handgeo and mode == 'pp_csv':
            self.save_handgeo_features_to_file_csv(fn + "_hg")
        if use_handgeo and mode == 'pp_binary':
            self.save_handgeo_features_to_file_binary(fn + "_hg")

    def load_from_buffer_raw_internal(self, array, point):
        '''Load data from a buffer (as a numpy ndarray).

        Args:

            array   The buffer.

        Returns:
        
            None.

        
        '''
        
        l = array.shape[0]
        d = array.shape[1]
        
        m = 18

        # --------- process timestamp ------

        if d == 133 or d == 93:
            
            # CAUTION: the early format, with only one timestamp.
            offset_base = 1
            ts = array[:, 0].flatten()
            tsc = np.zeros(l, np.float32)
            
        elif d == 134:
            
            # CATUION: the current format, with two timestamps.
            offset_base = 2
            ts = array[:, 1].flatten()
            tsc = array[:, 0].flatten()
            
            # Use an offset to reduce the size of tsc so that it can fit into
            # the float32 type.
            tsc -= 1514764800

        else:
            
            raise ValueError('Unknown data file format!: d = %d' % d)


        for i in range(l):

            # fix timestamp wraping over maximum of uint32
            if i > 0 and ts[i] < ts[i - 1]:
                ts[i] += 4294967295.0

        # fix timestamp offset and convert timestamp to millisecond
        # CAUTION: timestamp must start from 0! Other methods such as filtering
        # depend on this assumption!
        ts0 = ts[0]
        ts -= ts0
        ts /= 1000


        # --------- process point coordinate and joint coordinate ------

        offset_tip = offset_base
        offset_center = offset_base + 6
        offset_aux = offset_base + 6 + 9

        array_tip = np.zeros((l, 3), np.float32)
        
        # CAUTION: axis mapping: yzx -> xyz
        array_tip[:, 0] = array[:, offset_tip + 2]
        array_tip[:, 1] = array[:, offset_tip + 0]
        array_tip[:, 2] = array[:, offset_tip + 1]
        
        array_center = np.zeros((l, 3), np.float32)

        # CAUTION: axis mapping: yzx -> xyz
        array_center[:, 0] = array[:, offset_center + 2]
        array_center[:, 1] = array[:, offset_center + 0]
        array_center[:, 2] = array[:, offset_center + 1]

        data = np.zeros((l, m), np.float32)

        if point == 'tip':
            
            data[:, 0:3] = array_tip
            self.point = 'tip'

        elif point == 'center':
            
            data[:, 0:3] = array_center
            self.point = 'center'
            

        else:
            
            raise ValueError('Unknown point type: %s' % str(point))
        



        data_aux = np.zeros((l, 5, 5, 3), np.float32)
        
        # load joint positions
        for j in range(5):
            for k in range(5):
                index = j * 5 * 3 + k * 3
                data_aux[:, j, k, 0] = array[:, offset_aux + index + 2]
                data_aux[:, j, k, 1] = array[:, offset_aux + index + 0]
                data_aux[:, j, k, 2] = array[:, offset_aux + index + 1]



        # load confidences and valid flags
        confs = array[:, -2]
        valids = array[:, -1]

        self.len = l
        self.dim = m

        self.ts = ts.astype(np.float32)
        self.tsc = tsc.astype(np.float32)
        self.data = data

        self.data_aux = data_aux
        self.confs = confs.astype(np.float32)
        self.valids = valids.astype(np.float32)
        
        self.array_tip = array_tip.astype(np.float32)
        self.array_center = array_center.astype(np.float32)
        
        

    def load_from_file_raw_internal(self, fn, point):
        '''Load data from a raw text file obtained by the data collection client.

        Args:

            fn   The file name.

        Returns:
        
            None.
        
        CAUTION: Do not use this method directly. Instead, uses the class method
        FMSignalLeap.construct_from_file() instead.
        
        '''

        fn += '.txt'
        array = np.loadtxt(fn, np.float64, delimiter=',')

        self.load_from_buffer_raw_internal(array, point)

    def prepare_load_from_file_raw(self, array, point):
        '''Converting columns to object fields when loading from a raw file.
        
        CAUTION: This method is only for internal usage.
        
        '''

        l = array.shape[0]
        d = array.shape[1]
        
        assert d == 2 + 3 + 3 + 75 + 2, 'Wrong raw file format: d = %d' % d
        
        m = 18

        self.len = l
        self.dim = m

        self.ts = array[:, 1].flatten()
        self.tsc = array[:, 0].flatten()

        offset_tip = 2
        offset_center = 2 + 3
        offset_aux = 2 + 3 + 3
        
        self.array_tip = array[:, offset_tip:offset_tip + 3]
        self.array_center = array[:, offset_center:offset_center + 3]
 
        self.data = np.zeros((l, m), np.float32)
 
        if point == 'tip':
            self.data[:, 0:3] = self.array_tip
            self.point = 'tip'
        elif point == 'center':
            self.data[:, 0:3] = self.array_center
            self.point = 'center'
        else:
            raise ValueError('Unknown point type: %s' % str(point))

        self.data_aux = np.zeros((l, 5, 5, 3), np.float32)
        
        for j in range(5):
            for k in range(5):
                index = offset_aux + j * 5 * 3 + k * 3
                self.data_aux[:, j, k, :] = array[:, index:index + 3]
       
        # load confidences and valid flags
        self.confs = array[:, -2]
        self.valids = array[:, -1]


    def load_from_file_raw_csv(self,fn, point):
        '''Load data from a comma separated value (CSV) file in raw fromat.
        
        '''

        fn += '.csv'
        array = np.loadtxt(fn, np.float32, delimiter=',')

        self.prepare_load_from_file_raw(array, point)

    def load_from_file_raw_binary(self,fn, point):
        '''Load data from a raw a Numpy binary file (.npy) in raw fromat.
        
        '''
        
        fn += '.npy'
        array = np.load(fn)
        
        assert array.dtype == np.float32

        self.prepare_load_from_file_raw(array, point)

    def prepare_save_to_file_raw(self):
        '''Converting object fields to columns when saving to a raw file.
        
        CAUTION: This method is only for internal usage.
        
        '''
        
        d = 2 + 3 + 3 + 75 + 2
        array = np.zeros((self.len, d), np.float32)
        
        array[:, 0] = self.tsc
        array[:, 1] = self.ts
        
        offset_tip = 2
        offset_center = 2 + 3
        offset_aux = 2 + 3 + 3
        
        array[:, offset_tip:offset_tip + 3] = self.array_tip
        array[:, offset_center:offset_center + 3] = self.array_center

        for j in range(5):
            for k in range(5):
                index = offset_aux + j * 5 * 3 + k * 3

                array[:, index:index + 3] = self.data_aux[:, j, k, :]
                
        array[:, -2] = self.confs
        array[:, -1] = self.valids
        
        return array

    def save_to_file_raw_csv(self,fn):
        '''Save data to a comma separated value (CSV) file in raw fromat.
        
        CAUTION: For floating point numbers, six digits after the decimal
        point are saved.
        '''
        
        array = self.prepare_save_to_file_raw()
        fn += '.csv'
        np.savetxt(fn, array, fmt='%.6f', delimiter=', ')

    def save_to_file_raw_binary(self,fn):
        '''Save data to a raw a Numpy binary file (.npy) in raw fromat.
        
        '''
        
        array = self.prepare_save_to_file_raw()
        fn += '.npy'
        np.save(fn, array)

    # ---------------------------- hand geometry features ------------------

    def extract_handgeo_features(self):
        '''Extract hand geometry features.
        
        NOTE: This can only be done on a signal constructed by loading from
        a raw file, i.e., the "data_aux" field must be provided.
        
        '''

        l = self.len
        data_aux = self.data_aux

        handgeo_samples = np.zeros((100, 22), np.float32)
        ii = 0

        for i in range(l - 100, l):

            handgeo_sample = np.zeros(22, np.float32)

            # bones of index finger, mid finger, ring finger, and little finger
            for j in range(0, 4):

                for k in range(4):

                    handgeo_sample[j * 4 + k] = np.linalg.norm(
                        data_aux[i, j + 1, k + 1] - data_aux[i, j + 1, k])

            # bones on the thumb
            handgeo_sample[16] = np.linalg.norm(
                data_aux[i, 0, 2] - data_aux[i, 0, 1])
            handgeo_sample[17] = np.linalg.norm(
                data_aux[i, 0, 3] - data_aux[i, 0, 2])
            handgeo_sample[18] = np.linalg.norm(
                data_aux[i, 0, 4] - data_aux[i, 0, 3])

            # hand widths
            handgeo_sample[19] = np.linalg.norm(
                data_aux[i, 1, 1] - data_aux[i, 2, 1])
            handgeo_sample[20] = np.linalg.norm(
                data_aux[i, 2, 1] - data_aux[i, 3, 1])
            handgeo_sample[21] = np.linalg.norm(
                data_aux[i, 3, 1] - data_aux[i, 4, 1])

            handgeo_samples[ii] = handgeo_sample
            ii += 1

        handgeo = np.mean(handgeo_samples, 0)
        handgeo_std = np.std(handgeo_samples, 0)

        # print(handgeo_samples)
        # print(handgeo)
        # print(handgeo_std)

        self.handgeo = handgeo
        self.handgeo_std = handgeo_std


    def load_handgeo_features_from_file_csv(self, fn):
        '''Load hand geometry features from a comma separated value (CSV) file.
        
        '''
        
        fn += '.csv'
        array = np.loadtxt(fn, dtype=np.float32, delimiter=',')
        self.handgeo = array[:22]
        self.handgeo_std = array[22:]

    def load_handgeo_features_from_file_binary(self, fn):
        '''Load hand geometry features from a Numpy binary file (.npy).
        
        '''
        
        fn += '.npy'
        array = np.loadtxt(fn, dtype=np.float32, delimiter=',')
        self.handgeo = array[:22]
        self.handgeo_std = array[22:]


    def save_handgeo_features_to_file_csv(self, fn):
        '''Save hand geometry features to a comma separated value (CSV) file.
        '''
        
        array = np.concatenate((self.handgeo, self.handgeo_std))
        fn += '.csv'
        np.savetxt(fn, array, fmt='%.6f', delimiter=', ')
        
    def save_handgeo_features_to_file_binary(self, fn):
        '''Save hand geometry features to a Numpy binary file (.npy).
        '''
        
        array = np.concatenate((self.handgeo, self.handgeo_std))
        np.save(fn, array)
        


    # ---------------------------- preprocessing ---------------------------

    def fix_missing_samples(self):
        ''' Fix missing data samples by linear interpolation.
        
        The missing samples are mainly caused by the motion of the hand
        which are outside the field of the view of the sensor.
        
        CAUTION: This procedure assumes that the first sample is always valid!
        
        CAUTION: Since it is just linear interpolation, it will not work well
        with too many missing samples.
        
        This method is only used for preprocessing with raw_internal data.
        
        '''

        i = 1

        while i < self.len - 1:

            # find the start of a missing segment
            while i < self.len - 1:

                if self.valids[i] == 0:
                    break
                else:
                    i += 1

            # find the end of a missing segment
            j = i
            while j < self.len - 1:

                if self.valids[j] == 1:
                    break
                else:
                    j += 1

            # If missing points found between i and j, fix them.
            # If no missing point found, just skip.
            if i < j:

                #print(i, j)

                # fix data[i:j]
                start = self.data[i - 1]
                start_ts = self.ts[i - 1]
                end = self.data[j]
                end_ts = self.ts[j]
                
                start_aux = self.data_aux[i - 1]
                end_aux = self.data_aux[j]

                for k in range(i, j):

                    k_ts = self.ts[k]
                    rate = (k_ts - start_ts) * 1.0 / (end_ts - start_ts)
                    self.data[k] = start + (end - start) * rate
                    self.valids[k] = 1
                    
                    # also fix data_aux
                    self.data_aux[k] = start_aux + (end_aux - start_aux) * rate
                    

                # then find another missing sement in the next iteration
                i = j


    def resample(self, re_freq):
        '''Resample the signal with at a specified frequency.
        
        This method uses linear interpolation for resampling.
        
        '''

        data = self.data
        ts = self.ts
        l = self.len
        d = self.dim

        duration = ts[l - 1] - ts[0]

        step = 1000.0 / re_freq
        l_new = int(duration / 1000.0 * re_freq)


        ts_resample = np.arange(0, step * l_new, step, dtype=np.float32)

        data_resample = np.zeros((l_new, d), dtype=np.float32)
        for j in range(3):
            data_resample[:, j] = np.interp(ts_resample, ts, data[:, j])


        array_tip_resample = np.zeros((l_new, 3), dtype=np.float32)
        for j in range(3):
            array_tip_resample[:, j] = np.interp(ts_resample, ts, self.array_tip[:, j])

        array_center_resample = np.zeros((l_new, 3), dtype=np.float32)
        for j in range(3):
            array_center_resample[:, j] = np.interp(ts_resample, ts, self.array_center[:, j])

        data_aux_resample = np.zeros((l_new, 5, 5, 3), np.float32)
        for j in range(5):
            for k in range(5):
                for v in range(3):
                    data_aux_resample[:, j, k, v] \
                        = np.interp(ts_resample, ts, self.data_aux[:, j, k, v])

        

        confs_resample = np.interp(ts_resample, ts, self.confs)
        valids_resample = np.ones(l_new, dtype=np.float32)


        self.len = l_new
        self.data = data_resample
        self.ts = ts_resample

        self.data_aux = data_aux_resample
        self.confs = confs_resample.astype(np.float32)
        self.valids = valids_resample

        self.array_tip = array_tip_resample
        self.array_center = array_center_resample

    def offset(self, axis, value):
        
        l = self.len
        
        offset_v = (np.arange(0, l, 1) - l / 2) * value
        
        self.data[:, axis] += offset_v

        for j in range(5):
            for k in range(5):
                
                self.data_aux[:, j, k, axis] += offset_v
        
        self.array_tip[:, axis] += offset_v
        self.array_center[:, axis] += offset_v

    def translate(self, ox, oy, oz):
        '''Translate the coorindate reference frame.
        
        CAUTION: This method works only on raw signals.
        
        '''
        
        o = np.array((ox, oy, oz), dtype=np.float32)
        
        # position translation
        self.data[:, 0:3] -= o
        
        for j in range(5):
            for k in range(5):
                
                self.data_aux[:, j, k] -= o

        self.array_tip -= o
        self.array_center -= o

    def rotate_internal(self, R):
        '''Rotate the coorindate reference frame.
        
        CAUTION: This method works only on raw signals.
        
        '''

        # rotation
        pv = self.data[:, 0:3].T
        pv = np.matmul(R, pv)
        self.data[:, 0:3] = pv.T
        
        for j in range(5):
            for k in range(5):
        
                pv = self.data_aux[:, j, k].T
                pv = np.matmul(R, pv)
                self.data_aux[:, j, k] = pv.T

        pv = self.array_tip[:, 0:3].T
        pv = np.matmul(R, pv)
        self.array_tip[:, 0:3] = pv.T

        pv = self.array_center[:, 0:3].T
        pv = np.matmul(R, pv)
        self.array_center[:, 0:3] = pv.T

    def rotate(self, R):
        '''Rotate the coorindate reference frame.
        
        CAUTION: This method works only on raw signals.
        
        '''

        self.rotate_internal(R)
        
        self.estimate_velocity_and_acceleration()
        self.estimate_angular_position_speed_acceleration()


    def position_normalize(self, start, end):

        # Find the origin.

        pos = self.data[start:end, 0:3]
        o = np.mean(pos, axis=0)
        
        # translate
        
        self.translate(o[0], o[1], o[2])


    def pose_normalize(self, start, end, p_yaw=0, p_pitch=0, p_roll=0):
        '''Normalize the posture by translation and rotation.

        The three angles in the argument are offsets that applied during
        the normalization. This can be used in view angle purtubation for
        data augmentation or in correcting the pointing direction.

        Note that the new x axis is the average of the pointing direction,
        the new y axis is the horizontal direction, and the new z axis is
        determined by the new x and y axes.
        
        This method is used for preprocessing.
        
        '''
        
        data_aux = self.data_aux
        
        self.position_normalize(start, end)
        
        # Find the average pose.
        
        if self.point == 'tip':
            
            # use the direction of the middle segment of the index finger
            p0 = data_aux[start:end, 1, 1, :]
            p1 = data_aux[start:end, 1, 2, :]

        elif self.point == 'center':
            
            # use the direction of the last segment of the middle finger
            p0 = data_aux[start:end, 2, 0, :]
            p1 = data_aux[start:end, 2, 1, :]
        
        else:
            
            raise ValueError('Unknown point type: %s' % str(self.point))

        vx = np.mean(p1 - p0, axis=0)
        vz_t = np.asarray((0.0, 0.0, 1.0), np.float32)
        vy = np.cross(vz_t, vx)
        vz = np.cross(vx, vy)

        vx = vx / np.linalg.norm(vx)
        vy = vy / np.linalg.norm(vy)
        vz = vz / np.linalg.norm(vz)

        vx = vx.reshape((3, 1))
        vy = vy.reshape((3, 1))
        vz = vz.reshape((3, 1))

        R_g2l = np.concatenate((vx, vy, vz), axis=1)

        R_l2g = R_g2l.transpose()

        R_offset = euler_zyx_to_rotation_matrix(p_yaw, p_pitch, p_roll)

        R_l2g = np.matmul(R_offset, R_l2g)
        
        
        self.rotate_internal(R_l2g)

        # print('centeriod: (%f, %f, %f, %f, %f, %f)' % (xo, yo, zo, vx, vy, vz))


    def estimate_velocity_and_acceleration(self):
        '''Estimate velocity and acceleration from the position data samples.
        
        This method is used for preprocessing.
        '''

        data = self.data

        data[:, 3] = np.gradient(data[:, 0])
        data[:, 4] = np.gradient(data[:, 1])
        data[:, 5] = np.gradient(data[:, 2])

        data[:, 6] = np.gradient(data[:, 3])
        data[:, 7] = np.gradient(data[:, 4])
        data[:, 8] = np.gradient(data[:, 5])

        self.data = data

        
    def estimate_angular_position_speed_acceleration(self,
            p_yaw=0, p_pitch=0, p_roll=0):
        '''Estimate angular position.
        
        Angular positions are represented as quaternion.
        
        CAUTION: This should be called after posture normalization.
        
        CAUTION: currently angular speed is just the differential of angular
        position, i.e., differential of qx, qy, qz, not the real angular speed.
        Similarly, angular acceleration is just the second order differential
        of qx, qy, qz, not the real angular acceleration.
        
        CAUTION: The current code iterate through each sample, which is
        very slow.
        
        This method is used for preprocessing.
        
        '''

        data = self.data
        ts = self.ts
        
        # rotation matrix of each data sample
        rotms = np.zeros((self.len, 3, 3))

        # pose offset, for augmentation        
        R_offset = euler_zyx_to_rotation_matrix(p_yaw, p_pitch, p_roll)

        
        # quaternion of each data sample
        qs = [None] * self.len
        omega_pre = np.asarray((0.0, 0.0, 0.0))
        
        for i in range(self.len):
        
            joints = self.data_aux[i]
            
            # Origin is the near end of the index finger.
            #p0 = joints[1, 1]
            p0 = joints[1, 0]
            
            # Pointing direction is from the near end to the next joint along
            # the index finger.
            #p1 = joints[1, 2]
            p1 = joints[1, 1]
            
            # The side direction is from the near end of the index finger to
            # the near end of the little finger.
            #p2 = joints[4, 1]
            p2 = joints[4, 0]
            
            # derive the pose represented in three orthogonal vectors,
            # i.e., vx, vy, vz
            
            # Note that here vx is the general pointing direction, vz is the
            # general palm facing direction
            
            vx = p1 - p0
            vy_prime = p0 - p2
            vz = np.cross(vx, vy_prime)

            if np.linalg.norm(vx) <= 1e-6:
                
                print(i, p0, p1)

            # most of the time the palm is facing downward, so we flip the
            # axes here to set up a local reference frame where the z-axis
            # is always upward.
            
            # CAUTION: There are chances that the sensor wrongly recognize the
            # right hand as the palm facing upward. Similarly, there are cases
            # that the right hand is wrongly identified as the left hand. In
            # either cases, we just always make the z-axis upward.
                        
            if vz[2] < 0:
                
                vz = -vz
                
            vy = np.cross(vz, vx)
            vz = np.cross(vx, vy)
            
            vx = vx / np.linalg.norm(vx)
            vy = vy / np.linalg.norm(vy)
            vz = vz / np.linalg.norm(vz)
                
        
            vx = vx.reshape((3, 1))
            vy = vy.reshape((3, 1))
            vz = vz.reshape((3, 1))
            
            # CAUTION: this is just an approximation of a rotation matrix,
            # and it may not be northonormal!!!
            rotm = np.concatenate((vx, vy, vz), axis=1)
            
            q = Quaternion.construct_from_rotation_matrix(rotm)
            #print(q, q.norm())
#             u = rotation_matrix_to_angle_axis(rotm)
            
            if i > 0:
                  
                pv = qs[i - 1].to_vector()
                qv = q.to_vector()
                 
                p1 = np.linalg.norm(pv - qv)
                p2 = np.linalg.norm(pv + qv)
                 
                if p1 > p2:
                     
                    q = q.negate()
                
                
            
            # Now since it is a unit quaternion encoding the rotation, we
            # convert it back to a rotation matrix for future usage

            #rotm = q.to_rotation_matrix()
            
            qs[i] = q
            rotms[i] = rotm

            # Now we derive the three Tait-Bryan angles
            # CAUTION: yaw first, then pitch, then roll
            # i.e., z-y-x intrinsic rotation (east, north, sky)
            
#             yaw, pitch, roll, gimbal_lock \
#                 = rotation_matrix_to_euler_angles_zyx(rotm)
#  
#             data[i, 9] = roll
#             data[i, 10] = pitch
#             data[i, 11] = yaw
# 
#             assert not gimbal_lock

            # Tait-Bryan angles have singularity.
            # Use quaternion components instead.
            data[i, 9] = q.x
            data[i, 10] = q.y
            data[i, 11] = q.z

#             data[i, 9:12] = u


            if i > 0:
                
                timestep = ts[i] - ts[i - 1]
                if timestep > 0:
                    
                    p = qs[i - 1]
                    omega = Quaternion.differentiate_local(p, q, timestep)
                    
                else:
                    
                    omega = omega_pre

                omega_pre = omega

                data[i, 12] = omega[0] * 1000
                data[i, 13] = omega[1] * 1000
                data[i, 14] = omega[2] * 1000

        data[0, 12] = data[1, 12]
        data[0, 13] = data[1, 13]
        data[0, 14] = data[1, 14]

        
        
        ov = data[:, 12:15].T
        ov = np.matmul(R_offset, ov)
        data[:, 12:15] = ov.T
        
        data[:, 15] = np.gradient(data[:, 12])
        data[:, 16] = np.gradient(data[:, 13])
        data[:, 17] = np.gradient(data[:, 14])

        self.data = data
        self.dim = 18

        self.rotms = rotms
        self.qs = qs



    def prepare_trim_by_speed(self, threshold):
        '''Determine the start and the end when the hand starts move and stops.

        The trimming process is split into two parts to accommodate filtering.
        This method is the first part, which just return the start and end
        offsets without actually throwing away the data samples at the start
        and the end.
        
        This method does not modify the data.
        
        '''

        data = self.data
        l = self.len

        # CAUTION: trim is called before estimating speed, and hence speed is
        # independently calculated here.
        data[:, 3] = np.gradient(data[:, 0])
        data[:, 4] = np.gradient(data[:, 1])
        data[:, 5] = np.gradient(data[:, 2])

        start = 0
        for i in range(l):
            if np.linalg.norm(data[i, 3:6]) > threshold:

                start = i
                break
        if start > 5:
            start -= 5
        end = 0
        for i in range(l - 1, 0, -1):
            if np.linalg.norm(data[i, 3:6]) > threshold:
                end = i
                break
        if end < l - 5 - 1:
            end += 5

        #print(self.id_label, self.seq, l, start, l - end)



        return (start, end)

    def trim(self, start, end):
        '''Trim the start and the end where the hand does not move.

        The trimming process is split into two parts to accommodate filtering.
        This method is the second part, which throws away the data samples at
        according to the start and the end offset.

        This method actually modifies the data.
        
        '''

        l_new = end - start

        self.data = self.data[start:end, :]
        self.ts = self.ts[start:end]

        ts0 = self.ts[0]
        self.ts -= ts0

        self.len = l_new
        
        if hasattr(self, 'qs'):
            
            self.qs = self.qs[start:end]
            
        if hasattr(self, 'rotms'):
            
            self.rotms = self.rotms[start:end]

        if hasattr(self, 'array_tip'):
            
            self.array_tip = self.array_tip[start:end]

        if hasattr(self, 'array_center'):
            
            self.array_center = self.array_center[start:end]

        if hasattr(self, 'data_aux'):
            
            self.data_aux = self.data_aux[start:end]
            self.confs = self.confs[start:end]
            self.valids = self.valids[start:end]



    def amplitude_normalize(self):
        '''Normalize the amplitude of each group of sensor axes.
        
        NOTE: The x-y-z axes of one type, e.g., position, acc, etc., are
        nomalized together.
        
        '''

        data = self.data

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
 
        for j in range(self.dim):
            data[:, j] = np.divide(data[:, j] - mean[j], std[j])

#         for j in range(6):
#  
#             s = j * 3
#             e = (j + 1) * 3
#  
#             axes = data[:, s:e]
#             axes_n = np.linalg.norm(axes, axis=1)
#      
#      
#             mean_axes = np.mean(axes, axis=0)
#             std_axes_n = np.std(axes_n)
#              
#             for i in range(3):
#                  
#                 axes[:, i] = np.divide(axes[:, i] - mean_axes[i], std_axes_n)
#      
#             data[:, s:e] = axes


    def prepare_signal(self, trim_th=1, resample_freq=50):

        self.fix_missing_samples()

        self.resample(resample_freq)
        
        start, end = self.prepare_trim_by_speed(trim_th)
        
        self.pose_normalize(start, end)

        self.estimate_velocity_and_acceleration()

        self.estimate_angular_position_speed_acceleration()
        
        

    def preprocess_shape(self, 
            trim_th=1, filter_cut_freq=10, resample_freq=50):
        '''Preprocess the signal without amplitude normalization.
        
        '''

        self.prepare_signal(trim_th, resample_freq)

        start, end = self.prepare_trim_by_speed(trim_th)

        self.filter(resample_freq, filter_cut_freq)
    
        self.trim(start, end)
        

    def preprocess_shape_augment(self, 
            trim_th=2, filter_cut_freq=10, resample_freq=50,
            p_yaw=0, p_pitch=0, p_roll=0):
        
        self.fix_missing_samples()

        self.resample(resample_freq)

        start, end = self.prepare_trim_by_speed(trim_th)
        
        self.pose_normalize(start, end, p_yaw, p_pitch, p_roll)
        
        self.estimate_velocity_and_acceleration()

        self.estimate_angular_position_speed_acceleration(
            p_yaw, p_pitch, p_roll)

        
        self.filter(resample_freq, filter_cut_freq)
    
        self.trim(start, end)
        
        

    def preprocess(self, trim_th=2, filter_cut_freq=10, resample_freq=50):
        '''
        Preprocess the signal (including amplitude normalization)
        '''

        self.preprocess_shape(trim_th, filter_cut_freq, resample_freq)

        #self.extract_stat_feature()

        self.amplitude_normalize()

        #self.stat_feature_extract_post()

        # assert(self.len != self.data.shape[0])



    # ---------------------------- operations ---------------------------


    def align_to(self, template, window=50, penalty=0, method=DTW_method):
        '''Align the signal to a template (or another signal).
        
        '''
        ts_aligned, data_aligned, l_aligned, dist, a1start, a1end = FMSignal.align_signal_to(
            self, template, window, penalty, method)

        dim = self.dim

        signal_aligned = FMSignalLeap(l_aligned, dim, ts_aligned, data_aligned,
                                   user_label=self.user_label,
                                   id_label=self.id_label,
                                   seq=self.seq)

        signal_aligned.len_origin = self.len
        signal_aligned.dist = dist
        signal_aligned.a1start = a1start
        signal_aligned.a1end = a1end

        if self.stat is not None:
            signal_aligned.stat = self.stat.copy()

        if self.handgeo is not None:
            signal_aligned.handgeo = self.handgeo.copy()
            signal_aligned.handgeo_std = self.handgeo_std.copy()

        return signal_aligned

    def clone(self):
        ''' Deep copy of the signal.
        
        '''
        ts_clone = self.ts.copy()
        data_clone = self.data.copy()

        signal_clone = FMSignalLeap(self.len, self.dim, ts_clone, data_clone,
                                 user_label=self.user_label,
                                 id_label=self.id_label,
                                 seq=self.seq)

        # TODO: handle additional fields in raw format.


        if self.stat is not None:
            signal_clone.stat = self.stat.copy()

        if self.handgeo is not None:
            signal_clone.handgeo = self.handgeo.copy()
            signal_clone.handgeo_std = self.handgeo_std.copy()

        return signal_clone



    @classmethod
    def construct_from_file(cls, fn, mode, point, use_stat, use_handgeo,
                            user_label='', id_label='', seq=0):
        '''Factory method to construct a signal object from a file.
        
        CAUTION: In raw_internal mode, statistical features and hand geometry
        features must be extracted explicitly by calling the extraction
        method. This factory method does not call them!
        
        '''

        signal = FMSignalLeap(0, 0, None, None)
        signal.load_from_file(fn, mode, point,
            use_stat, use_handgeo, user_label, id_label, seq)

        return signal

















class FMSignalGlove(FMSignal):
    '''FMSignal signal collected by the data glove.

    This class is derived from the abstract base class FMSignal. The glove uses
    the BNO055 IMU.

    NOTE: The IMU is set to NDOF mode by default (see BNO055 datasheet
    section 3.3).

    NOTE: The acceleration is linear acceleration with gravity removed.
    Although the BNO055 IMU does provide absolute orientation, we do not
    use the absolute orientation directly obtained from the IMU. Instead,
    we use a very simple method to derive the orientation by integrating
    the angular speed. Since the signal usually has only a few seconds,
    this simple method is good enough.

    Additional Attributes:

        acc:        the linear acceleration directly obtained from the IMU.
        gyro:       the angular speed directly obtained from the IMU.
        gravity:    the gravity vector directly obtained from the IMU.

    NOTE: For a signal loaded from a raw text file, only the acceleration axes
    and the angular speed axes have valid data.

    See "The FMKit Dataset Format" document for more information.
    '''

    def __init__(self, length, dimension, ts, data,
                 user_label='', id_label='', seq=0):
        '''
        Default constructor.
        '''

        FMSignal.__init__(self, length, dimension, ts, data,
                          user_label=user_label, id_label=id_label, seq=seq)

        self.stat = None
        
        self.trajectory = None


    # ---------------------------- input / output ---------------------------

    def load_from_file(self, fn, mode, point='tip', use_stat=False,
                       user_label='', id_label='', seq=0):
        '''General interface to load the signal object from files.
        
        '''

        if mode == 'raw_internal':
            self.load_from_file_raw_internal(fn, point)
        elif mode == 'raw_csv':
            self.load_from_file_raw_csv(fn, point)
        elif mode == 'raw_binary':
            self.load_from_file_raw_binary(fn, point)
        elif mode == 'pp_csv':
            self.load_from_file_pp_csv(fn)
        elif mode == 'pp_binary':
            self.load_from_file_pp_binary(fn)
        else:
            raise ValueError('Unknown file mode %s!' % mode)

        self.user_label = user_label
        self.id_label = id_label
        self.seq = seq

        if use_stat and mode == 'pp_csv':
            self.load_stat_features_from_file_csv(fn + "_ss")
        if use_stat and mode == 'pp_binary':
            self.load_stat_features_from_file_binary(fn + "_ss")


    def save_to_file(self, fn, mode, use_stat):
        '''General interface to dump the signal object to files.
        
        '''

        if mode == 'raw_csv':
            self.save_to_file_raw_csv(fn)
        elif mode == 'raw_binary':
            self.save_to_file_raw_binary(fn)
        elif mode == 'pp_csv':
            self.save_to_file_pp_csv(fn)
        elif mode == 'pp_binary':
            self.save_to_file_pp_binary(fn)
        else:
            raise ValueError('Unknown file mode %s!' % mode)

        if use_stat and mode == 'pp_csv':
            self.save_stat_features_to_file_csv(fn + "_ss")
        if use_stat and mode == 'pp_binary':
            self.save_stat_features_to_file_binary(fn + "_ss")

    def load_from_buffer_raw_internal(self, array, point):
        '''Load from a file directly obtained by the data collection client.

        Args:

            array   The buffer, which is a numpy ndarrary.

        Returns:
        
            None.

        CAUTION: Do not use this method directly. Instead, uses the class 
        method FMSignalGlove.construct_from_file() instead.
        
        '''

        l = array.shape[0]
        d = array.shape[1]
        
        m = 18

        self.len = l
        self.dim = m
        
        # CAUTION: Currently the data has 18 sensor axes!
        data = np.zeros((l, m), dtype=np.float32)

        if d == 25:
            
            # CAUTION: the early format, with only one timestamp.
            ts = array[:, 0].flatten()
            tsc = np.zeros(l, dtype=np.float32)
            
            acc0 = array[:, 1:4]
            gyro0 = array[:, 4:7]
            gravity0 = array[:, 7:10]

            acc1 = array[:, 13:16]
            gyro1 = array[:, 16:19]
            gravity1 = array[:, 19:22]
            
            data[:, 6:9] = acc0
            data[:, 12:15] = gyro0
            
        elif d == 34:
            
            # CATUION: the current format, with two timestamps.
            ts = array[:, 1].flatten()
            tsc = array[:, 0].flatten()
            
            # CAUTION: tsc is in float64!
            self.tsc = tsc
            
            # Use an offset to reduce the size of tsc so that it can fit into
            # the float32 type.
            tsc -= 1514764800

            acc0 = array[:, 2:5]
            gyro0 = array[:, 5:8]
            gravity0 = array[:, 11:14]

            acc1 = array[:, 18:21]
            gyro1 = array[:, 21:24]
            gravity1 = array[:, 27:30]
        
        else:
            
            raise ValueError('Unknown format: d = %d' % d)
        
        
        if point == 'tip':
            data[:, 6:9] = acc0
            data[:, 12:15] = gyro0
        elif point == 'center':
            data[:, 6:9] = acc1
            data[:, 12:15] = gyro1
        else:
            raise ValueError('Unknown point type: %s' % str(point))


        # Fix any timestamp anormaly
        # CAUTION: The device timestamps in glove data are in millisecond.
        for i in range(l - 1):

            if ts[i + 1] < ts[i]:
                ts[i + 1] = ts[i] + 20
                

        self.ts = ts.astype(np.float32)
        self.tsc = tsc.astype(np.float32)
        self.data = data
        
        self.acc0 = acc0.astype(np.float32)
        self.gyro0 = gyro0.astype(np.float32)
        self.gravity0 = gravity0.astype(np.float32)

        self.acc1 = acc1.astype(np.float32)
        self.gyro1 = gyro1.astype(np.float32)
        self.gravity1 = gravity1.astype(np.float32)
        
        self.point = point
        
        
    def load_from_file_raw_internal(self, fn, point):
        '''Load from a file directly obtained by the data collection client.

        Args:

            fn   The file name.

        Returns:
        
            None.

        CAUTION: Do not use this method directly. Instead, uses the class 
        method FMSignalGlove.construct_from_file() instead.

        CAUTION: This method uses an internal file format, which may change
        in the future. All published data format uses a fixed raw format
        instead. See load_from_file_raw_csv() or load_from_file_raw_binary().

        '''

        fn += '.txt'
        array = np.loadtxt(fn, dtype=np.float64, delimiter=',')

        self.load_from_buffer_raw_internal(array, point)

    def prepare_load_from_file_raw(self, array, point):
        '''Converting columns to object fields when loading from a raw file.
        
        CAUTION: This method is only for internal usage.
        
        '''
        
        l = array.shape[0]
        d = array.shape[1]
        
        assert d == 20, 'Wrong raw file format: d = %d' % d
        
        m = 18

        self.len = l
        self.dim = m

        self.tsc = array[:, 0].flatten()
        self.ts = array[:, 1].flatten()

        self.acc0 = array[:, 2:5]
        self.gyro0 = array[:, 5:8]
        self.gravity0 = array[:, 8:11]
        
        self.acc1 = array[:, 11:14]
        self.gyro1 = array[:, 14:17]
        self.gravity1 = array[:, 17:20]

        self.data = np.zeros((l, m), np.float32)

        if point == 'tip':
            self.data[:, 6:9] = self.acc0
            self.data[:, 12:15] = self.gyro0
        elif point == 'center':
            self.data[:, 6:9] = self.acc1
            self.data[:, 12:15] = self.gyro1
        else:
            raise ValueError('Unknown point type: %s' % str(point))

    def load_from_file_raw_csv(self,fn, point):
        '''Load data from a comma separated value (CSV) file in raw fromat.
        
        '''
        
        fn += '.csv'
        array = np.loadtxt(fn, np.float32, delimiter=',')

        self.prepare_load_from_file_raw(array, point)

    def load_from_file_raw_binary(self,fn, point):
        '''Load data from a raw a Numpy binary file (.npy) in raw fromat.
        
        '''
        
        fn += '.npy'
        array = np.load(fn)
        
        assert array.dtype == np.float32

        self.prepare_load_from_file_raw(array, point)

    def prepare_save_to_file_raw(self):
        '''Converting object fields to columns when saving to a raw file.
        
        CAUTION: This method is only for internal usage.
        
        '''
        
        array = np.zeros((self.len, 20), dtype=np.float32)
        
        array[:, 0] = self.tsc
        array[:, 1] = self.ts
        
        array[:, 2:5] = self.acc0
        array[:, 5:8] = self.gyro0
        array[:, 8:11] = self.gravity0
        
        array[:, 11:14] = self.acc1
        array[:, 14:17] = self.gyro1
        array[:, 17:20] = self.gravity1

        return array

    def save_to_file_raw_csv(self,fn):
        '''Save data to a comma separated value (CSV) file in raw fromat.
        
        CAUTION: For floating point numbers, eight digits after the decimal
        point are saved.
        '''
        
        array = self.prepare_save_to_file_raw()
        fn += '.csv'
        np.savetxt(fn, array, fmt='%.8f', delimiter=', ')
        

    def save_to_file_raw_binary(self,fn):
        '''Save data to a raw a Numpy binary file (.npy) in raw fromat.
        
        '''
        
        array = self.prepare_save_to_file_raw()
        fn += '.npy'
        np.save(fn, array)


    def convert_axes_glove2_to_standard_glove(self, axes):
        
        n = axes.shape[0]
        
        temp = np.zeros((n, 1), np.float32)
        
        # x <= y, y <= -x
        
        temp[:, 0] = axes[:, 0]
        axes[:, 0] = -axes[:, 1]
        axes[:, 1] = temp[:, 0]
        
    
    def convert_from_glove2_to_standard_glove(self):
        '''Convert raw data columns from glove v2 to standard reference frame.

        CAUTION: Internal usage only!
        
        '''
        
        data = self.data
        
        n = data.shape[0]
        temp = np.zeros((n, 1), np.float32)

        # liear acceleration columns
        temp[:, 0] = data[:, 6]
        data[:, 6] = -data[:, 7]
        data[:, 7] = temp[:, 0]
        
        # angular speed columns
        temp[:, 0] = data[:, 12]
        data[:, 12] = -data[:, 13]
        data[:, 13] = temp[:, 0]
        
        self.convert_axes_glove2_to_standard_glove(self.acc0)
        self.convert_axes_glove2_to_standard_glove(self.gyro0)
        self.convert_axes_glove2_to_standard_glove(self.gravity0)
        
        self.convert_axes_glove2_to_standard_glove(self.acc1)
        self.convert_axes_glove2_to_standard_glove(self.gyro1)
        self.convert_axes_glove2_to_standard_glove(self.gravity1)
        
    def convert_from_glove1_to_standard_glove(self):
        '''Convert raw data columns from glove v1 to standard reference frame.

        CAUTION: Internal usage only!
        
        CAUTION: The glove v1 is already in standard reference frame.
        
        '''
        
        pass
    
    # ---------------------------- preprocessing ---------------------------

    def offset(self, axis, value):
        
        l = self.len
        
        offset_v = (np.arange(0, l, 1) - l / 2) * value
        
        self.data[:, axis] += offset_v

        self.trajectory[:, axis] += offset_v

    def translate(self, ox, oy, oz):
        '''Translate the coorindate reference frame.
        
        CAUTION: This method works only on raw signals.
        
        '''
        
        o = np.array((ox, oy, oz), dtype=np.float32)
        
        # position translation
        self.data[:, 0:3] -= o
        
    def rotate(self, R):
        '''Rotate the coorindate reference frame.
        
        CAUTION: This method is not implemented yet
        
        '''

        pass

    def estimate_angular_position(self):
        '''Estimate angular position by integration from angular speed.
        
        CAUTION: Currently the beginning of the signal is used as the initial
        pose and the integration goes back to the beginning.
        
        '''

        data = self.data
        l = self.len
        
        
        q = Quaternion.identity()
        
        # rotation matrix of each data sample
        rotms = np.zeros((l, 3, 3))
        
        # quaternion of each data sample
        qs = [None] * l
        
        gyro = data[:, 12:15]
        
        # given 50 Hz, one timestep is 20 ms, i.e., 0.02 second
        timestep = 0.02
        
        for i in range(0, l):
             
            qs[i] = q

            rotm = q.to_rotation_matrix()
 
            rotms[i] = rotm
            
            # CAUTION: Gyro output is in rad/s
            omega = gyro[i]
             
            q = Quaternion.integrate_local(q, omega, timestep)
            q.normalize()
            
            #z, y, x = rotation_matrix_to_euler_angles_zyx(rotm)
            
            data[i, 9] = q.x
            data[i, 10] = q.y
            data[i, 11] = q.z

#             yaw, pitch, roll, gimbal_lock \
#                 = rotation_matrix_to_euler_angles_zyx(rotm)
#   
#             data[i, 9] = roll
#             data[i, 10] = pitch
#             data[i, 11] = yaw
#              
#             assert not gimbal_lock
            
        self.rotms = rotms
        self.qs = qs




    def pose_normalize(self, start, end, p_yaw=0, p_pitch=0, p_roll=0):
        '''Normalize the posture using the average pointing direction.
        
        NOTE: this method expect a start index and an end index, where the
        signal segment between the start and the end is used to calculate the
        average pointing direction. Hence, it should be called after 
        trim_pre(). 
        
        '''
        
        data = self.data
        rotms = self.rotms
        qs = self.qs
        


        vi = np.array((1.0, 0, 0), np.float32).reshape((3, 1))
        #vj = np.array((0, 1.0, 0), np.float32).reshape((3, 1))
        vk = np.array((0, 0, 1.0), np.float32).reshape((3, 1))
        
        # find the approximated average pointing direction as vx
        # find the approximated average downward direction as vz
        vxs = np.matmul(rotms[start:end], vi)
        vzs = np.matmul(rotms[start:end], vk)

        vx = np.mean(vxs, axis=0).flatten()
        vz = np.mean(vzs, axis=0).flatten()

        vx = vx / np.linalg.norm(vx)
        vz = vz / np.linalg.norm(vz)
        
        vy = np.cross(vz, vx)
        vy = vy / np.linalg.norm(vy)
        vz = np.cross(vx, vy)
        vz = vz / np.linalg.norm(vz)

        vx = vx.reshape((3, 1))
        vy = vy.reshape((3, 1))
        vz = vz.reshape((3, 1))

        R_g2l = np.concatenate((vx, vy, vz), axis=1)
        normalize_rotation_matrix(R_g2l)

        R_l2g = R_g2l.transpose()

        R_offset = euler_zyx_to_rotation_matrix(p_yaw, p_pitch, p_roll)
        
        R_l2g = np.matmul(R_offset, R_l2g)

        for i in range(self.len):
            
            rotm = np.matmul(R_l2g, rotms[i])
            
            rotm = normalize_rotation_matrix(rotm)
            
            q = Quaternion.construct_from_rotation_matrix(rotm)
            
            rotms[i] = rotm
            qs[i] = q

            # Now we derive the three Tait-Bryan angles
            # CAUTION: yaw first, then pitch, then roll
            # i.e., z-y-x intrinsic rotation (east, north, sky)
            
#             yaw, pitch, roll, gimbal_lock \
#                 = rotation_matrix_to_euler_angles_zyx(rotm)
#    
#             data[i, 9] = roll
#             data[i, 10] = pitch
#             data[i, 11] = yaw

            # Tait-Bryan angles have singularity.
            # Use quaternion components instead.
            data[i, 9] = q.x
            data[i, 10] = q.y
            data[i, 11] = q.z

            # verify using the gravity vector
            #self.data[i, 3:6] = np.matmul(rotms[i], gravity0[i].reshape((3, 1))).reshape(3)

        # apply pose offset on angular speed
        ov = data[:, 12:15].T
        ov = np.matmul(R_offset, ov)
        data[:, 12:15] = ov.T
        


    def estimate_position_velocity_acceleration(self):
        '''Estimate position and velocity from linear acceleration.
        
        NOTE: This method depends on the orientation.
        
        '''

        data = self.data
        l = self.len

        trajectory = np.zeros((l, 3))

        acc0 = self.data[:, 6:9]

        qs = self.qs

        # given 50 Hz, one timestep is 20 ms, i.e., 0.02 second
        timestep = 0.02
        
        # dead reckoning using linear acceleration
        
        p = np.array((0, 0, 0))
        v_m = np.array((0, 0, 0))
        a_m = np.array((0, 0, 0))
        
        pp = np.array((200, 0, 0))
        
        for i in range(0, l):
 
            acc_local = acc0[i]
             
            q = qs[i]
             
            a_m = q.rotate_a_point(acc_local).reshape(3)
 
 
            data[i, 6:9] = a_m
            data[i, 3:6] = v_m
            data[i, 0:3] = p * 1000 # position is in mm
            
            pp_i = q.rotate_a_point(pp).reshape(3)
            trajectory[i, 0:3] = pp_i
            
            
            v_m = v_m + a_m * timestep

            # Now we add a correction term
            u = np.multiply(v_m, np.abs(v_m))
            v_m = v_m - 0.05 * u
             
            
            
            

            p = p + v_m * timestep + 0.5 * a_m * timestep * timestep

            # Now we add a similar correction term
            w = np.multiply(p, np.abs(p))
            p = p - 2 * w

        trajectory[:, 0] -= 200
        self.trajectory = trajectory

    def estimate_angular_acceleration(self):
        '''Estimate angular acceleration from angular speed.
        
        CAUTION: Currently angular speed is in IMU reference frame, not in the
        world reference frame. Also, angular acceleration is just calculated as
        the differential of angular speed.
        '''
        
        data = self.data

        data[:, 15] = np.gradient(data[:, 12])
        data[:, 16] = np.gradient(data[:, 13])
        data[:, 17] = np.gradient(data[:, 14])


    def prepare_trim_by_acc(self, threshold):
        '''Determine the start and the end when the hand starts move and stops.
        
        NOTE: The movement detection is based on acceleration change, and
        the default threshold is 5 mm/s^2.

        The trimming process is split into two parts to accommodate filtering.
        This method is the first part, which just return the start and end
        offsets without actually throwing away the data samples at the start
        and the end.

        This method does not modify the data.
        
        '''


        data = self.data
        l = self.len

        start_acc = np.zeros((3), np.float32)
        end_acc = np.zeros((3), np.float32)

        for v in range(3):
            start_acc[v] = (data[0, v + 6] 
                            + data[1, v + 6]
                            + data[2, v + 6]) / 3

        for v in range(3):
            end_acc[v] = (data[l - 1, v + 6] + data[l - 2, v + 6] + data[l - 3, v + 6]) / 3

        start = 0
        for i in range(l):
            if np.linalg.norm(data[i, 6:9] - start_acc) > threshold:

                start = i
                break

        end = 0
        for i in range(l - 1, 0, -1):
            if np.linalg.norm(data[i, 6:9] - end_acc) > threshold:
                end = i
                break

        return (start, end)

    def trim(self, start, end):
        '''Trim the start and the end where the hand does not move.

        The trimming process is split into two parts to accommodate filtering.
        This method is the second part, which throws away the data samples at
        according to the start and the end offset.

        This method actually modifies the data.
        
        '''

        l_new = end - start

        self.data = self.data[start:end, :]
        self.ts = self.ts[start:end]

        ts0 = self.ts[0]
        self.ts -= ts0

        self.len = l_new
        
        if self.qs is not None:
            
            self.qs = self.qs[start:end]
            
        if self.rotms is not None:
            
            self.rotms = self.rotms[start:end]
            
        if self.trajectory is not None:
            
            self.trajectory = self.trajectory[start:end]
            

    def amplitude_normalize(self):
        '''Normalize the amplitude of each group of sensor axes.
        
        NOTE: The x-y-z axes of one type, e.g., position, acc, etc., are
        nomalized together.
        
        '''

        data = self.data

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        for j in range(self.dim):
            data[:, j] = np.divide(data[:, j] - mean[j], std[j])
            
            
#         for j in range(6):
# 
#             s = j * 3
#             e = (j + 1) * 3
# 
#             axes = data[:, s:e]
#             axes_n = np.linalg.norm(axes, axis=1)
#     
#     
#             mean_axes = np.mean(axes, axis=0)
#             std_axes_n = np.std(axes_n)
#             
#             for i in range(3):
#                 
#                 axes[:, i] = np.divide(axes[:, i] - mean_axes[i], std_axes_n)
#     
#             data[:, s:e] = axes

    def preprocess_shape(self, trim_th=2, filter_cut_freq=10):
        '''Preprocess the signal without amplitude normalization.
        
        '''
        
        # CAUTION: The glove device samples at 50 Hz, so no need to resample.

        start, end = self.prepare_trim_by_acc(trim_th)

        if start >= end:
            
            print(self, start, end)
        
        assert start < end

        self.estimate_angular_position()
        self.pose_normalize(start, end)

 
        self.estimate_position_velocity_acceleration()
        self.estimate_angular_acceleration()

        sample_freq = self.len / (self.ts[self.len - 1]) * 1000.0
  
        self.filter(sample_freq, filter_cut_freq)

        start, end = self.prepare_trim_by_acc(trim_th)
        self.trim(start, end)

        #tn = time.time()
        
        #print(tn - t0)

    def preprocess_shape_augment(self, 
            trim_th=2, filter_cut_freq=10,
            p_yaw=0, p_pitch=0, p_roll=0):
        
        start, end = self.prepare_trim_by_acc(trim_th)

        self.estimate_angular_position()
        self.pose_normalize(start, end, p_yaw, p_pitch, p_roll)

 
        self.estimate_position_velocity_acceleration()
        self.estimate_angular_acceleration()

        sample_freq = self.len / (self.ts[self.len - 1]) * 1000.0
  
        self.filter(sample_freq, filter_cut_freq)

        start, end = self.prepare_trim_by_acc(trim_th)
        self.trim(start, end)




    def preprocess(self, trim_th=2, filter_cut_freq=10):
        '''Preprocess the signal (including amplitude normalization).
        
        '''
        
        self.preprocess_shape(trim_th, filter_cut_freq)


        #self.extract_stat_feature()
 
        self.amplitude_normalize()
 
        #self.stat_feature_extract_post()


    # ---------------------------- operations ---------------------------


    def align_to(self, template, window=50, penalty=0, method=DTW_method):
        '''Align the signal to a template (or just another signal).
        
        '''

        ts_aligned, data_aligned, l_aligned, dist, a1start, a1end \
            = FMSignal.align_signal_to(
                self, template, window, penalty, method)

        d = self.dim

        signal_aligned = FMSignalGlove(l_aligned, d, ts_aligned, data_aligned, \
                                       user_label=self.user_label, \
                                       id_label=self.id_label, \
                                       seq=self.seq)

        signal_aligned.len_origin = self.len
        signal_aligned.dist = dist
        signal_aligned.a1start = a1start
        signal_aligned.a1end = a1end

        if self.stat is not None:
            signal_aligned.stat = self.stat.copy()
        
        return signal_aligned

    def clone(self):
        '''Deep copy of the signal.
        
        '''
        ts_clone = self.ts.copy()
        data_clone = self.data.copy()

        signal_clone = FMSignalGlove(self.len, self.dim, ts_clone, data_clone,
                                  user_label=self.user_label, \
                                  id_label=self.id_label, \
                                  seq=self.seq)
        
        if self.stat is not None:
            signal_clone.stat = self.stat.copy()

        return signal_clone


    @classmethod
    def construct_from_file(cls, fn, mode, point, use_stat,
                            user_label='', id_label='', seq=0):
        '''
        Factory method to build the signal by loading a file.
        '''

        signal = FMSignalGlove(0, 0, None, None)
        signal.load_from_file(fn, mode, point, use_stat, user_label, id_label, seq)

        return signal



















class FMSignalTemplate(FMSignal):
    '''
    FMSignal signal template constructed from a collection of signals.

    In most cases, template is identical to a signal. However, there are a few
    additional fields including:

    variance:   the variance of the samples constructing the template.
                The variance is calculates at sample level, so it has the same
                dimension as the data.

    Note that this class is the abstract class. Do not instantiate it directly.
    Instead, use the derived class listed below such as FMSignalTemplateLeap or
    FMSignalTemplateGlove

    '''

    def __init__(self, length, dimension, ts, data,
                 user_label='', id_label='', seq=0,
                 variance=None, signals_aligned=None):
        '''Default constructor.
        '''

        FMSignal.__init__(self, length, dimension, ts, data,
                          user_label=user_label, id_label=id_label, seq=seq)

        self.variance = variance
        self.signals_aligned = signals_aligned

    def load_from_file(self, fn, mode,
                       user_label='', id_label='', seq=0):
        '''General interface to load the template object from a file.
        
        '''

        if mode == 'csv':
            self.load_from_file_csv(fn)
        elif mode == 'binary':
            self.load_from_file_binary(fn)
        else:
            raise ValueError('Unknown file mode %s!' % mode)

        self.user_label = user_label
        self.id_label = id_label
        self.seq = seq

    def save_to_file(self, fn, mode):
        '''General interface to save the template object to a file.
        
        '''

        if mode == 'csv':
            self.save_to_file_csv(fn)
        elif mode == 'binary':
            self.save_to_file_binary(fn)
        else:
            raise ValueError('Unknown file mode %s!' % mode)

    def load_from_file_csv(self, fn):
        '''Load the template from a CSV file.

        '''

        fn += '.csv'
        array = np.loadtxt(fn, dtype=np.float32, delimiter=',')

        ts = array[:, 0:1]
        data_and_varance = array[:, 1:]

        l = data_and_varance.shape[0]
        d = int(data_and_varance.shape[1] / 2)

        data = data_and_varance[:, :d]
        variance = data_and_varance[:, d:]

        self.len = l
        self.dim = d
        self.ts = ts[:l]
        self.data = data[:l, :]

        self.variance = variance
        self.signals_aligned = None

    def load_from_file_binary(self, fn):
        '''Load the template from a Numpy binary file (.npy).
        
        '''

        fn += ".npy"
        array = np.load(fn)

        assert array.dtype == np.float32

        l = array.shape[0]
        d = (array.shape[1] - 1) // 2

        ts = array[:, 0:1]
        data = array[:, 1:(d + 1)]
        variance = array[:, (d + 1):]

        self.len = l
        self.dim = d
        self.ts = ts[:l]
        self.data = data[:l, :]

        self.variance = variance
        self.signals_aligned = None

    def save_to_file_csv(self, fn):
        '''Save the template to a csv file.
        
        '''

        l = self.len

        array = np.concatenate((self.ts.reshape((l, 1)), 
                                self.data, 
                                self.variance), axis=1)
        
        fn += '.csv'
        np.savetxt(fn, array, fmt='%.6f', delimiter=', ')

    def save_to_file_binary(self, fn):
        '''Save the template to a Numpy binary file (.npy).
        
        '''
        
        l = self.len

        array = np.concatenate((self.ts.reshape((l, 1)), 
                                self.data, 
                                self.variance), axis=1)
        
        assert array.dtype == np.float32
        
        np.save(fn, array)



    # ---------------------------- statistical features ----------------------

    def load_stat_features_from_file_csv(self, fn):
        '''Load statistical features from a comma separated value (CSV) file.
        
        '''

        fn += '.csv'
        array = np.loadtxt(fn, dtype=np.float32, delimiter=',')
        shape = array.shape
        n = array.shape[0] // 2
        array1 = array[0:n]
        array2 = array[n:-1]
        self.stat = array1.reshape((shape[0] // 5, 5, self.dim))
        self.sstd = array2.reshape((shape[0] // 5, 5, self.dim))

    def load_stat_features_from_file_binary(self, fn):
        '''Load the statistical features from a Numpy binary file (.npy).
        
        '''

        fn += '.npy'
        array = np.load(fn)
        
        n = array.shape[0] // 2
        self.stat = array[0:n]
        self.sstd = array[n:array.shape[0]]
        

    def save_stat_features_to_file_csv(self, fn):
        '''Save statistical features to a comma separated value (CSV) file.
        
        An exception is raised if it does not have such features extracted.
        
        '''

        assert (self.stat is not None), \
            'The template does not have stat features.'
        shape = self.stat.shape
        array1 = self.stat.reshape((shape[0] * shape[1], shape[2]))
        array2 = self.sstd.reshape((shape[0] * shape[1], shape[2]))

        array = np.concatenate((array1, array2), axis=0)

        fn += '.csv'
        np.savetxt(fn, array, fmt='%.6f', delimiter=', ')



    def save_stat_features_to_file_binary(self, fn):
        '''Save the statistical features to a Numpy binary file (.npy).
        
        '''
        
        assert (self.stat is not None), \
            'The template does not have stat features.'
        array = np.concatenate((self.stat, self.sstd), axis=0)
        np.save(fn, array)


    # ---------------------------- operations ----------------------


    def update(self, new_signal, factor, use_stat):
        '''Update the template with a new signal.

        new_signal: the new signal S used to update the template.
        factor: update factor, i.e., T_{new} = (1 - factor) * T + S * factor

        CAUTION: The new signal must be aligned to the template
        
        '''

        # update the signal template
        self.data = self.data * (1 - factor) + new_signal.data * factor

        # update the statistical features
        if use_stat:
            self.stat = self.stat * (1 - factor) + new_signal.stat * factor


    @classmethod
    def prepare_stat_features_from_signals(cls, signals, template_index):
        '''Construct the statistical features of the template.
        
        This method uses the statistical features of a set of signals to
        compute the mean and standard deviation of them. They are the stat
        features of the template.

        NOTE: This method is only called by the derived class. Do not use
        this method directly.
        
        '''
        
        template = signals[template_index]

        k = len(signals)

        sshape = (k, ) + template.stat.shape

        sstat = np.zeros(sshape)

        for i, signal in enumerate(signals):

            assert (signal.stat is not None), 'Error in constructing ' \
                + 'statistical features of the template: ' \
                + 'stat of the signal is missing.'

            sstat[i] = signal.stat

        stat = np.mean(sstat, axis=0)
        sstd = np.std(sstat, axis=0)

        return stat, sstd


    @classmethod
    def prepare_construction_from_signals(cls, signals, template_index,
                                          window=50, penalty=0):
        '''Construct the template by aligning and average a set of signals.

            signals:        the set of signals
            template_index: the index of signal which others align to.
            window:         alignment window, see utils.dtw().
            penalty:        element level misalign penalty, see utils.dtw().

        NOTE: This method is only called by the derived class. Do not use
        this method directly.

        '''

        template = signals[template_index]
        signals_aligned = [template]

        k = len(signals)

        # construct signal template

        data = template.data.copy()
        ts = template.ts.copy()

        variance = np.zeros(template.data.shape, template.data.dtype)

        for signal in signals:

            if signal == template:

                continue

            signal_aligned = signal.align_to(template, window, penalty)
            signals_aligned.append(signal_aligned)

            data += signal_aligned.data

        data /= k

        for signal_aligned in signals_aligned:

            variance += np.square(signal_aligned.data - data)

        variance /= k


        return ts, data, variance, signals_aligned
















class FMSignalTemplateLeap(FMSignalTemplate, FMSignalLeap):
    '''Finger motion signal template from FMSignalLeap.

    This class is derived from both FMSignalTemplate and FMSignalLeap.

    '''

    def __init__(self, length, dimension, ts, data,
                 user_label='', id_label='', seq=0,
                 variance=None, signals_aligned=None):
        '''Default constructor.
        
        '''

        FMSignalTemplate.__init__(self, length, dimension, ts, data, \
                                  user_label=user_label, \
                                  id_label=id_label, \
                                  seq=seq, \
                                  variance=variance, \
                                  signals_aligned=signals_aligned)

        FMSignalLeap.__init__(self, length, dimension, ts, data,
                              user_label=user_label, \
                              id_label=id_label, \
                              seq=seq)

        self.stat = None
        self.handgeo = None
        self.handgeo_std = None

    # ---------------------------- input / output ---------------------------

    def load_from_file(self, fn, mode, use_stat, use_handgeo,
                       user_label='', id_label='', seq=0):
        '''General interface to dump the signal object to files.
        
        CAUTION: FMSignalTemplate overrides the method on loading and saving
        statistical features, but it does not override the method on loading
        and saving hand geometry features. For hand geometry features, the
        loading and saving are done by the corresponding methods provided by
        FMSignalLeap.
        
        '''

        FMSignalTemplate.load_from_file(
            self, fn, mode, user_label, id_label, seq)

        if use_stat and mode == 'csv':
            self.load_stat_features_from_file_csv(fn + '_ss')
        elif use_stat and mode == 'binary':
            self.load_stat_features_from_file_binary(fn + '_ss')

        if use_handgeo and mode == 'csv':
            self.load_handgeo_features_from_file_csv(fn + 'hg')
        if use_handgeo and mode == 'binary':
            self.load_handgeo_features_from_file_binary(fn + 'hg')

    def save_to_file(self, fn, mode, use_stat, use_handgeo):
        '''General interface to save the tem to files.

        CAUTION: FMSignalTemplate overrides the method on loading and saving
        statistical features, but it does not override the method on loading
        and saving hand geometry features. For hand geometry features, the
        loading and saving are done by the corresponding methods provided by
        FMSignalLeap.
        
        '''

        FMSignalTemplate.save_to_file(self, fn, mode)

        if use_stat and mode == 'csv':
            self.save_stat_features_to_file_csv(fn + '_ss')
        elif use_stat and mode == 'binary':
            self.save_stat_features_to_file_binary(fn + '_ss')

        if use_handgeo and mode == 'csv':
            self.save_handgeo_features_to_file_csv(fn + 'hg')
        if use_handgeo and mode == 'binary':
            self.save_handgeo_features_to_file_binary(fn + 'hg')

    # ---------------------------- operations ---------------------------

    @classmethod
    def prepare_handgeo_features_from_signals(cls, signals, template_index):
        '''Construct the hand geometry features of the template.
        
        NOTE: This method is only called by construct_from_signals(). Do not
        use this method directly.
        '''
        
        signal_t = signals[template_index]
        k = len(signals)

        assert (signal_t.handgeo is not None), 'Error in constructing ' \
            + 'hand geometry features of the template: ' \
            + 'handgeo of the signal is missing.'

        handgeo = signal_t.handgeo.copy()
        handgeo_std = signal_t.handgeo_std.copy()
        for signal in signals:

            if signal == signal_t:

                continue

            assert (signal.handgeo is not None), 'Error in constructing ' \
                + 'hand geometry features of the template: ' \
                + 'handgeo of the signal is missing.'

            handgeo += signal.handgeo
            handgeo_std += signal.handgeo_std

        handgeo /= k
        handgeo_std /= k

        return handgeo, handgeo_std

    @classmethod
    def construct_from_signals(cls, signals, template_index, 
                               use_stat, use_handgeo,
                               window=50, penalty=0):
        '''Factory method to construct the template by a set of signals.
        
        '''

        signal_t = signals[template_index]
        l = signal_t.len
        d = signal_t.dim

        # CAUTION: All aligned signals have the same length as the template
        # signal.
        tup = FMSignalTemplate.prepare_construction_from_signals( \
                signals, template_index, window, penalty)
        (ts, data, variance, signals_aligned) = tup

        template = cls(l, d, ts, data, 
                       signal_t.user_label, signal_t.id_label, signal_t.seq,
                       variance, signals_aligned)

        if use_stat:
            
            stat, sstd = cls.prepare_stat_features_from_signals(
                signals, template_index)
            
            template.stat = stat
            template.sstd = sstd

        else:
            
            template.stat = None
            template.sstd = None

        # hand geometry
        # CAUTION: We just calculated the arithmetic mean of the hand geometry
        # vector from the template building signals.
        if use_handgeo:

            handgeo, handgeo_std = cls.prepare_handgeo_features_from_signals(
                signals, template_index)

            template.handgeo = handgeo
            template.handgeo_std = handgeo_std

        else:
            template.handgeo = None
            template.handgeo_std = None


        return template

    @classmethod
    def construct_from_file(cls, fn, mode, use_stat, use_handgeo,
                            user_label='', id_label='', seq=0):
        '''
        Factory method to build the signal by loading a file.
        '''

        template = cls(0, 0, None, None)

        template.load_from_file(
            fn, mode, use_stat, use_handgeo, user_label, id_label, seq)

        return template



























class FMSignalTemplateGlove(FMSignalTemplate, FMSignalGlove):
    '''Finger motion signal template from FMSignalLeap.

    This class is derived from both FMSignalTemplate and FMSignalGlove.

    '''

    def __init__(self, length, dimension, ts, data,
                 user_label='', id_label='', seq=0,
                 variance=None, signals_aligned=None):
        '''Default constructor.
        
        '''

        FMSignalTemplate.__init__(self, length, dimension, ts, data, \
                                  user_label=user_label, \
                                  id_label=id_label, \
                                  seq=seq, \
                                  variance=variance, \
                                  signals_aligned=signals_aligned)

        FMSignalGlove.__init__(self, length, dimension, ts, data, \
                               user_label=user_label, \
                               id_label=id_label, \
                               seq=seq)

    # ---------------------------- input / output ---------------------------

    def load_from_file(self, fn, mode, use_stat,
                       user_label='', id_label='', seq=0):
        '''General interface to dump the signal object to files.
        
        CAUTION: FMSignalTemplate overrides the method on loading and saving
        statistical features.
        
        '''

        FMSignalTemplate.load_from_file(
            self, fn, mode, user_label, id_label, seq)

        if use_stat and mode == 'csv':
            self.load_stat_features_from_file_csv(fn + '_ss')
        elif use_stat and mode == 'binary':
            self.load_stat_features_from_file_binary(fn + '_ss')


    def save_to_file(self, fn, mode, use_stat):
        '''General interface to save the tem to files.

        CAUTION: FMSignalTemplate overrides the method on loading and saving
        statistical features.
        
        '''

        FMSignalTemplate.save_to_file(self, fn, mode)

        if use_stat and mode == 'csv':
            self.save_stat_features_to_file_csv(fn + '_ss')
        elif use_stat and mode == 'binary':
            self.save_stat_features_to_file_binary(fn + '_ss')


    # ---------------------------- operations ---------------------------

    @classmethod
    def construct_from_signals(cls, signals, template_index, use_stat,
                               window=50, penalty=0):
        '''Factory method to construct the template by aligning and average a set
        of signals.
        
        '''

        signal_t = signals[template_index]
        l = signal_t.len
        d = signal_t.dim

        tup = FMSignalTemplate.prepare_construction_from_signals( \
                signals, template_index, window, penalty)
        (ts, data, variance, signals_aligned) = tup

        template = cls(l, d, ts, data,
                       signal_t.user_label, signal_t.id_label, signal_t.seq,
                       variance, signals_aligned)

        if use_stat:
            
            stat, sstd = cls.prepare_stat_features_from_signals(
                signals, template_index)
            
            template.stat = stat
            template.sstd = sstd

        else:
            
            template.stat = None
            template.sstd = None

        return template

    @classmethod
    def construct_from_file(cls, fn, mode, use_stat,
                            user_label='', id_label='', seq=0):
        '''Factory method to build the signal by loading a file.
        '''

        template = cls(0, 0, None, None)

        template.load_from_file(fn, mode, use_stat, user_label, id_label, seq)

        return template














