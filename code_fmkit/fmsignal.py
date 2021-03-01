"""
This module is the core of the fmkit framework, which is designed to facilitate
researches on in-air-handwriting related research.

Author: Duo Lu <<duolu.cs@gmail.com>>

Version: 0.1
License: MIT

Updated on Feb. 7, 2020, version 0.1

Created on Aug 14, 2017, draft


The MIT License

Copyright 2017-2021 Duo Lu <<duolu.cs@gmail.com>>

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

"""


import time
import csv
import math

import numpy as np
#import scipy.stats
#import scipy.signal

from pyrotation import Quaternion
from pyrotation import euler_zyx_to_rotation_matrix
from pyrotation import rotation_matrix_to_euler_angles_zyx
from pyrotation import normalize_rotation_matrix

try:
    import fmkit_utilities

    DTW_METHOD = "c"
    # print('fmkit_utilities installed')
except ImportError:
    DTW_METHOD = "python"
    # print('fmkit_utilities not installed')


def dtw(series_1, series_2, window=0, penalty=0):
    """Dynamic Time Warping (DTW) on two multidimensional time series.

    **SEE**:
    [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping)

    **NOTE**: This is the python implementation. It iteratively accesses each
    element of the time series array, which could be very slow.

    Args:

        series_1 (ndarray): The first array, n-by-d numpy ndarray.
        series_2 (ndarray): The second array, m-by-d numpy ndarray.
        window (int): The window constraint, by default no constraint.
        penalty (float): The misalign penalty, by default zero.

    Returns:
        
        tuple: A tuple containing the following.

            int: The final DTW distance.
            ndarray: The (n + 1)-by-(m + 1) distance matrix.
            ndarray: The (n + 1)-by-(m + 1) direction matrix (warp path).
            ndarray: The alignment starting indices from 2 to 1.
            ndarray: The alignment ending indices from 2 to 1.
            ndarray: The alignment starting indices from 1 to 2.
            ndarray: The alignment ending indices from 1 to 2.
            ndarray: The generated time series by aligning 2 to 1.
    
    Raises:

        ValueError: If the input series have incompatible dimensions.

    """

    if not isinstance(series_1, np.ndarray) \
        or not isinstance(series_2, np.ndarray) \
        or len(series_1.shape) != 2 or len(series_2.shape) != 2 \
        or series_1.dtype != np.float32 or series_2.dtype != np.float32:

        raise ValueError('Input series must be l-by-d NumPy ndarrays!')

    n = series_1.shape[0]
    m = series_2.shape[0]

    d1 = series_1.shape[1]
    d2 = series_2.shape[1]

    if d1 != d2:
        raise ValueError('Input series must have the same dimension!')

    if series_1.dtype != np.float32 or series_2.dtype != np.float32:
        raise ValueError('Input series must have "dtype == np.float32"!')

    # NOTE: By default, the window is set to a sufficiently large value (m * 2)
    # to essentially remove the window constraint.
    if window <= 0:
        window = m * 2

    # These are the distance matrix and the direction table.
    dist_matrix = np.zeros((n + 1, m + 1), np.float32)
    direction = np.zeros((n + 1, m + 1), np.int32)

    # These are the index mapping to align series_2 to series_1.
    a2to1_start = np.zeros(n, np.int32)
    a2to1_end = np.zeros(n, np.int32)

    # These are the index mapping to align series_1 to series_2.
    a1to2_start = np.zeros(m, np.int32)
    a1to2_end = np.zeros(m, np.int32)

    # Initialization.
    dist_matrix.fill(1e6)

    dist_matrix[0, 0] = 0
    direction[0, 0] = 0

    #dist_matrix[:, 0] = 0
    #dist_matrix[0, :] = 0


    # find the warping path
    for i in range(1, n + 1):

        jj = int(float(m) / n * i)
        start = jj - window if jj - window > 1 else 1
        end = jj + window if jj + window < m + 1 else m + 1

        for j in range(start, end):

            # CAUTION: series_1[0] and series_2[0] mapps to dists[1][1],
            # and i, j here are indexing dists instead of the series,
            # i.e., dists[i][j] is comparing series_1[i - 1] and series_2[j - 1]
            cost = np.linalg.norm(series_1[i - 1] - series_2[j - 1])

            min_dist = dist_matrix[i - 1, j - 1]
            direction[i, j] = 1  # 1 stands for diagonal

            if dist_matrix[i - 1, j] + penalty < min_dist:

                min_dist = dist_matrix[i - 1, j] + penalty
                direction[i, j] = 2  # 2 stands for the i direction

            if dist_matrix[i][j - 1] + penalty < min_dist:

                min_dist = dist_matrix[i][j - 1] + penalty
                direction[i, j] = 4  # 4 stands for the j direction

            dist_matrix[i][j] = cost + min_dist

    # trace back the warping path to find element-wise mapping

    # print('warping path done')

    a2to1_start[n - 1] = m - 1
    a2to1_end[n - 1] = m - 1
    a1to2_start[m - 1] = n - 1
    a1to2_end[m - 1] = n - 1

    i = n
    j = m
    while True:

        if direction[i, j] == 2:  # the i direction

            i -= 1

            a2to1_start[i - 1] = j - 1
            a2to1_end[i - 1] = j - 1
            a1to2_start[j - 1] = i - 1

        elif direction[i, j] == 4:  # the j direction

            j -= 1

            a1to2_start[j - 1] = i - 1
            a1to2_end[j - 1] = i - 1
            a2to1_start[i - 1] = j - 1

        elif direction[i, j] == 1:  # the diagonal direction

            i -= 1
            j -= 1
            if i == 0 and j == 0:
                break

            a2to1_start[i - 1] = j - 1
            a2to1_end[i - 1] = j - 1
            a1to2_start[j - 1] = i - 1
            a1to2_end[j - 1] = i - 1

        else:  # direction[i][j] == 0, the corner
            break

    series_2to1 = np.zeros(series_1.shape, series_1.dtype)

    for i, jj, kk in zip(range(n), a2to1_start, a2to1_end):

        if jj == kk:

            series_2to1[i] = series_2[jj]

        else:

            series_2to1[i] = np.mean(series_2[jj : kk + 1, :], axis=0)

    return dist_matrix[n, m], dist_matrix[1:, 1:], direction, \
           a2to1_start, a2to1_end, a1to2_start, a1to2_end, series_2to1


def dtw_c(series_1, series_2, window=0, penalty=0):
    """Python wrapper of the Dynamic Time Warping (DTW) implemented in C.

    **SEE**:
    [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping)

    **SEE**: The "fmkit_cutils" package in this project has the implementation.

    **NOTE**: This is a Python wrapper of the C implementation, which is
    relative fast compared to the pure Python implementation.

    **NOTE**: The input series must have "dtype == np.float32".

    Args:

        series_1 (ndarray): The first array, n-by-d numpy ndarray.
        series_2 (ndarray): The second array, m-by-d numpy ndarray.
        window (int, optional): The window constraint, by default no constraint.
        penalty (float, optional): The misalign penalty, by default zero.

    Returns:
        
        tuple: A tuple containing the following.

            int: The final DTW distance.
            ndarray: The (n + 1)-by-(m + 1) distance matrix.
            ndarray: The (n + 1)-by-(m + 1) direction matrix (warp path).
            ndarray: The alignment starting indices from 2 to 1.
            ndarray: The alignment ending indices from 2 to 1.
            ndarray: The alignment starting indices from 1 to 2.
            ndarray: The alignment ending indices from 1 to 2.
            ndarray: The generated time series by aligning 2 to 1.
    
    Raises:

        ValueError: If the input series have incompatible dimensions.

    """

    if not isinstance(series_1, np.ndarray) \
        or not isinstance(series_2, np.ndarray) \
        or len(series_1.shape) != 2 or len(series_2.shape) != 2:

        raise ValueError('Input series must be l-by-d NumPy ndarrays!')
    
    n = series_1.shape[0]
    m = series_2.shape[0]

    d1 = series_1.shape[1]
    d2 = series_2.shape[1]

    if d1 != d2:
        raise ValueError('Input series must have the same dimension!')

    if series_1.dtype != np.float32 or series_2.dtype != np.float32:
        raise ValueError('Input series must have "dtype == np.float32"!')

    if window <= 0:
        window = m * 2

    # print('begin dtw_c()')

    # NOTE: It may be problematic to manage the memory and Python object
    # life-cycle in C code, and hence, we allocate space here in the wrapper.

    dist_matrix = np.zeros((n + 1, m + 1), np.float32)
    direction = np.zeros((n + 1, m + 1), np.int32)

    # These are the index mapping to align series_2 to series_1.
    a2to1_start = np.zeros(n, np.int32)
    a2to1_end = np.zeros(n, np.int32)

    # These are the index mapping to align series_1 to series_2.
    a1to2_start = np.zeros(m, np.int32)
    a1to2_end = np.zeros(m, np.int32)

    series_2to1 = np.zeros(series_1.shape, np.float32)

    # Initialization.
    dist_matrix.fill(1e6)

    dist = fmkit_utilities.dtw_c(
        series_1,
        series_2,
        series_1.shape[1],
        n,
        m,
        window,
        penalty,
        dist_matrix,
        direction,
        a2to1_start,
        a2to1_end,
        a1to2_start,
        a1to2_end,
        series_2to1,
    )

    # print('end dtw_c()')

    return dist, dist_matrix[1:, 1:], direction, \
        a2to1_start, a2to1_end, a1to2_start, a1to2_end, series_2to1

def normalize_warping_path_a2to1(a2to1_start, a2to1_end, n=100):
    """Normalize the warping path from l2-by-l1 to n-by-n.

    Args:

        a2to1_start (ndarray): The alignment starting indices from 2 to 1.
        a2to1_end (ndarray): The alignment ending indices from 2 to 1.
        n (int): The size of the warping path after normalization.

    Returns:

        (ndarray): The "a2to1_start" after normalization.
        (ndarray): The "a2to1_end" after normalization.

    Raises:

        ValueError: If the input arrays have wrong type or value.

    """

    if not isinstance(a2to1_start, np.ndarray) \
        or not isinstance(a2to1_end, np.ndarray) \
        or len(a2to1_start.shape) != 1 or len(a2to1_end.shape) != 1 \
        or a2to1_start.dtype != np.int or a2to1_end.dtype != np.int:

        raise ValueError('Input series must be two 1D NumPy arrays!')

    assert a2to1_start.shape[0] == a2to1_end.shape[0]
    assert np.all((a2to1_end - a2to1_start) >= 0)
    assert n > 0

    l1 = a2to1_start.shape[0]
    l2 = a2to1_end[-1]
    
    xp = np.arange(l1)
    x_n = np.linspace(0, l1 - 1, n)

    a2to1_start_n = np.interp(x_n, xp, a2to1_start)
    a2to1_start_n = a2to1_start_n / l2 * (n - 1)
    
    a2to1_end_n = np.interp(x_n, xp, a2to1_end)
    a2to1_end_n = a2to1_end_n / l2 * (n - 1)

    return a2to1_start_n.astype(np.int32), a2to1_end_n.astype(np.int32)

def warping_path_to_xy_sequences(a2to1_start, a2to1_end):
    """Convert the warping path from alignment indices to xy coordinates.

    Args:

        a2to1_start (ndarray): The alignment starting indices from 2 to 1.
        a2to1_end (ndarray): The alignment ending indices from 2 to 1.

    Returns:

        (list): The x coordinates.
        (list): The y coordinates.

    Raises:

        ValueError: If the input arrays have wrong type or value.


    """

    if not isinstance(a2to1_start, np.ndarray) \
        or not isinstance(a2to1_end, np.ndarray) \
        or len(a2to1_start.shape) != 1 or len(a2to1_end.shape) != 1 \
        or a2to1_start.dtype != np.int or a2to1_end.dtype != np.int:

        raise ValueError('Input series must be two 1D NumPy arrays!')

    assert a2to1_start.shape[0] == a2to1_end.shape[0]
    assert np.all((a2to1_end - a2to1_start) >= 0)

    xs = []
    ys = []
    
    for i, (start, end) in enumerate(zip(a2to1_start, a2to1_end)):
        
        for j in range(start, end + 1):
            
            xs.append(i)
            ys.append(j)
    
            #print(i, j, start, end)
    
    return xs, ys

class FMSignalDesc(object):
    """A descriptor of a set of signals of the same writing content.

    Typically, these signals are generated by users or imposters writing the 
    same content in multiple repetitions. If the contents are different, they
    are considered as different set of signals with different descriptors.

    **NOTE**: One descriptor object corresponds to one line in the meta file.

    Attributes:

        uid (int):  A unique ID that orders all descriptors in a collections.
                    Usually it is the line number of the descriptor in the meta
                    file.

        user (string): A label indicating which user generates the signals.

        cid (string): A label indicating the content for classification tasks.
                    Note that for spoofing attacks, i.e., different users are
                    asked to write the same content, the signals have the same
                    id_label. This label is used as the account ID for user
                    identification and authentication purpose, and hence it
                    gets the name "id_label".

        device (string): This indicates which type of device is used to obtain
                    the signals. Currently it is either "glove" or "leap".

        start (int): The start repetition sequence number (inclusive).

        end (int):  The end repetition sequence number (exclusive). These
                    sequence numbers allow the database to only load a
                    specified section of the data, which is very useful when
                    spliting the dataset into training and testing set or
                    dealing with data augmentation.

        fn_prefix (string): File name prefix, typically "user_cid". The full
                    file name is "fn_prefix" + "_" + "seq" + ".csv" or ".npy",
                    where "seq" indicate the specific repetition.

        content (string): The actual content that is written.



    """

    def __init__(self, uid, user, cid, 
        device, start, end, fn_prefix, content):
        """Constructor.

        See the class attributes for the meaning of the arguments.

        """

        self.uid = uid
        self.user = user
        self.cid = cid
        self.device = device

        self.start = start
        self.end = end

        self.fn_prefix = fn_prefix
        self.content = content

    @classmethod
    def construct_from_meta_file(cls, meta_fn):
        """Factory method to build list of FMSignalDesc from a metadata file.

        Args:

            meta_fn (string): The metadata file name.
        
        Returns:

            list: a list of descriptor objects.

        **NOTE**: The meta file contains a table with the following structure,
        (columns are seperated by commas):

        column |  type   |  content
        -------|---------|--------------
            0  |  int    |  uid
            1  |  string |  user
            2  |  string |  cid
            3  |  string |  device
            4  |  int    |  start repetition sequence number
            5  |  int    |  end repetition sequence number
            6  |  string |  file name prefix
            7  |  string |  content

        Typically, the file name prefix field is just "user_label_id_label".

        See the "data_example" and "meta_example" folders for examples.

        """

        descs = []

        with open(meta_fn, "r") as meta_fd:
            reader = csv.reader(meta_fd)

            for row in reader:

                if len(row) == 0:

                    continue

                if row[0].startswith("#"):

                    continue

                strs = []
                for column in row:
                    strs.append(column.lstrip().rstrip())

                desc = cls(
                    int(strs[0]),
                    strs[1],
                    strs[2],
                    strs[3],
                    int(strs[4]),
                    int(strs[5]),
                    strs[6],
                    strs[7],
                )

                descs.append(desc)

        return descs

    def __str__(self):
        """Convert a descriptor to a human readable string representation.

        **NOTE**: Currently it is "uid \\t user \\t cid \\t device".

        """

        return "%8d\t%20s\t%20s\t%8s\t" % (
            self.uid,
            self.user,
            self.cid,
            self.device,
        )


class FMSignal(object):
    """3D finger motion signal (after preprocessing, device agnostic).

    This is the data structure for a finger motion signal. The signal is a time
    series containing samples of physical states of one point on a hand, 
    captured by a 3D camera device or a wearable data glove device. This is the 
    class the abstracts the signal data structure after preprocessing, and it is
    device agnostic.

    Attributes:

        length (int): length of the time series (i.e., number of samples).
        dim (int): dimension of each sample (i.e., number of sensor axes).
        ts (ndarray): timestamps of each sample, in a len dimensional vector.
        data (ndarray): the actual time series data, in a len * dim matrix.

        user (str): the user who creates this signal.
        cid (str): the unique id indicating the content of the signal.
        seq (int): the sequence id in a set when loaded from a dataset.

    **NOTE**: There are several ways to create an FMSignal object.

        (1) Preprocess a raw signal, i.e., "raw_signal.preprocess()".
        (2) Load data from a file with the class method "construct_from_file()".
        (3) Deep copy from an existing FMSignal object, i.e., "signal.copy()".
        (4) Align to a template or another signal to generate an aligned signal,
            i.e., "signal.align_to(another_signal)"
        (5) Modify an existing FMSignal object to generate a new signal, which
            is basically only used for data agumentation.

    It is not recommended to directly construct an FMSignal object using its
    constructor since the attributes may be inconsistent. Instead, please only
    use those previously mentioned methods to create an FMSignal object.

    **NOTE**: Timestamp is always in ms and frequency is always in Hz.

    **NOTE**: "ts" and "data" both have "dtype == np.float32".

    Currently, a signal has the following 18 sensor axes, i.e., "dim" is 18,
    and the "data" field has the shape of (len, 18).

    axes  | description
    ------|------------
    0-2   | position in x-y-z
    3-5   | speed in x-y-z, currently just the derivative of the position
    6-8   | acceleration in x-y-z, currently just the derivative of speed
    9-11  | orientation, i.e., the x, y, z components of the  quaternion
    12-14 | angular speed, currently just the derivative of the orientation
    15-17 | angular acceleration, just the derivative of the angular speed

    A signal may also have the following optional attributes.

    attributes  | description
    ------------|------------
    len_origin  | The length before alignment (only for aligned signals).
    dist        | DTW alignment distant (only for aligned signals).
    a2to1_start | Alignment index start array (only for aligned signals).
    a2to1_end   | Alignment index end array (only for aligned signals).

    **NOTE**: "user", "cid", and "seq" are only needed to print information
    for debugging. Use the class "FMSignalDesc" to obtain the meta data of the
    signal for more details. Typically, the signal file is named as
    "user_cid_seq.txt" (or ".csv", or ".npy").

    For example, given a file "duolu_duolu_01.txt", the "user" label is "duolu",
    the "cid" label is also "duolu", indicating the content is about writing
    the string "duolu", and the "seq" is 1, indicating the repetition #1.

    Usually, for privacy issues, "user" label and "cid" label are anonymous
    strings since they only need to be distinctive instead of meaningful.
    For example, "user00_id00_01.txt" means the "user" label is "user00",
    the "id" label is "id00", and the seq is 1.

    **NOTE**: The raw signal file may have two different formats, i.e., either
    in Comma Separated Value (".csv") or in NumPy binary format (".npy").
    However, the content structure is the same, which is essentially a matrix,
    where the rows are the samples at a certain time and the columns are the
    data from a specific sensor axis. See the "Data Format Details" document for
    more information.

    """

    def __init__(self, length=0, dim=0, ts=None, data=None, 
                 user="", cid="", seq=0):
        """Constructor.

        See the class attributes for the meaning of the arguments.

        **NOTE**: This is only for internal usage. If an FMSignal object is
        needed, use the class method "construct_from_file()" to load data from
        a file, or use the "copy()" method to duplicate a signal, or use the
        "align_to()" method to obtain an aligned signal.

        """

        self.length = length
        self.dim = dim
        self.ts = ts
        self.data = data

        self.user = user
        self.cid = cid
        self.seq = seq

    @classmethod
    def construct_from_file(cls, fn, mode, user="", cid="", seq=0):
        """Construct a signal by loading data from a file.

        Args:

            fn (string): The file name (without extension).
            mode (string): The file format (currently either "csv" or "npy").
            user (string): The user who creates this signal.
            cid (string): The unique id indicating the content of the signal.
            seq (int): The sequence id in a set when loaded from a dataset.

        Returns:

            FMSignal: The constructed signal object.

        Raises:

            ValueError: If the "mode" is unknown.
            FileNotFoundError: If the file does not exist.

        """
        signal = FMSignal(0, 0, None, None)
        signal.load_from_file(fn, mode)

        signal.user = user
        signal.cid = cid
        signal.seq = seq

        return signal

    def copy(self):
        """Deep copy of the signal.

        Args:

            None (None): No arguments.

        Returns:

            None: No return value.

        """

        signal = FMSignal(self.length, self.dim, self.data.copy(), 
            self.ts.copy(), self.user, self.cid, self.seq)

        return signal

    def align_to(self, template, window=0, penalty=0, method=DTW_METHOD,
        keep_dist_matrix=False):
        """Get another signal by aligning the signal to a template signal.

        **NOTE**: The alignment is done using Dynamic Time Warping (DTW).

        Args:

            template (FMSignal): The template signal.
            window (int): The DTW alignment window.
            penalty (int): The DTW element-wise missalign penalty.
            method (string): Implementation method, either "c" or "python".
            keep_dist_matrix (bool): Indicating whether to keep the DTW result,
                                     i.e., the "dist_matrix".

        Returns:

            FMSignal: The constructed signal object.

        """

        length_new = template.length
        ts_new = template.ts.copy()

        #print(template.data.shape)
        #print(self.data.shape)

        if method == "python":

            tup = dtw(template.data, self.data, window, penalty)

        elif method == "c":

            tup = dtw_c(template.data, self.data, window, penalty)

        else:

            raise ValueError("Unkown DTW implementation method (%s)!" % method)


        (_dist, matrix, _d, a2to1_start, a2to1_end, _s, _e, data_new) = tup

        signal = FMSignal(length_new, self.dim, ts_new, data_new, 
            self.user, self.cid, self.seq)

        signal.len_origin = self.length    
        signal.a2to1_start = a2to1_start
        signal.a2to1_end = a2to1_end

        if keep_dist_matrix:
            signal.dist_matrix = matrix

        return signal

    # ---------------------------- input / output ---------------------------

    def load_from_file(self, fn, mode):
        """Load the signal from a file.

        **NOTE**: This is only for internal usage. If an FMSignal object is
        needed, use the class method "construct_from_file()" instead.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format (currently either "csv" or "npy").

        Returns:

            None: No return value.

        Raises:

            ValueError: if the "mode" is unknown.
            FileNotFoundError: if the file does not exist.

        """

        if mode == "csv":

            fn += ".csv"
            array = np.loadtxt(fn, dtype=np.float32, delimiter=",")

        elif mode == "npy":

            fn += ".npy"
            array = np.load(fn)

            assert array.dtype == np.float32

        else:

            raise ValueError("Unknown mode: " + mode)

        # NOTE: These "copy()" force the "ts" and "data" to have proper C-like
        # array in memory, which is crucial for "dtw_c()"!!!
        ts = array[:, 0:1].copy()
        data = array[:, 1:].copy()

        length = data.shape[0]
        dim = data.shape[1]

        self.length = length
        self.dim = dim
        self.ts = ts
        self.data = data

    def save_to_file(self, fn, mode):
        """Save the signal to a file.

        **NOTE**: Only six digits after the decimal point are kept when floating
        point numbers are converted to CSV strings.

        Args:

            fn (string): file name (without the ".csv" or ".npy" extension).
            mode (string): The file format (currently either "csv" or "npy").

        Returns:

            None: No return value.

        """

        l = self.length

        array = np.concatenate((self.ts.reshape((l, 1)), self.data), axis=1)

        if mode == "csv":

            fn += ".csv"
            np.savetxt(fn, array, fmt="%.6f", delimiter=", ")

        elif mode == "npy":

            assert array.dtype == np.float32
            # NOTE: NumPy library add the ".npy" file extension for us!
            np.save(fn, array)

        else:

            raise ValueError("Unknown mode: " + mode)


    # ---------------------------- operations ---------------------------

    def get_orientation(self):
        """Obtain orientation as a series of rotation matrices and quaternions.

        Args:

            None (None): No argument.

        Returns:

            tuple: A tuple of the following representing the orientation.

                ndarray: The rotation matrices (l-by-3-by-3).
                list: The unit quaternions.
        """

        l = self.length
        data = self.data

        rotms = np.zeros((l, 3, 3))
        qs = [None] * l

        for i in range(l):

            qx = data[i, 9]
            qy = data[i, 10]
            qz = data[i, 11]
            qw = math.sqrt(1 - qx * qx - qy * qy - qz * qz)

            q = Quaternion(qw, qx, qy, qz)
            qs[i] = q
            rotms[i] = q.to_rotation_matrix()

        return rotms, qs

    def filter(self, sample_freq, cut_freq):
        """Low-pass filtering on the signal (in place).

        **NOTE**: It is assumed that a hand can not move in very high frequency,
        so the high frequency components of the signal is filtered.

        **NOTE**: This method use NumPy FFT and IFFT. It modifies the signal.

        Args:

            sample_freq (float): sample frequency of this signal.
            cut_freq (float): low pass filtering cutoff frequency.

        Returns:

            None: No return value.

        """

        l = self.length
        data = self.data

        cut_l = int(cut_freq * l / sample_freq)

        dft_co = np.fft.fft(data, l, axis=0)

        for i in range(cut_l, l - cut_l):

            dft_co[i] = 0 + 0j

        ifft_c = np.fft.ifft(dft_co, l, axis=0)

        ifft = ifft_c.astype(np.float32)

        for i in range(l):
            self.data[i] = ifft[i]

    def amplitude_normalize(self):
        """Normalize the amplitude of each sensor axes.

        **NOTE**: The ratios of x-y-z axes of one type, e.g., position, acc, etc.,
        are not perserved.

        Args:

            None (None): No argument.

        Returns:

            None: No return value.

        """

        data = self.data

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        for j in range(self.dim):
            data[:, j] = np.divide(data[:, j] - mean[j], std[j])

    # ------------------- operations for data augmentation ---------------------

    def pertube_amplitude(self, axis, time, s_window, scale, sigma):
        """Pertube the signal at the specified time.

        **NOTE**: This method is used in data augmentation. It adds a Gaussian
        shape pertubation signal segment to the original signal. This method
        changes the signal.

        Args:

            axis (int): The specified axis to pertube.
            time (int): The specified time to pertube.
            s_window (int): The pertubation smooth window size (greater than 0).
            scale (float): The scale factor of the pertubation (pos or neg).
            sigma (float): The standard deviation of the Gaussian pertubation. 

        Returns:

            None: No return value.

        Raises:

            ValueError: If the input arguments are out of range.
        

        """

        l = self.length

        if axis < 0 or axis >= self.dim:
            raise ValueError('Bad input axis (%d)' % axis)
        if s_window <= 0 or time < 0 or time >= l:
            raise ValueError('Bad input time (%d)' % time)

        

        pertubation = scipy.signal.gaussian(s_window * 2, sigma) * scale

        # Calculate the start indices.
        s_data = time - s_window
        s_pertube = 0
        if s_data < 0:
            s_pertube = -s_data
            s_data = 0

        # Calculate the end indices.
        e_data = s_data + s_window * 2
        e_pertube = s_window
        if e_data > l:
            e_pertube -= e_data - l
            e_data = l


        self.data[s_data:e_data, axis] += pertubation[s_pertube:e_pertube]

    def pertube_amplitude_seg(self, axis, time, window, s_window, scale, sigma):
        """Pertube the signal at a specified time for a segment.

        **NOTE**: This method is used in data augmentation. It adds a flat
        pertubation signal segment to the original signal with Gaussian smooth
        edges on both sides. This method changes the signal.

        Args:

            axis (int): The specified axis to pertube.
            time (int): The specified time to pertube.
            window (int): The pertubation window size (greater than 0).
            s_window (int): The pertubation smooth window size (greater than 0).
            scale (float): The scale factor of the pertubation (pos or neg).
            sigma (float): The standard deviation of the Gaussian pertubation. 

        Returns:

            None: No return value.

        Raises:

            ValueError: If the input arguments are out of range.

        """

        l = self.length

        if axis < 0 or axis >= self.dim:
            raise ValueError('Bad input axis: ' + '%d' % axis)
        if window <= 0 or s_window <= 0:
            raise ValueError('Bad input window (%d) or s_window (%d)' 
                             % (window, s_window))
        if time < 0 or time + window >= l:
            raise ValueError('Bad input time (%d) or window (%d)' 
                             % (time, window))

        

        pertube_size = window + s_window * 2
        pertube_seg = np.ones(pertube_size, dtype=np.float32) * scale

        # NOTE: The peak of this Gaussian series is 1.
        smooth_edges = scipy.signal.gaussian(s_window * 2, sigma)
        se_left = smooth_edges[:s_window]
        se_right = smooth_edges[s_window:]

        pertube_seg[:s_window] = np.multiply(pertube_seg[:s_window], se_left)
        pertube_seg[-s_window:] = np.multiply(pertube_seg[:s_window], se_right)

        s_data = time - s_window
        s_pertube = 0
        if s_data < 0:
            s_pertube = -s_data
            s_data = 0

        e_data = time + window + s_window
        e_pertube = pertube_size
        if e_data >= l:
            e_pertube -= e_data - l
            e_data = l

        self.data[s_data:e_data, axis] += pertube_seg[s_pertube:e_pertube]

    def swap_segment(self, other, start, window, s_window):
        """Swap a segment of this signal with another signal.

        **NOTE**: This method is used in data augmentation. It generates two
        new signals, which are the signals after swapping. It is done on all
        axes. This method does not modify this signal or the other signal.

        Args:

            other (FMSignal): The other signal.
            start (int): The start of the segment to swap.
            window (int): The pertubation window size (greater than 0).
            s_window (int): The pertubation smooth window size (greater than 0).

        Returns:

            tuple: A tuple containing the following.

                FMSignal: This signal after swapping.
                FMSignal: The other signal after swapping.

        Raises:

            ValueError: If the input arguments are out of range.

        """

        end = start + window
        l_min = min(self.length, other.length)

        if window <= 0 or s_window <= 0:
            raise ValueError('Bad input window (%d) or s_window (%d)' 
                             % (window, s_window))
        if start < 0 or end >= l_min:
            raise ValueError('Bad input start (%d) or window (%d)' 
                             % (start, window))


        a = self.copy()
        b = other.copy()

        data_a = a.data
        ts_a = a.ts
        data_b = b.data
        ts_b = b.ts

        as1 = data_a[:start, :]
        as2 = data_a[start:end, :]
        as3 = data_a[end:, :]

        bs1 = data_b[:start, :]
        bs2 = data_b[start:end, :]
        bs3 = data_b[end:, :]

        ats1 = ts_a[:start]
        ats2 = ts_a[start:end]
        ats3 = ts_a[end:]

        bts1 = ts_b[:start]
        bts2 = ts_b[start:end]
        bts3 = ts_b[end:]

        data_a_new = np.concatenate([as1, bs2, as3], axis=0)
        ts_a_new = np.concatenate([ats1, bts2, ats3], axis=0)
        l_a = data_a_new.shape[0]

        data_b_new = np.concatenate([bs1, as2, bs3], axis=0)
        ts_b_new = np.concatenate([bts1, ats2, bts3], axis=0)
        l_b = data_b_new.shape[0]

        # Smooth the segment edges with linear interpolation.
        # TODO: Rewrite this using NumPy vectorized operation.
        if s_window != 0:

            # Left margin
            ml_start = max(0, start - s_window)
            ml_end = min(start + s_window, min(l_a, l_b))
            ml_length = ml_end - ml_start
            ml_factor = np.linspace(1, 0, ml_length, 
                endpoint=True, dtype=np.float32)

            # print(margin1_start, margin1_end)

            for i, ii in zip(range(ml_start, ml_end), range(ml_length)):

                data_a_new[i] = ml_factor[ii] * data_a[i] \
                    + (1 - ml_factor[ii]) * data_b[i]
                data_b_new[i] = ml_factor[ii] * data_b[i] \
                    + (1 - ml_factor[ii]) * data_a[i]

            # Right margin
            mr_start = max(0, end - s_window)
            mr_end = min(end + s_window, min(l_a, l_b))
            mr_length = mr_end - mr_start
            mr_factor = np.linspace(1, 0, mr_end - mr_start, 
                endpoint=True, dtype=np.float32)

            # print(margin2_start, margin2_end)

            for i, ii in zip(range(mr_start, mr_end), range(mr_length)):

                data_a_new[i] = mr_factor[ii] * data_b[i] \
                    + (1 - mr_factor[ii]) * data_a[i]
                data_b_new[i] = mr_factor[ii] * data_a[i] \
                    + (1 - mr_factor[ii]) * data_b[i]

        a.data = data_a_new
        a.ts = ts_a_new
        a.len = l_a
        b.data = data_b_new
        b.ts = ts_b_new
        b.len = l_b

        return a, b

    def resize(self, length_new):
        """Resize the signal to the specified new length.

        **NOTE**: This method stretches the signal in time by linear 
        interpolation. It is designed for temporal normalization. This method
        modifies the signal.

        Args:

            length_new (int): The specified new length.

        Returns:

            None: No return value.

        Raises:

            ValueError: If the input arguments are out of range.

        """

        if length_new < 0:
            raise ValueError('Bad new length (%d).' % length_new)

        l = self.length
        d = self.dim

        data_new = np.zeros((length_new, d), np.float32)
        ts_new = np.zeros((length_new), np.float32)

        data = self.data
        ts = self.ts

        xp = np.linspace(0, l - 1, num=l)
        x = np.linspace(0, l - 1, num=length_new)

        for j in range(d):

            data_new[:, j] = np.interp(x, xp, data[:, j])

        ts_new[:] = np.interp(x, xp, ts)

        self.length = length_new
        self.ts = ts_new
        self.data = data_new

    def resize_segment(self, start, window, seg_length_new):
        """Resize a segment of the signal and keep other parts untouched.

        **NOTE**: This method is used in data augmentation. This method changes
        the signal.

        Args:

            start (int): The start of the segment to stretch.
            window (int): The stretch window size (greater than 0).
            seg_length_new (int): The specified new segment length.

        Returns:

            None: No return value.

        Raises:

            ValueError: If the input arguments are out of range.

        """

        end = start + window

        if window <= 0:
            raise ValueError('Bad window size (%d).' % window)
        if seg_length_new < 0:
            raise ValueError('Bad new segment length (%d).' % seg_length_new)
        if start < 0 or end >= self.length:
            raise ValueError('Bad input start (%d) or window (%d).' 
                             % (start, window))

        sample_period = self.ts[1] - self.ts[0]

        l = self.length
        d = self.dim
        seg_l = end - start
        l_new = self.length - seg_l + seg_length_new
        delta_new = seg_l / seg_length_new

        seg_start_new = start
        seg_end_new = seg_start_new + seg_length_new

        data_new = np.zeros((l_new, d), np.float32)
        ts_new = np.zeros((l_new, 1), np.float32)

        data = self.data

        # TODO: rewrite this with array splice and linear interpolation

        data_new[0:start] = data[0:start]
        data_new[seg_end_new:l_new] = data[end:l]

        for i in range(seg_length_new):

            it = start + i * delta_new
            ii_old = int(math.floor(it))
            ii_new = start + i
            dt = it - ii_old

            # print(i, delta_new, it, ii)

            for v in range(d):
                rate = (
                    data[ii_old][v] - data[ii_old][v]
                    if ii_old + 1 < l
                    else data[ii_old][v] - data[ii_old - 1][v]
                )
                data_new[ii_new][v] = data[ii_old][v] + rate * dt

        for i in range(l_new):

            ts_new[i] = sample_period * i

        self.length = l_new
        self.dim = d
        self.ts = ts_new
        self.data = data_new

    def stretch_axis(self, axis, scale):
        """Stretch along a certain sensor axis.

        **NOTE**: This method is used in data augmentation. This method modifies
        the signal.

        Args:

            axis (int): The specified sensor axis to stretch.
            scale (float): The intensity of the stretching.

        Returns:

            None: No return value.

        Raises:

            ValueError: If the input arguments are out of range.

        """

        if axis < 0 or axis >= self.dim:
            raise ValueError('Bad axis (%d).' % axis)

        l = self.length

        offset_v = (np.arange(0, l, 1) - l / 2) * scale

        self.data[:, axis] += offset_v


    def shift_temporally(self, shift):
        """Shift the signal in time.

        **NOTE**: This method is used in data augmentation. This method modifies
        the signal.

        Args:

            shift (int): The specified shift in time.

        Returns:

            None: No return value.

        """

        data = self.data
        l = self.length
        d = self.dim

        xp = np.arange(0, l, 1, dtype=np.float32)
        x = xp + shift

        data_shift = np.zeros((l, d), dtype=np.float32)

        # TODO: Rewrite this using array splice.

        for j in range(d):

            data_shift[:, j] = np.interp(x, xp, data[:, j])

        self.data = data_shift

    # -------------- operations on an aligned signal ---------------------------

    def load_alignment_index(self, fn, mode):
        """Load the alignment indicies from a NumPy binary file (.npy).

        **NOTE**: The alignment indicies are the "a2to1_start" and "a2to1_end" 
        attributes.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format (currently either "csv" or "npy").

        Returns:

            None: no return value.

        Raises:

            ValueError: If the "mode" is unknown.
            FileNotFoundError: If the file does not exist.

        """

        if mode == "csv":

            fn += ".csv"
            array = np.loadtxt(fn, dtype=np.int32, delimiter=",")

        elif mode == "npy":

            fn += ".npy"
            array = np.load(fn)

            assert array.dtype == np.int32

        else:

            raise ValueError("Unknown mode: " + mode)

        self.a2to1_start = array[:, 0]
        self.a2to1_end = array[:, 1]

        self.length_origin = self.a2to1_end[-1]

    def save_alignment_index(self, fn, mode):
        """Save the alignment indicies to a NumPy binary file (.npy).

        **NOTE**: The alignment indicies are the "a2to1_start" and "a2to1_end" 
        attributes.

        Args:

            fn (string): file name (without the ".csv" or ".npy" extension).
            mode (string): The file format (currently either "csv" or "npy").

        Returns:

            None: No return value.

        """

        l = self.length

        column_1 = self.a2to1_start.reshape((l, 1))
        column_2 = self.a2to1_end.reshape((l, 1))

        array = np.concatenate((column_1, column_2), axis=1)

        if mode == "csv":

            fn += ".csv"
            np.savetxt(fn, array, fmt="%d", delimiter=", ")

        elif mode == "npy":

            # NOTE: NumPy library add the ".npy" file extension for us!
            np.save(fn, array)

        else:

            raise ValueError("Unknown mode: " + mode)

    def distance_to(self, other):
        """Calculate the element-wise absolute distance of two signals.

        **NOTE**: The other signal must be already aligned to this signal, or
        their lengths must be the same.

        Args:

            other (FMSignal): The other signal.

        Returns:

            ndarray: An array of the absolute difference of the two signals.

        Raises:

            ValueError: If their lengths are different.

        """

        if self.length != other.length:
            raise ValueError("Incompatible signal lengths %d and %d." 
                             % (self.length, other.length))

        return np.absolute(self.data - other.data)

    def all_close_to(self, signal):
        """Check whether this signal is almost identical to another signal.

        **NOTE**: The criteria of "identical" is defined by "np.allclose()".
        The two signals must have the same type and length.

        Args:

            signal (FMSignal): The other signal to compare.

        Returns:

            bool: True if they are almost identical; False otherwise.

        """

        if not isinstance(signal, FMSignal):
            return False

        if self.length != signal.length:
            return False

        # NOTE: The CSV format only stores six digits after the decimal point.
        # Hence, "atol" can not be smaller than 1e-6.
        r1 = np.allclose(self.ts, signal.ts, atol=1e-6)
        r2 = np.allclose(self.data, signal.data, atol=1e-6)

        return r1 and r2

    def __str__(self):
        """Convert the meta info of the signal to a human readable string.

        Currently, the string is "user_cid_seq".

        Args:

            None (None): No arguments.

        Returns:

            None: No return value.

        """

        return "%s_%s_%d" % (self.user, self.cid, self.seq)


class FMSignalLeap():
    """Raw finger motion signal collected by the Leap Motion controller.

    This class represents the raw signal obtained from the Leap Motion sensor.
    An FMSignal object can be obtained by preprocessing this raw signal, where
    the preprocessing is essentially the main purpose of this FMSignalLeap 
    class. It has the following additional attributes.

    Attributes:

        length (int): The number of samples in this signal.
        ts (ndarray): Timestamp obtained from the device (1D vector).
        tsc (ndarray): Timestamp obtained from the client computer (1D vector).
        tip (ndarray): Position of the index finger tip (length * 3 matrix).
        center (ndarray): Position of the palm center (length * 3 matrix).
        joints (ndarray): Position of each joint (length * 5 * 5 * 3 tensor).
        confs (ndarray): Confidence value of each sample (1D vector).
        valids (ndarray): Whether the sample is valid (1D vector).
        data (ndarray): The data after preprocessing.
        rotms (ndarray): The orientation of each smple in rotation matrices.
        qs (list): The orientation of each smple in unit quaternions.
        trajectory (ndarray): The motion trajectory of the selected point.

    **NOTE**: The "joints" is a length * 5 * 5 * 3 tensor, i.e., 5 fingers, 5 
    joints on each finger, and 3 coordinates for each joint. The sequence of 
    fingers are [thumb, index finger, middle finger, ring finger, little 
    finger]. The sequence of joints are from the end of the palm to the tip of
    the finger. For example, "joints[:, 1, 0, :]" is the trajectory of the palm
    end of the index finger, and "joints[:, 1, 4, :]" is the trajectory of the
    tip of the index finger. Note that the thumb has only four joints, i.e.,
    "joints[:, 0, 0, :]" and "joints[:, 0, 1, :]" are identical.

    **NOTE**: The preprocessed signal contains the motion of only one point on
    the hand, which can be derived either from the center of the hand or the tip
    of the index finger. This is controlled by the "point" argument of the
    "preprocess()" method.

    **NOTE**: The raw signal file may have two different formats, i.e., either
    in Comma Separated Value (".csv") or in NumPy binary format (".npy").
    However, the content structure is the same, which is essentially a matrix,
    where the rows are the samples at a certain time and the columns are the
    data from a specific sensor axis. See the "Data Format Details" document for
    more information. There is another format called "raw_internal", which is
    used to resolve format issues in the data collected at the early stage of
    this project. It is now obsolete.
        
    """

    def __init__(self, user="", cid="", seq=0):
        """Constructor.

        See the attributes of FMSignal for the meaning of the arguments.

        **NOTE**: This is only for internal usage. If an FMSignalLeap object is
        needed, use the class method "construct_from_file()" to load data from
        a file, or use the "copy()" method to duplicate a signal.

        """

        self.user = user
        self.cid = cid
        self.seq = seq

        self.length = 0

        self.ts = None
        self.tsc = None

        self.tip = None
        self.center = None
        self.joints = None
        
        self.confs = None
        self.valids = None

        self.data = None
        self.qs = None
        self.rotms = None
        self.trajectory = None

    @classmethod
    def construct_from_file(cls, fn, mode, user="", cid="", seq=0):
        """Construct a signal by loading data from a file.

        Args:

            fn (string): The file name (without extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".
            user (string): The user who creates this signal.
            cid (string): The unique id indicating the content of the signal.
            seq (int): The sequence id in a set when loaded from a dataset.

        Returns:

            FMSignalLeap: The constructed signal object.

        Raises:

            ValueError: If the "mode" is wrong.
            FileNotFoundError: If the file does not exist.

        """

        signal = FMSignalLeap(user, cid, seq)
        signal.load_from_file(fn, mode)

        return signal


    # ---------------------------- input / output ---------------------------

    def load_from_file(self, fn, mode):
        """General interface to load the raw signal from a file.

        **NOTE**: This is only for internal usage. If an FMSignalLeap object is
        needed, use the class method "construct_from_file()" instead.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".

        Returns:

            None: No return value.

        Raises:

            ValueError: if the "mode" is unknown.
            FileNotFoundError: if the file does not exist.

        """

        if mode == "raw_internal":

            fn += ".txt"
            # NOTE: The "dtype" needs to be np.float64 for the raw timestamps!
            array = np.loadtxt(fn, np.float64, delimiter=",")
            self.load_from_buffer_raw_internal(array)

        elif mode == "raw_csv":

            fn += ".csv"
            array = np.loadtxt(fn, np.float32, delimiter=",")
            self.load_from_buffer_raw(array)

        elif mode == "raw_npy":

            fn += ".npy"
            array = np.load(fn)
            assert array.dtype == np.float32
            self.load_from_buffer_raw(array)

        else:
            raise ValueError("Unknown file mode %s!" % mode)

    def save_to_file(self, fn, mode):
        """General interface to save the raw signal to a file.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".

        Returns:

            None: No return value.

        """

        if mode == "raw_csv":

            array = self.save_to_buffer_raw()
            fn += ".csv"
            np.savetxt(fn, array, fmt="%.6f", delimiter=", ")

        elif mode == "raw_npy":

            array = self.save_to_buffer_raw()
            # NOTE: NumPy library add the ".npy" file extension for us!
            np.save(fn, array)

        else:
            raise ValueError("Unknown file mode %s!" % mode)

    def load_from_buffer_raw_internal(self, array):
        """Load data from a NumPy ndarray in the "raw_internal" format.

        **NOTE**: This is only for internal usage.

        Args:

            array (ndarray): The buffer.

        Returns:

            None: No Return value.


        """

        l = array.shape[0]
        m = array.shape[1]


        # --------- process timestamp ------

        if m == 133 or m == 93:

            # NOTE: This is the old format, with only one timestamp.
            offset_base = 1
            ts = array[:, 0].flatten()
            tsc = np.zeros(l, np.float32)

        elif m == 134:

            # CATUION: the current format, with two timestamps.
            offset_base = 2
            ts = array[:, 1].flatten()
            tsc = array[:, 0].flatten()

            # Use an offset to reduce the size of tsc so that it can fit into
            # the float32 type.
            tsc -= 1514764800

        else:

            raise ValueError("Unknown data file format!: m = %d" % m)

        for i in range(l):

            # fix timestamp wraping over maximum of uint32
            if i > 0 and ts[i] < ts[i - 1]:
                ts[i] += 4294967295.0

        # Fix timestamp offset and convert timestamp to millisecond
        # CAUTION: timestamp must start from 0! Other methods such as filtering
        # depend on this assumption!
        ts0 = ts[0]
        ts -= ts0
        ts /= 1000

        # --------- process point coordinate and joint coordinate ------

        offset_tip = offset_base
        offset_center = offset_base + 6
        offset_joints = offset_base + 6 + 9

        data_tip = np.zeros((l, 3), np.float32)
        # NOTE: Axes mapping: yzx -> xyz
        data_tip[:, 0] = array[:, offset_tip + 2]
        data_tip[:, 1] = array[:, offset_tip + 0]
        data_tip[:, 2] = array[:, offset_tip + 1]

        data_center = np.zeros((l, 3), np.float32)
        # NOTE: Axes mapping: yzx -> xyz
        data_center[:, 0] = array[:, offset_center + 2]
        data_center[:, 1] = array[:, offset_center + 0]
        data_center[:, 2] = array[:, offset_center + 1]

        data_joints = np.zeros((l, 5, 5, 3), np.float32)

        # Load joint positions
        for j in range(5):
            for k in range(5):
                index = j * 5 * 3 + k * 3
                # NOTE: Axes mapping: yzx -> xyz
                data_joints[:, j, k, 0] = array[:, offset_joints + index + 2]
                data_joints[:, j, k, 1] = array[:, offset_joints + index + 0]
                data_joints[:, j, k, 2] = array[:, offset_joints + index + 1]

        # Load confidences and valid flags
        confs = array[:, -2]
        valids = array[:, -1]

        self.length = l

        self.ts = ts.astype(np.float32)
        self.tsc = tsc.astype(np.float32)

        self.tip = data_tip
        self.center = data_center
        self.joints = data_joints

        self.confs = confs.astype(np.float32)
        self.valids = valids.astype(np.float32)



    def load_from_buffer_raw(self, array):
        """Load data from a NumPy ndarray.

        **NOTE**: This is only for internal usage.

        Args:

            array (ndarray): The buffer.

        Returns:

            None: No return value.

        """

        l = array.shape[0]
        m = array.shape[1]

        assert m == 2 + 3 + 3 + 75 + 2, "Wrong raw file format: m = %d" % m

        self.length = l

        self.ts = array[:, 1].flatten()
        self.tsc = array[:, 0].flatten()

        offset_tip = 2
        offset_center = 2 + 3
        offset_joints = 2 + 3 + 3

        self.tip = array[:, offset_tip : offset_tip + 3]
        self.center = array[:, offset_center : offset_center + 3]

        # Load joint positions
        self.joints = np.zeros((l, 5, 5, 3), np.float32)

        for j in range(5):
            for k in range(5):
                index = offset_joints + j * 5 * 3 + k * 3
                self.joints[:, j, k, :] = array[:, index : index + 3]

        # Load confidences and valid flags
        self.confs = array[:, -2]
        self.valids = array[:, -1]


    def save_to_buffer_raw(self):
        """Save the raw signal to a NumPy ndarray.

        **NOTE**: This is only for internal usage.

        Args:

            None (None): No argument.

        Returns:

            ndarray: The buffer.

        """

        m = 2 + 3 + 3 + 75 + 2
        array = np.zeros((self.length, m), np.float32)

        array[:, 0] = self.tsc
        array[:, 1] = self.ts

        offset_tip = 2
        offset_center = 2 + 3
        offset_joints = 2 + 3 + 3

        array[:, offset_tip : offset_tip + 3] = self.tip
        array[:, offset_center : offset_center + 3] = self.center

        for j in range(5):
            for k in range(5):
                index = offset_joints + j * 5 * 3 + k * 3

                array[:, index : index + 3] = self.joints[:, j, k, :]

        array[:, -2] = self.confs
        array[:, -1] = self.valids

        return array




    # ---------------------------- preprocessing ---------------------------

    def fix_missing_samples(self):
        """Fix missing data samples by linear interpolation.

        **NOTE**: This is only for internal usage. The missing samples are 
        mainly caused by the motion of the hand which are outside the field of 
        the view of the sensor. This procedure assumes that the first sample is
        always valid! Since it uses linear interpolation, it will not work well
        with too many missing samples. 
        
        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            None (None): No argument.

        Returns:

            None: No return value.


        """

        l = self.length

        i = 0

        while i < l - 1:

            # Find the start index of a missing segment.
            while i < l - 1:

                if self.valids[i] == 0:
                    break
                else:
                    i += 1

            # Find the end index of a missing segment.
            j = i
            while j < l - 1:

                if self.valids[j] == 1:
                    break
                else:
                    j += 1

            # If missing points found between i and j, fix them.
            # If no missing point found, just skip.
            if i < j:

                # print(i, j)

                start_ts = self.ts[i - 1]
                end_ts = self.ts[j]

                start_tip = self.tip[i - 1]
                end_tip = self.tip[j]
                length_tip = (end_tip - start_tip)

                start_center = self.center[i - 1]
                end_center = self.center[j]
                length_center = (end_center - start_center)

                start_joints = self.joints[i - 1]
                end_joints = self.joints[j]
                length_joints = (end_joints - start_joints)

                for k in range(i, j):

                    k_ts = self.ts[k]
                    rate = (k_ts - start_ts) * 1.0 / (end_ts - start_ts)
                    self.tip[k] = start_tip + length_tip * rate
                    self.center[k] = start_center + length_center * rate
                    self.joints[k] = start_joints + length_joints * rate

                    self.valids[k] = 1

                # Find another missing sement in the next iteration
                i = j

    def resample(self, re_freq):
        """Resample the signal at a specified frequency.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal. This method uses linear interpolation
        for resampling.

        Args:

            re_freq (float): Resampling frequency.

        Returns:

            None: No return value.

        """

        ts = self.ts
        l = self.length

        duration = ts[l - 1] - ts[0]

        step = 1000.0 / re_freq
        l_new = int(duration / 1000.0 * re_freq)

        ts_re = np.arange(0, step * l_new, step, dtype=np.float32)

        a_tip_re = np.zeros((l_new, 3), dtype=np.float32)
        for j in range(3):
            a_tip_re[:, j] = np.interp(ts_re, ts, self.tip[:, j])

        a_center_re = np.zeros((l_new, 3), dtype=np.float32)
        for j in range(3):
            a_center_re[:, j] = np.interp(ts_re, ts, self.center[:, j])

        data_joints_re = np.zeros((l_new, 5, 5, 3), np.float32)
        for j in range(5):
            for k in range(5):
                for v in range(3):
                    data_joints_re[:, j, k, v] = np.interp(
                        ts_re, ts, self.joints[:, j, k, v]
                    )

        confs_resample = np.interp(ts_re, ts, self.confs).astype(np.float32)
        valids_resample = np.ones(l_new, dtype=np.float32)

        # NOTE: Make sure they all have "dtype == np.float32"!
        # print(ts_re.dtype)
        # print(a_tip_re.dtype)
        # print(a_center_re.dtype)
        # print(data_joints_re.dtype)
        # print(confs_resample.dtype)
        # print(valids_resample.dtype)

        self.length = l_new
        self.ts = ts_re

        self.tip = a_tip_re
        self.center = a_center_re
        self.joints = data_joints_re
        self.confs = confs_resample
        self.valids = valids_resample


    def translate(self, t):
        """Translate the signal coorindate reference frame.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            t (ndarray): The 3D translation vector.

        Returns:

            None: No return value.

        """

        self.tip += t
        self.center += t

        for j in range(5):
            for k in range(5):

                self.joints[:, j, k] += t


    def rotate(self, R):
        """Rotate the signal coorindate reference frame.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            R (ndarray): The 3-by-3 rotation matrix.

        Returns:

            None: No return value.

        """

        pv = self.tip[:, 0:3].T
        pv = np.matmul(R, pv)
        self.tip[:, 0:3] = pv.T

        pv = self.center[:, 0:3].T
        pv = np.matmul(R, pv)
        self.center[:, 0:3] = pv.T

        for j in range(5):
            for k in range(5):

                pv = self.joints[:, j, k].T
                pv = np.matmul(R, pv)
                self.joints[:, j, k] = pv.T

    def pose_normalize(self, point, start, end, p_yaw=0, p_pitch=0, p_roll=0):
        """Normalize the position and orientation of the signal.

        **NOTE**: The new x-axis is the average of the pointing direction,
        the new z-axis is the vertical up direction of the leap motion sensor
        (assuming the sensor is placed on a horizontal surface), and the new 
        y-axis is determined by the new x-axis and y-axis.

        **NOTE**: The three angles in the argument are offsets that applied 
        during the normalization in radians. This can be used in orientation
        purtubation for data augmentation or correcting the pointing direction.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            point (string): The point on the hand, either "tip" or "center".
            start (int): The start time index of the signal for normalization.
            end (int): The end time index of the signal for normalization.
            p_yaw (float): The pertubation yaw angle in radian.
            p_pitch (float): The pertubation pitch angle in radian.
            p_roll (float): The pertubation roll angle in radian.

        Returns:

            None: No return value.

        Raises:

            ValueError: If the "point" is wrong or the "start" or "end" is bad.

        """

        l = self.length

        if start < 0 and end >= l and end <= start:

            raise ValueError("Bad start and end indices: %d, %d." \
                % (start, end))

        # Find the average position.

        if point == "tip":

            # Use the tip of the index finger.
            pos = self.tip[start:end]

        elif point == "center":

            # Use the center of the palm.
            pos = self.center[start:end]

        else:

            raise ValueError("Unknown point type: %s." % str(point))

        pos_mean = np.mean(pos, axis=0)

        # Position normalization.
        self.translate(-pos_mean)

        # Find the average orientation.

        data_joints = self.joints

        if point == "tip":

            # Use the direction of the middle segment of the index finger.
            p0 = data_joints[start:end, 1, 1, :]
            p1 = data_joints[start:end, 1, 2, :]

        elif point == "center":

            # Use the direction of the last segment of the middle finger.
            p0 = data_joints[start:end, 2, 0, :]
            p1 = data_joints[start:end, 2, 1, :]

        else:

            raise ValueError("Unknown point type: %s" % str(point))

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

        # Orientation normalization.
        # NOTE: "p_yaw", "p_pitch", "p_roll" are orientation pertubation. They
        # are all in radian.

        R_g2l = np.concatenate((vx, vy, vz), axis=1)

        R_l2g = R_g2l.transpose()

        R_offset = euler_zyx_to_rotation_matrix(p_yaw, p_pitch, p_roll)

        R_l2g = np.matmul(R_offset, R_l2g)

        self.rotate(R_l2g)

        # NOTE: Make sure they all have "dtype == np.float32"!
        # print(self.ts.dtype)
        # print(self.tip.dtype)
        # print(self.center.dtype)
        # print(self.joints.dtype)
        # print(self.confs.dtype)
        # print(self.valids.dtype)

        # print('center: (%f, %f, %f, %f, %f, %f)' % (xo, yo, zo, vx, vy, vz))

    def estimate_linear_states(self, point):
        """Estimate position, velocity and acceleration.

        **NOTE**: For the Leap Motion device, only the position is directly
        obtained from the device. The velocity and acceleration are derived
        by differentiating the the position.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            point (string): The point on the hand, either "tip" or "center".

        Returns:

            None: No return value.

        """

        data = self.data

        # print(data.dtype)

        if point == "tip":

            # Use the tip of the index finger.
            data[:, 0:3] = self.tip
            self.trajectory = self.tip.copy()

        elif point == "center":

            # Use the center of the palm.
            data[:, 0:3] = self.center
            self.trajectory = self.center.copy()

        else:

            raise ValueError("Unknown point type: %s" % str(point))

        data[:, 3] = np.gradient(data[:, 0])
        data[:, 4] = np.gradient(data[:, 1])
        data[:, 5] = np.gradient(data[:, 2])

        data[:, 6] = np.gradient(data[:, 3])
        data[:, 7] = np.gradient(data[:, 4])
        data[:, 8] = np.gradient(data[:, 5])

        # print(data.dtype)
        self.data = data

    def estimate_angular_states(self, p_yaw=0, p_pitch=0, p_roll=0):
        """Estimate orientation, angular speed, and angular acceleration.

        **NOTE**: Orientations are represented as the qx, qy, and qz 
        components of a unit quaternion. It is obtained by the position of the
        joints of the hand.

        **NOTE**: Currently, the angular speed is the relative local 
        differential of the angular position. The angular acceleration is just 
        the differential of the angular speed, not the real angular 
        acceleration. In this way, even if the reference frame changes, the
        angular speed will not change. Hence, we add three purtubation angles
        to change the local reference frame a bit as needed, mainly for data 
        augmentation usage.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal. This method should be called after 
        the pose normalization step.

        **NOTE**: Currently, this method iterates through each sample, which is
        relatively slow.

        Args:

            p_yaw (float): The pertubation yaw angle in radian.
            p_pitch (float): The pertubation pitch angle in radian.
            p_roll (float): The pertubation roll angle in radian.

        Returns:

            None: No return value.

        """

        data = self.data
        ts = self.ts

        # rotation matrix of each data sample
        rotms = np.zeros((self.length, 3, 3))

        # pose offset, for augmentation
        R_offset = euler_zyx_to_rotation_matrix(p_yaw, p_pitch, p_roll)

        # quaternion of each data sample
        qs = [None] * self.length
        omega_pre = np.asarray((0.0, 0.0, 0.0))

        for i in range(self.length):

            joints = self.joints[i]

            # Origin is the near end of the index finger.
            # p0 = joints[1, 1]
            p0 = joints[1, 0]

            # Pointing direction is from the near end to the next joint along
            # the index finger.
            # p1 = joints[1, 2]
            p1 = joints[1, 1]

            # The side direction is from the near end of the index finger to
            # the near end of the little finger.
            # p2 = joints[4, 1]
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

            # if vz[2] < 0:

            #     vz = -vz

            vy = np.cross(vz, vx)
            vz = np.cross(vx, vy)

            vx = vx / np.linalg.norm(vx)
            vy = vy / np.linalg.norm(vy)
            vz = vz / np.linalg.norm(vz)

            vx = vx.reshape((3, 1))
            vy = vy.reshape((3, 1))
            vz = vz.reshape((3, 1))

            # CAUTION: this is just an approximation of a rotation matrix,
            # and it may not be perfectly northonormal!!!
            rotm = np.concatenate((vx, vy, vz), axis=1)

            q = Quaternion.construct_from_rotation_matrix(rotm)
            # print(q, q.norm())
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

            # rotm = q.to_rotation_matrix()

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

        #print(data.dtype)
        self.data = data
        self.dim = 18

        self.rotms = rotms
        self.qs = qs

    def filter(self, sample_freq, cut_freq):
        """Low-pass filtering on the signal.

        **NOTE**: It is assumed that a hand can not move in very high frequency,
        so the high frequency components of the signal is filtered. This method
        uses NumPy FFT and IFFT.
        
        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            sample_freq (float): sample frequency of this signal.
            cut_freq (float): low pass filtering cutoff frequency.

        Returns:

            None: No return value.

        """

        l = self.length
        data = self.data

        cut_l = int(cut_freq * l / sample_freq)

        dft_co = np.fft.fft(data, l, axis=0)

        for i in range(cut_l, l - cut_l):

            dft_co[i] = 0 + 0j

        ifft_c = np.fft.ifft(dft_co, l, axis=0)

        # NOTE: This must be converted to "np.float32"!
        ifft = ifft_c.real.astype(np.float32)

        #print(ifft.dtype)
        self.data = ifft

    def prepare_trim_by_velocity(self, point, threshold):
        """Determine the indicies for the start and end of the finger motion.

        **NOTE**: The trimming process is split into two parts to accommodate
        other preprocessing steps. This method is the first part, which returns
        the start and end indices without actually throwing away the data 
        samples when the hand is not moving.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method **DOES NOT** modify the signal.

        Args:

            point (string): The point on the hand, either "tip" or "center".
            threshold (float): Threshold to detect the hand motion.

        Returns:

            None: No return value.

        **NOTE**: The threshold is relative. Typically some value between 0.2 
        to 0.5 would be good enough.

        """

        if point == "tip":

            # Use the tip of the index finger.
            pos = self.tip

        elif point == "center":

            # Use the center of the palm.
            pos = self.center

        else:

            raise ValueError("Unknown point type: %s" % str(point))

        vel = pos.copy()
        l = self.length

        # The "start" and "end" are determined by the relative velocity. It is 
        # calculated here by taking gradient of the position and then 
        # normalizing it.
        vel[:, 0] = np.gradient(pos[:, 0])
        vel[:, 1] = np.gradient(pos[:, 1])
        vel[:, 2] = np.gradient(pos[:, 2])

        v = np.linalg.norm(vel, axis=1)

        v_std = np.std(v)

        # NOTE: We cannot subtract the mean of velocity here.
        vn = np.divide(v, v_std)

        # Determine the start
        start = 0
        for i in range(l):
            if abs(vn[i]) > threshold:
                start = i - 5
                break
        start = max(start, 0)
        end = 0
        for i in range(l - 1, 0, -1):
            if abs(vn[i]) > threshold:
                end = i + 5
                break
        end = min(end, l)

        # print(self.id_label, self.seq, l, start, l - end)

        return (start, end)

    def trim(self, start, end):
        """Trim the start and the end where the hand does not move.

        **NOTE**: The trimming process is split into two parts to accommodate
        other steps of preprocessing. This method is the second part, which 
        throws away the data samples given the start and end indices.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        """

        l_new = end - start

        # NOTE: These "copy()" force the "ts" and "data" to have proper C-like
        # array in memory, which is crucial for "dtw_c()"!!!

        self.data = self.data[start:end, :].copy()
        self.ts = self.ts[start:end].copy()

        ts0 = self.ts[0]
        self.ts -= ts0

        self.length = l_new

        if self.qs is not None:
            self.qs = self.qs[start:end]

        if self.rotms is not None:
            self.rotms = self.rotms[start:end]

        if self.trajectory is not None:
            self.trajectory = self.trajectory[start:end]

        self.tip = self.tip[start:end]
        self.center = self.center[start:end]
        self.joints = self.joints[start:end]
        self.confs = self.confs[start:end]
        self.valids = self.valids[start:end]


    def preprocess(self, point, threshold=0.2, cut_freq=10, re_freq=50,
                   p_yaw=0, p_pitch=0, p_roll=0):
        """Preprocess the signal.

        **NOTE**: This method follows all the preprocessing steps. See the
        "Data Processing" document for details.

        Args:

            point (string): The point on the hand, either "tip" or "center".
            threshold (float): Threshold to trim the signal without hand motion.
            cut_freq (float): Low pass filtering cutoff frequency.
            re_freq (float): Resampling frequency.
            p_yaw (float): The pertubation yaw angle in radian.
            p_pitch (float): The pertubation pitch angle in radian.
            p_roll (float): The pertubation roll angle in radian.

        Returns:

            FMSignal: The preprocessed signal.

        **NOTE**: The default argument values are typically good enough.

        """

        self.fix_missing_samples()

        self.resample(re_freq)

        start, end = self.prepare_trim_by_velocity(point, threshold)

        self.pose_normalize(point, start, end, p_yaw, p_pitch, p_roll)

        # For the Leap Motion device, position is directly observable.

        dim = 18
        data = np.zeros((self.length, dim), dtype=np.float32)

        self.dim = dim
        self.data = data

        self.estimate_linear_states(point)

        self.estimate_angular_states(p_yaw, p_pitch, p_roll)

        self.filter(re_freq, cut_freq)

        self.trim(start, end)

        assert self.ts.dtype == np.float32
        assert self.data.dtype == np.float32

        signal = FMSignal(self.length, self.dim, self.ts, self.data,
                          self.user, self.cid, self.seq)

        return signal

    def all_close_to(self, signal):
        """Check whether this signal is almost identical to the other signal.

        **NOTE**: The criteria of "identical" is defined by "np.allclose()".
        The two signals must have the same type and length.

        Args:

            signal (FMSignalLeap): The other signal to compare.

        Returns:

            bool: True if they are almost identical; False otherwise.

        """

        if not isinstance(signal, FMSignalLeap):
            return False

        if self.length != signal.length:
            return False

        # NOTE: The CSV format only stores six digits after the decimal point.
        # Hence, "atol" can not be smaller than 1e-6.
        r1 = np.allclose(self.ts, signal.ts, atol=1e-6)
        r2 = np.allclose(self.tsc, signal.tsc, atol=1e-6)
        r3 = np.allclose(self.tip, signal.tip, atol=1e-6)
        r4 = np.allclose(self.center, signal.center, atol=1e-6)
        r5 = np.allclose(self.joints, signal.joints, atol=1e-6)
        r6 = np.allclose(self.confs, signal.confs, atol=1e-6)
        r7 = np.allclose(self.valids, signal.valids, atol=1e-6)

        return r1 and r2 and r3 and r4 and r5 and r6 and r7




class FMSignalGlove():
    """Raw finger motion signal collected by the data glove.

    This class represents the raw signal obtained from the data glove. The glove
    uses two BNO055 Inertial Measurement Units (IMU), one on the tip of the
    index finger and the other on the tip of the thumb.

    Attributes:

        length (int): The number of samples in this signal.
        ts (ndarray): Timestamp obtained from the device (1D vector).
        tsc (ndarray): Timestamp obtained from the client computer (1D vector).
        acc0 (ndarray): The linear acceleration obtained from the first IMU.
        gyro0 (ndarray): The angular speed obtained from the first IMU.
        gravity0 (ndarray): The gravity vector obtained from the first IMU.
        acc1 (ndarray): The linear acceleration obtained from the second IMU.
        gyro1 (ndarray): The angular speed obtained from the second IMU.
        gravity1 (ndarray): The gravity vector obtained from the second IMU.
        data (ndarray): The data after preprocessing.
        rotms (ndarray): The orientation of each smple in rotation matrices.
        qs (list): The orientation of each smple in unit quaternions.
        trajectory (ndarray): The motion trajectory of the selected IMU.

    **NOTE**: The raw sensor data such as "acc0" is a length * 3 matrix.

    **NOTE**: "data", "rotms", "qs", and "trajectory" are only available after
    preprocessing. Since a preprocessed signal (i.e., FMSignal object) contains
    the motion of only one point, in the preprocessing procedure, one IMU must
    be selected (i.e., by the "point" argument of "preprocess()"). The "rotms"
    is a length * 3 * 3 tensor. The "qs" is a list of Quaternion objects from
    the "pyrotation" module. The "trajectory" is a length * 3 matrix.

    **NOTE**: Although the BNO055 IMU can provide absolute orientation fused
    by the accelerometer, gyroscope. Additionally, it can provide linear 
    acceleration with gravity removed. In our case, The IMU is set to NDOF mode 
    (default mode, see BNO055 datasheet section 3.3). We use the linear
    acceleration but not the absolute orientation. Instead, we use a simple 
    method to derive the orientation by integrating the angular speed. Since 
    the signal usually has only a few seconds, this simple method is good
    enough. The raw file contains additional columns. See the "Data Format 
    Details" document for more information.
    
    **NOTE**: The raw signal file may have two different formats, i.e., either
    in Comma Separated Value (".csv") or in NumPy binary format (".npy").
    However, the content structure is the same, which is essentially a matrix,
    where the rows are the samples at a certain time and the columns are the
    data from a specific sensor axis. See the "Data Format Details" document for
    more information. There is another format called "raw_internal", which is
    used to resolve format issues in the data collected at the early stage of
    this project. It is now obsolete.

    """

    def __init__(self, user="", cid="", seq=0):
        """Constructor.

        See the attributes of FMSignal for the meaning of the arguments.

        **NOTE**: This is only for internal usage. If an FMSignalGlove object is
        needed, use the class method "construct_from_file()" to load data from
        a file, or use the "copy()" method to duplicate a signal.

        """

        self.user = user
        self.cid = cid
        self.seq = seq

        self.length = 0

        self.ts = None
        self.tsc = None

        self.acc0 = None
        self.gyro0 = None
        self.gravity0 = None

        self.acc1 = None
        self.gyro1 = None
        self.gravity1 = None

        self.data = None
        self.qs = None
        self.rotms = None
        self.trajectory = None

    @classmethod
    def construct_from_file(cls, fn, mode, user="", cid="", seq=0):
        """Construct a signal by loading data from a file.

        Args:

            fn (string): The file name (without extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".
            user (string): The user who creates this signal.
            cid (string): The unique id indicating the content of the signal.
            seq (int): The sequence id in a set when loaded from a dataset.

        Returns:

            FMSignalGlove: The constructed signal object.

        Raises:

            ValueError: If the "mode" is wrong.
            FileNotFoundError: If the file does not exist.

        """

        signal = cls(user, cid, seq)
        signal.load_from_file(fn, mode)

        return signal

    # ---------------------------- input / output ---------------------------

    def load_from_file(self, fn, mode):
        """General interface to load the raw signal from a file.

        **NOTE**: This is only for internal usage. If an FMSignalLeap object is
        needed, use the class method "construct_from_file()" instead.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".

        Returns:

            None: No return value.

        Raises:

            ValueError: if the "mode" is unknown.
            FileNotFoundError: if the file does not exist.

        """

        if mode == "raw_internal":

            fn += ".txt"
            # NOTE: The "dtype" needs to be np.float64 for the raw timestamps!
            array = np.loadtxt(fn, np.float64, delimiter=",")
            self.load_from_buffer_raw_internal(array)

        elif mode == "raw_csv":

            fn += ".csv"
            array = np.loadtxt(fn, np.float32, delimiter=",")
            self.load_from_buffer_raw(array)

        elif mode == "raw_npy":

            fn += ".npy"
            array = np.load(fn)
            assert array.dtype == np.float32
            self.load_from_buffer_raw(array)

        else:
            raise ValueError("Unknown file mode %s!" % mode)

    def save_to_file(self, fn, mode):
        """General interface to save the raw signal to a file.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".

        Returns:

            None: No return value.

        """

        if mode == "raw_csv":

            array = self.save_to_buffer_raw()
            fn += ".csv"
            np.savetxt(fn, array, fmt="%.6f", delimiter=", ")

        elif mode == "raw_npy":

            array = self.save_to_buffer_raw()
            # NOTE: NumPy library add the ".npy" file extension for us!
            np.save(fn, array)

        else:
            raise ValueError("Unknown file mode %s!" % mode)

    def load_from_buffer_raw_internal(self, array):
        """Load from a file directly obtained by the data collection client.

        Args:

            array (ndarray): The buffer, which is a numpy ndarrary.

        Returns:

            None: No return value.

        CAUTION: Do not use this method directly. Instead, uses the class
        method FMSignalGlove.construct_from_file() instead.

        """

        l = array.shape[0]
        m = array.shape[1]

        self.length = l

        if m == 25:

            # NOTE: This is the early format, with only one timestamp.
            ts = array[:, 0].flatten()
            tsc = np.zeros(l, dtype=np.float32)

            acc0 = array[:, 1:4]
            gyro0 = array[:, 4:7]
            gravity0 = array[:, 7:10]

            acc1 = array[:, 13:16]
            gyro1 = array[:, 16:19]
            gravity1 = array[:, 19:22]

        elif m == 34:

            # NOTE: This is the current format, with two timestamps.
            ts = array[:, 1].flatten()
            tsc = array[:, 0].flatten()

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

            raise ValueError("Unknown format: m = %d" % m)

        # Fix any timestamp anormaly
        # NOTE: The device timestamps in glove data are in millisecond.
        for i in range(l - 1):

            if ts[i + 1] < ts[i]:
                ts[i + 1] = ts[i] + 20

        self.ts = ts.astype(np.float32)
        self.tsc = tsc.astype(np.float32)

        self.acc0 = acc0.astype(np.float32)
        self.gyro0 = gyro0.astype(np.float32)
        self.gravity0 = gravity0.astype(np.float32)

        self.acc1 = acc1.astype(np.float32)
        self.gyro1 = gyro1.astype(np.float32)
        self.gravity1 = gravity1.astype(np.float32)

    def load_from_buffer_raw(self, array):
        """Load data from a NumPy ndarray.

        **NOTE**: This is only for internal usage.

        Args:

            array (ndarray): The buffer.

        Returns:

            None: No return value.

        """

        l = array.shape[0]
        m = array.shape[1]

        assert m == 20, "Wrong raw file format: m = %d" % m

        self.length = l

        self.tsc = array[:, 0].flatten()
        self.ts = array[:, 1].flatten()

        self.acc0 = array[:, 2:5]
        self.gyro0 = array[:, 5:8]
        self.gravity0 = array[:, 8:11]

        self.acc1 = array[:, 11:14]
        self.gyro1 = array[:, 14:17]
        self.gravity1 = array[:, 17:20]

    def save_to_buffer_raw(self):
        """Save the raw signal to a NumPy ndarray.

        **NOTE**: This is only for internal usage.

        Args:

            None (None): No argument.

        Returns:

            ndarray: The buffer.

        """

        array = np.zeros((self.length, 20), dtype=np.float32)

        array[:, 0] = self.tsc
        array[:, 1] = self.ts

        array[:, 2:5] = self.acc0
        array[:, 5:8] = self.gyro0
        array[:, 8:11] = self.gravity0

        array[:, 11:14] = self.acc1
        array[:, 14:17] = self.gyro1
        array[:, 17:20] = self.gravity1

        return array



    # ---------------------------- preprocessing ---------------------------

    def convert_axes_to_standard_glove(self, axes):
        """Convert a set of xyz data series to the standard glove axes.

        Args:

            axes (ndarray): The original data series.
        
        Returns:

            None: No return value.

        **NOTE**: This method is designed to handle "raw_internal" signals only.

        """

        n = axes.shape[0]

        temp = np.zeros((n, 1), np.float32)

        # x <= -y', y <= x'
        # (x, y) is standard glove. (x', y') is glove2 or glove3.

        temp[:, 0] = axes[:, 0]
        axes[:, 0] = -axes[:, 1]
        axes[:, 1] = temp[:, 0]

    def convert_to_standard_glove(self):
        """Convert raw data columns to standard glove reference frame.

        Args:

            none (None): No arguments.
        
        Returns:

            None: No return value.

        **NOTE**: This method is designed to handle "raw_internal" signals only.

        """

        self.convert_axes_to_standard_glove(self.acc0)
        self.convert_axes_to_standard_glove(self.gyro0)
        self.convert_axes_to_standard_glove(self.gravity0)

        self.convert_axes_to_standard_glove(self.acc1)
        self.convert_axes_to_standard_glove(self.gyro1)
        self.convert_axes_to_standard_glove(self.gravity1)

    def estimate_angular_states(self, point):
        """Estimate orientation, angular speed, and angular acceleration.

        **NOTE**: Orientations are represented as the qx, qy, and qz 
        components of a unit quaternion. For the data glove device, the angular
        speed is directly obtained from the sensor, and the angular po

        **NOTE**: Currently the beginning of the signal is used as the initial
        pose and the integration goes through the signal.

        Args:

            point (string): The point on the hand, either "tip" or "center".

        Returns:

            None: No return value.

        """

        data = self.data
        l = self.length

        if point == "tip":
            gyro = self.gyro0
        elif point == "center":
            gyro = self.gyro1
        else:
            raise ValueError("Unknown point type: %s" % str(point))


        q = Quaternion.identity()

        # rotation matrix of each data sample
        rotms = np.zeros((l, 3, 3))

        # quaternion of each data sample
        qs = [None] * l


        # Given 50 Hz, one timestep is 20 ms, i.e., 0.02 second
        timestep = 0.02

        for i in range(0, l):

            qs[i] = q

            rotm = q.to_rotation_matrix()

            rotms[i] = rotm


            data[i, 9] = q.x
            data[i, 10] = q.y
            data[i, 11] = q.z

            # z, y, x = rotation_matrix_to_euler_angles_zyx(rotm)

            # yaw, pitch, roll, gimbal_lock \
            #     = rotation_matrix_to_euler_angles_zyx(rotm)

            # data[i, 9] = roll
            # data[i, 10] = pitch
            # data[i, 11] = yaw

            # assert not gimbal_lock

            # NOTE: Gyro output is in rad/s
            omega = gyro[i]

            q = Quaternion.integrate_local(q, omega, timestep)
            q.normalize()

        data[:, 12:15] = gyro

        data[:, 15] = np.gradient(data[:, 12])
        data[:, 16] = np.gradient(data[:, 13])
        data[:, 17] = np.gradient(data[:, 14])

        self.data = data
        self.rotms = rotms
        self.qs = qs

    def pose_normalize(self, start, end, p_yaw=0, p_pitch=0, p_roll=0):
        """Normalize the position and orientation of the signal.

        **NOTE**: This method expect a start index and an end index, where the
        signal segment between the start and the end is used to calculate the
        average pointing direction. Hence, it should be called after
        "prepare_trim_by_xxx()".

        **NOTE**: This normalization step depends on the orientation. Hence,
        it must be called after "estimate_angular_states()"

        Args:

            start (int): The start time index of the signal for normalization.
            end (int): The end time index of the signal for normalization.
            p_yaw (float): The pertubation yaw angle in radian.
            p_pitch (float): The pertubation pitch angle in radian.
            p_roll (float): The pertubation roll angle in radian.

        Returns:

            None: No return value.

        Raises:

            ValueError: If the "point" is wrong or the "start" or "end" is bad.

        """

        data = self.data
        rotms = self.rotms
        qs = self.qs

        vi = np.array((1.0, 0, 0), np.float32).reshape((3, 1))
        # vj = np.array((0, 1.0, 0), np.float32).reshape((3, 1))
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

        for i in range(self.length):

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
            # self.data[i, 3:6] = np.matmul(rotms[i], gravity0[i].reshape((3, 1))).reshape(3)

        # apply pose offset on angular speed
        ov = data[:, 12:15].T
        ov = np.matmul(R_offset, ov)
        data[:, 12:15] = ov.T

        data[:, 15] = np.gradient(data[:, 12])
        data[:, 16] = np.gradient(data[:, 13])
        data[:, 17] = np.gradient(data[:, 14])

    def estimate_linear_states(self, point, wv=0.2, wp=5):
        """Estimate position, velocity and acceleration.

        **NOTE**: For the data glove device, the linear acceleration is directly
        obtained from the sensor. The position and velocity are derived by
        integrating the linear acceleration.

        **NOTE**: This method depends on the orientation. Hence, it must be
        called after "estimate_angular_states()"

        """

        data = self.data
        l = self.length

        if point == "tip":
            acc = self.acc0
        elif point == "center":
            acc = self.acc1
        else:
            raise ValueError("Unknown point type: %s" % str(point))

        trajectory = np.zeros((l, 3))

        qs = self.qs

        # given 50 Hz, one timestep is 20 ms, i.e., 0.02 second
        timestep = 0.02

        # dead reckoning using linear acceleration

        p = np.array((0, 0, 0))
        v_m = np.array((0, 0, 0))
        a_m = np.array((0, 0, 0))

        pp = np.array((200, 0, 0))

        for i in range(1, l):

            acc_local = acc[i]

            q = qs[i]

            a_m = q.rotate_a_point(acc_local).reshape(3)



            pp_i = q.rotate_a_point(pp).reshape(3)
            trajectory[i, 0:3] = pp_i

            v_m = v_m + a_m * timestep
            

            # Now we add a correction term
            u = np.multiply(v_m, np.abs(v_m))
            v_m = v_m - wv * u

            p = p + v_m * timestep + 0.5 * a_m * timestep * timestep

            #print(p)

            # Now we add a similar correction term.
            w = np.multiply(p, np.abs(p))
            p = p - wp * w

            data[i, 6:9] = a_m
            data[i, 3:6] = v_m
            data[i, 0:3] = p * 1000  # position is in mm


        trajectory[:, 0] -= 200
        self.trajectory = trajectory

    def filter(self, sample_freq, cut_freq):
        """Low-pass filtering on the signal.

        **NOTE**: It is assumed that a hand can not move in very high frequency,
        so the high frequency components of the signal is filtered. This method
        uses NumPy FFT and IFFT.
        
        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            sample_freq (float): sample frequency of this signal.
            cut_freq (float): low pass filtering cutoff frequency.

        Returns:

            None: No return value.

        """

        l = self.length
        data = self.data

        cut_l = int(cut_freq * l / sample_freq)

        dft_co = np.fft.fft(data, l, axis=0)

        for i in range(cut_l, l - cut_l):

            dft_co[i] = 0 + 0j

        ifft_c = np.fft.ifft(dft_co, l, axis=0)

        # NOTE: This must be converted to "np.float32"!
        ifft = ifft_c.real.astype(np.float32)

        for i in range(l):
            self.data[i] = ifft[i]


    def prepare_trim_by_acc(self, point, threshold):
        """Determine the indicies for the start and end of the finger motion.

        **NOTE**: The trimming process is split into two parts to accommodate
        other preprocessing steps. This method is the first part, which returns
        the start and end indices without actually throwing away the data 
        samples when the hand is not moving.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method **DOES NOT** modify the signal.

        Args:

            point (string): The point on the hand, either "tip" or "center".
            threshold (float): Threshold to detect the hand motion.

        Returns:

            None: No return value.

        **NOTE**: The threshold is relative. Typically some value between 0.2 
        to 0.5 would be good enough.

        """

        if point == "tip":
            data = self.acc0
        elif point == "center":
            data = self.acc1
        else:
            raise ValueError("Unknown point type: %s" % str(point))

        l = self.length

        acc = data.copy()
        a = np.linalg.norm(acc, axis=1)

        a_std = np.std(a)

        # NOTE: We cannot subtract the mean of acceleration here.
        an = np.divide(a, a_std)

        start = 0
        for i in range(l):
            if abs(an[i]) > threshold:
                start = i - 5
                break
        start = max(start, 0)
        end = 0
        for i in range(l - 1, 0, -1):
            if abs(an[i]) > threshold:
                end = i + 5
                break
        end = min(end, l)

        return (start, end)

    def trim(self, start, end):
        """Trim the start and end of the signal where the hand does not move.

        **NOTE**: The trimming process is split into two parts to accommodate
        other steps of preprocessing. This method is the second part, which 
        throws away the data samples given the start and end indices.

        **NOTE**: This method is only used in the preprocessing procedure.
        This method modifies the signal.

        Args:

            start (int): The start index, i.e., data[start:end, :] will be kept.
            end (int): The end index, i.e., data[start:end, :] will be kept.

        Returns:

            None: No return value.

        """

        l_new = end - start

        # NOTE: These "copy()" force the "ts" and "data" to have proper C-like
        # array in memory, which is crucial for "dtw_c()"!!!
        self.data = self.data[start:end, :].copy()
        self.ts = self.ts[start:end].copy()

        ts0 = self.ts[0]
        self.ts -= ts0

        self.length = l_new

        if self.qs is not None:
            self.qs = self.qs[start:end]

        if self.rotms is not None:
            self.rotms = self.rotms[start:end]

        if self.trajectory is not None:
            self.trajectory = self.trajectory[start:end]

    def preprocess(self, point, threshold=0.2, cut_freq=10, 
                   p_yaw=0, p_pitch=0, p_roll=0):
        """Preprocess the signal.

        **NOTE**: This method follows all the preprocessing steps. See the
        "Data Processing" document for details.

        Args:

            point (string): The point on the hand, either "tip" or "center".
            threshold (float): Threshold to trim the signal without hand motion.
            cut_freq (float): Low pass filtering cutoff frequency.
            p_yaw (float): The pertubation yaw angle in radian.
            p_pitch (float): The pertubation pitch angle in radian.
            p_roll (float): The pertubation roll angle in radian.

        Returns:

            FMSignal: The preprocessed signal.

        **NOTE**: The default argument values are typically good enough.

        """

        dim = 18
        data = np.zeros((self.length, dim), dtype=np.float32)

        self.dim = dim
        self.data = data

        start, end = self.prepare_trim_by_acc(point, threshold)

        self.estimate_angular_states(point)
        self.pose_normalize(start, end, p_yaw, p_pitch, p_roll)

        self.estimate_linear_states(point)

        sample_freq = self.length / (self.ts[self.length - 1]) * 1000.0

        #self.filter(sample_freq, cut_freq)

        #self.trim(start, end)

        assert self.ts.dtype == np.float32
        assert self.data.dtype == np.float32

        signal = FMSignal(self.length, self.dim, self.ts, self.data,
                          self.user, self.cid, self.seq)

        return signal


    def all_close_to(self, signal):
        """Check whether this signal is almost identical to the other signal.

        **NOTE**: The criteria of "identical" is defined by "np.allclose()".
        The two signals must have the same type and length.

        Args:

            signal (FMSignalGlove): The other signal to compare.

        Returns:

            bool: True if they are almost identical; False otherwise.

        """

        if not isinstance(signal, FMSignalGlove):
            return False

        if self.length != signal.length:
            return False

        # NOTE: The CSV format only stores six digits after the decimal point.
        # Hence, "atol" can not be smaller than 1e-6.
        r1 = np.allclose(self.ts, signal.ts, atol=1e-6)
        r2 = np.allclose(self.tsc, signal.tsc, atol=1e-6)
        r3 = np.allclose(self.acc0, signal.acc0, atol=1e-6)
        r4 = np.allclose(self.gyro0, signal.gyro0, atol=1e-6)
        r5 = np.allclose(self.gravity0, signal.gravity0, atol=1e-6)
        r6 = np.allclose(self.acc1, signal.acc1, atol=1e-6)
        r7 = np.allclose(self.gyro1, signal.gyro1, atol=1e-6)
        r8 = np.allclose(self.gravity1, signal.gravity1, atol=1e-6)

        return r1 and r2 and r3 and r4 and r5 and r6 and r7 and r8

class FMSignalTemplate(FMSignal):
    """FMSignal signal template constructed from a collection of signals.

    In most cases, template is identical to a signal (preprocessed, sensor
    agnostic) with a few additional attributes. Note that this class is derived
    from the class "FMSignal".

    Attributes:

        variance (ndarray): The variance of the signal (length * 18 matrix).
        signals_aligned (ndarray): The signals for constructing the template.   


    **NOTE**: The variance is calculates at sample level, so it has the same
    length and dimension as the data, i.e., a length * 18 matrix.

    **NOTE**: The "signals_aligned" is only valid if it is constructed from a
    collection of signals, i.e., by calling "construct_from_signals()". It is
    not valid if the template is loaded from a file.

    """

    def __init__(self, length=0, dim=0, ts=None, data=None, 
                 variance=None, signals_aligned=None, user="", cid="", seq=0):
        """Constructor.
        
        See the attributes of FMSignal for the meaning of the arguments.

        """

        FMSignal.__init__(self, length, dim, ts, data, user, cid, seq)

        self.variance = variance
        self.signals_aligned = signals_aligned

    @classmethod
    def construct_from_signals(cls, signals, template_index, 
                               window=50, penalty=0):
        """Construct the template by aligning and average a set of signals.

        Args:

            signals (list): The collection of signals (FMSignal objects).
            template_index (int): The index indicating the alignment.
            window (int): Alignment window, see "dtw()".
            penalty (int): Element-wise misalignment penalty, see "dtw()".

        **NOTE**: The "signals" should be a collection of signals

        """

        signal_t = signals[template_index]
        signals_aligned = [signal_t]

        k = len(signals)

        # construct signal template

        length = signal_t.length
        dim = signal_t.dim

        ts = signal_t.ts.copy()
        data = signal_t.data.copy()

        variance = np.zeros(signal_t.data.shape, signal_t.data.dtype)

        for signal in signals:

            if signal == signal_t:

                continue

            signal_aligned = signal.align_to(signal_t, window, penalty)
            signals_aligned.append(signal_aligned)

            data += signal_aligned.data

        data /= k

        for signal_aligned in signals_aligned:

            variance += np.square(signal_aligned.data - data)

        variance /= k

        template = cls(length, dim, ts, data, variance, signals_aligned,
                       signal_t.user, signal_t.cid, signal_t.seq)

        return template

    @classmethod
    def construct_from_file(cls, fn, mode, user="", cid="", seq=0):
        """Factory method to build the signal by loading a file.

        Args:

            fn (string): The file name (without extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".
            user (string): The user who creates this signal.
            cid (string): The unique id indicating the content of the signal.
            seq (int): The sequence id in a set when loaded from a dataset.

        Returns:

            FMSignalTemplate: The constructed signal object.

        Raises:

            ValueError: If the "mode" is wrong.
            FileNotFoundError: If the file does not exist.

        """

        template = cls()

        template.load_from_file(fn, mode)

        template.user = user
        template.cid = cid
        template.seq = seq


        return template


    # ---------------------------- input / output ---------------------------

    def load_from_file(self, fn, mode):
        """General interface to load the template from a file.

        **NOTE**: This is only for internal usage. If an FMSignalLeap object is
        needed, use the class method "construct_from_file()" instead.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".

        Returns:

            None: No return value.

        Raises:

            ValueError: if the "mode" is unknown.
            FileNotFoundError: if the file does not exist.

        """
        if mode == "csv":

            fn += ".csv"
            array = np.loadtxt(fn, dtype=np.float32, delimiter=",")

            assert array.shape[1] == 37

            l = array.shape[0]
            d = 18

            ts = array[:, 0:1]
            data_and_variance = array[:, 1:]

            data = data_and_variance[:, :d]
            variance = data_and_variance[:, d:]

        elif mode == "npy":

            fn += ".npy"
            array = np.load(fn)

            assert array.dtype == np.float32
            assert array.shape[1] == 37

            l = array.shape[0]
            d = 18

            ts = array[:, 0:1]
            data = array[:, 1:(d + 1)]
            variance = array[:, (d + 1):]

        else:
            raise ValueError("Unknown file mode %s!" % mode)

        assert d == 18

        self.length = l
        self.dim = d
        self.ts = ts
        self.data = data

        self.variance = variance

    def save_to_file(self, fn, mode):
        """General interface to save the template to a file.

        Args:

            fn (string): The file name (without the ".csv" or ".npy" extension).
            mode (string): The file format, either "raw_csv", or "raw_npy".

        Returns:

            None: No return value.

        """

        l = self.length

        array_tup = (self.ts.reshape((l, 1)), self.data, self.variance)
        array = np.concatenate(array_tup, axis=1)

        if mode == "csv":

            fn += ".csv"
            np.savetxt(fn, array, fmt="%.6f", delimiter=", ")


        elif mode == "npy":

            assert array.dtype == np.float32
            # NOTE: NumPy library add the ".npy" file extension for us!
            np.save(fn, array)

        else:
            raise ValueError("Unknown file mode %s!" % mode)


    # ---------------------------- operations ----------------------

    def update(self, new_signal, factor):
        """Update the template with a new signal.

        Args:

            new_signal (FMSignal): The new signal S used to update the template.
            factor (float): The update factor, 
                            i.e., T_{new} = (1-factor)*T_{old} + S*factor.

        Returns:

            None: No return value.

        **NOTE**: The "new_signal" must be already aligned to the template.

        """

        # update the signal template
        self.data = self.data * (1 - factor) + new_signal.data * factor

    def all_close_to(self, template):
        """Check whether this template is almost identical to another template.

        **NOTE**: The criteria of "identical" is defined by "np.allclose()".
        The two templates must have the same type and length.

        Args:

            template (FMSignalTemplate): The other template to compare.

        Returns:

            bool: True if they are almost identical; False otherwise.

        """

        if not isinstance(template, FMSignal):
            return False

        if self.length != template.length:
            return False

        # NOTE: The CSV format only stores six digits after the decimal point.
        # Hence, "atol" can not be smaller than 1e-6.
        r1 = np.allclose(self.ts, template.ts, atol=1e-6)
        r2 = np.allclose(self.data, template.data, atol=1e-6)
        r3 = np.allclose(self.variance, template.variance, atol=1e-6)

        return r1 and r2 and r3



