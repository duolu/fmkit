'''
This module contains the visuzlaization code of the fmkit framework, which is 
designed to facilitate researches on in-air-handwriting related research.

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

'''

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider, Button, RadioButtons
import colorsys

from fmsignal import FMSignal
from fmsignal import FMSignalLeap
from fmsignal import FMSignalGlove


COLORS = ('k', 'r', 'g', 'b', 'c', 'm', 'y')


def signal_vis(signal, start_col=0, nr_cols=18, normalized=False):
    '''Visualize a preprocessed signal by plotting the specified columns.
    
    Args:

        signal (FMSignal): The signal to be visualized.
        start_col (int): The start index of the sensor axes on the plot.
        nr_cols (int): The number of sensor axes on the plot.
        normalized (bool): Indicating whether the amplitude of the signal is
                           normalized or not. 

    Returns:

        None: No return value.

    **NOTE**: Both "nr_cols" and "start_col" must be a multiple of 3.

    '''

    d = signal.dim

    assert isinstance(signal, FMSignal)
    assert nr_cols % 3 == 0 and start_col % 3 == 0
    assert start_col < d and start_col + nr_cols <= d
    
    fig = plt.figure()
    axes = []
    
    l = signal.length
    
    for ii, i in enumerate(range(start_col, start_col + nr_cols)):
        
        data_column = signal.data[:, i]

        ax = fig.add_subplot(nr_cols, 1, ii + 1)
    
        ax.plot(data_column, color='k')
        if normalized:
            ax.set_ylim(-3.5, 3.5)
        else:
            ymax = np.max(data_column)
            ymin = np.min(data_column)
            dist = max(abs(ymin), abs(ymax))
            ylim_max = dist * 1.2
            ylim_min = -dist * 1.2

            ax.set_ylim(ylim_min, ylim_max)

        ax.set_xlim(0, l)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color(COLORS[i // 3])

        major_ticks = np.arange(0, l, 20)
        ax.set_xticks(major_ticks)
        ax.grid()
        
        #ax.tick_params(colors='w')
    
        axes.append(ax)
    
    
    plt.show()

def signal_vis_compact(signal, normalized=False):
    '''Visualize a signal by plotting three columns together.

    Args:

        signal (FMSignal): The signal to be visualized.
        normalized (bool): Indicating whether the amplitude of the signal is
                           normalized or not. 

    Returns:

        None: No return value.

    '''

    assert isinstance(signal, FMSignal)

    fig = plt.figure()
    axes = []
    
    l = signal.length
    d = signal.dim
    
    n = d // 3
    
    for i in range(n):
        
        ax = fig.add_subplot(n, 1, i + 1)
    
        data_x = signal.data[:, i * 3 + 0]
        data_y = signal.data[:, i * 3 + 1]
        data_z = signal.data[:, i * 3 + 2]
    
        ax.plot(data_x, c='r')
        ax.plot(data_y, c='g')
        ax.plot(data_z, c='b')

        data_x_max = np.max(data_x)
        data_x_min = np.min(data_x)
        data_y_max = np.max(data_y)
        data_y_min = np.min(data_y)
        data_z_max = np.max(data_z)
        data_z_min = np.min(data_z)

        if normalized:
            ax.set_ylim(-3.5, 3.5)
        else:
            ymax = max(data_x_max, data_y_max, data_z_max)
            ymin = min(data_x_min, data_y_min, data_z_min)
            dist = max(abs(ymin), abs(ymax))
            ylim_max = dist * 1.2
            ylim_min = -dist * 1.2

            ax.set_ylim(ylim_min, ylim_max)
        
        ax.set_xlim(0, l)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color(COLORS[i])

        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        
        major_ticks = np.arange(0, l, 20)
        ax.set_xticks(major_ticks)
        ax.grid()
        
        #ax.tick_params(color='w')

    
        axes.append(ax)
    
    
    plt.show()


def signal_vis_comparison(signals, start_col=0, nr_cols=18, normalized=False):
    '''Visualize multiple signals on the same plot.

    Args:

        signals (FMSignal): The signals to be visualized.
        start_col (int): The start index of the sensor axes on the plot.
        nr_cols (int): The number of sensor axes on the plot.
        normalized (bool): Indicating whether the amplitudes of all signals are
                           normalized or not. 

    Returns:

        None: No return value.

    '''

    assert len(signals) > 0
    for signal in signals:
        assert isinstance(signal, FMSignal)


    fig = plt.figure()
    axes = []

    lmax = -1e6
    for signal in signals:
        if signal.length > lmax:
            lmax = signal.length
    
    for ii, i in enumerate(range(start_col, start_col + nr_cols)):
        
        ax = fig.add_subplot(nr_cols, 1, ii + 1)
    
        ymax = -1e6
        ymin = 1e6

        for j, signal in enumerate(signals):
            
                data_column = signal.data[:, i]

                ax.plot(data_column)

                data_column_max = np.max(data_column)
                if data_column_max > ymax:
                    ymax = data_column_max
                data_column_min = np.min(data_column)
                if data_column_min < ymin:
                    ymin = data_column_min
        
        if normalized:
            ax.set_ylim(-3.5, 3.5)
        else:
            dist = max(abs(ymin), abs(ymax))
            ylim_max = dist * 1.2
            ylim_min = -dist * 1.2

            ax.set_ylim(ylim_min, ylim_max)

        ax.set_xlim(0, lmax)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
            ax.spines[axis].set_color(COLORS[i // 3])

        major_ticks = np.arange(0, lmax, 20)
        ax.set_xticks(major_ticks)
        ax.grid()
        axes.append(ax)
    
    
    plt.show()

def signal_vis_comparison_side_by_side(signals_0, signals_1, 
        start_col=0, nr_cols=18, normalized=False):
    '''Visualize multiple signals on the same plot.

    Args:

        signals_0 (FMSignal): The first set of signals to be visualized.
        signals_1 (FMSignal): The second set of signals to be visualized.
        start_col (int): The start index of the sensor axes on the plot.
        nr_cols (int): The number of sensor axes on the plot.
        normalized (bool): Indicating whether the amplitudes of all signals are
                           normalized or not. 

    Returns:

        None: No return value.

    '''

    assert len(signals_0) > 0 and len(signals_1)
    for signal in signals_0:
        assert isinstance(signal, FMSignal)
    for signal in signals_1:
        assert isinstance(signal, FMSignal)


    fig = plt.figure()
    axes_left = []
    axes_right = []

    lmax = -1e6
    for signal in signals_0:
        if signal.length > lmax:
            lmax = signal.length
    for signal in signals_1:
        if signal.length > lmax:
            lmax = signal.length
    
    for ii, i in enumerate(range(start_col, start_col + nr_cols)):
        
        ax_left = fig.add_subplot(nr_cols, 2, 2 * ii + 1)
        ax_right = fig.add_subplot(nr_cols, 2, 2 * ii + 2)
    
        ymax = -1e6
        ymin = 1e6

        for signal_0 in signals_0:

            d0 = signal_0.data[:, i]
            ax_left.plot(d0)

            d_max = np.max(d0)
            if d_max > ymax:
                ymax = d_max
            d_min = np.min(d0)
            if d_min < ymin:
                ymin = d_min

        for signal_1 in signals_1:
            
            d1 = signal_1.data[:, i]
            ax_right.plot(d1)

            d_max = np.max(d1)
            if d_max > ymax:
                ymax = d_max
            d_min = np.min(d1)
            if d_min < ymin:
                ymin = d_min

        
        if normalized:
            ax_left.set_ylim(-3.5, 3.5)
            ax_right.set_ylim(-3.5, 3.5)
        else:
            dist = max(abs(ymin), abs(ymax))
            ylim_max = dist * 1.2
            ylim_min = -dist * 1.2

            ax_left.set_ylim(ylim_min, ylim_max)
            ax_right.set_ylim(ylim_min, ylim_max)

        ax_left.set_xlim(0, lmax)
        ax_right.set_xlim(0, lmax)


        for axis in ['top','bottom','left','right']:
            ax_left.spines[axis].set_linewidth(1)
            ax_left.spines[axis].set_color(COLORS[i // 3])
            ax_right.spines[axis].set_linewidth(1)
            ax_right.spines[axis].set_color(COLORS[i // 3])

        major_ticks = np.arange(0, lmax, 20)
        ax_left.set_xticks(major_ticks)
        ax_right.set_xticks(major_ticks)
        ax_left.grid()
        ax_right.grid()

        axes_left.append(ax_left)
        axes_right.append(ax_right)
    
    
    plt.show()






def plot_xyz_axes(ax, R, t, scale=1):
    '''Plot three line segments representing the x-y-z axes.

    Args:

        ax (Axes in matplotlib): The axes object to plot.
        R (3-by-3 ndarray): The rotation matrix (local to world).
        t (3-by-1 ndarray): The translation vector (local to world).
        scale (float): The length of the line segment. 

    Returns:

        None: No return value.

    **NOTE**: This is designed for internal usage only.

    '''

    x_axis_local = np.asarray([[0, 0, 0], [scale, 0, 0]]).T
    x_axis = np.matmul(R, x_axis_local) + t

    y_axis_local = np.asarray([[0, 0, 0], [0, scale, 0]]).T
    y_axis = np.matmul(R, y_axis_local) + t

    z_axis_local = np.asarray([[0, 0, 0], [0, 0, scale]]).T
    z_axis = np.matmul(R, z_axis_local) + t

    ax.plot(x_axis[0, :], x_axis[1, :], x_axis[2, :], color='r')
    ax.plot(y_axis[0, :], y_axis[1, :], y_axis[2, :], color='g')
    ax.plot(z_axis[0, :], z_axis[1, :], z_axis[2, :], color='b')




def trajectory_vis(signal, show_local_axes=False, interval=10):
    '''Visualize the signal as the trajectory of a point in 3D.

    Args:

        signal (FMSignal): The signal to be visualized.
        show_local_axes (bool): Indicating whether the orientation is shown.
        interval (int): Indicating how many samples to plot the orientation 
                        x-y-z axes once.

    Returns:

        None: No return value.

    '''
    
    if isinstance(signal, FMSignal) :
        trajectory = signal.data[:, 0:3]
        rotms, _qs = signal.get_orientation()
    elif isinstance(signal, FMSignalLeap) \
        or isinstance(signal, FMSignalGlove):
        trajectory = signal.trajectory
        rotms = signal.rotms
        assert trajectory is not None and rotms is not None
    else:
        raise ValueError('Wong signal: %s.' % signal.__class__)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.view_init(elev=30, azim=160)

    tx = trajectory[:, 0]
    ty = trajectory[:, 1]
    tz = trajectory[:, 2]

    tx_max = np.max(tx)
    tx_min = np.min(tx)
    ty_max = np.max(ty)
    ty_min = np.min(ty)
    tz_max = np.max(tz)
    tz_min = np.min(tz)

    dx = max(abs(tx_max), abs(tx_min))
    dy = max(abs(ty_max), abs(ty_min))
    dz = max(abs(tz_max), abs(tz_min))
    dist = max(dx, dy, dz)
    lim = dist * 1.2

    l = signal.length
    
    ax.plot(tx, ty, tz, color='k', markersize=0.1)
    
    if show_local_axes:

        

        for i in range(0, l, interval):
            
            R = rotms[i]
            t = trajectory[i].reshape((3,1))
            
            plot_xyz_axes(ax, R, t, lim / 10)
    
    ax.tick_params(color='w')
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
   
    # ax.zaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    # ax.xaxis.set_major_locator(plt.NullLocator())
    
    # ax.zaxis.set_major_formatter(plt.NullFormatter())
    # ax.yaxis.set_major_formatter(plt.NullFormatter())
    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    
    
    plt.show()

def trajectorie_vis_comparison(signals):
    '''Visualize multiple trajectories in the same plot.

    Args:

        signals (list): The list of signals to visualize.

    Returns:

        None: No return value.
    
    '''
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.view_init(elev=30, azim=160)

    dist_max = -1e6

    for signal in signals:

        if isinstance(signal, FMSignal) :
            trajectory = signal.data[:, 0:3]
        elif isinstance(signal, FMSignalLeap) \
            or isinstance(signal, FMSignalGlove):
            trajectory = signal.trajectory
            assert trajectory is not None
        else:
            raise ValueError('Wong signal: %s.' % signal.__class__)

        tx = trajectory[:, 0]
        ty = trajectory[:, 1]
        tz = trajectory[:, 2]

        tx_max = np.max(tx)
        tx_min = np.min(tx)
        ty_max = np.max(ty)
        ty_min = np.min(ty)
        tz_max = np.max(tz)
        tz_min = np.min(tz)

        dx = max(abs(tx_max), abs(tx_min))
        dy = max(abs(ty_max), abs(ty_min))
        dz = max(abs(tz_max), abs(tz_min))
        dist = max(dx, dy, dz)
        if dist > dist_max:
            dist_max = dist

        l = signal.length
    
        ax.plot(tx, ty, tz, markersize=0.1)
    
    lim = dist_max * 1.2

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    #plt.axis('equal')
    
    plt.show()


def plot_handgeo(ax, signal, i):
    '''Plot the skeleton shape of the hand.

    Args:

        signal (FMSignalLeap): The raw signal containing the "joints".
        i (int): 

    Returns:

        None: No return value.

    **NOTE**: This is designed for internal usage only.

    '''

    joints = signal.joints[i]

    
    x1 = []
    y1 = []
    z1 = []

    # Set up hand skeleton points.
    for j in range(5):
        for k in range(5):
            x1.append(joints[j][k][0])
            y1.append(joints[j][k][1])
            z1.append(joints[j][k][2])
        for k in range(4, -1, -1):
            x1.append(joints[j][k][0])
            y1.append(joints[j][k][1])
            z1.append(joints[j][k][2])

    ax.plot(x1, y1, z1, color='b', markersize=0.2)



def trajectory_animation(signal, speed=1, seg_length=-1, show_hand=False):
    '''Animate the signal trajectory in 3D.

    Args:

        signal (FMSignal): The signal to be visualized.
        speed (float): Indicating the animation speed. "speed=1" is 1x.
        seg_length (int): The animated segment length in number of samples.
        show_hand (bool): Indicating whether the hand geometry is shown.

    Returns:

        None: No return value.

    **NOTE**: The "speed" can be either a fraction number between 0 and 1, 
    which means slow down, or integers greater than 1, which means speed up.
    If it is greater than 1 and it is not an integer, it is rounded to the
    closest integer.

    '''

    if isinstance(signal, FMSignal) :
        trajectory = signal.data[:, 0:3]
    elif isinstance(signal, FMSignalLeap) \
        or isinstance(signal, FMSignalGlove):
        trajectory = signal.trajectory
        assert trajectory is not None
    else:
        raise ValueError('Wong signal: %s.' % signal.__class__)

    matplotlib.interactive(True)
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.view_init(elev=30, azim=160)

    tx = trajectory[:, 0]
    ty = trajectory[:, 1]
    tz = trajectory[:, 2]

    tx_max = np.max(tx)
    tx_min = np.min(tx)
    ty_max = np.max(ty)
    ty_min = np.min(ty)
    tz_max = np.max(tz)
    tz_min = np.min(tz)

    dx = max(abs(tx_max), abs(tx_min))
    dy = max(abs(ty_max), abs(ty_min))
    dz = max(abs(tz_max), abs(tz_min))
    dist = max(dx, dy, dz)
    lim = dist * 1.2


    l = signal.length
    
    for i in range(1, l):

        # speed up by subsampling
        if speed > 1 and i % int(round(speed)) != 0:
            continue
    
        ax.clear()
    
        if seg_length <= 0:
            s = 0
        else:
            s = i - seg_length
            if s < 0:
                s = 0
    
#         ax.plot(data[0:i, 0], data[0:i, 1], data[0:i, 2],
#                  color='k', markersize=0.2)

        ax.plot(tx[s:i], ty[s:i], tz[s:i], color='k', markersize=0.1)

        if isinstance(signal, FMSignalLeap) and show_hand:
            plot_handgeo(ax, signal, i)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        if i == 1:
            plt.pause(1)
        else:
            plt.pause(0.001)
            if speed < 1:
                plt.pause(0.02 / speed)
    
    matplotlib.interactive(False)
    
    

    ax.plot(tx, ty, tz, color='k', markersize=0.1)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig.canvas.draw()

    plt.show()
    






def orientation_animation(signal, speed=1, seg_length=-1):  
    '''Animate the orientation of the tracked point.

    Args:

        signal (FMSignal): The signal to be visualized.
        speed (float): Indicating the animation speed. "speed=1" is 1x.
        seg_length (int): The animated segment length in number of samples.

    Returns:

        None: No return value.

    '''

    if isinstance(signal, FMSignal) :
        rotms, _qs = signal.get_orientation()
    elif isinstance(signal, FMSignalLeap) \
        or isinstance(signal, FMSignalGlove):
        rotms = signal.rotms
    else:
        raise ValueError('Wong signal: %s.' % signal.__class__)

    matplotlib.interactive(True)
    
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.view_init(elev=30, azim=160)
    
    l = signal.length

    scale = 100
    lim = scale * 1.2

    xv = np.asarray((scale, 0, 0), np.float32).reshape((3, 1))
    points_x = np.zeros((l, 3))

    for i in range(l):
        
        R = rotms[i]
        xv_R = np.matmul(R, xv)
        points_x[i] = xv_R[:, 0]

    ox = points_x[:, 0]
    oy = points_x[:, 1]
    oz = points_x[:, 2]

    for i in range(l):
    
        # speed up by subsampling
        if speed > 1 and i % int(round(speed)) != 0:
            continue
    
        ax.clear()
    
        if seg_length <= 0:
            s = 0
        else:
            s = i - seg_length
            if s < 0:
                s = 0
        
        ax.plot(ox[s:i], oy[s:i], oz[s:i], color='k', markersize=0.1)

        R = rotms[i]
        t = np.asarray((0, 0, 0)).reshape((3, 1))
        plot_xyz_axes(ax, R, t, scale=scale)
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        if i == 1:
            plt.pause(1)
        else:
            plt.pause(0.001)
            if speed < 1:
                plt.pause(0.02 / speed)
    
    matplotlib.interactive(False)
    
    ax.plot(ox, oy, oz, color='k', markersize=0.1)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig.canvas.draw()
    
    plt.show()





def alignment_vis(signal, template, aligned_signal, plot_3d=False, col=0):
    '''Visualize the warping path and DTW distance matrix of an aligned signal.

    Args:

        signal (FMSignal): The signal that is aligned to the template.
        template (FMSignal): The template.
        aligned_signal (FMSignal): The aligned signal.
        plot_3d (bool): Indicating whether the plot is in 3D or in 2D.
        col (int): Indicating the sensor axis to be plotted.

    Returns:

        None: No return value.

    **NOTE**: The "aligned_signal" must have the "dist_matrix" attribute and the
    two attributes for the warping path (i.e., "a2to1_start" and "a2to1_end").
    This means when the "aligned_signal" is generated by calling the 
    "align_to()", the "keep_dist_matrix" parameter must be set to "True".

    '''

    # NOTE: Make a copy of "dist_matrix" in case we want to modify it.
    dist_matrix = aligned_signal.dist_matrix.copy()
    a2to1_start = aligned_signal.a2to1_start
    a2to1_end = aligned_signal.a2to1_end

    # Normalize dists_matrix to [0, 1] manually
    dmin = np.amin(dist_matrix)
    dmax = np.amax(dist_matrix)
    
    delta = dmax - dmin
    
    dist_matrix = (dist_matrix - dmin) / delta

    # n is the length of the template.
    n = dist_matrix.shape[0]
    # m is the length of the signal.
    m = dist_matrix.shape[1]

    fig = plt.figure(figsize=(8, 8))

    if plot_3d:

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        

        xs, ys = np.meshgrid(np.arange(0, n), np.arange(0, m))

        zs = dist_matrix.T

        surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

        ax.set_xlabel('template')
        ax.set_ylabel('signal')
        ax.set_zlabel('distance')

    else:

        # Mark the warping path.
        
        for i in range(len(a2to1_start)):
            for j in range(a2to1_start[i], a2to1_end[i] + 1):
                dist_matrix[i][j] = 1

        ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
        ax.set_axis_off()

        ax_s = fig.add_axes([0.1, 0.3, 0.2, 0.6])
        ax_t = fig.add_axes([0.3, 0.1, 0.6, 0.2])

        s_col = signal.data[:, col]
        t_col = template.data[:, col]

        dmax = max(np.max(s_col), np.max(t_col))
        dmin = min(np.min(s_col), np.min(t_col))
        dist = max(abs(dmax), abs(dmin)) * 1.2

        ax_s.plot(s_col, range(m))
        ax_t.plot(range(n), t_col)

        ax_s.set_xlim(-dist, dist)
        ax_s.set_ylim(0, m)
        ax_t.set_ylim(-dist, dist)
        ax_t.set_xlim(0, n)

        for axis in ['top','bottom','left','right']:
            ax_s.spines[axis].set_linewidth(1)
            ax_s.spines[axis].set_color(COLORS[col // 3])

            ax_t.spines[axis].set_linewidth(1)
            ax_t.spines[axis].set_color(COLORS[col // 3])

        major_ticks = np.arange(0, m, 20)
        ax_s.set_yticks(major_ticks)
        major_ticks = np.arange(0, n, 20)
        ax_t.set_xticks(major_ticks)

        ax_s.grid()
        ax_t.grid()

        ax.imshow(dist_matrix.T, origin='lower', aspect='auto')

        ax_s.set_ylabel('signal')
        ax_t.set_xlabel('template')


    #major_ticks = np.arange(0, 60, 20)
    #ax.set_xticks(major_ticks)
    #ax.set_xticklabels([0, 1, 2, 5])
    #ax.set_aspect('equal')
    #ax.set_xticks([0, 5])
    
    plt.show()













