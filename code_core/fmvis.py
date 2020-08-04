'''
Visualization code for finger motion signals, shape, and hand geometry

@author: duolu
'''

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import colorsys

from fmutils import normalize_warping_path
from fmutils import warping_path_to_index_sequences

colors = ('k', 'r', 'g', 'b', 'c', 'm', 'y')

colors_other = ('k', 'b', 'k', 'k', 'k', 'k')


def signal_vis(signal, nr_column, start_column=0):
    '''Visualize a signal by plotting the specified columns.
    
    '''
    
    fig = plt.figure()
    axes = []
    
    l = signal.len
    
    for i in range(nr_column):
        
        ax = fig.add_subplot(nr_column, 1, i + 1)
    
        # ax.plot(signal.ts[0:len], signal.data[0:len, i])
        #ax.plot(signal.data[0:l, start_column + i], color=colors[i // 3])
        ax.plot(signal.data[0:l, start_column + i], color='b')
        ax.set_ylim(-3.4, 3.4)
        ax.set_xlim(0, signal.len)
        
        major_ticks = np.arange(0, l, 20)
        ax.set_xticks(major_ticks)
        ax.grid()
        
        #ax.tick_params(colors='w')
    
        axes.append(ax)
    
    
    plt.show()

def signal_vis_three_column(signal):
    '''
    Visualize a signal by plotting three columns together.
    '''

    fig = plt.figure()
    axes = []
    
    l = signal.len
    d = signal.dim
    
    n = d // 3
    
    for i in range(n):
        
        ax = fig.add_subplot(n, 1, i + 1)
    
        start_col = i * 3
        end_col = (i + 1) * 3
    
        # ax.plot(signal.ts[0:len], signal.data[0:len, i])
        ax.plot(signal.data[0:l, start_col], c='r')
        ax.plot(signal.data[0:l, start_col + 1], c='g')
        ax.plot(signal.data[0:l, start_col + 2], c='b')
        #ax.set_ylim(-3, 3)
        ax.set_xlim(0, signal.len)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        major_ticks = np.arange(0, l, 20)
        ax.set_xticks(major_ticks)
        ax.grid()
        
        ax.tick_params(colors='w')

    
        axes.append(ax)
    
    
    plt.show()


def signals_vis_same_plot(signals, nr_column, start_column=0):
    '''
    Visualize multiple signals on the same plot.
    '''
    
    fig = plt.figure()
    axes = []
    
    for i in range(nr_column):
        
        ax = fig.add_subplot(nr_column, 1, i + 1)
    
        for j, signal in enumerate(signals):
            
            
                data = signal.data
                l = signal.len

                ax.plot(signal.ts[:l], data[0:l, start_column + i])
                #ax.plot(signal.data[0:l, start_column + i], color=colors_other[j])
                
        #ax.set_ylim(-3, 3)
        ax.grid()
        axes.append(ax)
    
    
    plt.show()

def signals_vis_side_by_side(signals, nr_column):
    '''
    Visualize multiple signals side by side.
    '''
    
    fig = plt.figure()
    axes = []
    
    nr_signals = len(signals)
    
    ts_max = 0;
    l_max = 0
    for signal in signals:
        
        if signal.ts[-1] > ts_max:
            ts_max = signal.ts[-1]
            
        if signal.len > l_max:
            l_max = signal.len
    
    for i in range(nr_column):
        
        for signal, j in zip(signals, range(nr_signals)):
                data_t = signal.data.T
                l = signal.len

                ax = fig.add_subplot(nr_column * nr_signals, 1, i * nr_signals + j + 1)
                
                hue = (j * 20 % 256) / 256
                c = np.asarray(colorsys.hsv_to_rgb(hue, 1, 1))
                
                ax.plot(signal.ts[:l], data_t[i][:l], color=c)
                #ax.plot(data_t[i][:l], color=c)
    
                ax.set_xlim(0, ts_max)
                #ax.set_xlim(0, l_max)
    
        axes.append(ax)
    
    
    plt.show()







def plot_xyz_axes_rotm(ax, R, t, scale=1):

    x_axis_local = np.asarray([[0, 0, 0], [scale, 0, 0]]).T
    x_axis = np.matmul(R, x_axis_local) + t

    y_axis_local = np.asarray([[0, 0, 0], [0, scale, 0]]).T
    y_axis = np.matmul(R, y_axis_local) + t

    z_axis_local = np.asarray([[0, 0, 0], [0, 0, scale]]).T
    z_axis = np.matmul(R, z_axis_local) + t

    ax.plot(x_axis[0, :], x_axis[1, :], x_axis[2, :], color='r')
    ax.plot(y_axis[0, :], y_axis[1, :], y_axis[2, :], color='g')
    ax.plot(z_axis[0, :], z_axis[1, :], z_axis[2, :], color='b')


def axes_vis_3d(R, t):
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    plot_xyz_axes_rotm(ax, R, t)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.axis('equal')
    
    plt.show()


def shape_vis_3d(signal, angle=False, interval=10, lim=200):
    '''
    Visualize the signal as the trajectory of a point in 3D.
    '''
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')


    data = signal.data
    l = signal.len
    
    ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2],
             color='k', markersize=0.1)

    t1 = 43
     
    t2 = 64
     
 
    ax.plot(data[0:t1, 0], data[0:t1, 1], data[0:t1, 2],
             color='r', markersize=0.4)
  
    ax.plot(data[t1:t2, 0], data[t1:t2, 1], data[t1:t2, 2],
             color='k', markersize=0.4)
 
    ax.plot(data[t2:, 0], data[t2:, 1], data[t2:, 2],
             color='b', markersize=0.4)

#     ax.plot(data[0:15, 0], data[0:15, 1], data[0:15, 2],
#              color='r', markersize=0.4)
#  
#     ax.plot(data[33:63, 0], data[33:63, 1], data[33:63, 2],
#              color='g', markersize=0.4)
#  
#     ax.plot(data[82:113, 0], data[82:113, 1], data[82:113, 2],
#              color='b', markersize=0.4)
# 
#     ax.plot(data[135:174, 0], data[135:174, 1], data[135:174, 2],
#              color='c', markersize=0.4)
# 
#     ax.plot(data[193:212, 0], data[193:212, 1], data[193:212, 2],
#              color='m', markersize=0.4)
#     ax.plot(data[218:225, 0], data[218:225, 1], data[218:225, 2],
#              color='m', markersize=0.4)
# 
#     ax.plot(data[247:284, 0], data[247:284, 1], data[247:284, 2],
#              color='y', markersize=0.4)
    
    
    if angle:
        for i in range(0, l, interval):
            
            R = signal.rotms[i]
            t = signal.data[i, 0:3].reshape((3,1))
            
            plot_xyz_axes_rotm(ax, R, t, lim / 50)
    
    ax.tick_params(colors='w')
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #ax.axis('equal')
    
    #ax.zaxis.set_major_locator(plt.NullLocator())
    #ax.yaxis.set_major_locator(plt.NullLocator())
    #ax.xaxis.set_major_locator(plt.NullLocator())
    
#     ax.zaxis.set_major_formatter(plt.NullFormatter())
#     ax.yaxis.set_major_formatter(plt.NullFormatter())
#     ax.xaxis.set_major_formatter(plt.NullFormatter())
    
    
    plt.show()

def shapes_vis_3d_same_plot(signals, lim=200):
    '''
    Visualize multiple signals as the trajectories of a point in 3D.
    
    '''
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')


    for signal in signals:
    
        data = signal.data
        l = signal.len
    
        ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2],
            markersize=0.2)
    
    ax.tick_params(colors='w')
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    #plt.axis('equal')
    
    plt.show()

def axes_3d_animation(signal):
    '''
    Animate the local reference frame by three axes.
    
    Note that this only animates the rotation and it relies on the rotation 
    matrix, i.e., signal.rotms must exist.
    
    '''
    matplotlib.interactive(True)
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    l = signal.len


    t = np.asarray((0, 0, 0), np.float32).reshape((3, 1))
    
    xv = np.asarray((1, 0, 0), np.float32).reshape((3, 1))
    points_x = np.zeros((l, 3))

    yv = np.asarray((0, -1, 0), np.float32).reshape((3, 1))
    points_y = np.zeros((l, 3))

    for i in range(l):
        
        R = signal.rotms[i]
        xv_R = np.matmul(R, xv)
        points_x[i] = xv_R[:, 0]
        yv_R = np.matmul(R, yv)
        points_y[i] = yv_R[:, 0]
        

    
    for i in range(l):

        R = signal.rotms[i]
        
        start = 0 if i < 50 else i - 50
        end = i
        
        ax.clear()
        
        plot_xyz_axes_rotm(ax, R, t)
        
        if i > 0:
            ax.plot(points_x[start:end, 0], 
                    points_x[start:end, 1], 
                    points_x[start:end, 2],
                    color='k', markersize=1)

            ax.plot(points_y[start:end, 0], 
                    points_y[start:end, 1], 
                    points_y[start:end, 2],
                    color='k', markersize=1)


        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        plt.pause(0.05)

    matplotlib.interactive(False)
    
    plot_xyz_axes_rotm(ax, R, t)
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


def trajectory_3d_animation(signal, speed=1, angle=False, lim=200):
    '''
    Animate the signal trajectory in 3D.
    '''
    
    matplotlib.interactive(True)
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    

    #plt.pause(10)

    data = signal.data
    l = signal.len
    
    for i in range(l):

        # speed up by subsampling
        if i % speed != 0:
            continue
    
        ax.clear()
    
        start = i - 100
        if start < 0:
            start = 0
    
#         ax.plot(data[0:i, 0], data[0:i, 1], data[0:i, 2],
#                  color='k', markersize=0.2)

        ax.plot(data[start:i, 0], data[start:i, 1], data[start:i, 2],
                 color='k', markersize=0.2)
 
        if angle:
            
            R = signal.rotms[i]
            t = signal.data[i, 0:3].reshape((3,1))
            
            plot_xyz_axes_rotm(ax, R, t, 10)
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        plt.pause(0.001)
    
    matplotlib.interactive(False)
    
    ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2],
             color='b', markersize=0.2)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    


def _plot_handgeo(ax, signal, i):
    '''
    Internal hand geometry plot function use for both visualization and 
    animation.
    '''

    joints = signal.data_aux[i]

    
    x1 = []
    y1 = []
    z1 = []

    # set up hand skeleton
    for j in range(5):
        for k in range(5):
            x1.append(joints[j][k][0])
            y1.append(joints[j][k][1])
            z1.append(joints[j][k][2])
        for k in range(4, -1, -1):
            x1.append(joints[j][k][0])
            y1.append(joints[j][k][1])
            z1.append(joints[j][k][2])

    ax.plot(x1, y1, z1, color='k', markersize=0.2)


def handgeo_vis(signal, i=0):
    '''
    Visualize the hand geometry in 3D.
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    _plot_handgeo(ax, signal, i)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def shape_handgeo_3d_animation(signal, speed=2):  
    '''
    Animate the signal trajectory as well as hand geometry in 3D.
    '''
    
    matplotlib.interactive(True)
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')


    data = signal.data
    
    l = signal.len
    
    for i in range(l):
    
        # speed up by subsampling
        if i % speed != 0:
            continue
    
        ax.clear()
    
        ax.plot(data[0:i, 0], data[0:i, 1], data[0:i, 2],
                 color='k', markersize=0.2)
        
        _plot_handgeo(ax, signal, i)
        
        
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-200, 200)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        plt.pause(0.001)
    
    matplotlib.interactive(False)
    
    ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2],
             color='b', markersize=0.2)
    
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

# def handgeo_rotation_animation(signal):
#     
#     matplotlib.interactive(True)
#     
#     fig = plt.figure()
# 
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
# 
#     #plt.pause(10)
# 
#     l = signal.len
#     
#     for i in range(0, 90):
#     
#         angle = 1 * math.pi / 180
#     
#         ax.clear()
#     
#         signal.rotate(0, 0, angle)
#     
#         ax.plot(signal.data[0:l, 0], signal.data[0:l, 1], signal.data[0:l, 2],
#              color='b', markersize=0.2)
# 
#         _plot_handgeo(ax, signal, 0)
# 
#         ax.set_xlim(-200, 200)
#         ax.set_ylim(-200, 200)
#         ax.set_zlim(-200, 200)
#         
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#     
#         #plt.axis('equal')
# 
#         plt.pause(0.03)
# 
#     matplotlib.interactive(False)
#     
#     ax.plot(signal.data[0:l, 0], signal.data[0:l, 1], signal.data[0:l, 2],
#          color='b', markersize=0.2)
# 
#     ax.set_xlim(-200, 200)
#     ax.set_ylim(-200, 200)
#     ax.set_zlim(-200, 200)
#     
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
# 
#     #plt.axis('equal')
# 
#     plt.show()


def warping_path_vis(a1start, a1end):
    
    wp, a1start_100, a1end_100 = normalize_warping_path(a1start, a1end)
    
    xs, ys = warping_path_to_index_sequences(a1start_100, a1end_100)
    
    path = np.zeros((100, 100))
    
    path[ys, xs] += 1

    print(path)
    plt.imshow(path, origin='lower')
    
    #plt.plot(xp_int, wp_int)
    
    #plt.plot(wp)
    #plt.plot(a1start_100)
    #plt.plot(a1end_100)
    
    plt.show()

def dtw_vis(dists_matrix, a1start, a1end):
    '''
    Visualize the warping path calculated by DTW
    '''
    # normalize dists_matrix to [0, 1] manually
    dmin = np.amin(dists_matrix)
    dmax = np.amax(dists_matrix)
    
    delta = dmax - dmin
    
    dists_matrix = (dists_matrix - dmin) / delta

    # mark the warping path
    
    for i in range(len(a1start)):
        for j in range(a1start[i], a1end[i] + 1):
            dists_matrix[i][j] = 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.imshow(dists_matrix, origin='lower')
    ax.tick_params(colors='w')
    
    #major_ticks = np.arange(0, 60, 20)
    #ax.set_xticks(major_ticks)
    #ax.set_xticklabels([0, 1, 2, 5])
    #ax.set_aspect('equal')
    #ax.set_xticks([0, 5])
    
    plt.show()


def alignment_path_vis(signals_aligned):

    path = np.zeros((100, 100))

    for signal in signals_aligned:
        
        a1start = signal.a1start
        a1end = signal.a1end
        
        wp, a1start_100, a1end_100 = normalize_warping_path(a1start, a1end)
        
        xs, ys = warping_path_to_index_sequences(a1start_100, a1end_100)
        
        path[ys, xs] += 1

    plt.imshow(path, origin='lower')
    
    plt.show()


def ttv_vis(diff_signal, col, th1, th2):



    pass


def activation_vis(activation):

    plt.imshow(activation)
    
    plt.show()
    


class FMVisualizerLeap(object):

    def __init__(self, signal):

        self.signal = signal
        l = signal.len

        fig = plt.figure(figsize=(15, 10), dpi=100, facecolor='w', edgecolor='k')
        
        ax = fig.add_axes([0.0, 0.2, 0.9, 0.7], projection='3d')

        self.fig = fig
        self.ax = ax

        axcolor = 'lightgoldenrodyellow'

        ax_frame = fig.add_axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
        
        s_frame = Slider(ax_frame, 'frame', 0, l, valinit=0, valstep=1)
        s_frame.on_changed(self.update_frame)
    
        self.ax_frame = ax_frame
        self.s_frame = s_frame
    
    
    
        self.update_frame(0)
    
    def plot_hand(self, ax, signal, i):
        '''
        Internal hand geometry plot function use for both visualization and 
        animation.
        '''
    
        joints = signal.data_aux[i]
    
        
        x1 = []
        y1 = []
        z1 = []
    
        # set up hand skeleton
        for j in range(5):
            for k in range(5):
                x1.append(joints[j][k][0])
                y1.append(joints[j][k][1])
                z1.append(joints[j][k][2])
            for k in range(4, -1, -1):
                x1.append(joints[j][k][0])
                y1.append(joints[j][k][1])
                z1.append(joints[j][k][2])
    
        ax.plot(x1, y1, z1, color='k', markersize=0.2)


    def adjust_axes(self, ax):
        
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-200, 200)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_aspect('equal')
    
    
    
    def update_frame(self, val):
    
    
        i = int(self.s_frame.val)
        
        data = self.signal.data
    
        self.ax.clear()
        
        
#         self.plot_hand(self.ax, self.signal, i)
#         
#         self.ax.plot(data[0:i, 0], data[0:i, 1], data[0:i, 2],
#                  color='k', markersize=0.2)

        self.ax.plot(data[:, 0], data[:, 1], data[:, 2],
                  color='k', markersize=0.2)

        self.ax.scatter(data[i, 0], data[i, 1], data[i, 2],
                  c='r', s=2)

        self.adjust_axes(self.ax)
        
        #print(self.signal.confs[i], self.signal.valids[i])
    
    
    def run(self):
        
        plt.show()

    
    
    pass












