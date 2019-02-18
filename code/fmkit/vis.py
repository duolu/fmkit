'''
Visualization code for finger motion signals, shape, and hand geometry

@author: duolu
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import colorsys
import numpy as np


from fmkit.core import *
from fmkit.utils import dtw
from fmkit.utils import euclidean_distance

COLOR_LIST = ['b', 'r', 'g', 'c', 'm', 'y', 'k']


def signal_vis(signal, nr_column):
    '''
    Visualize a signal by plotting the specified columns.
    '''
    
    fig = plt.figure()
    axes = []
    
    l = signal.l
    
    for i in range(nr_column):
        
        ax = fig.add_subplot(nr_column, 1, i + 1)
    
        #ax.plot(signal.ts[0:l], signal.data[0:l, i])
        ax.plot(signal.data[0:l, i])
        #ax.set_ylim(-3.5, 3.5)
        #ax.set_xlim(0, 400)
        
        ax.grid()
    
        axes.append(ax)
    
    
    plt.show()

def signals_vis_same_plot(signals, nr_column):
    '''
    Visualize multiple signals on the same plot.
    '''
    
    fig = plt.figure()
    axes = []
    
    for i in range(nr_column):
        
        ax = fig.add_subplot(nr_column, 1, i + 1)
    
        for signal in signals:
                data_t = signal.data.T
                l = signal.l

                ax.plot(signal.ts[:l], data_t[i][:l])
    
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
    for signal in signals:
        
        if signal.ts[-1] > ts_max:
            ts_max = signal.ts[-1]
    
    for i in range(nr_column):
        
        for signal, j in zip(signals, range(nr_signals)):
                data_t = signal.data.T
                l = signal.l

                ax = fig.add_subplot(nr_column * nr_signals, 1, i * nr_signals + j + 1)
                
                hue = (j * 20 % 256) / 256
                c = np.asarray(colorsys.hsv_to_rgb(hue, 1, 1))
                
                ax.plot(signal.ts[:l], data_t[i][:l], color=c)
    
                ax.set_xlim(0, ts_max)
    
        axes.append(ax)
    
    
    plt.show()


def signals_vis_comparing_two_batch(batch1, batch2, column_index = 0):
    '''
    Visualize two batches of multiple signals side by side in two figures.
    '''

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    for signal in batch1:
        l = signal.l
        ax1.plot(signal.ts[:l, column_index], signal.data[:l, column_index])
        
    #ax1.set_xlim(xlim)

    ax2 = fig.add_subplot(2, 1, 2)
    for signal in batch2:
        l = signal.l
        ax2.plot(signal.ts[:l, column_index], signal.data[:l, column_index])
        
    #ax2.set_xlim(xlim)

    plt.show()



def shape_vis_3d(signal):
    '''
    Visualize the signal as the trajectory of a point in 3D.
    '''
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')


    data = signal.data
    l = signal.l
    
    ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2], 
             color = 'b', markersize = 0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def shapes_vis_3d_same_plot(signals):
    '''
    Visualize multiple signals as the trajectories of a point in 3D.
    '''
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')


    for signal in signals:
    
        data = signal.data
        l = signal.l
    
        ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2], 
            markersize = 0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def trajectory_3d_animation(signal, speed=10):
    '''
    Animate the signal trajectory in 3D.
    '''
    
    matplotlib.interactive(True)
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')


    data = signal.data
    l = signal.l
    
    for i in range(l):

        # speed up by subsampling
        if i % speed != 0:
            continue
    
        ax.clear()
    
        ax.plot(data[0:i, 0], data[0:i, 1], data[0:i, 2], 
                 color = 'b', markersize = 0.2)
        
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-200, 200)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        plt.pause(0.001)
    
    matplotlib.interactive(False)
    
    ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2], 
             color = 'b', markersize = 0.2)
    
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    


def handgeo_vis(signal):
    '''
    Visualize the hand geometry in 3D.
    '''
    data_aux = signal.data_aux
    
    joints = data_aux[0]

    
    x1 = []
    y1 = []
    z1 = []

    for j in range(5):
        for k in range(5):
            x1.append(joints[j][k][0])
            y1.append(joints[j][k][1])
            z1.append(joints[j][k][2])
        for k in range(4, -1, -1):
            x1.append(joints[j][k][0])
            y1.append(joints[j][k][1])
            z1.append(joints[j][k][2])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot(x1, y1, z1, color = 'b', markersize = 0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def shape_handgeo_3d_animation(signal, speed = 10):  
    '''
    Animate the signal trajectory as well as hand geometry in 3D.
    '''
    
    matplotlib.interactive(True)
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')


    data = signal.data
    
    l = signal.l
    
    for i in range(l):
    
        # speed up by subsampling
        if i % speed != 0:
            continue
    
        ax.clear()
    
        ax.plot(data[0:i, 0], data[0:i, 1], data[0:i, 2], 
                 color = 'r', markersize = 0.2)
        
        joints = signal.data_aux[i]
        x1 = []
        y1 = []
        z1 = []
    
        for j in range(5):
            for k in range(5):
                x1.append(joints[j][k][0])
                y1.append(joints[j][k][1])
                z1.append(joints[j][k][2])
            for k in range(4, -1, -1):
                x1.append(joints[j][k][0])
                y1.append(joints[j][k][1])
                z1.append(joints[j][k][2])
        ax.plot(x1, y1, z1, color = 'b', markersize = 0.2)
        
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_zlim(-200, 200)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        plt.pause(0.001)
    
    matplotlib.interactive(False)
    
    ax.plot(data[0:l, 0], data[0:l, 1], data[0:l, 2], 
             color = 'r', markersize = 0.2)
    
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    
    

def warping_path_vis(dists_matrix, a1start, a1end):
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

    
    plt.imshow(dists_matrix)
    
    plt.show()




















