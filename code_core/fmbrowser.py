'''
Created on Mar 25, 2020

@author: duolu
'''

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.widgets import TextBox
import colorsys

import codecs # for display non-ascii characters

import json

from fmsignal import FMSignalDesc, FMSignalLeap
from fmsignal import FMSignalLeap
from fmsignal import FMSignalGlove

from pyrotation import euler_zyx_to_rotation_matrix

from pyrotation import *

class FMBrowser(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        
        self.load_config('./fmbrowser.conf')
        
        self.load_dataset_meta()
        
        self.load_signal_meta()
        
        self.load_current_signal()
        
        self.remembered_signals = []
        
        
        
        
        self.setup_ui()
        
        self.update_all()
        
        pass
    
    
    def load_config(self, config_fn):

        
        with open(config_fn, mode='r') as fd:
            
            config_str = fd.read()
            config = json.loads(config_str)
        
            self.data_folder = config['data_folder']
        
            self.meta_folder = config['meta_folder']
            self.devices = config['devices']
            self.nr_devices = len(self.devices)
            self.device_id = 0
            self.device = self.devices[self.device_id]

            self.datasets = config['datasets']
            self.nr_datasets = len(self.datasets)
            self.dataset_id = 0
            self.dataset = self.datasets[self.dataset_id]

            self.mode = config['mode']
        
        
        self.signal_id = 0
        self.rep = 0
        
        pass
    
    
    
    
    def load_dataset_meta(self):
        '''
        
        NOTE: This must be called every time when the current device or 
              dataset is changed.
        '''
        
        meta_fn = self.meta_folder + '/meta_' + self.device + '_' + self.dataset
        
        self.descs = FMSignalDesc.construct_from_meta_file(meta_fn)
        
        self.nr_signals = len(self.descs)
        
        if self.signal_id >= self.nr_signals:
            
            self.signal_id = 0
        
        pass
    
    def load_signal_meta(self):
        '''
        
        NOTE: This must be called every time when the current signal is 
              changed.
        '''
        
        self.desc = self.descs[self.signal_id]
        
        self.nr_repetitions = self.desc.end - self.desc.start
        
        if self.rep > self.nr_repetitions:
            
            self.rep = 0
    
    
    def load_current_signal(self):
    
        
    
        mode = self.mode
        
        sub_folder = self.device + '_' + self.dataset + '_' + mode
    
        fn = self.data_folder + '/' + sub_folder + \
            '/' + self.desc.fn_prefix + '_%d' % (self.rep + self.desc.start)
    
        print(fn)

        if self.device == 'leap':
            
            self.signal = FMSignalLeap.construct_from_file(fn, 
                mode=mode, point='tip', use_stat=False, use_handgeo=False)
            
        elif self.device == 'glove':
            
            self.signal = FMSignalGlove.construct_from_file(fn, 
                mode=mode, point='tip', use_stat=False)
            
            #self.signal.convert_from_glove2_to_standard_glove()
        
        else:
            
            print('BUG in load_current_signal() !!!')
            exit()
    
        self.signal.preprocess_shape(filter_cut_freq=10)
        
        #self.signal.amplitude_normalize()
        
        
        
    
    
    def setup_ui(self):
        
        fig = plt.figure(figsize=(16, 9))
        fig.canvas.mpl_connect('close_event', self.on_close)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.fig = fig
        
        
        self.setup_browsing_top_buttons()
        self.setup_browsing_middle_buttons()
        self.setup_browsing_panels()
        self.setup_browsing_bottom_buttons()
        
        #print('setup done')
        
        pass
    
    
    def setup_browsing_top_buttons(self):
    
        fig = self.fig
        
        ax_button_device_prev = fig.add_axes([0.02, 0.88, 0.03, 0.05])
        ax_text_device = fig.add_axes([0.06, 0.88, 0.06, 0.05])
        ax_button_device_next = fig.add_axes([0.13, 0.88, 0.03, 0.05])

        button_device_prev = Button(ax_button_device_prev, '<')
        button_device_next = Button(ax_button_device_next, '>')

        button_device_prev.on_clicked(self.on_button_device_prev)
        button_device_next.on_clicked(self.on_button_device_next)
    
        ax_button_dataset_prev = fig.add_axes([0.17, 0.88, 0.03, 0.05])
        ax_text_dataset = fig.add_axes([0.21, 0.88, 0.10, 0.05])
        ax_button_dataset_next = fig.add_axes([0.32, 0.88, 0.03, 0.05])

        button_dataset_prev = Button(ax_button_dataset_prev, '<')
        button_dataset_next = Button(ax_button_dataset_next, '>')

        button_dataset_prev.on_clicked(self.on_button_dataset_prev)
        button_dataset_next.on_clicked(self.on_button_dataset_next)
    
        ax_button_index_prev = fig.add_axes([0.36, 0.88, 0.03, 0.05])
        ax_text_index = fig.add_axes([0.40, 0.88, 0.06, 0.05])
        ax_button_index_next = fig.add_axes([0.47, 0.88, 0.03, 0.05])

        button_index_prev = Button(ax_button_index_prev, '<')
        button_index_next = Button(ax_button_index_next, '>')

        button_index_prev.on_clicked(self.on_button_index_prev)
        button_index_next.on_clicked(self.on_button_index_next)

        ax_button_rep_prev = fig.add_axes([0.51, 0.88, 0.03, 0.05])
        ax_text_rep = fig.add_axes([0.55, 0.88, 0.03, 0.05])
        ax_button_rep_next = fig.add_axes([0.59, 0.88, 0.03, 0.05])

        button_rep_prev = Button(ax_button_rep_prev, '<')
        button_rep_next = Button(ax_button_rep_next, '>')
    
        button_rep_prev.on_clicked(self.on_button_rep_prev)
        button_rep_next.on_clicked(self.on_button_rep_next)

        
        #ax_button_reload = fig.add_axes([0.81, 0.88, 0.08, 0.05])


        ax_button_remember = fig.add_axes([0.81, 0.88, 0.08, 0.05])
        ax_button_forget = fig.add_axes([0.91, 0.88, 0.07, 0.05])

        button_remember = Button(ax_button_remember, 'remember')
        button_forget = Button(ax_button_forget, 'forget')

        button_remember.on_clicked(self.on_button_remember)
        button_forget.on_clicked(self.on_button_forget)
    
    
        self.button_device_prev = button_device_prev
        self.ax_text_device = ax_text_device
        self.button_device_next = button_device_next
        
        self.button_dataset_prev = button_dataset_prev
        self.ax_text_dataset = ax_text_dataset
        self.button_dataset_next = button_dataset_next
        
        self.button_index_prev = button_index_prev
        self.ax_text_index = ax_text_index
        self.button_index_next = button_index_next
        
        self.button_rep_prev = button_rep_prev
        self.ax_text_rep = ax_text_rep
        self.button_rep_next = button_rep_next
    
        self.button_remember = button_remember
        self.button_forget = button_forget
    
        pass


    def setup_browsing_middle_buttons(self):
        
        
        
        pass
    
    def reset_panel_states(self):
        
        self.origin_x = 0
        self.origin_y = 0
        self.origin_z = 0
        self.origin_roll = 0
        self.origin_pitch = 0
        self.origin_yaw = 0
        
        self.y_offset = 0
        
        self.show_motion = False
        self.pointer = self.pointer_front
    
    
    def setup_browsing_panels(self):

        self.show_aux = False
        self.show_pointer = False
        self.show_motion = False
        
        self.trajectory_lim = 200
        self.trajectory_scale = 0

        self.col_group = 0
        self.col_texts = ['pos', 'velocity', 'acc', 'orientation', 'angular velocity', 'angular acc']
    
        self.pointer = 0
        self.pointer_front = 0
        self.pointer_rear = 1
    
        fig = self.fig
    
        self.timer = fig.canvas.new_timer(interval=200)
        self.timer.add_callback(self.on_timer_expire)
    
        ax_trajectory = fig.add_axes([0.02, 0.21, 0.46, 0.58], 
                projection='3d')

        self.reset_panel_states()
    
    
        ax_signals = []
    
        
        ax_button_axes_prev = fig.add_axes([0.52, 0.67, 0.05, 0.05])
        ax_text_axes = fig.add_axes([0.6, 0.67, 0.25, 0.05])
        ax_button_axes_next = fig.add_axes([0.88, 0.67, 0.05, 0.05])
    
        button_axes_prev = Button(ax_button_axes_prev, '<')
        button_axes_next = Button(ax_button_axes_next, '>')

        button_axes_prev.on_clicked(self.on_button_axes_prev)
        button_axes_next.on_clicked(self.on_button_axes_next)

        for i in range(3):
    
            ax = fig.add_axes([0.52, 0.50 - 0.15 * i + 0.02, 0.41, 0.12])
            
            ax_signals.append(ax)
    
        self.ax_trajectory = ax_trajectory
        
        self.ax_button_axes_prev = ax_button_axes_prev
        self.ax_text_axes = ax_text_axes
        self.ax_button_axes_next = ax_button_axes_next
        
        self.button_axes_prev = button_axes_prev
        self.button_axes_next = button_axes_next
        
        self.ax_signals = ax_signals
        

    def setup_browsing_bottom_buttons(self):
    
        fig = self.fig
    

        ax_slider_pointer = fig.add_axes([0.52, 0.14, 0.41, 0.05])
        ax_slider_front = fig.add_axes([0.52, 0.08, 0.41, 0.05])
        ax_slider_rear = fig.add_axes([0.52, 0.02, 0.41, 0.05])
        
        slider_pointer = Slider(ax_slider_pointer, 'pointer', 0, 1, valinit=0, valstep=0.01)
        slider_front = Slider(ax_slider_front, 'front', 0, 1, valinit=0, valstep=0.01)
        slider_rear = Slider(ax_slider_rear, 'rear', 0, 1, valinit=1, valstep=0.01)
    
        slider_pointer.on_changed(self.on_slider_pointer_change)
        slider_front.on_changed(self.on_slider_front_change)
        slider_rear.on_changed(self.on_slider_rear_change)
    
        self.slider_pointer = slider_pointer
        self.slider_front = slider_front
        self.slider_rear = slider_rear
    


        ax_slider_x = fig.add_axes([0.05, 0.14, 0.08, 0.05])
        ax_slider_y = fig.add_axes([0.05, 0.08, 0.08, 0.05])
        ax_slider_z = fig.add_axes([0.05, 0.02, 0.08, 0.05])
        
        slider_x = Slider(ax_slider_x, 'x', -1, 1, valinit=0, valstep=0.01)
        slider_y = Slider(ax_slider_y, 'y', -1, 1, valinit=0, valstep=0.01)
        slider_z = Slider(ax_slider_z, 'z', -1, 1, valinit=0, valstep=0.01)
    
        slider_x.on_changed(self.on_slider_x_change)
        slider_y.on_changed(self.on_slider_y_change)
        slider_z.on_changed(self.on_slider_z_change)
    
        self.slider_x = slider_x
        self.slider_y = slider_y
        self.slider_z = slider_z



        ax_slider_roll = fig.add_axes([0.18, 0.14, 0.08, 0.05])
        ax_slider_pitch = fig.add_axes([0.18, 0.08, 0.08, 0.05])
        ax_slider_yaw = fig.add_axes([0.18, 0.02, 0.08, 0.05])
        
        slider_roll = Slider(ax_slider_roll, 'roll', -180, 180, valinit=0, valstep=1)
        slider_pitch = Slider(ax_slider_pitch, 'pitch', -180, 180, valinit=0, valstep=1)
        slider_yaw = Slider(ax_slider_yaw, 'yaw', -180, 180, valinit=0, valstep=1)
    
        slider_roll.on_changed(self.on_slider_roll)
        slider_pitch.on_changed(self.on_slider_pitch)
        slider_yaw.on_changed(self.on_slider_yaw)
    
        self.slider_roll = slider_roll
        self.slider_pitch = slider_pitch
        self.slider_yaw = slider_yaw

        
        
        ax_slider_y_offset = fig.add_axes([0.35, 0.14, 0.08, 0.05])
        ax_slider_scale = fig.add_axes([0.35, 0.08, 0.08, 0.05])
        
        ax_button_show_aux = fig.add_axes([0.30, 0.02, 0.05, 0.05])
        ax_button_show_pointer = fig.add_axes([0.36, 0.02, 0.05, 0.05])
        ax_button_show_motion = fig.add_axes([0.42, 0.02, 0.05, 0.05])
        
        slider_y_offset = Slider(ax_slider_y_offset, 'y_offset', -1, 1, valinit=0, valstep=0.01)
        slider_scale = Slider(ax_slider_scale, 'scale', -1, 1, valinit=0, valstep=0.01)
        
        button_show_aux = Button(ax_button_show_aux, 'aux')
        button_show_pointer = Button(ax_button_show_pointer, 'pointer')
        button_show_motion = Button(ax_button_show_motion, 'motion')
    
    
        slider_y_offset.on_changed(self.on_slider_y_offset)
        slider_scale.on_changed(self.on_slider_scale)
        
        button_show_aux.on_clicked(self.on_button_show_aux)
        button_show_pointer.on_clicked(self.on_button_show_pointer)
        button_show_motion.on_clicked(self.on_button_show_motion)
    
        self.slider_y_offset = slider_y_offset
        self.slider_scale = slider_scale
        self.button_show_hand = button_show_aux
        self.button_show_pointer = button_show_pointer
        self.button_show_motion = button_show_motion


    
    
    def update_all(self):
        
        #print('update all')

        self.update_top_text()
        
        self.update_trajectory()
        self.update_signal_plot()
        self.update_signal_plot_text()

        self.fig.canvas.draw()
        
        pass
    
    def update_top_text(self):
        
        device_str = str(self.device)
        self.ax_text_device.clear()
        self.ax_text_device.text(0.5, 0.5, device_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_text_device.axis('off')

        dataset_str = str(self.dataset)
        self.ax_text_dataset.clear()
        self.ax_text_dataset.text(0.5, 0.5, dataset_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_text_dataset.axis('off')

        signal_id_str = '%d / %d' % (self.signal_id, self.nr_signals - 1)
        self.ax_text_index.clear()
        self.ax_text_index.text(0.5, 0.5, signal_id_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_text_index.axis('off')

        rep_str = '%d / %d' % (self.rep, self.nr_repetitions - 1)
        self.ax_text_rep.clear()
        self.ax_text_rep.text(0.5, 0.5, rep_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_text_rep.axis('off')
        
        
        pass
    
    def plot_aux(self, ax, signal, i):
        '''
        Internal hand geometry plot function use for both visualization and 
        animation.
        '''
    
        if isinstance(signal, FMSignalLeap): 
    
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
        
            ax.plot(x1, y1, z1, color='b', markersize=0.1)

    def update_trajectory(self):
        
        signal = self.signal
        ax = self.ax_trajectory
        
        if isinstance(signal, FMSignalLeap):
            
            data = signal.data
            
        elif isinstance(signal, FMSignalGlove):
        
            data = signal.trajectory
        
        l = signal.len

        p = int(self.pointer * l)
        if p >= l:
            p = l - 1
        s = int(self.pointer_front * l)
        e = int(self.pointer_rear * l)
        
        ax.clear()
        ax.plot(data[s:e, 0], data[s:e, 1], data[s:e, 2],
                 color='k', markersize=0.1)

        if self.show_pointer:
            
            ax.scatter(data[p, 0], data[p, 1], data[p, 2],
                 color='r', s=15)

        
        for rs in self.remembered_signals:
            
            data = rs.trajectory
            l = signal.len
            ax.plot(data[s:e, 0], data[s:e, 1], data[s:e, 2],
                     markersize=0.1)
            
        
        
        if self.show_aux:
            
            
            
            self.plot_aux(ax, signal, p)
        
        

        lim = self.trajectory_lim * math.exp(self.trajectory_scale) 
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        
        

    def update_signal_plot(self):

        signal = self.signal
        ax_signals = self.ax_signals
        
        data = signal.data
        l = signal.len

        xs = np.arange(0, l)
        
        p = int(self.pointer * l)
        s = int(self.pointer_front * l)
        e = int(self.pointer_rear * l)

        # calculate the mean and range of each column for adjust the
        # scale and offset of the plot.
        means = np.zeros(3)
        ranges = np.zeros(3)
        
        for i in range(3):
            
            col = self.col_group * 3 + i
            
            means[i] = np.mean(data[0:l, col])
            mins = np.min(data[0:l, col])
            maxs = np.max(data[0:l, col])
            ranges[i] = math.fabs(maxs - mins)
        
        y_range = np.max(ranges) * 1.5

        if y_range < 1:
            y_range = 1
        
        for i in range(3):
            
            col = self.col_group * 3 + i
            ax = ax_signals[i]
            ax.clear()
            ax.plot(xs[s:e], data[s:e, col], color='k')
            
        for rs in self.remembered_signals:
            
            for i in range(3):
                
                col = self.col_group * 3 + i
                ax = ax_signals[i]
                ax.plot(rs.data[:, col])

        for i in range(3):

            ax = ax_signals[i]
            ax.axvline(x = p, color='r')
            
        for i in range(3):
            
            ax = ax_signals[i]
            
            ax.set_xlim(0, l)
            ax.set_ylim(means[i] - y_range / 2, means[i] + y_range / 2)

            major_ticks = np.arange(0, l, 20)
            ax.set_xticks(major_ticks)
            ax.grid()
        
        
        pass

    def update_signal_plot_text(self):
        
        
        axes_str = str(self.col_texts[self.col_group])
        self.ax_text_axes.clear()
        self.ax_text_axes.text(0.5, 0.5, axes_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_text_axes.axis('off')
        
        
        pass


    def on_button_axes_prev(self, val):
        
        self.col_group = (self.col_group - 1) % 6
        
        self.update_signal_plot()
        self.update_signal_plot_text()
        
        self.fig.canvas.draw()
        
        pass

    def on_button_axes_next(self, val):
        
        self.col_group = (self.col_group + 1) % 6
        
        self.update_signal_plot()
        self.update_signal_plot_text()
        
        self.fig.canvas.draw()
        


    def on_button_remember(self, val):
        
        if self.signal not in self.remembered_signals:
            
            print('remember')

            self.remembered_signals.append(self.signal)
        
        self.update_all()

    def on_button_forget(self, val):
        
        self.remembered_signals = []
        
        self.update_all()




    def on_slider_pointer_change(self, val):
        
        self.pointer = self.slider_pointer.val
        
        self.update_all()

    def on_slider_front_change(self, val):
        
        self.pointer_front = self.slider_front.val
        
        self.update_all()

    def on_slider_rear_change(self, val):
        
        self.pointer_rear = self.slider_rear.val
        
        self.update_all()


    def modify_current_signal(self):


        new_offset = -(self.slider_y_offset.val * 0.01 * self.trajectory_lim)
        
        self.signal.offset(1, new_offset - self.y_offset)
        self.y_offset = new_offset

        lim = self.trajectory_lim * math.exp(self.trajectory_scale)
        
        new_x = self.slider_x.val * lim
        delta_x = new_x - self.origin_x
        self.origin_x = new_x

        new_y = self.slider_y.val * lim
        delta_y = new_y - self.origin_y
        self.origin_y = new_y

        new_z = self.slider_z.val * lim
        delta_z = new_z - self.origin_z
        self.origin_z = new_z

        new_roll = self.slider_roll.val / 180.0 * math.pi
        delta_roll = self.origin_roll - new_roll
        self.origin_roll = new_roll

        new_pitch = self.slider_pitch.val / 180.0 * math.pi
        delta_pitch = self.origin_pitch - new_pitch
        self.origin_pitch = new_pitch

        new_yaw = self.slider_yaw.val / 180.0 * math.pi
        delta_yaw = self.origin_yaw - new_yaw
        self.origin_yaw = new_yaw
        
        R = euler_zyx_to_rotation_matrix(delta_yaw, delta_pitch, delta_roll)

        self.signal.rotate(R)
        
        self.signal.translate(-delta_x, -delta_y, -delta_z)




    def on_slider_x_change(self, val):
        
        self.modify_current_signal()
        
        self.update_all()

    def on_slider_y_change(self, val):
        
        self.modify_current_signal()
        
        self.update_all()

    def on_slider_z_change(self, val):
        
        self.modify_current_signal()

        self.update_all()


    def on_slider_roll(self, val):
        
        self.modify_current_signal()
        
        self.update_all()
        

    def on_slider_pitch(self, val):
        
        self.modify_current_signal()
        
        self.update_all()

    def on_slider_yaw(self, val):
        
        self.modify_current_signal()
        
        self.update_all()

    def on_slider_y_offset(self, val):
        
        self.modify_current_signal()
        
        self.update_all()

    def on_slider_scale(self, val):
        
        self.trajectory_scale = self.slider_scale.val
        
        self.update_all()


    def on_button_show_aux(self, val):
        
        self.show_aux = not self.show_aux
        
        self.update_all()

    def on_button_show_pointer(self, val):
        
        self.show_pointer = not self.show_pointer
        
        self.update_all()

    def on_button_show_motion(self, val):

        if not self.show_motion:

            self.show_motion = True
        
            self.pointer = self.pointer_front
        
            self.timer.start()
            
        else:
            
            self.show_motion = False
            
            self.timer.stop()
        

    def on_timer_expire(self):
        
        if not self.show_motion:
            
            self.timer.stop()
            
        else:
        
            if self.pointer < self.pointer_rear - 0.01:
                
                self.pointer += 0.01
            
            else:
                
                self.pointer = self.pointer_front
                self.show_motion = False
                self.timer.stop()
        
            #self.update_trajectory()
            self.update_signal_plot()
            self.fig.canvas.draw()
        
        pass

    def on_button_align(self, val):
        
        signals_aligned = []
        
        for rs in self.remembered_signals:
            
            signals_aligned.append(rs.align_to(self.signal))
        
        self.remembered_signals = signals_aligned
        
        self.update_all()



    def on_button_device_prev(self, val):
        
        self.device_id = (self.device_id - 1) % self.nr_devices
        
        self.device = self.devices[self.device_id]
        
        self.load_dataset_meta()
        self.load_signal_meta()
        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()
        
        self.update_all()

    def on_button_device_next(self, val):
        
        self.device_id = (self.device_id + 1) % self.nr_devices
        
        self.device = self.devices[self.device_id]
        
        self.load_dataset_meta()
        self.load_signal_meta()
        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()
        
        self.update_all()

    def on_button_dataset_prev(self, val):
        
        self.dataset_id = (self.dataset_id - 1) % self.nr_datasets
        
        self.dataset = self.datasets[self.dataset_id]

        self.load_dataset_meta()
        self.load_signal_meta()
        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()
        
        self.update_all()

    def on_button_dataset_next(self, val):
        
        self.dataset_id = (self.dataset_id + 1) % self.nr_datasets
        
        self.dataset = self.datasets[self.dataset_id]

        self.load_dataset_meta()
        self.load_signal_meta()
        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()
        
        self.update_all()

    def on_button_index_prev(self, val):
        
        self.signal_id = (self.signal_id - 1) % self.nr_signals

        self.load_signal_meta()
        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()
        
        self.update_all()

    def on_button_index_next(self, val):
        
        self.signal_id = (self.signal_id + 1) % self.nr_signals

        self.load_signal_meta()
        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()

        self.update_all()
        

    def on_button_rep_prev(self, val):
        
        self.rep = (self.rep - 1) % self.nr_repetitions

        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()
        
        self.update_all()

    def on_button_rep_next(self, val):
        
        self.rep = (self.rep + 1) % self.nr_repetitions

        self.load_current_signal()
        
        self.reset_panel_states()
        self.modify_current_signal()
        
        self.update_all()










    def on_key_press(self, event):
    
        # TODO: Add a few keys to navigate the dataset.

        #print('press', event.key)

        
        if event.key == ' ':
            
            pass
        

    




    def on_close(self, evt):
        
        
        pass




    def run(self):
        
        #print('running')
        
        plt.show()
        
    
    
    pass





if __name__ == '__main__':

    browser = FMBrowser()
    browser.run()




















    
