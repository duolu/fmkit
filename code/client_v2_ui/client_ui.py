'''
Created on Nov 17, 2019

@author: duolu


TODO:

    * design the UI, buttons, plot, etc.
    * read and show the word list, move forward, backward, etc.
    * process the word list in chs and eng, remove redundant words, store it in a file






'''

from __future__ import print_function


import codecs # for display non-ascii characters

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

import threading
import time

import sys
sys.path.insert(0, "./lib")
import Leap


import serial

class ClientLeap(object):

    def __init__(self):
        
        N = 2000
        self.N = N
        
        self.ltss = np.zeros((N, 1), np.float64)
        self.tss = np.zeros((N, 1), np.float64)
        self.tip_co = np.zeros((N, 6), np.float32)
        self.hand_co = np.zeros((N, 9), np.float32)
        self.joint_series = np.zeros((N, 5, 5, 3), np.float32)
        self.bone_geo = np.zeros((N, 5, 4, 2), np.float32)
        self.confs = np.zeros((N, 1), np.float32)
        self.valids = np.zeros((N, 1), np.uint32)

        self.t2d = np.zeros((N, 2), np.float32)

        self.has_data = False

        self.stop_flag = False
        self.client_stop = False
        
        self.sem = threading.Semaphore(0)
        
        self.fn = './sample.csv'

        self.controller = Leap.Controller()
        #device_list = self.controller.devices
        
        #assert(len(device_list) > 0)

    def capture_start(self, fn):
        
        self.fn = fn
        self.stop_flag = False
        
        self.sem.release()
    
    def capture_stop(self):
        
        self.stop_flag = True
    
    def close(self):
        
        self.client_stop = True
        self.sem.release()
    
    def capture(self):
        
        N = self.N
        controller = self.controller

        # zeroize the arraies
        self.ltss = np.zeros((N, 1), np.float64)
        self.tss = np.zeros((N, 1), np.float64)
        self.tip_co = np.zeros((N, 6), np.float32)
        self.hand_co = np.zeros((N, 9), np.float32)
        self.joint_series = np.zeros((N, 5, 5, 3), np.float32)
        self.bone_geo = np.zeros((N, 5, 4, 2), np.float32)
        self.confs = np.zeros((N, 1), np.float32)
        self.valids = np.zeros((N, 1), np.uint32)
        
        out_of_range = 0
        
        self.has_data = False
        
        init_frame = controller.frame()
        frame_id = init_frame.id
        
        
        # wait until the first frame is ready
        while True:
        
            init_frame = controller.frame()
            if frame_id == init_frame.id:
                continue
            else:
                frame_id = init_frame.id
                break
        
        frame = init_frame
    
        # actual length of the finger motion data
        i = -1
        self.l = 0

        while not self.stop_flag:
        
            # Retrieve a frame
            frame = controller.frame()
        
            if frame_id == frame.id:
                continue
            else:
                frame_id = frame.id
            
            if i >= N:
                self.l = N
                break;
            else:
                i += 1
                self.l += 1

            #frame_str = "Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
            #    frame.id - init_frame.id, (frame.timestamp - init_frame.timestamp) / 1000, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures()))
            #print(frame_str)
        
            self.tss[i] = frame.timestamp
            self.valids[i] = 1
        
            # Get hands
            if not frame.hands:
        
                out_of_range += 1
                self.valids[i] = 0
                continue
        
            self.has_data = True
        
            hand = frame.hands[0]
        
            # Get estimation confidence
            self.confs[i] = hand.confidence
        
            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction
        
            hand_pos = (hand.palm_position.x, hand.palm_position.y, hand.palm_position.z,
                direction.pitch, direction.roll, direction.yaw,
                normal.pitch, normal.roll, normal.yaw)
        
            for j in range(9):
        
                self.hand_co[i][j] = hand_pos[j]
        
            #print("\tpitch: %f degrees, roll: %f degrees, yaw: %f degrees" % (direction.pitch * Leap.RAD_TO_DEG, normal.roll * Leap.RAD_TO_DEG, direction.yaw * Leap.RAD_TO_DEG))
        
            # Get index finger tip
            ifinger = hand.fingers[1]
            tbone = ifinger.bone(3)
            tip_end = tbone.next_joint
            tip_start = tbone.prev_joint
            tip_pos = (tip_end.x, tip_end.y, tip_end.z,
                tip_end.x - tip_start.x, tip_end.y - tip_start.y, tip_end.z - tip_start.z)
        
            for j in range(6):
        
                self.tip_co[i, j] = tip_pos[j]
        
            # Get fingers
            for j in range(5):
        
                finger = hand.fingers[j]
        
                #print("\t\t%s finger, id: %d, length: %fmm, width: %fmm" % (finger_names[finger.type], finger.id, finger.length, finger.width))
        
                # Get bones and joints
        
                bone = finger.bone(0)
                bone_start = bone.prev_joint.to_tuple()
        
                for v in range(3):
                    self.joint_series[i][j][0][v] = bone_start[v]
        
                for k in range(4):
                    bone = finger.bone(k)
                    bone_pos = bone.next_joint.to_tuple()
                    #print("\t\t\tBone: %s, start: %s, end: %s, direction: %s" % (bone_names[bone.type], bone.prev_joint, bone.next_joint, bone.direction))
                    for v in range(3):
                        self.joint_series[i][j][k + 1][v] = bone_pos[v]
                    bone_length = bone.length
                    bone_width = bone.width
                    self.bone_geo[i][j][k][0] = bone_length
                    self.bone_geo[i][j][k][1] = bone_width
    
        self.stop_flag = True
        self.l = self.l - 1
        print('capture stopped')
        print("# of frames: %d, last ts: %d, out of range: %d" % \
              (self.l, self.tss[self.l - 1], out_of_range))
    
    def project(self):
        
        self.t2d = np.zeros((self.N, 2))
        
        if not self.has_data:
            
            return
        
        t3d = self.tip_co[:, 0:3].copy()
        t3d[:, 0] = self.tip_co[:, 2]
        t3d[:, 1] = self.tip_co[:, 0]
        t3d[:, 2] = self.tip_co[:, 1]
        
        v3d = self.tip_co[:, 3:6].copy()
        v3d[:, 0] = self.tip_co[:, 5]
        v3d[:, 1] = self.tip_co[:, 3]
        v3d[:, 2] = self.tip_co[:, 4]
        
        v = np.mean(v3d, axis=0)
        nv = np.linalg.norm(v)
        v /= nv
        
        up = np.asarray((0, 0, 1), dtype=np.float32)
        
        x_axis = -v
        y_axis = np.cross(up, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = x_axis.reshape((1, 3))
        y_axis = y_axis.reshape((1, 3))
        z_axis = z_axis.reshape((1, 3))
        
        R = np.concatenate((x_axis, y_axis, z_axis))
        
        pt3d = np.matmul(R, t3d.transpose())
        
        pt3d = pt3d.transpose()
        
        #print(x_axis, y_axis, z_axis)
        
        self.t2d = pt3d[:, 1:3]
        
    
    
    def check_sanity(self):
        
        
        
        
        return True
    
    
    def save_to_file(self, fn):
        
        fd = open(fn, 'w')
        for i in range(0, self.l):
        
            tip = tuple(self.tip_co[i])
            hand = tuple(self.hand_co[i])
            confidence = self.confs[i]
            valid = self.valids[i]
            ts = self.tss[i]
        
            # tip contains three positions and three orientations of the finger tip
            tip_str = "%8.04f, %8.04f, %8.04f, %8.04f, %8.04f, %8.04f" % tip
            
            # hand contains three positions, three hand directions, and three normal vector of the center of the hand
            hand_str = "%8.04f, %8.04f, %8.04f, %8.04f, %8.04f, %8.04f, %8.04f, %8.04f, %8.04f" % hand
        
            # one hand contains five fingers, each with positions of five joints, in total 25 3D vectors
            finger_strs = []
            bgeo_strs = []
        
            for j in range(5):
        
                for k in range(5):
        
                    joint = tuple(self.joint_series[i][j][k])
        
                    joint_str = "%8.04f, %8.04f, %8.04f" % joint
                    finger_strs.append(joint_str)
        
            for j in range(5):
        
                for k in range(4):
        
                    bgeo = tuple(self.bone_geo[i][j][k])
        
                    bgeo_str = "%8.04f, %8.04f" % bgeo
                    bgeo_strs.append(bgeo_str)
        
        
            fd.write('%d' % ts)
            fd.write(',\t')
            fd.write(tip_str)
            fd.write(',\t\t')
            fd.write(hand_str)
            fd.write(',\t\t')
        
            for joint_str in finger_strs:
        
                fd.write(joint_str)
                fd.write(',\t')
        
            for bgeo_str in bgeo_strs:
        
                fd.write(bgeo_str)
                fd.write(',\t')
        
            fd.write("%8.04f,\t" % confidence)
            fd.write("%d" % valid)
        
            fd.write("\n")
        
        fd.flush()
        fd.close()    
    
    def run(self):
        
        print('client_leap started.')
        
        while not self.client_stop:
            
            self.sem.acquire()
            
            if self.client_stop:
                
                break
            
            print('run once' + self.fn)
        
            self.capture()
             
            self.save_to_file(self.fn)
        
        
        
        print('client_leap closed.')


    pass


class ClientGlove(object):
    
    def __init__(self):
        
        self.stop_flag = False
        self.client_stop = False
        
        self.sem = threading.Semaphore(0)
        
        self.fn = './sample.csv'
        
        self.ser = serial.Serial('/dev/ttyUSB0', 115200)
        
        self.data = np.zeros((2000, 33), np.float32)
        self.l = 0
        
    def capture_start(self, fn):
        
        self.fn = fn
        self.stop_flag = False
        
        self.sem.release()
    
    def capture_stop(self):
        
        self.stop_flag = True
    
    def close(self):
        
        self.client_stop = True
        self.sem.release()
        

    def recv_payload(self, payload_len, ser):
    
        #print(payload_len)
    
        payload = ser.read(payload_len)
    
        #print(payload)
    
        ts = np.frombuffer(payload[0:4], dtype=np.int32)[0]
    
        sample = np.frombuffer(payload[4:payload_len], dtype=np.float32)
    
#         print(ts)
#         print("  ", end="")
#         for j in range(16):
#      
#             print("%6.2f, " % sample[j], end="")
#         print()
#         print("  ", end="")
#         for j in range(16, 32):
#      
#             print("%6.2f, " % sample[j], end="")
#         print()
    
        return (ts, sample)
    
    def capture(self):
        
        N = 2000
        
        # 0 ---> initial state, expecting magic1 (character 'D', decimal 68, hex 0x44)
        # 1 ---> after magic1 is received, expecting magic2 (character 'L', decimal 76, hex 0x4C)
        # 2 ---> after magic2 is received, expecting a length (decimal 132, hex 0x84)
        # 3 ---> after length is received, expecting an opcode (decimal 133, hex 0x85)
        
        state = 0
        
        self.data = np.zeros((N, 33), np.float32)
        
        i = 0
        msg_len = 0
        
        self.ser.reset_input_buffer()
        
        while not self.stop_flag:
        
            s = self.ser.read(1)
        
            # cast the received byte to an integer    
            c = s[0]
        
            #print(c)
            
        
            if state == 0:
        
                if c == 'D': # ASCII 68 is 'D'
                    state = 1
                else:
                    state = 0
        
            elif state == 1:
        
                if c == 'L': # ASCII 76 is 'L'
                    state = 2
                else:
                    state = 0
        
            elif state == 2:
        
                msg_len = c
                state = 3
        
        
            elif state == 3:
        
                #if c == 133:
        
                    ts, sample = self.recv_payload(132, self.ser)
                    self.data[i, 0] = ts
                    self.data[i, 1:] = sample
        
                    i += 1
                    self.l = i
                    
                    if i >= N:
                        break
        
                    state = 0
        
            else:
        
                print("Unknown state! Program is corrupted!")
    
        
        print(self.l)
        
    def save_to_file(self, fn):        
        
        np.savetxt(fn, self.data, fmt='%.8f', delimiter=', ')
        


    def run(self):
        
        print('client_glove started.')
        
        while not self.client_stop:
            
            self.sem.acquire()
            
            if self.client_stop:
                
                break
            
            print('run once' + self.fn)
        
            self.capture()
             
            self.save_to_file(self.fn)
        
        print('client_glove closed.')

        self.ser.close()


    pass




class ClientUI(object):
    
    def __init__(self, c_leap, c_glove):
    
        self.c_leap = c_leap
        self.c_glove = c_glove
   
        self.load_meta_files()
    
        self.nr_clients = 10
        self.client_id = 0
    
        self.lan_list = ['English', 'Chinese']
        self.lan_index = 0
        
        self.lan_dict = {'English' : self.words_eng,
                         'Chinese' : self.words_chs}

        self.group_list = ['group %d' % x for x in range(100)]
        self.group_index = 0
        
        self.nr_groups = 100
        self.group_size = 100

        self.word_list = self.words_eng
        self.word_index = 0

        self.started = False
        
        self.info_str = 'Stopped'
        
        self.warning_str = ''

        pass
    
    def load_meta_files(self):
        
        words_eng = []
        words_chs = []
        
        with open('./meta/en_10k_random.txt', 'r') as fp_eng:
            
            word = fp_eng.readline().strip()
            words_eng.append(word)
            while word:
                word = fp_eng.readline().strip()
                words_eng.append(word)
                

        with codecs.open('./meta/cn_10k_random.txt', encoding='utf-8') as fp_chs:
            
            word = fp_chs.readline().strip()
            words_chs.append(word)
            while word:
                word = fp_chs.readline().strip()
                words_chs.append(word)

        self.words_eng = words_eng
        self.words_chs = words_chs
    
    def on_close(self, evt):
        
        if self.c_leap is not None:
            
            self.c_leap.close()
            
        if self.c_glove is not None:
            
            self.c_glove.close()
        
        pass

    def on_prev_client(self, val):
    
        self.client_id = (self.client_id - 1) % self.nr_clients
        
        self.update_text()

    def on_next_client(self, val):
    
        self.client_id = (self.client_id + 1) % self.nr_clients
        
        self.update_text()

    
    def on_prev_lan(self, val):
    
        if self.lan_index > 0:
            
            self.lan_index -= 1
            
            lan_str = self.lan_list[self.lan_index]
            self.word_list = self.lan_dict[lan_str]
            
            self.update_text()
            

    def on_next_lan(self, val):
    
        if self.lan_index < len(self.lan_list) - 1:
            
            self.lan_index += 1

            lan_str = self.lan_list[self.lan_index]
            self.word_list = self.lan_dict[lan_str]
            
            self.update_text()


    def on_prev_group(self, val):
    
        self.group_index = (self.group_index - 1) % self.nr_groups
        
        self.word_index = 0
        
        self.update_text()

    def on_next_group(self, val):
    
        self.group_index = (self.group_index + 1) % self.nr_groups
        
        self.word_index = 0
        
        self.update_text()

    def on_prev_word(self, val):
        
        self.warning_str = ''
    
        if self.word_index > 0:
    
            self.word_index = self.word_index - 1
            
        else:
            self.warning_str = 'This is the first word.'
        
        self.update_text()

    def on_next_word(self, val):
    
        self.warning_str = ''
    
        if self.word_index < self.group_size:
    
            self.word_index = self.word_index + 1
            
        else:
            self.warning_str = 'This is the last word.'
        
        self.update_text()

    def on_start_stop(self, val):
    
        self.warning_str = ''
    
        if self.started:
            
            self.started = False
            self.info_str = 'Stopped'
            
            self.update_text()
            
            if self.c_leap is not None:
                
                self.c_leap.capture_stop()
            
            if self.c_glove is not None:
            
                self.c_glove.capture_stop()
            
            self.update_trajectory()
            
        else:
            
            self.started = True
            self.info_str = 'Started'
            
            self.update_text()
            
            ii = self.group_index * self.group_size + self.word_index
            
            lan_str = self.lan_list[self.lan_index]
            fn_leap = './data_leap/%s/client%d_word%d' % \
                (lan_str, self.client_id, ii)
            fn_glove = './data_glove/%s/client%d_word%d' % \
                (lan_str, self.client_id, ii)
            
            if self.c_leap is not None:
                
                self.c_leap.capture_start(fn_leap)
            
            if self.c_glove is not None:
            
                self.c_glove.capture_start(fn_glove)

        
    
    def on_key_press(self, event):
    
        #print('press', event.key)
        
        if event.key == ' ':
            
            self.on_start_stop(None)
        
        if not self.started:
            
            if event.key == 'a':
                
                self.on_prev_word(None)
                
            elif event.key == 'd':
                
                self.on_next_word(None)
    
            elif event.key == 'q':
                
                self.on_prev_group(None)
                
            elif event.key == 'e':
                
                self.on_next_group(None)
    
    
    def update_text(self):
        
        client_str = 'client %d' % self.client_id
        self.ax_client_t.clear()
        self.ax_client_t.text(0.5, 0.5, client_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_client_t.axis('off')
        
        lan_str = self.lan_list[self.lan_index]
        self.ax_lan_t.clear()
        self.ax_lan_t.text(0.5, 0.5, lan_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_lan_t.axis('off')

        self.ax_group_t.clear()
        self.ax_group_t.text(0.5, 0.5, self.group_list[self.group_index],
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_group_t.axis('off')

        ii = self.group_index * self.group_size + self.word_index
        ii_str = 'word %d' % ii
        self.ax_word_t.clear()
        self.ax_word_t.text(0.5, 0.5, ii_str,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_word_t.axis('off')

        label_font = {
            'family': 'SimHei',
            'size': 20,
        }

        self.ax_label_t.clear()
        self.ax_label_t.text(0.5, 0.5, 
                      self.word_list[ii],
                      size=20,
                      fontdict=label_font,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_label_t.axis('off')

        if self.started:
            
            info_font = {
                'color': 'blue',
            }
            
        else:
            
            info_font = {
                'color': 'red',
            }


        
        info_str = self.info_str
        self.ax_info_t.clear()
        self.ax_info_t.text(0.5, 0.5, info_str,
                      fontdict=info_font,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_info_t.axis('off')

        warning_font = {
            'color': 'red',
        }

        self.ax_warning_t.clear()
        self.ax_warning_t.text(0.5, 0.5, self.warning_str,
                      fontdict=warning_font,
                      horizontalalignment='center',
                      verticalalignment='center')
        self.ax_warning_t.axis('off')

        
        self.fig.canvas.draw_idle()
    
    def plot_hand(self):
        
        pass
    
    def update_trajectory(self):
        
        if self.c_leap is not None:
        
            self.c_leap.project()
            t2d = self.c_leap.t2d
            tip_co = self.c_leap.tip_co
            l = self.c_leap.l
            
            self.ax_trajectory_2d.clear()
            self.ax_trajectory_2d.plot(t2d[1:l-1, 0], t2d[1:l-1, 1])
            self.ax_trajectory_2d.set_xlim(-300, 300)
            self.ax_trajectory_2d.set_ylim(0, 600)
            
            self.ax_trajectory_3d.clear()
            self.ax_trajectory_3d.plot(tip_co[:l, 2], tip_co[:l, 0], tip_co[:l, 1])
            self.ax_trajectory_3d.set_xlim(-300, 300)
            self.ax_trajectory_3d.set_ylim(-300, 300)
            self.ax_trajectory_3d.set_zlim(0, 600)
        
        if self.c_glove is not None:
            
            print('update signal plot')
            
            data = self.c_glove.data
            l = self.c_glove.l
            
            for j in range(3):
                
                ax = self.ax2[j]
                ax.clear()
                #ax.plot(data[:l, 0], data[:l, j + 1])
                ax.plot(data[:l, j + 1])

            for j in range(3):
                
                ax = self.ax2[j + 3]
                ax.clear()
                #ax.plot(data[:l, 0], data[:l, j + 1])
                ax.plot(data[:l, j + 1 + 16])
                
            self.fig1.canvas.draw_idle()
            
            pass
        
        pass
    
    def setup_ui(self):
    
        fig = plt.figure()
        
        fig1 = plt.figure()

        fig.canvas.mpl_connect('close_event', self.on_close)

        fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        #axamp = fig.add_axes([0.25, .5, 0.50, 0.02])
        
        #samp = Slider(axamp, 'Amp', 0, 1, valinit=0)
        
        
        #samp.on_changed(self.update_slider)
        
        
        ax_info_t = fig.add_axes([0.02, 0.88, 0.3, 0.1])

        ax_client_b_p = fig.add_axes([0.02, .8, 0.05, 0.05])
        ax_client_b_n = fig.add_axes([0.22, .8, 0.05, 0.05])
        
        ax_lan_b_p = fig.add_axes([0.02, .7, 0.05, 0.05])
        ax_lan_b_n = fig.add_axes([0.22, .7, 0.05, 0.05])

        ax_group_b_p = fig.add_axes([0.02, .6, 0.05, 0.05])
        ax_group_b_n = fig.add_axes([0.22, .6, 0.05, 0.05])

        ax_word_b_p = fig.add_axes([0.02, .5, 0.05, 0.05])
        ax_word_b_n = fig.add_axes([0.22, .5, 0.05, 0.05])

        ax_client_t = fig.add_axes([0.10, .8, 0.09, 0.05])
        ax_lan_t = fig.add_axes([0.10, .7, 0.09, 0.05])
        ax_group_t = fig.add_axes([0.10, .6, 0.09, 0.05])
        ax_word_t = fig.add_axes([0.10, .5, 0.09, 0.05])

        ax_label_t = fig.add_axes([0.0, 0.3, 0.3, 0.2])

        ax_ss_b = fig.add_axes([0.05, 0.2, 0.19, 0.1])

        ax_warning_t = fig.add_axes([0.05, 0.1, 0.19, 0.1])
        
        
        ax_trajectory_2d = fig.add_axes([0.35, 0.1, 0.3, 0.8])
        
        ax_trajectory_3d = fig.add_axes([0.65, 0.1, 0.3, 0.8], 
            projection='3d')
        
        # widgets

        client_b_p = Button(ax_client_b_p, '<-')
        client_b_n = Button(ax_client_b_n, '->')
        
        lan_b_p = Button(ax_lan_b_p, '<-')
        lan_b_n = Button(ax_lan_b_n, '->')
    
        group_b_p = Button(ax_group_b_p, '<-')
        group_b_n = Button(ax_group_b_n, '->')
    
        word_b_p = Button(ax_word_b_p, '<-')
        word_b_n = Button(ax_word_b_n, '->')
        
        ss_b = Button(ax_ss_b, 'start/stop')

        client_b_p.on_clicked(self.on_prev_client)
        client_b_n.on_clicked(self.on_next_client)

        lan_b_p.on_clicked(self.on_prev_lan)
        lan_b_n.on_clicked(self.on_next_lan)

        group_b_p.on_clicked(self.on_prev_group)
        group_b_n.on_clicked(self.on_next_group)
    
        word_b_p.on_clicked(self.on_prev_word)
        word_b_n.on_clicked(self.on_next_word)
    
        ss_b.on_clicked(self.on_start_stop)
    
    
        # save the reference of the widgets
        
        self.fig = fig
        self.fig1 = fig1
        
        


        # axes

        self.ax2 = []
        for j in range(6):

            ax = fig1.add_subplot(6, 1, j + 1)
            self.ax2.append(ax)


        
        self.ax_info_t = ax_info_t

        self.ax_client_b_p = ax_client_b_p
        self.ax_client_b_n = ax_client_b_n

        self.ax_lan_b_p = ax_lan_b_p
        self.ax_lan_b_n = ax_lan_b_n

        self.ax_group_b_p = ax_group_b_p
        self.ax_group_b_n = ax_group_b_n

        self.ax_word_b_p = ax_word_b_p
        self.ax_word_b_n = ax_word_b_n

        self.ax_client_t = ax_client_t
        self.ax_lan_t = ax_lan_t
        self.ax_group_t = ax_group_t
        self.ax_word_t = ax_word_t
        
        self.ax_label_t = ax_label_t
        
        self.ax_ss_t = ax_ss_b
        
        self.ax_warning_t = ax_warning_t
        
        self.ax_trajectory_2d = ax_trajectory_2d
        self.ax_trajectory_3d = ax_trajectory_3d
        
        # bottons

        self.client_b_p = client_b_p
        self.client_b_n = client_b_n
        
        self.lan_b_p = lan_b_p
        self.lan_b_n = lan_b_n

        self.group_b_p = group_b_p
        self.group_b_n = group_b_n

        self.word_b_p = word_b_p
        self.word_b_n = word_b_n
    
        self.ss_b = ss_b
    
        # update widget states
    
    
        self.update_text()
        

    
    def run(self):
        
        
        plt.show()
        
    
    
    pass












if __name__ == '__main__':
    
    c_leap = ClientLeap()
     
    x_leap = threading.Thread(target=c_leap.run)
    x_leap.start()

    c_glove = ClientGlove()
    
    x_glove = threading.Thread(target=c_glove.run)
    x_glove.start()

    
    #ui = ClientUI(c_leap, None)
    #ui = ClientUI(None, c_glove)
    ui = ClientUI(c_leap, c_glove)
    
    ui.setup_ui()
    ui.run()
    
    x_leap.join(1)
    x_glove.join(1)
    
    pass







