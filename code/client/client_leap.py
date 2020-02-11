'''
Created on Feb 5, 2020

@author: duolu
'''

import numpy as np
import threading
import time


import sys
sys.path.insert(0, "./lib")
import Leap



class ClientLeap(object):

    def __init__(self):
        
        N = 5000
        self.N = N
        
        self.l = 0
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

    def capture_start(self, fn, fn_bk):
        
        self.fn = fn
        self.fn_bk = fn_bk
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
        
        self.out_of_range = 0
        self.different_hand = -1
        self.hand_id = -1
        self.hand_up = 0
        self.hand_left = 0
        self.max_z = 0
        self.min_z = 1000
        
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
            
            if i >= N - 1:
                self.l = N
                break;
            else:
                i += 1
                self.l += 1

            #frame_str = "Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
            #    frame.id - init_frame.id, (frame.timestamp - init_frame.timestamp) / 1000, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures()))
            #print(frame_str)
        
            self.ltss[i] = time.time()
            self.tss[i] = frame.timestamp
            self.valids[i] = 1
        
            # Get hands
            if not frame.hands:
        
                self.out_of_range += 1
                self.valids[i] = 0
                continue
        
        
            self.has_data = True
        
            hand = frame.hands[0]

            if self.different_hand < 0:
                
                self.different_hand = 0
                self.hand_id = hand.id
                
            else:
                
                if self.hand_id != hand.id:
                    
                    self.different_hand = 1
            
            if hand.palm_normal.y > 0:
                
                self.hand_up = 1
        
            if not hand.is_right:
                
                self.hand_left = 1
        
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
        
            if tip_pos[1] > self.max_z:
                
                self.max_z = tip_pos[1]
                
            if tip_pos[1] < self.min_z:
                
                self.min_z = tip_pos[1]
        
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
              (self.l, self.tss[self.l - 1], self.out_of_range))
    
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
        
        if self.out_of_range > 0:
            
            return False, 'Hand is out of range! Please rewrite!'
        
        if self.different_hand > 0:
            
            return False, 'Hand tracking inconsistent! Please rewrite!'
        
        if self.hand_up > 0:
            
            return False, 'Palm must face downward! Please rewrite!'

        if self.hand_left > 0:
            
            return False, 'Must use the right hand! Please rewrite!'
        
#         print(self.min_z, self.max_z)
#         if self.max_z - self.min_z < 100:
#             
#             return False, 'Too small! Please rewrite!'
        
        return True, ''
    
    
    def save_to_file(self, fn):
        
        fd = open(fn, 'w')
        for i in range(0, self.l):
        
            tip = tuple(self.tip_co[i])
            hand = tuple(self.hand_co[i])
            confidence = self.confs[i]
            valid = self.valids[i]
            ts = self.tss[i]
            lts = self.ltss[i]
        
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
        
            
            fd.write('%f' % lts)
            fd.write(',\t')
            fd.write('%f' % ts)
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
            self.save_to_file(self.fn_bk)
        
        
        
        print('client_leap closed.')


    pass



if __name__ == '__main__':
    pass