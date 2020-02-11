'''
Created on Feb 5, 2020

@author: duolu
'''

import numpy as np
import threading
import time

import serial


class ClientGlove(object):
    
    def __init__(self, version=3):
        
        self.stop_flag = False
        self.client_stop = False
        
        if version < 1 or version > 3:
            raise ValueError('Unknown glove version: %d' % version)

        self.version = version
        

        if self.version == 1 or self.version == 2:
            
            self.offset1 = 1
            self.offset2 = 1 + 12
            
            self.MAGIC_1 = 'A'
            self.MAGIC_2 = 'F'
            
        if self.version == 3:
            
            self.offset1 = 2
            self.offset2 = 2 + 16

            self.MAGIC_1 = 'D'
            self.MAGIC_2 = 'L'

        
        self.sem = threading.Semaphore(0)
        
        self.fn = './sample.csv'
        
        #self.ser = serial.Serial('/dev/ttyUSB0', 115200)
        self.ser = serial.Serial('/dev/ttyACM0', 115200)

        N = 5000
        self.N = N
        
        self.data = np.zeros((N, 33), np.float32)
        self.l = 0
    
    
    
    
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
        
        N = self.N
        
        # 0 ---> initial state, expecting magic1 (character 'D', decimal 68, hex 0x44)
        # 1 ---> after magic1 is received, expecting magic2 (character 'L', decimal 76, hex 0x4C)
        # 2 ---> after magic2 is received, expecting a length (decimal 132, hex 0x84)
        # 3 ---> after length is received, expecting an opcode (decimal 133, hex 0x85)
        
        state = 0
        
        if self.version == 1 or self.version == 2:
            
            self.data = np.zeros((N, 25), np.float64)
            
        elif self.version == 3:
            
            self.data = np.zeros((N, 34), np.float64)
            
        else:
            
            raise ValueError('Unknown glove version: %d' % self.version)
        
        i = 0
        msg_len = 0
        
        self.ser.reset_input_buffer()
        
        while not self.stop_flag:
        
            s = self.ser.read(1)
            
        
            # cast the received byte to an integer    
            c = s[0]
        
            #print(c)
            
        
            if state == 0:
        
                if c == self.MAGIC_1:
                    state = 1
                else:
                    state = 0
        
            elif state == 1:
        
                if c == self.MAGIC_2:
                    state = 2
                else:
                    state = 0
        
            elif state == 2:
        
                msg_len = c
                state = 3
        
        
            elif state == 3:
        
                #if c == 133:
        
                    if self.version == 1 or self.version == 2:
                        
                        ts, sample = self.recv_payload(100, self.ser)
                        
                        self.data[i, 0] = ts
                        self.data[i, 1:] = sample
        
                    if self.version == 3:
                        
                        ts, sample = self.recv_payload(132, self.ser)
                        self.data[i, 0] = time.time()
                        self.data[i, 1] = ts
                        self.data[i, 2:] = sample
        
                    i += 1
                    self.l = i
                    
                    if i >= N:
                        break
        
                    state = 0
        
            else:
        
                print("Unknown state! Program is corrupted!")
    
        self.data = self.data[0:self.l, :]

        print('glove capture OK, %d samples are obtained.' % self.l)
    
    
    def check_sanity(self):
        
            
        
        check1 = np.mean(np.absolute(self.data[:, self.offset1]))
        check2 = np.mean(np.absolute(self.data[:, self.offset2]))
        
        print(check1, check2)
        
        if check1 < 0.01 or check2 < 0.01:
            
            return False, 'IMU error!'

        
        return True, ''
    
    
    def save_to_file(self, fn):        
        
        np.savetxt(fn, self.data, fmt='%.8f', delimiter=', ')
        
        #self.data = np.zeros((N, 34), np.float64)

    def run(self):
        
        print('client_glove started.')
        
        while not self.client_stop:
            
            self.sem.acquire()
            
            if self.client_stop:
                
                break
            
            print('run once' + self.fn)
        
            self.capture()
             
            self.save_to_file(self.fn)
            self.save_to_file(self.fn_bk)
        
        print('client_glove closed.')

        self.ser.close()


    pass



if __name__ == '__main__':
    pass