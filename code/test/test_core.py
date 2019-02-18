'''
Created on Jul 12, 2018

@author: duolu
'''

from fmkit.core import *
from fmkit.vis import *

class TestCore(object):
    
    '''
    Common code for testing the fmkit.core module.
    '''

    def assemble_fn(self, folder, user_label, id_label, seq):
    
        fn_prefix = folder + '/'
        
        # postfix are appended by FMSignal.load_from_file according to the 
        # mode.
        fn_postfix = ''
        
        fn_body = user_label + '_' + id_label + '_' + ('%02d' % seq)
        
        fn = fn_prefix + fn_body + fn_postfix

        return fn
    
        
        

class TestCoreFMSignalLeap(TestCore):

    '''
    Testing code for finger motion signals obtained by the Leap Motion device.
    '''

    def test_load_from_file(self, folder='../data_temp/leap', 
            user_label='fmcode', id_label='fmcode', seq=1, 
            mode='raw', use_stat=True, use_handgeo=True, verbose=False):
    
        fn = self.assemble_fn(folder, user_label, id_label, seq)
        
        signal = FMSignalLeap.construct_from_file(fn, 
            mode=mode, use_stat=use_stat, use_handgeo=use_handgeo,
            user_label=user_label, id_label=id_label, seq=seq)
    
        if verbose:
            print('l = %d, d = %d' % (signal.l, signal.d))
            print(signal.data)
    
        return signal
    
    
    def test_load_from_multiple_files(self, folder='../data_temp/leap', 
            user_label='luduo', id_label='luduo', start_seq=1, end_seq=10, 
            mode='raw', use_stat=True, use_handgeo=True, verbose=False):
        
        signals = []
    
        for i in range(start_seq, end_seq + 1):
            
            fn = self.assemble_fn(folder, user_label, id_label, i)
            
            signal = FMSignalLeap()
            signal.load_from_file(fn, 
                mode=mode, use_stat=use_stat, use_handgeo=use_handgeo,
                user_label=user_label, id_label=id_label, seq=i)
    
            #signal.preprocess()
    
            signals.append(signal)
        
        
            if verbose:
                print('l = %d, d = %d' % (signal.l, signal.d))

        return signals

    def test_load_template_from_file(self, folder='../data_temp/leap', 
            user_label='fmcode', id_label='fmcode', seq=1, 
            mode='raw', use_stat=True, use_handgeo=True, verbose=False):
    
        fn = self.assemble_fn(folder, user_label, id_label, seq)
        
        signal = FMSignalTemplateLeap.construct_from_file(fn, 
            mode=mode, use_stat=use_stat, use_handgeo=use_handgeo,
            user_label=user_label, id_label=id_label, seq=seq)
    
        if verbose:
            print('l = %d, d = %d' % (signal.l, signal.d))
            print(signal.data)
    
        return signal
    
    def test_save_to_file(self, signal, folder='../data_temp/leap', 
            user_label='fmcode', id_label='fmcode', seq=1, 
            mode='csv', use_stat=True, use_handgeo=True):
        
        fn = self.assemble_fn(folder, user_label, id_label, seq)
        signal.save_to_file(fn, mode, use_stat, use_handgeo)

    def test_construct_template(self, folder='../data_temp/leap', 
            user_label='luduo', id_label='luduo', start_seq=1, end_seq=10, 
            mode='raw', use_stat=True, use_handgeo=True, verbose=False):
        
        signals = self.test_load_from_multiple_files(folder, 
            user_label, id_label, start_seq, end_seq, 
            mode, use_stat, use_handgeo, verbose)

        for signal in signals:
            
            signal.preprocess()
        
        template = FMSignalTemplateLeap.construct_from_signals(signals, 0, 
            use_stat, use_handgeo)
        
        return template


class TestCoreFMSignalGlove(TestCore):

    '''
    Testing code for finger motion signals obtained by the Leap Motion device.
    '''

    def test_load_from_file(self, folder='../data_temp/glove', 
            user_label='fmcode', id_label='fmcode', seq=1, 
            mode='raw', use_stat=True,
            verbose=False):
    
        fn = self.assemble_fn(folder, user_label, id_label, seq)
        
        signal = FMSignalGlove.construct_from_file(fn, 
            mode=mode, use_stat=use_stat,
            user_label=user_label, id_label=id_label, seq=seq)
    
        if verbose:
            print('l = %d, d = %d' % (signal.l, signal.d))
            print(signal.data)
    
        return signal
    
    
    def test_load_from_multiple_files(self, folder='../data_temp/glove', 
            user_label='luduo', id_label='luduo', start_seq=1, end_seq=10, 
            mode='raw', use_stat=True, verbose=False):
        
        signals = []
    
        for i in range(start_seq, end_seq + 1):
            
            fn = self.assemble_fn(folder, user_label, id_label, i)
            
            signal = FMSignalGlove()
            signal.load_from_file(fn, 
                mode=mode, use_stat=use_stat,
                user_label=user_label, id_label=id_label, seq=i)
    
            #signal.preprocess()
    
            signals.append(signal)
        
        
            if verbose:
                print('l = %d, d = %d' % (signal.l, signal.d))

        return signals

    def test_load_template_from_file(self, folder='../data_temp/glove', 
            user_label='fmcode', id_label='fmcode', seq=1, 
            mode='raw', use_stat=True, verbose=False):
    
        fn = self.assemble_fn(folder, user_label, id_label, seq)
        
        signal = FMSignalTemplateGlove.construct_from_file(fn, 
            mode=mode, use_stat=use_stat,
            user_label=user_label, id_label=id_label, seq=seq)
    
        if verbose:
            print('l = %d, d = %d' % (signal.l, signal.d))
            print(signal.data)
    
        return signal

    def test_save_to_file(self, signal, folder='../data_temp/glove', 
            user_label='fmcode', id_label='fmcode', seq=1, 
            mode='csv', use_stat=True):
        
        fn = self.assemble_fn(folder, user_label, id_label, seq)
        signal.save_to_file(fn, mode, use_stat)
    
    def test_construct_template(self, folder='../data_temp/glove', 
            user_label='luduo', id_label='luduo', start_seq=1, end_seq=10, 
            mode='raw', use_stat=True, verbose=False):
        
        signals = self.test_load_from_multiple_files(folder, 
            user_label, id_label, start_seq, end_seq, 
            mode, use_stat, verbose)
        
        for signal in signals:
            
            signal.preprocess()
        
        template = FMSignalTemplateGlove.construct_from_signals(signals, 0, 
            use_stat)
        
        return template


def test_core_leap_signal_load_and_save():
    
    tester = TestCoreFMSignalLeap()
    
    signal = tester.test_load_from_file(mode='raw')
    tester.test_save_to_file(signal, mode='csv', use_stat=False, use_handgeo=False)
    tester.test_save_to_file(signal, mode='binary', use_stat=False, use_handgeo=False)
    
    #print(signal.stat)
    #print(signal.handgeo)
    
    signal.preprocess()
    
    tester.test_save_to_file(signal, id_label='fmcode-processed', 
        mode='csv')
    tester.test_save_to_file(signal, id_label='fmcode-processed', 
        mode='binary')

    signal_pcsv = tester.test_load_from_file(id_label='fmcode-processed', 
        mode='csv')

    signal_pbinary = tester.test_load_from_file(id_label='fmcode-processed', 
        mode='binary')

    # CAUTION: There are tiny differences between the data in memory and text
    # representation in CSV file.
    print(np.all(signal.data - signal_pcsv.data < 0.00001))
    print(np.all(signal.data == signal_pbinary.data))
    
    print(np.all(signal.stat - signal_pcsv.stat < 0.00001))
    print(np.all(signal.stat == signal_pbinary.stat))

    print(np.all(signal.handgeo - signal_pcsv.handgeo < 0.00001))
    print(np.all(signal.handgeo == signal_pbinary.handgeo))

def test_core_leap_template_load_and_save():
    
    tester = TestCoreFMSignalLeap()
    
    template = tester.test_construct_template()
    
    tester.test_save_to_file(template, id_label='fmcode-template', 
        mode='csv')
    tester.test_save_to_file(template, id_label='fmcode-template', 
        mode='binary')

    template_pcsv = tester.test_load_template_from_file(id_label='fmcode-template', 
        mode='csv')

    template_pbinary = tester.test_load_template_from_file(id_label='fmcode-template', 
        mode='binary')

    # CAUTION: There are tiny differences between the data in memory and text
    # representation in CSV file.
    print(np.all(template.data - template_pcsv.data < 0.00001))
    print(np.all(template.data == template_pbinary.data))
    
    print(np.all(template.stat - template_pcsv.stat < 0.00001))
    print(np.all(template.stat == template_pbinary.stat))

    print(np.all(template.handgeo - template_pcsv.handgeo < 0.00001))
    print(np.all(template.handgeo == template_pbinary.handgeo))


def test_core_glove_signal_load_and_save():
    
    tester = TestCoreFMSignalGlove()
    
    signal = tester.test_load_from_file(mode='raw')
    tester.test_save_to_file(signal, mode='csv', use_stat=False)
    tester.test_save_to_file(signal, mode='binary', use_stat=False)
    
    signal.preprocess()
    
    tester.test_save_to_file(signal, id_label='fmcode-processed', mode='csv')
    tester.test_save_to_file(signal, id_label='fmcode-processed', mode='binary')


    signal_pcsv = tester.test_load_from_file(id_label='fmcode-processed', 
                                             mode='csv')

    signal_pbinary = tester.test_load_from_file(id_label='fmcode-processed', 
                                             mode='binary')

    # CAUTION: There are tiny differences between the data in memory and text
    # representation in CSV file.
    print(np.all(signal.data - signal_pcsv.data < 0.00001))
    print(np.all(signal.data == signal_pbinary.data))

    print(np.all(signal.stat - signal_pcsv.stat < 0.00001))
    print(np.all(signal.stat == signal_pbinary.stat))

    t_leap = TestCoreFMSignalLeap()
    
    template = t_leap.test_construct_template()
    
    t_leap.test_save_to_file(template, id_label='fmcode-template', 
        mode='csv')
    t_leap.test_save_to_file(template, id_label='fmcode-template', 
        mode='binary')

    template_pcsv = t_leap.test_load_template_from_file(id_label='fmcode-template', 
        mode='csv')

    template_pbinary = t_leap.test_load_template_from_file(id_label='fmcode-template', 
        mode='binary')

    # CAUTION: There are tiny differences between the data in memory and text
    # representation in CSV file.
    print(np.all(template.data - template_pcsv.data < 0.00001))
    print(np.all(template.data == template_pbinary.data))
    
    print(np.all(template.stat - template_pcsv.stat < 0.00001))
    print(np.all(template.stat == template_pbinary.stat))

    print(np.all(template.handgeo - template_pcsv.handgeo < 0.00001))
    print(np.all(template.handgeo == template_pbinary.handgeo))

def test_core_glove_template_load_and_save():
    
    tester = TestCoreFMSignalGlove()
    
    template = tester.test_construct_template()
    
    tester.test_save_to_file(template, id_label='fmcode-template', 
        mode='csv')
    tester.test_save_to_file(template, id_label='fmcode-template', 
        mode='binary')

    template_pcsv = tester.test_load_template_from_file(id_label='fmcode-template', 
        mode='csv')

    template_pbinary = tester.test_load_template_from_file(id_label='fmcode-template', 
        mode='binary')

    # CAUTION: There are tiny differences between the data in memory and text
    # representation in CSV file.
    print(np.all(template.data - template_pcsv.data < 0.00001))
    print(np.all(template.data == template_pbinary.data))
    
    print(np.all(template.stat - template_pcsv.stat < 0.00001))
    print(np.all(template.stat == template_pbinary.stat))






if __name__ == '__main__':
    
     
    #test_core_leap_signal_load_and_save()
    
    #test_core_leap_template_load_and_save()
    
    #test_core_glove_signal_load_and_save()
    
    test_core_glove_template_load_and_save()
    
    pass
































