'''
Created on Jul 12, 2018

@author: manifd
'''

from fmkit.core import *
from fmkit.vis import *
from test.test_core import *

def test_signal_vis():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')
    
    #signal.preprocess()
    
    signal_vis(signal, 9)

def test_signals_vis_same_plot():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal_1 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_2 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=2, mode='raw')
    
    
    signal_1.preprocess()
    signal_2.preprocess()
    
    signals_vis_same_plot([signal_1, signal_2], 9)

def test_signals_vis_same_plot_multiple_files():

    t_leap = TestCoreFMSignalLeap()
    
    signals = t_leap.test_load_from_multiple_files()

    for signal in signals:
        
        signal.preprocess()

    signals_vis_same_plot(signals, 9)



def test_signals_vis_side_by_side():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal_1 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_2 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=2, mode='raw')

    signal_1.preprocess()
    signal_2.preprocess()
    
    signals_vis_side_by_side([signal_1, signal_2], 3)

def test_signals_vis_comparing_two_batch():

    t_leap = TestCoreFMSignalLeap()
    
    signals_1 = t_leap.test_load_from_multiple_files(user_label='luduo', id_label='luduo')
    
    signals_2 = t_leap.test_load_from_multiple_files(user_label='luduo', id_label='jisuanji')

    
    for signal in signals_1:
        
        signal.preprocess()

    for signal in signals_2:
        
        signal.preprocess()


    signals_vis_comparing_two_batch(signals_1, signals_2)






def test_shape_vis_3d():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal.preprocess_shape()
    
    print(signal.l)
    
    #signal.trim_post(25, 375)
    
    shape_vis_3d(signal)


def test_shapes_vis_3d_same_plot():


    t_leap = TestCoreFMSignalLeap()
    
    signal_1 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_2 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=2, mode='raw')

    signal_1.preprocess_shape()
    signal_2.preprocess_shape()
    
    
    shapes_vis_3d_same_plot([signal_1, signal_2])




def test_shapes_vis_3d_same_plot_multiple_files():
    
    t_leap = TestCoreFMSignalLeap()
    
    signals = t_leap.test_load_from_multiple_files()

    for signal in signals:
        
        signal.preprocess_shape()


    shapes_vis_3d_same_plot(signals)


def test_trajectory_3d_animation():

    t_leap = TestCoreFMSignalLeap()
    
    signal = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal.preprocess_shape()
    
    trajectory_3d_animation(signal)


def test_handgeo_vis():

    t_leap = TestCoreFMSignalLeap()
    
    signal = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    handgeo_vis(signal)


def test_shape_handgeo_3d_animation():

    t_leap = TestCoreFMSignalLeap()
    
    signal = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    shape_handgeo_3d_animation(signal)




def test_warping_path_vis():
    
    
    t_leap = TestCoreFMSignalLeap()
    
    signals_1 = t_leap.test_load_from_multiple_files(user_label='luduo', id_label='luduo')
    
    signals_2 = t_leap.test_load_from_multiple_files(user_label='luduo', id_label='jisuanji')


    for signal in signals_1:
        
        signal.preprocess()

    for signal in signals_2:
        
        signal.preprocess()

        
    signal_1 = signals_1[0]
    signal_2 = signals_1[3]
    
    #signals_vis_side_by_side([signal_1, signal_2], 9)
    #signals_vis_same_plot([signal_1, signal_2], 9)
    
    print(signal_1.l, signal_1.d, signal_1.data.dtype, signal_1.data.shape)
    print(signal_2.l, signal_2.d, signal_2.data.dtype, signal_2.data.shape)
    
    tup = dtw_c(signal_1.data, signal_2.data, signal_1.l, signal_2.l)
    
    
    (dist, dists, dir, a1start, a1end, a2start, a2end, data_new) = tup

    print(dists)
    print(dists.shape)

    dists_matrix = dists[1:, 1:]
    

    warping_path_vis(dists_matrix, a1start, a1end)













# cross module test functions (mainly test the core and utils)



def test_signal_alignment():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal_1 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_2 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=2, mode='raw')

    signal_1.preprocess()
    signal_2.preprocess()

    template = signal_1
    signal_aligned = signal_2.align_to(template)

    signals_vis_same_plot([template, signal_aligned], 9)

    #shapes_vis_3d_same_plot([fmc0, fmc1])


def test_template():
    
    t_leap = TestCoreFMSignalLeap()
    
    signals = t_leap.test_load_from_multiple_files()

    for signal in signals:
        
        signal.preprocess()

    template = FMSignalTemplateLeap.construct_from_signals(signals[0:5], 0, False, False)
    
    fmcs_aligned = template.signals_aligned

    signals_vis_comparing_two_batch([template], fmcs_aligned, 0)




def test_amplitude_normalize():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal_1 = t_leap.test_load_from_file()

    signal_2 = t_leap.test_load_from_file()

    signal_2.fix_missing_points()
    signal_2.amplitude_normalize()

    signals_vis_side_by_side([signal_1, signal_2], 3)


def test_pertube():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal = t_leap.test_load_from_file()

    signal_pertube = t_leap.test_load_from_file()

    signal.preprocess()
    signal_pertube.preprocess()
    
    signal_pertube.pertube(100, 0, 20, 0.2, 3)
    
    signals_vis_same_plot([signal, signal_pertube], 3)
    shapes_vis_3d_same_plot([signal, signal_pertube])


def test_stretch():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal_1 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_2 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_1.preprocess()
    signal_2.preprocess()

    signal_2.stretch(250)
    #print(fmc.l)


    signals_vis_side_by_side([signal_1, signal_2], 3)
    #signals_vis_comparing_two_batch(fmcs1, fmcs2)

def test_stretch_seg():
    
    t_leap = TestCoreFMSignalLeap()
    
    signal = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_streth_1 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_streth_2 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal.preprocess()
    signal_streth_1.preprocess()
    signal_streth_2.preprocess()

    signal_streth_1.stretch_segment(50, 100, 30)
    signal_streth_2.stretch_segment(0, 50, 30)

    signals_vis_same_plot([signal, signal_streth_1, signal_streth_2], 3)


def test_swap_segment():
    
    
    t_leap = TestCoreFMSignalLeap()
    
    signal_1 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=1, mode='raw')

    signal_2 = t_leap.test_load_from_file(user_label='luduo', id_label='luduo', 
        seq=2, mode='raw')

    signal_1.preprocess()
    signal_2.preprocess()


    template = signal_1
    
    signal_aligned = signal_2.align_to(template)
    
    seg_start = int(template.l * 0)
    seg_end = int(template.l * 2/3)
    
    print(seg_start * 20, seg_end * 20)
    
    template_ss, signal_ss = template.swap_segment(signal_aligned, seg_start, seg_end, 5)
    
    #signals_vis_same_plot(signals_aligned, 3)
    signals_vis_same_plot([template, template_ss], 3)
    signals_vis_same_plot([signal_aligned, signal_ss], 3)
    
    signals_vis_side_by_side([template, template_ss, signal_aligned, signal_ss], 1)





if __name__ == '__main__':
    
    print('vis.py')
    #test_signal_vis()
    #test_signals_vis_same_plot()
    #test_signals_vis_same_plot_multiple_files()
    #test_signals_vis_side_by_side()
    #test_signals_vis_comparing_two_batch()
    
    
    #test_shape_vis_3d()
    #test_shapes_vis_3d_same_plot()
    #test_shapes_vis_3d_same_plot_multiple_files()
    
    
    #test_trajectory_3d_animation()
    #test_handgeo_vis()
    #test_shape_handgeo_3d_animation()
    
    
    
    #test_warping_path_vis()
    
    
    
    #test_signal_alignment()
    #test_template()
    
    #test_amplitude_normalize()
    #test_pertube()
    
    #test_stretch()
    test_stretch_seg()
    #test_swap_segment()
    

    
    pass



