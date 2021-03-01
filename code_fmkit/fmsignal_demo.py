'''
This module contains the demonstration code of the fmkit framework, which is 
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

import sys
import random

from fmsignal import *
from fmsignal_vis import *


# ---------------------------- signal input / output ---------------------------

def make_fn(folder, user, cid, seq):

    fn_folder = folder + '/'
    
    # postfix are appended by FMSignal.load_from_file according to the 
    # mode.
    
    fn_body = user + '_' + cid + '_' + ('%d' % seq)
    
    fn = fn_folder + fn_body

    return fn

def make_fns(folder, user, cid, sequences):

    fns = []

    for i in sequences:

        fn = make_fn(folder, user, cid, i)
        fns.append(fn)

    return fns

def load_one_signal(folder, user, cid, seq, mode):

    fn = make_fn(folder, user, cid, seq)
    signal = FMSignal.construct_from_file(fn, mode, user, cid, seq)

    return signal

def load_signals(folder, user, cid, sequences, mode):

    signals = []

    fns = make_fns(folder, user, cid, sequences)
    for seq, fn in zip(sequences, fns):

        signal = FMSignal.construct_from_file(fn, mode, user, cid, seq)
        signals.append(signal)
    
    return signals

def save_one_signal(signal, folder, mode):

    fn = make_fn(folder, signal.user, signal.cid, signal.seq)
    signal.save_to_file(fn, mode)

def save_signals(signals, folder, mode):

    for signal in signals:
        save_one_signal(signal, folder, mode)

def load_one_template(folder, user, cid, seq, mode):

    fn = make_fn(folder, user, cid, seq)
    template = FMSignalTemplate.construct_from_file(fn, mode, user, cid, seq)

    return template

def save_one_template(template, folder, mode):

    fn = make_fn(folder, template.user, template.cid, template.seq)
    template.save_to_file(fn, mode)

# ----------------------- raw signal input / output ---------------------------

def load_one_raw_signal(signal_class, folder, user, cid, seq, mode):

    fn = make_fn(folder, user, cid, seq)
    raw_signal = signal_class.construct_from_file(fn, mode, user, cid, seq)

    return raw_signal

def load_raw_signals(signal_class, folder, user, cid, sequences, mode):

    raw_signals = []

    fns = make_fns(folder, user, cid, sequences)
    for seq, fn in zip(sequences, fns):

        raw_signal = signal_class.construct_from_file(fn, mode, user, cid, seq)
        raw_signals.append(raw_signal)
    
    return raw_signals

def save_one_raw_signal(raw_signal, folder, mode):

    fn = make_fn(folder, raw_signal.user, raw_signal.cid, raw_signal.seq)
    raw_signal.save_to_file(fn, mode)

def save_raw_signals(raw_signals, folder, mode):

    for raw_signal in raw_signals:
        save_one_signal(raw_signal, folder, mode)


# ----------------------- helper function for demo -----------------------------

FOLDER = '../data_demo'
USER = "alice"
#USER = "bob"
CID = 'FMKit'
#CID = '123456'
PP_MODE_LOAD = 'npy'
PP_MODE_SAVE = 'npy'
RAW_MODE_LOAD = 'raw_internal'
RAW_MODE_SAVE = 'raw_npy'

LEAP_PP_SUBFOLDER = 'leap_pp'
GLOVE_PP_SUBFOLDER = 'glove_pp'
LEAP_TEMPLATE_SUBFOLDER = 'leap_template'
GLOVE_TEMPLATE_SUBFOLDER = 'glove_template'
LEAP_RAW_SUBFOLDER = 'leap_raw'
GLOVE_RAW_SUBFOLDER = 'glove_raw'

DEVICE = 'leap'
#DEVICE = 'glove'

def load_one_demo_signal_pp(device=DEVICE, folder=FOLDER, 
        user=USER, cid=CID, seq=0, mode=PP_MODE_LOAD):

    if device == 'leap':
        folder = folder + '/' + LEAP_PP_SUBFOLDER
    elif device == 'glove':
        folder = folder + '/' + GLOVE_PP_SUBFOLDER
    else:
        raise ValueError('Unknown device: %s!' % device)

    signal = load_one_signal(folder, user, cid, seq, mode)
    return signal

def load_demo_signals_pp(device=DEVICE, folder=FOLDER, 
        user=USER, cid=CID, sequences=range(0, 10), mode=PP_MODE_LOAD):

    if device == 'leap':
        folder = folder + '/' + LEAP_PP_SUBFOLDER
    elif device == 'glove':
        folder = folder + '/' + GLOVE_PP_SUBFOLDER
    else:
        raise ValueError('Unknown device: %s!' % device)

    signals = load_signals(folder, user, cid, sequences, mode)
    return signals

def save_one_demo_signal_pp(signal, device=DEVICE, folder=FOLDER, 
        mode=PP_MODE_SAVE):

    if device == 'leap':
        folder = folder + '/' + LEAP_PP_SUBFOLDER + '/temp'
    elif device == 'glove':
        folder = folder + '/' + GLOVE_PP_SUBFOLDER + '/temp'
    else:
        raise ValueError('Unknown device: %s!' % device)

    save_one_signal(signal, folder, mode)

def save_demo_signals_pp(signals, device=DEVICE, folder=FOLDER, 
        mode=PP_MODE_SAVE):

    if device == 'leap':
        folder = folder + '/' + LEAP_PP_SUBFOLDER + '/temp'
    elif device == 'glove':
        folder = folder + '/' + GLOVE_PP_SUBFOLDER + '/temp'
    else:
        raise ValueError('Unknown device: %s!' % device)

    save_signals(signals, folder, mode)


def load_one_demo_template(device=DEVICE, folder=FOLDER, 
        user=USER, cid=CID, seq=0, mode=PP_MODE_LOAD):

    if device == 'leap':
        folder = folder + '/' + LEAP_TEMPLATE_SUBFOLDER
    elif device == 'glove':
        folder = folder + '/' + GLOVE_TEMPLATE_SUBFOLDER
    else:
        raise ValueError('Unknown device: %s!' % device)

    template = load_one_template(folder, user, cid, seq, mode)
    return template

def save_one_demo_template(template, device=DEVICE, folder=FOLDER, 
        mode=PP_MODE_SAVE):

    if device == 'leap':
        folder = folder + '/' + LEAP_TEMPLATE_SUBFOLDER
    elif device == 'glove':
        folder = folder + '/' + GLOVE_TEMPLATE_SUBFOLDER
    else:
        raise ValueError('Unknown device: %s!' % device)

    save_one_template(template, folder, mode)


def load_one_demo_signal_raw(device=DEVICE, folder=FOLDER, 
        user=USER, cid=CID, seq=0, mode=RAW_MODE_LOAD):

    if device == 'leap':
        folder = folder + '/' + LEAP_RAW_SUBFOLDER
        signal = load_one_raw_signal(FMSignalLeap, folder, 
            user, cid, seq, mode)
    elif device == 'glove':
        folder = folder + '/' + GLOVE_RAW_SUBFOLDER
        signal = load_one_raw_signal(FMSignalGlove, folder, 
            user, cid, seq, mode)
    else:
        raise ValueError('Unknown device: %s!' % device)

    return signal

def load_demo_signals_raw(device=DEVICE, folder=FOLDER, 
        user=USER, cid=CID, sequences=range(0, 10), mode=RAW_MODE_LOAD):

    if device == 'leap':
        folder = folder + '/' + LEAP_RAW_SUBFOLDER
        signals = load_raw_signals(FMSignalLeap, folder, 
            user, cid, sequences, mode)
    elif device == 'glove':
        folder = folder + '/' + GLOVE_RAW_SUBFOLDER
        signals = load_raw_signals(FMSignalGlove, folder, 
            user, cid, sequences, mode)
    else:
        raise ValueError('Unknown device: %s!' % device)

    return signals

def save_one_demo_signal_pp(signal, device=DEVICE, folder=FOLDER, 
        mode=RAW_MODE_SAVE):

    if device == 'leap':
        folder = folder + '/' + LEAP_RAW_SUBFOLDER + '/temp'
    elif device == 'glove':
        folder = folder + '/' + GLOVE_RAW_SUBFOLDER + '/temp'
    else:
        raise ValueError('Unknown device: %s!' % device)

    save_one_raw_signal(signal, folder, mode)

def save_demo_signals_pp(signals, device=DEVICE, folder=FOLDER, 
        mode=RAW_MODE_SAVE):

    if device == 'leap':
        folder = folder + '/' + LEAP_RAW_SUBFOLDER + '/temp'
    elif device == 'glove':
        folder = folder + '/' + GLOVE_RAW_SUBFOLDER + '/temp'
    else:
        raise ValueError('Unknown device: %s!' % device)

    save_raw_signals(signals, folder, mode)




def fix_glove2(device, folder=FOLDER, 
               user=USER, cid=CID, seq_start=0, seq_end=10):

    assert device == 'glove'

    folder = folder + '/' + GLOVE_RAW_SUBFOLDER

    fns = make_fns(folder, user, cid, seq_start, seq_end)

    for fn in fns:

        fn += '.txt'
        print(fn)

        array = np.loadtxt(fn, dtype=np.float32, delimiter=", ")
        n = array.shape[0]
        col_start = 1
        for i in range(4):

            col_x = col_start + i * 3
            col_y = col_start + i * 3 + 1
            temp = np.zeros((n, 1), np.float32)
            # x <= -y', y <= x'
            temp[:, 0] = array[:, col_x]
            array[:, col_x] = -array[:, col_y]
            array[:, col_y] = temp[:, 0]

        np.savetxt(fn, array, fmt="%.6f", delimiter=", ")


# ----------------------- demo signal input / output ---------------------------

def demo_raw_signal_io(device=DEVICE, user=USER, cid=CID):

    print('Demo raw signal I/O')
    print('device="%s", user="%s", cid="%s".' % (device, user, cid))

    raw_signals = load_demo_signals_raw(device, user=user, cid=cid)

    raw_folder = FOLDER + '/' + device + '_raw'

    for raw_signal in raw_signals:

        save_one_raw_signal(raw_signal, raw_folder, 'raw_csv')
        save_one_raw_signal(raw_signal, raw_folder, 'raw_npy')

    rss_csv = load_demo_signals_raw(device, user=user, cid=cid, mode='raw_csv')
    rss_npy = load_demo_signals_raw(device, user=user, cid=cid, mode='raw_npy')

    for i, (rs_txt, rs_csv, rs_npy) \
        in enumerate(zip(raw_signals, rss_csv, rss_npy)):

        # print(raw_signal.length)
        # print(raw_signal_csv.length)
        # print(raw_signal_npy.length)

        print('seq=%d' % i, 'raw_csv', rs_txt.all_close_to(rs_csv))
        print('seq=%d' % i, 'raw_npy', rs_txt.all_close_to(rs_npy))

    print('Done.')

def demo_raw_signal_preprocess(device=DEVICE, user=USER, cid=CID):

    print('Demo raw signal preprocess')
    print('device="%s", user="%s", cid="%s".' % (device, user, cid))

    raw_signals = load_demo_signals_raw(device, user=user, cid=cid)
    pp_signals = []

    pp_folder = FOLDER + '/' + device + '_pp'

    for i, raw_signal in enumerate(raw_signals):

        print('seq=%d' % i, 'preprocessing')

        pp_signal = raw_signal.preprocess(point='tip')
        pp_signals.append(pp_signal)

        save_one_signal(pp_signal, pp_folder, 'csv')
        save_one_signal(pp_signal, pp_folder, 'npy')

    pss_csv = load_demo_signals_pp(device, user=user, cid=cid, mode='csv')
    pss_npy = load_demo_signals_pp(device, user=user, cid=cid, mode='npy')

    for i, (pp_signal, ps_csv, ps_npy) \
        in enumerate(zip(pp_signals, pss_csv, pss_npy)):

        print('seq=%d' % i, 'raw_csv', pp_signal.all_close_to(ps_csv))
        print('seq=%d' % i, 'raw_npy', pp_signal.all_close_to(ps_npy))

    print('Done.')


# ----------------------- demo signal visualization ----------------------------



def demo_one_signal_vis(device=DEVICE, user=USER, cid=CID):

    signal = load_one_demo_signal_pp(device=device, user=user, cid=cid)

    signal_vis(signal)


def demo_one_signal_vis_with_selected_axes(device=DEVICE, user=USER, cid=CID):

    signal = load_one_demo_signal_pp(device=device, user=user, cid=cid)

    signal_vis(signal, 9, 3)


def demo_one_signal_vis_compact(device=DEVICE, user=USER, cid=CID):

    signal = load_one_demo_signal_pp(device=device, user=user, cid=cid)

    signal_vis_compact(signal)


def demo_signals_comparison(device=DEVICE, user=USER):

    signal_0 = load_one_demo_signal_pp(device=device, user=user, seq=0)
    signal_1 = load_one_demo_signal_pp(device=device, user=user, seq=1)

    signal_vis_comparison([signal_0, signal_1], start_col=9, nr_cols=3)


def demo_signals_comparison_ten_signals(device=DEVICE, user=USER):

    signals = load_demo_signals_pp(device=device, user=user)

    signal_vis_comparison(signals, start_col=9, nr_cols=3)


# ----------------------- demo trajectory visualization ------------------------

def demo_one_trajectory_vis(device=DEVICE, user=USER, cid=CID):

    signal = load_one_demo_signal_pp(device=device, user=user, cid=cid)

    trajectory_vis(signal)


def demo_one_trajectory_vis_with_orientation(device=DEVICE, user=USER, cid=CID):

    signal = load_one_demo_signal_pp(device=device, user=user, cid=cid)

    trajectory_vis(signal, show_local_axes=True, interval=10)


def demo_trajectories_comparison(device=DEVICE, user=USER, cid=CID):

    signal_0 = load_one_demo_signal_pp(device=device, user=user, cid=cid, seq=0)
    signal_1 = load_one_demo_signal_pp(device=device, user=user, cid=cid, seq=1)

    trajectorie_vis_comparison([signal_0, signal_1])

def demo_trajectories_comparison_ten_signals(device=DEVICE, user=USER, cid=CID):

    signals = load_demo_signals_pp(device=device, user=user, cid=cid)

    trajectorie_vis_comparison(signals)


def demo_one_trajectory_animation(device=DEVICE, user=USER, cid=CID):

    signal = load_one_demo_signal_pp(device=device, user=user, cid=cid)

    trajectory_animation(signal, seg_length=30)

def demo_one_trajectory_animation_with_hand(user=USER, cid=CID):

    raw_signal = load_one_demo_signal_raw(device='leap', user=user, cid=cid)
    raw_signal.preprocess(point='tip')

    trajectory_animation(raw_signal, seg_length=30)

def demo_orientation_animation(device='glove', user=USER, cid='123456'):

    raw_signal = load_one_demo_signal_raw(device=device, user=user, cid=cid)
    raw_signal.preprocess(point='tip')

    orientation_animation(raw_signal, seg_length=20)

# ----------------------- demo signal alignment and template -------------------

def demo_signal_alignment(device=DEVICE, user=USER, cid=CID):

    signal_0 = load_one_demo_signal_pp(device=device, user=user, cid=cid, seq=0)
    signal_1 = load_one_demo_signal_pp(device=device, user=user, cid=cid, seq=1)

    signal_0.amplitude_normalize()
    signal_1.amplitude_normalize()

    signal_1_aligned = signal_1.align_to(signal_0)

    aligned = [signal_0, signal_1_aligned]
    unaligned = [signal_0, signal_1]

    signal_vis_comparison_side_by_side(aligned, unaligned, start_col=0, nr_cols=9)


def demo_warping_path(device=DEVICE, user=USER, cid=CID):

    signal_0 = load_one_demo_signal_pp(device=device, user=user, cid=cid, seq=0)
    signal_1 = load_one_demo_signal_pp(device=device, user=user, cid=cid, seq=1)

    signal_0.amplitude_normalize()
    signal_1.amplitude_normalize()

    signal_1_aligned = signal_1.align_to(signal_0, keep_dist_matrix=True)

    alignment_vis(signal_1, signal_0, signal_1_aligned, plot_3d=False, col=9)


def demo_template_construction(device=DEVICE, user=USER, cid=CID):

    signals = load_demo_signals_pp(device=device, user=user, cid=cid)

    template = FMSignalTemplate.construct_from_signals(signals, 0)

    signals_alinged = template.signals_aligned

    signal_vis_comparison_side_by_side(signals_alinged, [template], start_col=0, nr_cols=9)
    signal_vis_comparison_side_by_side(signals_alinged, signals, start_col=0, nr_cols=9)


def demo_template_io(device=DEVICE, user=USER, cid=CID):

    signals = load_demo_signals_pp(device=device, user=user, cid=cid)

    template = FMSignalTemplate.construct_from_signals(signals, 0)

    save_one_demo_template(template, device)

    template_load = load_one_demo_template(device, user=user, cid=cid, seq=0)

    print('Template save and load done:', template.all_close_to(template_load))

# ----------------------- demo signal augmentation -----------------------------

def demo_pose_pertubation():

    pass

def demo_amplitude_pertubation():

    pass

def demo_resize_segment():

    pass

def demo_swap_segment():

    pass

def demo_frequency_pertubation():

    pass






def demo(device):

    # Your demo code starts here.

    pass



if __name__ == '__main__':
    





    # demo_raw_signal_io(device, user="alice", cid='FMKit')
    # demo_raw_signal_io(device, user="alice", cid='123456')

    # demo_raw_signal_preprocess(device, user="alice", cid='FMKit')
    # demo_raw_signal_preprocess(device, user="alice", cid='123456')

    # demo_one_signal_visualization(device)
    # demo_one_signal_visualization_selected_axes(device)
    # demo_one_signal_visualization_compact(device)

    # demo_signals_comparison(device)
    # demo_signals_comparison_ten_signals(device)
    
    # demo_one_trajectory_vis()
    # demo_one_trajectory_vis_with_orientation()
    # demo_trajectories_comparison()
    # demo_trajectories_comparison_ten_signals()
    
    # demo_one_trajectory_animation()
    # demo_one_trajectory_animation_with_hand()
    # demo_orientation_animation()

    # demo_signal_alignment()
    # demo_warping_path()

    # demo_template_construction()
    demo_template_io()
    
    pass



