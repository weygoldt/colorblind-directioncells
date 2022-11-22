import os
from pathlib import Path

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from matplotlib.patches import Rectangle
from scipy import interpolate
from scipy import signal as fp
from vxtools.summarize.structure import SummaryFile

import modules.functions as fs
from modules.plotstyle import PlotStyle

p = Path('../data/data2')
folder_names = np.sort([x.name for x in p.iterdir() if x.is_dir()])
folder_id = np.char.split(folder_names, '_')
recs = [int(i[2][3]) for i in folder_id]
camera_files = sorted(p.glob('*/Camera.hdf5'))[:]
#recs = [0,1,2,3,4]
f = SummaryFile('../data/data2/Summary.hdf5')


def read_hdf5_file(file):

    with h5py.File(file, 'r') as f:
        right_eye_pos = f['eyepos_ang_re_pos_0']
        right_eye_time = f['eyepos_ang_re_pos_0_attr_time']
        left_eye_pos = f['eyepos_ang_le_pos_0']
        left_eye_time = f['eyepos_ang_le_pos_0_attr_time']

        r_eye_pos = np.ravel(right_eye_pos)
        r_eye_time = np.ravel(right_eye_time)
        l_eye_pos = np.ravel(left_eye_pos)
        l_eye_time = np.ravel(left_eye_time)

    return r_eye_pos, r_eye_time, l_eye_pos, l_eye_time


rec_velos = []
# get the data
for file in range(len(camera_files)):
    r_eye_pos, r_eye_time, l_eye_pos, l_eye_time = read_hdf5_file(
        camera_files[file])
    one_rec = fs.data_one_rec_id(f, recs[file])
    start_time, stop_time, target_dur, ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(
        one_rec)

    interp = 0.05
    einterp = interpolate.interp1d(r_eye_time, r_eye_pos)
    int_eye = []
    times = []
    velos = []
    t_interp = []

    # interpolate velocity in phases
    for st, td in zip(start_time[1:-1], target_dur[1:-1]):
        phase = np.arange(0, td, interp) + st
        eye_interp = np.array(einterp(phase))
        v = fs.velocity1d(phase, eye_interp)
        int_eye.append(eye_interp)
        times.append(phase)
        t_interp.append(phase[1:-1])
        velos.append(v)

    # extrapolate empty values
    extrap = interpolate.interp1d(
        np.ravel(t_interp), np.ravel(velos), fill_value='extrapolate')
    full_velos = extrap(np.ravel(times))
    times = np.asarray(np.ravel(times))

    # find peaks (i.e. scaccades)
    sacc_test = fp.find_peaks(abs(full_velos), prominence=6)[0]
    saccs_idx = np.array(fs.flatten([np.arange(x-2, x+2) if (x > 2) &
                                    (x < sacc_test[-2]) else [x] for x in sacc_test]))

    # remove saccades froma rray
    new_velo = np.delete(full_velos, saccs_idx)
    new_times = np.delete(times, saccs_idx)

    # prepare interpolating the missing saccades
    sacc_interpolator = interpolate.interp1d(
        new_times, new_velo, fill_value='extrapolate')
    times = []
    velos = []

    # interpolate the missing saccades
    for st, td in zip(start_time[1:-1], target_dur[1:-1]):
        phase = np.arange(0, td, interp) + st
        eye_veloc = sacc_interpolator(phase)
        times.append(phase)
        velos.append(eye_veloc)

    velos = np.asarray(velos)
    rec_velos.append(velos)
