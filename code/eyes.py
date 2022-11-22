#look at me 

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


r_eye_pos, r_eye_time, l_eye_pos, l_eye_time = read_hdf5_file(camera_files[0])

one_rec = fs.data_one_rec_id(f, recs[0])
start_time, stop_time, target_dur, ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(
        one_rec)

interp = 0.05 
einterp = interpolate.interp1d(r_eye_time, r_eye_pos)
int_eye = []
times = []
velos = []
# calculate the the pmean for the velocity
for st, td in zip(start_time[1:-1], target_dur[1:-1]):
    phase = np.arange(0, td, interp) + st
    eye_interp = np.array(einterp(phase))
    v = fs.velocity1d(phase, eye_interp)
    int_eye.append(eye_interp)
    times.append(phase)
    velos.append(v)

# getting rid of the saccades 

r_velos = np.ravel(np.abs(velos))
q75, q25 = np.nanpercentile(r_velos, [75, 25])
iqr = q75 - q25
saccade_cut_off = q75+2*iqr
sacc = fp.find_peaks(r_velos, height=saccade_cut_off)[0]
saccs_idx = [sacc-1, sacc, sacc+1]
new_velo = np.delete(r_velos, saccs_idx)
new_time = np.delete(np.ravel(times), saccs_idx)


inter_vel = []
interp = 0.05 
vinterp = interpolate.interp1d(new_time, new_velo)

for st, td in zip(start_time[1:-1], target_dur[1:-1]):
    phase = np.arange(0, td, interp) + st
    i_vel = np.array(vinterp(phase))
    inter_vel.append(i_vel)

plt.plot(np.ravel(times), np.ravel(inter_vel))
plt.plot(np.ravel(times), np.ravel(velos))
plt.show()

# second try 
velos1 = np.copy(velos)
q75, q25 = np.nanpercentile(np.abs(velos1), [75, 25])
iqr = q75 - q25
saccade_cut_off = q75 + 2*iqr

for v in range(len(velos1)):
    sacc = fp.find_peaks(np.abs(velos1[v]), height=saccade_cut_off)[0]
    velos1[v][sacc-1] = np.nanmean(velos1[v])
    velos1[v][sacc+1] = np.nanmean(velos1[v])
    velos1[v][sacc]   = np.nanmean(velos1[v])

plt.plot(np.ravel(times), np.ravel(velos1))
plt.plot(np.ravel(times), np.ravel(velos))
plt.show()


