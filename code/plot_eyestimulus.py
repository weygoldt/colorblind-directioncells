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

ps = PlotStyle()

f = SummaryFile('../data/data2/Summary.hdf5')

camera_file = '../data/data2/Camera.hdf5'


def read_hdf5_file(file):
    r_eye_pos = []
    r_eye_time = []
    l_eye_pos = []
    l_eye_time = []
    with h5py.File(file, 'r') as f:
        right_eye_pos = f['eyepos_ang_re_pos_0']
        right_eye_time = f['eyepos_ang_re_pos_0_attr_time']
        left_eye_pos = f['eyepos_ang_le_pos_0']
        left_eye_time = f['eyepos_ang_le_pos_0_attr_time']
        r_eye_pos.append(np.ravel(right_eye_pos))
        r_eye_time.append(np.ravel(right_eye_time))
        l_eye_pos.append(np.ravel(left_eye_pos))
        l_eye_time.append(np.ravel(left_eye_time))

    return r_eye_pos[0], r_eye_time[0], l_eye_pos[0], l_eye_time[0]


one_rec = fs.data_one_rec_id(f, 0)
start_time, stop_time, target_dur,  ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(
    one_rec)
ri_pos, ri_time, le_pos, le_time = read_hdf5_file(camera_file)

embed()
exit()
# take the absute value:
ri_pos_abs = np.abs(np.diff(ri_pos))

sacc = fp.find_peaks(ri_pos_abs, height=1.5)[0]
postive_peaks = []
negative_peaks = []
diff_right = np.diff(ri_pos)

for i in sacc:
    if diff_right[i] > 0:
        postive_peaks.append(ri_time[i])
    elif diff_right[i] < 0:
        negative_peaks.append(ri_time[i])
interp = 0.2
einterp = interpolate.interp1d(ri_time, ri_pos)
times = []
int_eye = []
velos = []
# calculate the the pmean for the velocity
for st, td in zip(start_time[1:-1], target_dur[1:-1]):
    phase = np.arange(0, td, interp) + st
    try:
        eye_interp = np.array(einterp(phase))
    except:
        embed()
    v = fs.velocity1d(phase, eye_interp)
    int_eye.append(eye_interp)
    times.append(phase)
    velos.append(v)

# plotting stimulus and eye tracking data only from the right eye
fig, ax = plt.subplots(figsize=(22*ps.cm, 15*ps.cm))
ax.set_xlim(4227.5, 4249.9)
ax.set_xticks(np.arange(start_time[100], 4250, 4))
ax.plot(ri_time, ri_pos,)
ax.set_xlabel('Time in [s]')
ax.set_ylabel('Deflection in [mm]')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i in range(len(start_time)):
    ax.vlines(start_time[i], -20, 10, linestyles='dashed',
              colors='k',)
    ax.text(start_time[i] + ((stop_time[i]-start_time[i])/2) -
            0.5, 10, f"{ang_veloc[i]}", clip_on = True )
    red = Rectangle(
        (start_time[i], -20), ((stop_time[i]-start_time[i])/2), 4, facecolor=(rgb_1[i], 0, 0))
    green = Rectangle((start_time[i] + (stop_time[i]-start_time[i])/2, -20),
                      ((stop_time[i]-start_time[i])/2), 4, facecolor=(0, rgb_2[i], 0), )
    
    ax.text(start_time[i] + 0.4, -22,  f"{rgb_1[i]:.2f}",  clip_on =True)
    ax.text(stop_time[i] - 1.7,  -22,  f"{rgb_2[i]:.2f}" , clip_on =True)
    ax.add_patch(red)
    ax.add_patch(green)

# fs.doublesave('../plots/eyestimulus')
plt.show()
