# load essential packages
import os
import matplotlib.pyplot as plt
import numpy as np
import modules.functions as fs
from vxtools.summarize.structure import SummaryFile
from IPython import embed
from modules.plotstyle import PlotStyle
import h5py
from scipy import signal as fp
from matplotlib.patches import Rectangle
from scipy import interpolate
ps = PlotStyle()
f = SummaryFile('../data2/Summary.hdf5')

camera_file = '../data2/Camera.hdf5'


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

# plotting stimulus and eye tracking data only from the right eye 
fig, ax = plt.subplots()
ax.set_xlim(4227.5, 4249.9)
ax.set_xticks(np.arange(start_time[100], 4250 , 4.07))
ax.plot(ri_time, ri_pos)
ax.set_xlabel('Time in [s]')
ax.set_ylabel('Deflection in [mm]')
for i in range(len(start_time)):
    ax.vlines(start_time[i], -20, 10, linestyles='dashed', colors='k')
    ax.text(start_time[i] + ((stop_time[i]-start_time[i])/2) -
            0.5, 10, f"{ang_veloc[i]}", clip_on=True)
    red = Rectangle(
        (start_time[i], -20), ((stop_time[i]-start_time[i])/2), 4, facecolor=(rgb_1[i], 0, 0))
    green = Rectangle((start_time[i] + (stop_time[i]-start_time[i])/2, -20),
                      ((stop_time[i]-start_time[i])/2), 4, facecolor=(0, rgb_2[i], 0))
    ax.add_patch(red)
    ax.add_patch(green)
plt.show()


## removing saccades from the velocity of the eye movement to calculate the mean velocity 
interp = 0.1
einterp = interpolate.interp1d(ri_time, ri_pos)

velo_right = np.diff(ri_pos)
# getting rid of the saccades 
velo_right[sacc-1] = 0
velo_right[sacc+1] = 0
velo_right[sacc]   = 0 

## check for saccads to be dealt with 
"""
fig, ax = plt.subplots()
#ax.set_xlim(4227.5, 4249.9)
#ax.set_xticks(np.arange(start_time[100], 4250 , 4.07))
ax.plot(ri_time[:-1], velo_right)
ax.plot(ri_time[:-1], np.diff(ri_pos))
for i in range(len(start_time)):
    ax.vlines(start_time[i], -20, 10, linestyles='dashed', colors='k')
    ax.text(start_time[i] + ((stop_time[i]-start_time[i])/2) -
            0.5, 10, f"{ang_veloc[i]}", clip_on=True)
    red = Rectangle(
        (start_time[i], -20), ((stop_time[i]-start_time[i])/2), 4, facecolor=(rgb_1[i], 0, 0))
    green = Rectangle((start_time[i] + (stop_time[i]-start_time[i])/2, -20),
                      ((stop_time[i]-start_time[i])/2), 4, facecolor=(0, rgb_2[i], 0))
    ax.add_patch(red)
    ax.add_patch(green)
plt.show()
"""

# interplotate to the right time 


int_eye = []
# calculate the the pmean for the velocity 
for st, td in zip(start_time[1:-1], target_dur[1:-1]):
    phase = np.arange(0, td, interp) + st
    eye_interp = np.array(einterp(phase))
    int_eye.append(eye_interp)


mean_vel = np.asarray([np.mean(x) for x in int_eye])

red_unique = np.unique(np.array(rgb_1)[~np.isnan(np.array(rgb_1))])

clock = np.array([1 if x > 0 else 0 for x in ang_veloc])
# counterclockwise motion regressor
cclock = np.array([1 if x < 0 else 0 for x in ang_veloc])

clock_stim = np.array(clock, dtype=float)
cclock_stim = np.array(cclock, dtype=float)

clock_stim[clock_stim == 0] = np.nan
cclock_stim[cclock_stim == 0] = np.nan

red_contr = np.array(rgb_1)
green_contr = np.array(rgb_2)

red_clock_stim = red_contr*clock_stim
green_clock_stim = green_contr*clock_stim

red_cclock_stim = red_contr*cclock_stim
green_cclock_stim = green_contr*cclock_stim

class rg_acivity:
    def __init__(self, mean_eye, red_stim, green_stim): 
        inx_red = np.arange(len(red_stim))
        self.contr1_index = np.unique(red_stim[~np.isnan(red_stim)])
        contr2 = np.unique(green_stim[~np.isnan(green_stim)])

        index_contrast1 = np.arange(len(red_stim[1:-1]))
        self.contr2_index = []
        pmean_eye_contrast = []

        for c in self.contr1_index:
            idx = index_contrast1[red_stim[1:-1] == c]
            pmean_contrast = mean_eye[idx]
            self.contr2_index.append(green_stim[idx])
            pmean_eye_contrast.append(pmean_contrast)

        self.pepmean_eye_contrast = np.array(pmean_eye_contrast)
        self.contr1_index = np.array(self.contr1_index)
        self.contr2_index = np.array(self.contr2_index)
        self.contr1 = red_stim
    
rg_clock_data = rg_acivity(mean_vel, red_clock_stim, green_clock_stim)
rg_cclock_data = rg_acivity(mean_vel, red_cclock_stim, green_cclock_stim)

fig, ax = plt.subplots(2,3, sharex=True, sharey=True, figsize=(20*ps.cm, 20*ps.cm))
ax = fs.flatten(ax)

colors = [ps.orange, ps.red]
labels = ['clockwise-selective cells', 'countercl.-selective cells']

for lab, color, rg_clock in zip(labels, colors, [rg_clock_data, rg_cclock_data]):

    pass
embed()
exit()
