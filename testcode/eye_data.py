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

f = SummaryFile('../data2/Summary.hdf5')

camera_file = '../data2/Camera.hdf5'

def read_hdf5_file(file):

    r_eye_pos = []
    r_eye_time = []
    l_eye_pos = []
    l_eye_time = []
    with h5py.File(file, 'r') as f:
        right_eye_pos  = f['eyepos_ang_re_pos_0']
        right_eye_time = f['eyepos_ang_re_pos_0_attr_time']
        left_eye_pos  = f['eyepos_ang_le_pos_0']
        left_eye_time = f['eyepos_ang_le_pos_0_attr_time'] 
        r_eye_pos.append(np.ravel(right_eye_pos)) 
        r_eye_time.append(np.ravel(right_eye_time))
        l_eye_pos.append(np.ravel(left_eye_pos)) 
        l_eye_time.append(np.ravel(left_eye_time))

    return r_eye_pos[0], r_eye_time[0], l_eye_pos[0], l_eye_time[0]

one_rec = fs.data_one_rec_id(f, 0)
start_time, stop_time, ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(one_rec)
ri_pos, ri_time, le_pos, le_time = read_hdf5_file(camera_file)
#plt.plot(ri_time[:-1], np.diff(ri_pos))

# take the absute value:
ri_pos_abs = np.abs(np.diff(ri_pos))
#plt.plot(ri_time[:-1], ri_pos_abs)

sacc = fp.find_peaks(ri_pos_abs, height=1.5)[0]
postive_peaks = []
negative_peaks = []
diff_right = np.diff(ri_pos)

for i in sacc:
    if diff_right[i] > 0:
        postive_peaks.append(ri_time[i])
    elif diff_right[i] < 0:
        negative_peaks.append(ri_time[i])

fig, ax = plt.subplots()

ax.plot(ri_time, ri_pos)
#ax.scatter(postive_peaks, np.ones_like(postive_peaks))
#ax.scatter(negative_peaks, np.ones_like(negative_peaks)*-1)
for i in range(len(start_time)):
    ax.axvspan (start_time[i], stop_time[i], color ='b', alpha=0.6)
    ax.text(start_time[i]+stop_time[i]/2, 10, f"{ang_veloc}")

plt.show()


embed()
exit()
    



