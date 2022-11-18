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

    return r_eye_pos, r_eye_time, l_eye_pos, l_eye_time

ri_pos, ri_time, le_pos, le_time = read_hdf5_file(camera_file)
embed()
exit()
plt.plot(ri_time[:-1], np.diff(ri_pos))

# take the absute value:
ri_pos_abs = np.abs(np.diff(ri_pos))
plt.plot(ri_time[:-1], ri_pos_abs)


