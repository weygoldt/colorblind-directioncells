import numpy as np 
import matplotlib.pyplot as plt
from  IPython import embed
import h5py as h5
from vxtools.summarize.structure import SummaryFile
from functions import find_closest as fc

f = SummaryFile('data/Summary.hdf5')
# get all rois 
rois = f.rois()

rois_0 = rois[0]

for i, roi in enumerate(f.rois()[:20]):
    dff = roi.dff
    plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()), color='black', linewidth='1.')
plt.show()

# get one recording 
one_rec = []
for r in rois:
    if r.rec_id == 1:
        one_rec.append(r)
time = one_rec[0].times




roi = 550 
start_time = []
end_time = []
angul_vel = []
angul_pre = []
rgb_1 = []
rgb_2 = []
len_phases = len(one_rec[roi].rec.h5group['display_data']['phases']) - 1

for  grp in one_rec[roi].rec.h5group['display_data']['phases'].values():
    start_time.append(grp.attrs['start_time'])
    end_time.append(grp.attrs['end_time'])
    if grp == one_rec[roi].rec.h5group['display_data']['phases']['0'] or grp == one_rec[roi].rec.h5group['display_data']['phases'][f'{len_phases}']:
        angul_vel.append(np.nan)
        angul_pre.append(np.nan)
        rgb_1.append(np.nan)
        rgb_2.append(np.nan)
        continue
    angul_vel.append(grp.attrs['angular_velocity'])
    angul_pre.append(grp.attrs['angular_period'])
    rgb_1.append(grp.attrs['rgb01'])
    rgb_2.append(grp.attrs['rgb02'])

start_time = np.sort(start_time)
end_time = np.sort(end_time)

start_indx = fc(time, start_time[3])
print(start_indx)

# time index for the first snippet 
bline=[]
dt = time[1]-time[0]
# first time indx snippet 
b_start = fc(time, start_time[0])
b_end = fc(time, end_time[0])
bline.append(np.arange(b_start, b_end))
time_snippets = []

for st, end in zip(start_time[1:], end_time[1:]):
    #print(st, end)
    start_inx = fc(time, st)
    end_inx = fc(time, end)
    time_snippets.append(np.arange(start_inx, end_inx))    





    baseline = start_time[0]

plt.plot(time, one_rec[0].zscore)
plt.scatter(np.sort(start_time), np.ones_like(start_time), c='r')
plt.scatter(np.sort(end_time), np.ones_like(end_time))
plt.show()


for i, roi in enumerate(one_rec[:20]):
    dff = roi.dff
    plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()), color='black', linewidth='1.')
plt.show()

