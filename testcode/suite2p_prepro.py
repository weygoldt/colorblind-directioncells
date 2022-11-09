import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from vxtools.summarize.structure import SummaryFile

f = SummaryFile('data/Summary.hdf5')
# get all rois
rois = f.rois()

rois_0 = rois[0]

for i, roi in enumerate(f.rois()[:20]):
    dff = roi.dff
    plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()),
             color='black', linewidth='1.')
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
for indx, grp in enumerate(one_rec[roi].rec.h5group['display_data']['phases'].values()):
    if indx == 0 or indx == len(one_rec[roi].rec.h5group['display_data']['phases'])-1:
        continue
    print(indx)
    try:
        start_time.append(grp.attrs['start_time'])
        end_time.append(grp.attrs['end_time'])
        angul_vel.append(grp.attrs['angular_velocity'])
        angul_pre.append(grp.attrs['angular_period'])
        rgb_1.append(grp.attrs['rgb01'])
        rgb_2.append(grp.attrs['rgb02'])
    except:
        embed()


plt.plot(time, one_rec[0].zscore)
plt.scatter(np.sort(start_time), np.ones_like(start_time), c='r')
plt.scatter(np.sort(end_time), np.ones_like(end_time))
plt.show()

embed()
exit()
