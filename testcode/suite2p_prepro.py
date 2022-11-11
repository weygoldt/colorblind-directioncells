from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from tqdm import tqdm
from vxtools.summarize.structure import SummaryFile

import functions as fs

# get data
f = SummaryFile('../data/Summary.hdf5')  # import HDF5 file
one_rec = fs.data_one_rec_id(f, 5)  # extract one recording
times = one_rec[0].times  # get the time axis
start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2 = fs.get_attributes(
    one_rec)

# make matrix of dffs
roi_dffs = np.empty((len(one_rec), len(one_rec[0].dff)))
for i, roi in enumerate(one_rec):
    roi_dffs[i, :] = roi.dff

# convert dff matrix to mean dff matrix
mean_dffs = fs.get_mean_dffs(roi_dffs, times, (start_time, end_time))

# get indices for stimulus phase series repeats
inx = fs.repeats(angul_vel)

# compute correlation coefficient of thresholded active ROIs
active_spear = fs.active_rois(mean_dffs, inx, threshold=0.3)

# get dffs for active ROIs
active_dff = [one_rec[int(ar[0])].dff for ar in active_spear]

# get all dffs
dffs = np.array([roi.dff for roi in one_rec])

# plot raster of all dffs
plt.imshow(dffs)
plt.show()

# plot dff lineplot for all dffs
for i, dff in enumerate(active_dff):
    plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()),
             color='black', linewidth='1.')
plt.show()

"""
rposs = []
rnegs = []
for roi in tqdm(range(len(one_rec))):

    # make logical arrays for left and right turning motion
    mean_zscore = mean_zscore_roi(one_rec, roi)[1:-1]
    angveloc_pos = [1 if a == 30.0 else 0 for a in angul_vel][1:-1]
    angveloc_neg = [1 if a == -30.0 else 0 for a in angul_vel][1:-1]

    # compute correlation
    rposs.append(spearmanr(mean_zscore, angveloc_pos))
    rnegs.append(spearmanr(mean_zscore, angveloc_neg))


for i, s in enumerate(rposs):
    plt.scatter(i, s[0])

"""
"""
# convert to numpy array
rposs = np.array(rposs)
rnegs = np.array(rnegs)

# threshold
rposs[rposs < 0.3] = 0
rnegs[rnegs < 0.3] = 0

# threshold by significance

# find indices, which are rois
index_array = np.arange(len(rposs))
pos_rois = index_array[rposs != 0]
neg_rois = index_array[rnegs != 0]


# plot
rois = pos_rois
for r in rois:
    mean_zscores = mean_zscore_roi(one_rec, r)
    z_vel_30 = [z for z, a in zip(mean_zscores, angul_vel) if a == 30.0]
    z_vel_minus30 = [z for z, a in zip(mean_zscores, angul_vel) if a == -30.0]
    z_vel_0 = [z for z, a in zip(mean_zscores, angul_vel) if a == 0.0]

    fig, ax = plt.subplots()
    ax.boxplot(z_vel_0, positions=[1])
    ax.boxplot(z_vel_minus30, positions=[2])
    ax.boxplot(z_vel_30, positions=[3])
    ax.set_xticklabels(['0vel', '-30vel', '30vel'])
    plt.show()
"""

"""
plt.plot(time, one_rec[roi].zscore)
plt.scatter(np.sort(start_time), np.ones_like(start_time), c='r')
plt.scatter(np.sort(end_time), np.ones_like(end_time))
plt.scatter((start_time) + (end_time - start_time)/2, mean_zscore, c='k')
plt.show()


"""

"""
roi = 10
z_vel_30 = [z for z, a in zip(mean_zscore, angul_vel) if a == 30.0]
z_vel_minus30 = [z for z, a in zip(mean_zscore, angul_vel) if a == -30.0]
z_vel_0 = [z for z, a in zip(mean_zscore, angul_vel) if a == 0.0]
"""

"""
correlate repeats with each other:
take first two repeats of the same phase series to find cells that respond over time
use dff
"""
