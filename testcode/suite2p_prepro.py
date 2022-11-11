import matplotlib.pyplot as plt
import numpy as np
from vxtools.summarize.structure import SummaryFile

import functions as fs
from plotstyle import PlotStyle

ps = PlotStyle()

# activity threshold (min spearman correlation)
thresh = 0.6

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
inx_all = fs.repeats(times)

# compute correlation coefficient of thresholded active ROIs
sorted_rois = fs.sort_rois(mean_dffs, inx)

# threshold them
thresh_rois = fs.thresh_correlations(sorted_rois, thresh)

# get dffs for all ROIs
sorted_dffs = np.array([roi_dffs[int(roi), :] for roi in sorted_rois[:, 0]])

# get dffs for active ROIs only
active_dffs = np.array([roi_dffs[int(roi), :] for roi in thresh_rois[:, 0]])


def meanstack(roi_dffs, times, inx):

    meanstack_times = times[inx[0][0]:inx[0][1]]
    meanstack_dffs = np.empty((len(roi_dffs[:, 0]), inx[0][1]))
    for roi in range(len(roi_dffs[:, 0])):
        dff = roi_dffs[roi, :]
        split_dff = np.array([dff[x[0]:x[1]] for x in inx])
        mean_dff = np.mean(split_dff, axis=0)
        meanstack_dffs[roi, :] = mean_dff

    return meanstack_times, meanstack_dffs


meanstack_times, meanstack_dffs = meanstack(sorted_dffs, times, inx_all)

# make stimulus arrays
time_idx = np.arange(len(times))
start_idx = [fs.find_on_time(times, x) for x in start_time]
stop_idx = [fs.find_on_time(times, x) for x in end_time]

# get movement data of stimulus
new_start_idx = start_idx[inx[0][0]:inx[0][1]]
new_stop_idx = stop_idx[inx[0][0]:inx[0][1]]
new_angul_vel = angul_vel[inx[0][0]:inx[0][1]]

no_motion = meanstack_times[[[new_start_idx[x], new_stop_idx[x]]
                             for x in range(len(new_angul_vel)) if new_angul_vel[x] == 0.0]]

left_motion = meanstack_times[[[new_start_idx[x], new_stop_idx[x]]
                               for x in range(len(new_angul_vel)) if new_angul_vel[x] < 0.0]]

right_motion = meanstack_times[[[new_start_idx[x], new_stop_idx[x]]
                                for x in range(len(new_angul_vel)) if new_angul_vel[x] > 0.0]]


def plot_filled_ranges(ax, ranges, style):
    for r in ranges:
        ax.axvspan(r[0], r[1], **style)


style_nomo = dict(color='black', lw=0, zorder=10, alpha=0.2)
style_left = dict(color='blue', lw=0, zorder=10, alpha=0.2)
style_right = dict(color='red', lw=0, zorder=10, alpha=0.2)

# plot raster of all dffs
extent = (np.min(meanstack_times), np.max(meanstack_times),
          np.min(thresh_rois[:, 0]), np.max(thresh_rois[:, 0]))

fig, ax = plt.subplots()
ax.imshow(meanstack_dffs, cmap='binary', aspect='auto', extent=extent)
plot_filled_ranges(ax, no_motion, style_nomo)
plot_filled_ranges(ax, left_motion, style_left)
plot_filled_ranges(ax, right_motion, style_right)
plt.show()

# plot dff lineplot for all dffs
fig, ax = plt.subplots()
for i, dff in enumerate(active_dffs):
    ax.plot(times, i + (dff - dff.min()) / (dff.max() - dff.min()),
            color='black', linewidth='1.')
plot_filled_ranges(ax, no_motion, style_nomo)
plot_filled_ranges(ax, left_motion, style_left)
plot_filled_ranges(ax, right_motion, style_right)
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
