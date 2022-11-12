# Essentials -------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy import interpolate
from sklearn.metrics import auc
from vxtools.summarize.structure import SummaryFile

import functions as fs
from plotstyle import PlotStyle

ps = PlotStyle()

# activity threshold (min spearman correlation)

prob_thresh = 0.05  # probability threshold for non-mean data

# get data
f = SummaryFile('../data/Summary.hdf5')  # import HDF5 file
one_rec = fs.data_one_rec_id(f, 5)  # extract one recording
times = one_rec[0].times  # get the time axis
start_times, stop_times, ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(
    one_rec)

# Getting actual dff data ------------------------------------------------------

# remove first and last start time because baseline
start_times = start_times[1:-1]
stop_times = stop_times[1:-1]

# new time for interpolation
new_end_time = stop_times-start_times[0]
new_start_time = start_times-start_times[0]
new_times = np.arange(new_start_time[0], new_end_time[-1]+0.1, 0.1)

# convert stim onset timestamps to indices
stimstart = fs.find_on_time(new_times, new_start_time[0])
stimstop = fs.find_on_time(new_times, new_end_time[-1])

# make matrix of dffs for this roi
roi_dffs = np.empty((len(one_rec), len(new_times)))
for i, roi in enumerate(one_rec):
    dff = roi.dff
    f = interpolate.interp1d(times-start_times[0], dff)
    dff_interp = f(new_times)
    roi_dffs[i, :] = dff_interp

# reassing these variables
# (if something unexpected happens, keep this is mind!)
times = new_times
start_times = new_start_time
stop_times = new_end_time

# convert dff matrix to mean dff matrix
mean_dffs, mean_times = fs.get_mean_dffs(
    roi_dffs, times, (start_times, stop_times))

# Working with and converting start and stop of stimulus -----------------------

# get indices for stimulus phase series repeats
inx_mean = fs.repeats((start_times, stop_times))

# convert repeat start and stop indices to timestamps
repeat_starts = np.array(start_times)[[int(i[0]) for i in inx_mean]]
repeat_stops = np.array(stop_times)[[int(i[1]) for i in inx_mean]]

# convert repeat start and stop timestamp to indices on full time vector
repeat_starts_idx = [fs.find_on_time(times, x) for x in repeat_starts]
repeat_stops_idx = [fs.find_on_time(times, x) for x in repeat_stops]

# reformat to fit inx shape
inx_full = np.array(
    [[x, y] for x, y in zip(repeat_starts_idx, repeat_stops_idx)]
)

# check if phase series repeats are of same length

# these MUST be of same length!
repeat_df_mean = np.sum(np.diff([x[1]-x[0] for x in inx_mean]))
# these may not be!
repeat_df_full = np.sum(np.diff([x[1]-x[0] for x in inx_full]))

# if they are not, cut the ends off
# (its just a few datapoints so should not matter much)
if repeat_df_full != 0:

    # get the shortest one
    imin = np.min([x[1]-x[0] for x in inx_full])

    # cut the others short
    for i, x in enumerate(inx_full):
        if x[1]-x[0] != imin:
            di = (x[1]-x[0])-imin
            inx_full[i][1] = inx_full[i][1] - di

# Compute corrcoeffs and threshold them ----------------------------------------

# compute correlation coefficient of ROIs
sorted_mean_rois = fs.sort_rois(mean_dffs, inx_mean)

# get dffs for all ROIs
sorted_dffs = np.array([roi_dffs[int(roi), :]
                       for roi in sorted_mean_rois[:, 0]])

# threshold the means based on correlation coefficient

# make a histpgram
counts, edges = np.histogram(sorted_mean_rois[:, 1], bins=50, density=True)

# compute a gaussian KDE
xkde, kde = fs.kde1d(
    sorted_mean_rois[:, 1],
    0.04,
    xlims=[edges.min(), edges.max()],
    resolution=200,
)

# find where right auc matches threshold
gradient, idx = fs.find_right_tail(xkde, kde, prob_thresh, plot=True)
thresh = xkde[idx]
thresh_mean_rois = fs.thresh_correlations(sorted_mean_rois, thresh)

# get dffs for active ROIs only for raster (means per stim)
# and lineplot (interp datapoints)
active_mean_dffs = np.array([mean_dffs[int(roi), :]
                             for roi in thresh_mean_rois[:, 0]])
active_dffs = np.array([roi_dffs[int(roi), :]
                        for roi in thresh_mean_rois[:, 0]])

# Make a meanstack -------------------------------------------------------------

# stack 3 repeats and compute mean

# ... of all dffs
meanstack_t_sorted_dffs, meanstack_d_sorted_dffs = fs.meanstack(
    sorted_dffs, times, inx_full)

# ... of the normal but thresholded dffs
meanstack_t_active_dffs, meanstack_d_active_dffs = fs.meanstack(
    active_dffs, times, inx_full)

# ... of the mean dffs
meanstack_t_active_mean_dffs, meanstack_d_active_mean_dffs = fs.meanstack(
    active_mean_dffs, mean_times, inx_mean)


# Encode stimulus contrast -----------------------------------------------------

# remove nans at start and stop
rgb_1 = rgb_1[1:-1]
rgb_2 = rgb_2[1:-1]

# luminance contrast: max(all)_stripe1 - max(all)_stripe2
stripe1 = np.array([np.max(x) for x in rgb_1])
stripe2 = np.array([np.max(x) for x in rgb_2])
lum_contr = abs(stripe1-stripe2)[inx_mean[0][0]:inx_mean[0][1]]

# color contrast: red_stripe1 - red_stripe2 (for red and green)
red = np.array(
    [abs(x[0]-y[0]) for x, y in zip(rgb_1, rgb_2)]
)[inx_mean[0][0]:inx_mean[0][1]]

green = np.array(
    [abs(x[1]-y[1]) for x, y in zip(rgb_1, rgb_2)]
)[inx_mean[0][0]:inx_mean[0][1]]


# Encode rotation direction ----------------------------------------------------

# make no rot = 0, counterclock = -1, clockwise = 1
ang_veloc = ang_veloc[1:-1]  # cut nans
rots = np.zeros_like(ang_veloc, dtype=int)
for i, rot in enumerate(ang_veloc):
    if rot > 0:
        rots[i] = 1
    elif rot == 0:
        rots[i] = 0
    else:
        rots[i] = -1
rots = rots[inx_mean[0][0]:inx_mean[0][1]]  # cut to first repeat


# specify here what you want to plot -------------------------------------------

plot_times = meanstack_t_active_dffs
plot_dffs = meanstack_d_active_dffs
plot_rois = thresh_mean_rois[:, 0]

# make matrix out of luminance contrast (where rotation was ON)
# lum_mask = np.arange(len(lum_contr))
# lum_mask = lum_mask[rots == 0]
# lum_contr[lum_mask] = np.nan
lum_img = np.empty((len(plot_rois), len(lum_contr)))
for i in range(len(lum_img[:, 0])):
    lum_img[i, :] = lum_contr


# Plot distribution of corrcoeffs ----------------------------------------------

fig, ax = plt.subplots()
ax.bar(edges[:-1],
       counts,
       width=np.diff(edges),
       edgecolor="white",
       facecolor="k",
       alpha=0.2,
       linewidth=0,
       align="edge",
       zorder=20,
       )
ax.plot(xkde, kde, zorder=10, c="k")
ax.fill_between(
    xkde[:idx+1],
    np.zeros_like(xkde[:idx+1]),
    kde[:idx+1],
    facecolor="lightblue",
    alpha=0.7,
)
ax.fill_between(
    xkde[idx:],
    np.zeros_like(xkde[idx:]),
    kde[idx:],
    facecolor="lightgreen",
    alpha=0.7,
)

ax.set_xlim(edges.min(), edges.max())
ax.set_ylabel("PDF")
ax.set_xlabel("Spearmans r")
plt.show()

# Plot rasterplot --------------------------------------------------------------

# plot raster of all dffs
extent = (np.min(plot_times), np.max(plot_times),
          0, len(plot_rois))

fig, ax = plt.subplots()
ax.imshow(plot_dffs,
          cmap='binary',
          aspect='auto',
          extent=extent,
          )

# plot contrast
ax.imshow(lum_img,
          cmap="Blues",
          alpha=0.3,
          aspect="auto",
          extent=extent,
          )

plt.show()


# Plot stacked lineplot --------------------------------------------------------

# plot dff lineplot for all dffs
fig, ax = plt.subplots()
for i, dff in enumerate(plot_dffs):
    ax.plot(plot_times, i + (dff - dff.min()) / (dff.max() - dff.min()),
            color='black', linewidth='1.')
plt.show()
