# Essentials -------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from vxtools.summarize.structure import SummaryFile

import functions as fs
from plotstyle import PlotStyle

ps = PlotStyle()

# activity threshold (min spearman correlation)

thresh = 0.2  # threshold for non-mean data
meanthresh = 0.6  # threshold for mean rois

# get data
f = SummaryFile('../data/Summary.hdf5')  # import HDF5 file
one_rec = fs.data_one_rec_id(f, 5)  # extract one recording
times = one_rec[0].times  # get the time axis
start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2 = fs.get_attributes(
    one_rec)

# Getting actual dff data ------------------------------------------------------

# remove first and last start time because baseline
start_time = start_time[1:-1]
end_time = end_time[1:-1]

# new time for interpolation
new_end_time = end_time-start_time[0]
new_start_time = start_time-start_time[0]
new_times = np.arange(new_start_time[0], new_end_time[-1]+0.1, 0.1)

# convert stim onset timestamps to indices
stimstart = fs.find_on_time(new_times, new_start_time[0])
stimstop = fs.find_on_time(new_times, new_end_time[-1])

# make matrix of dffs for this roi
roi_dffs = np.empty((len(one_rec), len(new_times)))
for i, roi in enumerate(one_rec):
    dff = roi.dff
    f = interpolate.interp1d(times-start_time[0], dff)
    dff_interp = f(new_times)
    roi_dffs[i, :] = dff_interp

# reassing these variables (if something unexpected happens, keep this is mind!)
times = new_times
start_time = new_start_time
end_time = new_end_time

# convert dff matrix to mean dff matrix
mean_dffs, mean_times = fs.get_mean_dffs(
    roi_dffs, times, (start_time, end_time))

# Working with and converting start and stop of stimulus -----------------------

# get indices for stimulus phase series repeats
inx_mean = fs.repeats((start_time, end_time))

# convert repeat start and stop indices to timestamps
repeat_starts = np.array(start_time)[[int(i[0]) for i in inx_mean]]
repeat_stops = np.array(end_time)[[int(i[1]) for i in inx_mean]]

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

# if they are not, cut the ends off (its just a few datapoints so should not matter much)
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
thresh_mean_rois = fs.thresh_correlations(sorted_mean_rois, meanthresh)

# get dffs for active ROIs only for raster (means per stim) and lineplot (interp datapoints)
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


# specify here what you want to plot -------------------------------------------

plot_times = meanstack_t_active_dffs
plot_dffs = meanstack_d_active_dffs
plot_rois = thresh_mean_rois[:, 0]


# Plot rasterplot --------------------------------------------------------------

# plot raster of all dffs
extent = (np.min(plot_times), np.max(plot_times),
          np.min(plot_rois), np.max(plot_rois))

fig, ax = plt.subplots()
ax.imshow(plot_dffs,
          cmap='binary',
          aspect='auto',
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
