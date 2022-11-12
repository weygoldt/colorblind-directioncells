# Essentials -------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
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


# Working with and converting start and stop of stimulus -----------------------

# remove first and last start time because baseline
start_time = start_time[1:-1]
end_time = end_time[1:-1]

# convert stim onset timestamps to indices
stimstart = fs.find_on_time(times, start_time[0])
stimstop = fs.find_on_time(times, end_time[-1])

# rewrite start and stop times (this will end in chaos!!!)
times = times[stimstart:stimstop]
start_time[0] = times[0]
end_time[-1] = times[-1]

# get indices for stimulus phase series repeats
inx_mean = fs.repeats((start_time, end_time))

# convert repeat start and stop indices to timestamps
repeat_starts = np.array(start_time)[[int(i[0]) for i in inx_mean]]
repeat_stops = np.array(end_time)[[int(i[1]) for i in inx_mean]]

# convert repeat start and stop timestamp to indices on full time vector
repeat_starts_idx = [fs.find_on_time(times, x) for x in repeat_starts]
repeat_stops_idx = [fs.find_on_time(times, x) for x in repeat_stops]

# reformat to fit inx shape
inx_full = np.array([[x, y]
                    for x, y in zip(repeat_starts_idx, repeat_stops_idx)])


# Getting actual dff data ------------------------------------------------------

# make matrix of dffs for this roi
roi_dffs = np.empty((len(one_rec), len(one_rec[0].dff[stimstart:stimstop])))
for i, roi in enumerate(one_rec):
    roi_dffs[i, :] = roi.dff[stimstart:stimstop]

# convert dff matrix to mean dff matrix
mean_dffs, mean_times = fs.get_mean_dffs(
    roi_dffs, times, (start_time, end_time))


# Compute corrcoeffs and threshold them ----------------------------------------

# compute correlation coefficient of thresholded active ROIs
sorted_mean_rois = fs.sort_rois(mean_dffs, inx_mean)
sorted_rois = fs.sort_rois(roi_dffs, inx_full)

# get dffs for all ROIs
sorted_dffs = np.array([roi_dffs[int(roi), :] for roi in sorted_rois[:, 0]])

# threshold them based on correlation coefficient
thresh_mean_rois = fs.thresh_correlations(sorted_mean_rois, meanthresh)
thresh_rois = fs.thresh_correlations(sorted_rois, thresh)

# get dffs for active ROIs only
active_mean_dffs = np.array([mean_dffs[int(roi), :]
                            for roi in thresh_mean_rois[:, 0]])
active_dffs = np.array([roi_dffs[int(roi), :]
                        for roi in thresh_rois[:, 0]])


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
plot_rois = thresh_rois[:, 0]


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

for t1, t2 in zip(repeat_starts, repeat_stops):
    ax.axvline(t1, color="blue")
    ax.axvline(t2, color="green")
plt.show()


# Plot stacked lineplot --------------------------------------------------------

# plot dff lineplot for all dffs
fig, ax = plt.subplots()
for i, dff in enumerate(plot_dffs):
    ax.plot(plot_times, i + (dff - dff.min()) / (dff.max() - dff.min()),
            color='black', linewidth='1.')
plt.show()
