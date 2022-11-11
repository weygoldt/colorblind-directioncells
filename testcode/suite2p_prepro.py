# %% Essentials ----------------------------------------------------------------

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


# %% Working with and converting start and stop of stimulus --------------------

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
inx = fs.repeats((start_time, end_time))

# convert repeat start and stop indices to timestamps
repeat_starts = np.array(start_time)[[int(i[0]) for i in inx]]
repeat_stops = np.array(end_time)[[int(i[1]) for i in inx]]

# convert repeat start and stop timestamp to indices on full time vector
repeat_starts_idx = [fs.find_on_time(times, x) for x in repeat_starts]
repeat_stops_idx = [fs.find_on_time(times, x) for x in repeat_stops]


# %% Getting actual dff data ---------------------------------------------------

# make matrix of dffs for this roi
roi_dffs = np.empty((len(one_rec), len(one_rec[0].dff[stimstart:stimstop])))
for i, roi in enumerate(one_rec):
    roi_dffs[i, :] = roi.dff[stimstart:stimstop]

# convert dff matrix to mean dff matrix
mean_dffs, mean_times = fs.get_mean_dffs(
    roi_dffs, times, (start_time, end_time))


# %% Compute corrcoeffs and threshold them -------------------------------------

# compute correlation coefficient of thresholded active ROIs
sorted_mean_rois = fs.sort_rois(mean_dffs, inx)
sorted_rois = fs.sort_rois(roi_dffs, inx)

# threshold them
thresh_rois = fs.thresh_correlations(sorted_rois, thresh)

# get dffs for active ROIs only
active_dffs = np.array([roi_dffs[int(roi), :] for roi in thresh_rois[:, 0]])


# %% Make a meanstack ----------------------------------------------------------

# get dffs for all ROIs
sorted_dffs = np.array([roi_dffs[int(roi), :] for roi in sorted_rois[:, 0]])

# stack 3 repeats and compute mean
meanstack_times, meanstack_dffs = fs.meanstack(active_dffs, mean_times, inx)


# %% Plot rasterplot -----------------------------------------------------------

# plot raster of all dffs
extent = (np.min(times), np.max(times),
          0, len(roi_dffs[:, 0])-1)

fig, ax = plt.subplots()
ax.imshow(roi_dffs,
          cmap='binary',
          aspect='auto',
          extent=extent,
          )
for t1, t2 in zip(repeat_starts, repeat_stops):
    ax.axvline(t1, color="blue")
    ax.axvline(t2, color="green")
plt.show()


# %% Plot stacked lineplot -----------------------------------------------------

# plot dff lineplot for all dffs
fig, ax = plt.subplots()
for i, dff in enumerate(roi_dffs):
    ax.plot(times, i + (dff - dff.min()) / (dff.max() - dff.min()),
            color='black', linewidth='1.')
plt.show()
