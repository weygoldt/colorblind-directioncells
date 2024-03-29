import os
from pathlib import Path

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from matplotlib.patches import Rectangle
from scipy import interpolate
from scipy import signal as fp
from vxtools.summarize.structure import SummaryFile

import modules.functions as fs
from modules.plotstyle import PlotStyle

ps = PlotStyle()
p = Path('../data/data2')
folder_names = np.sort([x.name for x in p.iterdir() if x.is_dir()])
folder_id = np.char.split(folder_names, '_')
recs = [int(i[2][3]) for i in folder_id]
camera_files = sorted(p.glob('*/Camera.hdf5'))[:]
#recs = [0,1,2,3,4]
f = SummaryFile('../data/data2/Summary.hdf5')


def read_hdf5_file(file):

    with h5py.File(file, 'r') as f:
        right_eye_pos = f['eyepos_ang_re_pos_0']
        right_eye_time = f['eyepos_ang_re_pos_0_attr_time']
        left_eye_pos = f['eyepos_ang_le_pos_0']
        left_eye_time = f['eyepos_ang_le_pos_0_attr_time']

        r_eye_pos = np.ravel(right_eye_pos)
        r_eye_time = np.ravel(right_eye_time)
        l_eye_pos = np.ravel(left_eye_pos)
        l_eye_time = np.ravel(left_eye_time)

    return r_eye_pos, r_eye_time, l_eye_pos, l_eye_time


pmean_recs = []
rec_phases = []

for file, _ in enumerate(camera_files):

    one_rec = fs.data_one_rec_id(f, recs[file])
    start_time, stop_time, target_dur,  ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(
        one_rec)
    ri_pos, ri_time, le_pos, le_time = read_hdf5_file(camera_files[file])

    ri_pos_abs = np.abs(np.diff(ri_pos))

    q75, q25 = np.percentile(ri_pos_abs, [75, 25])

    iqr = q75 - q25

    saccade_cut_off = q75+5*iqr
    sacc = fp.find_peaks(ri_pos_abs, height=saccade_cut_off)[0]
    """
    # check for saccade filte is doing his job
    # plt.plot(ri_time[:-1], ri_pos_abs)
    # plt.hlines(saccade_cut_off, xmin=ri_time[1], xmax=ri_time[-1])
    """
    # removing saccades from the velocity of the eye movement to calculate the mean velocity
    interp = 0.2
    einterp = interpolate.interp1d(ri_time, ri_pos)

    velo_right = np.diff(ri_pos)
    # getting rid of the saccades
    velo_right[sacc-1] = 0
    velo_right[sacc+1] = 0
    velo_right[sacc] = 0

    # check for saccads to be dealt with

    fig, ax = plt.subplots()
    #ax.set_xlim(4227.5, 4249.9)
    #ax.set_xticks(np.arange(start_time[100], 4250 , 4.07))
    ax.plot(ri_time[:-1], np.diff(ri_pos))
    ax.plot(ri_time[:-1], velo_right)
    ax.plot(ri_time, ri_pos)
    ax.scatter(ri_time[sacc], np.ones_like(velo_right[sacc])*-3)

    # interplotate to the right time
    times = []
    int_eye = []
    velos = []
    # calculate the the pmean for the velocity
    for st, td in zip(start_time[1:-1], target_dur[1:-1]):
        phase = np.arange(0, td, interp) + st
        try:
            eye_interp = np.array(einterp(phase))
        except:
            embed()
        v = fs.velocity1d(phase, eye_interp)
        int_eye.append(eye_interp)
        times.append(phase)
        velos.append(v)

    rec_phases.append(int_eye)
    mean_vel = np.asarray([abs(np.nanmean(x))for x in velos])
    pmean_recs.append(mean_vel)


pmean_recs = np.asarray(pmean_recs)

motion = np.asarray([1 if x != 0 else 0 for x in ang_veloc], dtype=float)
motion[motion == 0] = np.nan

red_contr_chroma = []
green_contr_chroma = []
#iterate across channels and append chromatic contrasts
for rgb1, rgb2 in zip(rgb_1[1:-1], rgb_2[1:-1]):
    if rgb2 == 0:
        red_contr_chroma.append(rgb1)
    else:
        red_contr_chroma.append(np.nan)
    if rgb1 == 0:
        green_contr_chroma.append(rgb2)
    else:
        green_contr_chroma.append(np.nan)

red_contr_chroma = np.array(red_contr_chroma)
green_contr_chroma = np.array(green_contr_chroma)
uniq_conr = np.unique(green_contr_chroma[~np.isnan(green_contr_chroma)]) 

eye_mean = []
for c in uniq_conr:
    contrast = []
    for i in range(len(pmean_recs[:, 0])):
        x = pmean_recs[i,:][red_contr_chroma==c]
        contrast.append(x)
    eye_mean.append(contrast)

eye_mean_mean = np.mean(eye_mean, axis=1)

for i in range(len(uniq_conr)):
    plt.scatter( np.ones_like(eye_mean_mean[i])*uniq_conr[i],  eye_mean_mean[i] )

plt.show()


clock = np.array([1 if x > 0 else np.nan  for x in ang_veloc])
cclock = np.array([1 if x < 0 else np.nan  for x in ang_veloc])
acromatic_c = abs(np.array(rgb_1) - np.array(rgb_2)) [1:-1] * clock[1:-1]
acromatic_cc = abs(np.array(rgb_1) - np.array(rgb_2)) [1:-1] * cclock[1:-1]

uniq_acromatic = np.unique(acromatic_c)
eye_mean = []
for c in uniq_acromatic:
    contrast = []
    for i in range(len(pmean_recs[:, 0])):
        x = pmean_recs[i,:][acromatic_c==c]
        contrast.append(x)

    eye_mean.append(contrast)

eye_mean_mean = np.mean(np.array(eye_mean, dtype='object'), axis=1)

for i in range(len(uniq_acromatic)):
    plt.scatter( np.ones_like(eye_mean_mean[i])*uniq_acromatic[i],  eye_mean_mean[i] )

plt.show()
embed()
exit()




red_contr = np.array(rgb_1)
green_contr = np.array(rgb_2)

red_motion_stim = red_contr * motion
green_motion_stim = green_contr * motion


class rg_activity:
    def __init__(self, mean_eye, contr1, contr2):

        self.__contr1 = contr1
        self.__contr2 = contr2
        self.__index = np.arange(len(self.__contr1))
        self.contr1 = np.unique(self.__contr1[~np.isnan(self.__contr1)])
        self.contr2 = np.unique(self.__contr2[~np.isnan(self.__contr2)])
        self.contr1_index = []
        self.contr2_index = []
        self.eye_dff = []

        for c1 in self.contr1:
            self.contr1_index.append(c1)
            idx = self.__index[self.__contr1 == c1]
            pmean_contrast = np.array(mean_eye)[:, idx]
            self.contr2_index.append(self.__contr2[idx])
            self.eye_dff.append(pmean_contrast)

        self.contr1_index = np.array(self.contr1_index)
        self.contr2_index = np.array(self.contr2_index)

rg_motion_data = rg_activity(
    pmean_recs, red_motion_stim[1:-1], green_motion_stim[1:-1])

gr_motion_data = rg_activity(
    pmean_recs, green_motion_stim[1:-1], red_motion_stim[1:-1],)

# ------------------------- THE PLOT -----------------------------------#

fig = plt.figure(figsize=(30*ps.cm, 20*ps.cm))

# build the subfigures
(subfig_l, subfig_r) = fig.subfigures(1, 2, hspace=0.05, width_ratios=[1, 1])

gs0 = gridspec.GridSpec(2, 3, figure=subfig_l)
stim_axs1 = [subfig_l.add_subplot(i) for i in gs0]

gs1 = gridspec.GridSpec(2, 3, figure=subfig_r)
stim_axs2 = [subfig_r.add_subplot(i) for i in gs1]

# plot the left side of the plot
colors = [ps.red]

for color, rg_clock in zip(colors, [rg_motion_data]):
    for i1, rds in enumerate(rg_clock.contr1):
        zscores = []
        contr = []
        for dff in np.array(rg_clock.eye_dff)[rg_clock.contr1 == rds]:

            for i in range(len(dff[:, 0])):
                eye_dff = dff[i, :]
                # the data
                contrsts2 = np.array(rg_clock.contr2_index)[
                    rg_clock.contr1 == rds][0]

                # sort the data
                sort_roi_zscores = eye_dff[np.argsort(contrsts2)]
                sort_contrsts2 = contrsts2[np.argsort(contrsts2)]

                zscores.append(sort_roi_zscores)
                contr.append(sort_contrsts2)

        # compute means for each category
        contrs = np.unique(fs.flatten(contr))
        meanzscore = []
        stdzscore = []
        for c2 in contrs:
            meanzscore.append(
                np.mean(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))
            stdzscore.append(
                np.std(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))

        meanzscore = np.asarray(meanzscore)
        stdzscore = np.asarray(stdzscore)

        stim_axs1[i1].axhline(0, color="darkgray", lw=1, ls="dashed")
        stim_axs1[i1].plot(contrs, meanzscore,
                           c=color, lw=2.5, zorder=100)
        stim_axs1[i1].fill_between(
            contrs, meanzscore-stdzscore, meanzscore+stdzscore, color=color, lw=1, alpha=0.2, label="_")
        stim_axs1[i1].set_title(f"red: {np.round(rds,2)}")

# plot the right side of the plot
colors = [ps.green]
for color, gr_clock in zip(colors, [gr_motion_data]):
    for i1, rds in enumerate(gr_clock.contr1):
        zscores = []
        contr = []
        for dff in np.array(gr_clock.eye_dff)[gr_clock.contr1 == rds]:
            for i in range(len(dff[:, 0])):
                eye_dff = dff[i, :]
                # the data
                contrsts2 = np.array(rg_clock.contr2_index)[
                    rg_clock.contr1 == rds][0]

                # sort the data
                sort_roi_zscores = eye_dff[np.argsort(contrsts2)]
                sort_contrsts2 = contrsts2[np.argsort(contrsts2)]

                zscores.append(sort_roi_zscores)
                contr.append(sort_contrsts2)

        # compute means for each category
        contrs = np.unique(fs.flatten(contr))
        meanzscore = []
        stdzscore = []
        for c2 in contrs:
            meanzscore.append(
                np.mean(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))
            stdzscore.append(
                np.std(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))

        meanzscore = np.asarray(meanzscore)
        stdzscore = np.asarray(stdzscore)

        stim_axs2[i1].axhline(0, color="darkgray", lw=1, ls="dashed")
        stim_axs2[i1].plot(contrs, meanzscore, 
                           c=color, lw=2.5, zorder=100)
        stim_axs2[i1].fill_between(
            contrs, meanzscore-stdzscore, meanzscore+stdzscore, color=color, lw=1, alpha=0.2, label="_")
        stim_axs2[i1].set_title(f"green: {np.round(rds,2)}")

# remove axes on right and top of plots
[x.spines["right"].set_visible(False) for x in np.asarray(stim_axs1)]
[x.spines["top"].set_visible(False) for x in np.asarray(stim_axs1)]
[x.spines["right"].set_visible(False) for x in np.asarray(stim_axs2)]
[x.spines["top"].set_visible(False) for x in np.asarray(stim_axs2)]

# turn y axes off where not needed
yaxes_off = [1, 2, 4, 5]
[x.get_yaxis().set_visible(False) for x in np.asarray(stim_axs1)[yaxes_off]]
[x.get_yaxis().set_visible(False) for x in np.asarray(stim_axs2)[yaxes_off]]
[x.spines["left"].set_visible(False) for x in np.asarray(stim_axs1)[yaxes_off]]
[x.spines["left"].set_visible(False) for x in np.asarray(stim_axs2)[yaxes_off]]

# turn x axes off where they are not needed
xaxes_off = [0, 1, 2]
[x.get_xaxis().set_visible(False) for x in np.asarray(stim_axs1)[xaxes_off]]
[x.get_xaxis().set_visible(False) for x in np.asarray(stim_axs2)[xaxes_off]]
[x.spines["bottom"].set_visible(False)
 for x in np.asarray(stim_axs1)[xaxes_off]]
[x.spines["bottom"].set_visible(False)
 for x in np.asarray(stim_axs2)[xaxes_off]]

# set all axes to same tick ranges
x_range = np.arange(0,1.5,0.5)
#y_range = np.arange(-1,2.1,1)
#[x.set_yticks(y_range) for x in np.asarray(stim_axs1)]
[x.set_xticks(x_range) for x in np.asarray(stim_axs1)]
#[x.set_yticks(y_range) for x in np.asarray(stim_axs2)]
[x.set_xticks(x_range) for x in np.asarray(stim_axs2)]

# set bounds to make axes nicer
x_bounds = (0,1)
#y_bounds = (-1,2)
#[x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs1)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs1)]
#[x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs2)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs2)]

# make axes shared
stim_axs1[0].get_shared_x_axes().join(stim_axs1[0], *stim_axs1)
stim_axs1[0].get_shared_y_axes().join(stim_axs1[0], *stim_axs1)
stim_axs2[0].get_shared_x_axes().join(stim_axs2[0], *stim_axs2)
stim_axs2[0].get_shared_y_axes().join(stim_axs2[0], *stim_axs2)

# add labels
subfig_l.supylabel('Mean Velocity ')
subfig_l.supxlabel('green contrast')
subfig_r.supxlabel('red contrast')

# get legend handles
handles1, labels1 = stim_axs1[i1].get_legend_handles_labels()
handles2, labels2 = stim_axs2[i1].get_legend_handles_labels()

# add legends
subfig_l.legend(handles1, labels1, loc='upper center', ncol=2)
subfig_r.legend(handles2, labels2, loc='upper center', ncol=2)

# adjust margins
plt.subplots_adjust(left=0.175, right=0.975, top=0.87,
                    bottom=0.1, hspace=0.17, wspace=0.15)
# fs.doublesave("../plots/eye_contrast")

plt.show()
