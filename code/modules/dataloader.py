import os
from pathlib import Path

import h5py
import numpy as np
from IPython import embed
from scipy import interpolate
from scipy import signal as fp
from tqdm.autonotebook import tqdm
from vxtools.summarize.structure import SummaryFile

from . import functions as fs
from .termcolors import TermColor as tc


class SingleFish:

    def __init__(self, SummaryFile, recordings, overwrite=False, behav=False):

        self.behav = behav

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

        f = SummaryFile
        self.fish_id = f.rois()[0].fish_id

        self.rec_nos = np.array(recordings)
        self.rec_nos_cached = np.array([])
        self.dataroot = os.path.dirname(f.file_path)
        self.type = ["raw", ]

        # check if recordings in file are the same
        if os.path.exists(self.dataroot+'/rec_nos.npy'):
            self.rec_nos_cached = np.load(self.dataroot+'/rec_nos.npy')

        # check if the supplied recordings are already processed on disk
        if np.array_equal(self.rec_nos, self.rec_nos_cached) & (overwrite == False):

            # load all the data from files
            self.rec_nos = np.load(self.dataroot + '/rec_nos.npy')
            self.dffs = np.load(self.dataroot + '/dffs.npy')
            self.zscores = np.load(self.dataroot + '/zscores.npy')
            self.rois = np.load(self.dataroot + '/rois.npy')
            self.recs = np.load(self.dataroot + '/recs.npy')
            self.fish = np.load(self.dataroot + '/fish.npy')
            self.times = np.load(self.dataroot + '/times.npy')
            self.start_times = np.load(self.dataroot + '/start_times.npy')
            self.stop_times = np.load(self.dataroot + '/stop_times.npy')
            self.target_durs = np.load(self.dataroot + '/target_durs.npy')
            self.ang_velocs = np.load(self.dataroot + '/ang_velocs.npy')
            self.ang_periods = np.load(self.dataroot + '/ang_periods.npy')
            self.red = np.load(self.dataroot + '/red.npy')
            self.green = np.load(self.dataroot + '/green.npy')
            if behav == True:
                self.eye_velocs = np.load(self.dataroot + '/eye_velocs.npy')

        # if not the case, recompute and save the stuff
        else:

            # collect camera files
            if behav == True:
                p = Path(self.dataroot)
                folder_names = np.sort(
                    [x.name for x in p.iterdir() if x.is_dir()])
                folder_id = np.char.split(folder_names, '_')

                recs = np.asarray([int(i[2][3]) for i in folder_id])
                camera_files = np.array(sorted(p.glob('*/Camera.hdf5')))

                rec_velos = []

                # get the data
                for file in range(len(camera_files)):
                    r_eye_pos, r_eye_time, l_eye_pos, l_eye_time = read_hdf5_file(
                        camera_files[file])

                    one_rec = fs.data_one_rec_id(f, recs[file])

                    start_time, stop_time, target_dur, ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(
                        one_rec)

                    interp = 0.05
                    einterp = interpolate.interp1d(r_eye_time, r_eye_pos)
                    int_eye = []
                    times = []
                    velos = []
                    t_interp = []

                    # interpolate velocity in phases
                    for st, td in zip(start_time[1:-1], target_dur[1:-1]):
                        phase = np.arange(0, td, interp) + st
                        eye_interp = np.array(einterp(phase))
                        v = fs.velocity1d(phase, eye_interp)
                        int_eye.append(eye_interp)
                        times.append(phase)
                        t_interp.append(phase[1:-1])
                        velos.append(v)

                    # extrapolate empty values
                    extrap = interpolate.interp1d(
                        np.ravel(t_interp), np.ravel(velos), fill_value='extrapolate')
                    full_velos = extrap(np.ravel(times))
                    times = np.asarray(np.ravel(times))

                    # find peaks (i.e. scaccades)
                    sacc_test = fp.find_peaks(abs(full_velos), prominence=6)[0]
                    saccs_idx = np.array(fs.flatten([np.arange(x-2, x+2) if (x > 2) &
                                                    (x < sacc_test[-2]) else [x] for x in sacc_test]))

                    # remove saccades froma rray
                    new_velo = np.delete(full_velos, saccs_idx)
                    new_times = np.delete(times, saccs_idx)

                    # prepare interpolating the missing saccades
                    sacc_interpolator = interpolate.interp1d(
                        new_times, new_velo, fill_value='extrapolate')
                    times = []
                    velos = []

                    # interpolate the missing saccades
                    for st, td in zip(start_time[1:-1], target_dur[1:-1]):
                        phase = np.arange(0, td, interp) + st
                        eye_veloc = sacc_interpolator(phase)
                        times.append(phase)
                        velos.append(np.abs(eye_veloc))

                    velos_norm = [(np.array(x) - np.min(velos)) /
                                  (np.max(velos) - np.min(velos)) for x in velos]

                    velos_norm = np.asarray(velos_norm)

                    rec_velos.append(velos_norm)

                self.eye_velocs = np.asarray(rec_velos)

            interp_dt = 0.1

            all_dffs = []
            all_zscores = []

            self.rois = []
            self.recs = []
            self.fish = []
            self.start_times = []
            self.stop_times = []
            self.target_durs = []
            self.ang_velocs = []
            self.ang_periods = []
            self.red = []
            self.green = []

            for rec_no in tqdm(self.rec_nos, desc=f"{tc.succ('[ roimatrix.__init__ ]')} Loading data ..."):

                # get recording data
                one_rec = fs.data_one_rec_id(
                    f, rec_no)  # extract one recording
                times = one_rec[0].times                 # get the time axis
                start_time, stop_time, target_dur, ang_veloc, ang_period, red, green = fs.get_attributes(
                    one_rec)

                # reformat time arrays
                start_time = start_time[1:-1]
                stop_time = stop_time[1:-1]
                target_dur = target_dur[1:-1]
                ang_veloc = ang_veloc[1:-1]
                ang_period = ang_period[1:-1]
                red = red[1:-1]
                green = green[1:-1]

                # make empty matrix
                rois_dffs = []
                rois_zscores = []

                self.red.append(red)
                self.green.append(green)
                self.ang_velocs.append(ang_veloc)
                self.ang_periods.append(ang_period)
                self.target_durs.append(target_dur)

                # make dffs
                for i, roi in enumerate(one_rec):

                    self.rois.append(i)
                    self.recs.append(rec_no)
                    self.fish.append(self.fish_id)

                    # get rois from dataset
                    dff = roi.dff
                    zscore = roi.zscore

                    # normalize
                    dff = (dff - dff.min()) / (dff.max() - dff.min())

                    # create interpolate function
                    dffinterp = interpolate.interp1d(times, dff)
                    zscinterp = interpolate.interp1d(times, zscore)

                    # interpolate for each phase
                    roi_dffs = []
                    roi_zscores = []

                    for st, td in zip(start_time, target_dur):

                        phase_t = np.arange(0, td, interp_dt) + st
                        dff_interp = np.array(dffinterp(phase_t))
                        zscore_interp = np.array(zscinterp(phase_t))

                        roi_dffs.append(dff_interp)
                        roi_zscores.append(zscore_interp)

                    # check if all are same len now
                    if len(np.unique([len(x) for x in roi_dffs])) != 1:
                        raise Exception(
                            'Dffs for all phases are not the same length!')
                    else:
                        rois_dffs.append(roi_dffs)
                        rois_zscores.append(roi_zscores)

                all_dffs.extend(rois_dffs)
                all_zscores.extend(rois_zscores)

            # convert to numpy arrays
            self.red = np.array(self.red)
            self.green = np.array(self.green)
            self.ang_velocs = np.array(self.ang_velocs)
            self.ang_periods = np.array(self.ang_periods)
            self.target_durs = np.array(self.target_durs)
            self.rois = np.array(self.rois)
            self.recs = np.array(self.recs)

            match = True

            # check if all stimulus arrays between recordings contain same stim data
            for i in range(len(self.red[0, :])):
                if len(np.unique(self.red[:, i])) != 1:
                    match = False
                    print(tc.err('Check out reds!'))

            for i in range(len(self.green[0, :])):
                if len(np.unique(self.green[:, i])) != 1:
                    match = False
                    print(tc.err('Check out greens!'))

            for i in range(len(self.ang_velocs[0, :])):
                if len(np.unique(self.ang_velocs[:, i])) != 1:
                    match = False
                    print(tc.err('Check out ang_velocs!'))

            for i in range(len(self.ang_periods[0, :])):
                if len(np.unique(self.ang_periods[:, i])) != 1:
                    match = False
                    print(tc.err('Check out ang_periods!'))

            for i in range(len(self.target_durs[0, :])):
                if len(np.unique(self.target_durs[:, i])) != 1:
                    match = False
                    print(tc.err('Check out target durations!'))

            # check if rois and recs match the number of dffs
            if len(all_dffs) != len(self.rois):
                match = False
                print(tc.err('Dff matrix dimensions does not match number of rois!'))

            if len(all_dffs) != len(self.recs):
                match = False
                print(tc.err('Dff matrix dimensions does not match number of recs!'))

            if match:
                self.red = self.red[0]
                self.green = self.green[0]
                self.ang_velocs = self.ang_velocs[0]
                self.ang_periods = self.ang_periods[0]
                self.target_durs = self.target_durs[0]
            else:
                raise Exception(
                    f"{tc.err('ERROR')} The stimulus data does not match!")

            # make data arrays
            self.times = np.arange(0, np.sum(self.target_durs), interp_dt)
            self.dffs = np.array(all_dffs)
            self.zscores = np.array(all_zscores)

            # make start and stop times
            self.start_times = np.cumsum(self.target_durs) - self.target_durs
            self.stop_times = np.cumsum(self.target_durs) - interp_dt

            # load all the data from files
            np.save(self.dataroot + '/rec_nos.npy', self.rec_nos)
            np.save(self.dataroot + '/dffs.npy', self.dffs)
            np.save(self.dataroot + '/zscores.npy', self.zscores)
            np.save(self.dataroot + '/rois.npy', self.rois)
            np.save(self.dataroot + '/recs.npy', self.recs)
            np.save(self.dataroot + '/fish.npy', self.fish)
            np.save(self.dataroot + '/times.npy', self.times)
            np.save(self.dataroot + '/start_times.npy', self.start_times)
            np.save(self.dataroot + '/stop_times.npy', self.stop_times)
            np.save(self.dataroot + '/target_durs.npy', self.target_durs)
            np.save(self.dataroot + '/ang_velocs.npy', self.ang_velocs)
            np.save(self.dataroot + '/ang_periods.npy', self.ang_periods)
            np.save(self.dataroot + '/red.npy', self.red)
            np.save(self.dataroot + '/green.npy', self.green)
            if behav == True:
                np.save(self.dataroot + '/eye_velocs.npy', self.eye_velocs)


class MultiFish:
    def __init__(self, fishes):

        # track class operations here
        self.type = ["raw"]

        # stim data should be the same
        self.times = fishes[0].times
        self.start_times = fishes[0].start_times
        self.stop_times = fishes[0].stop_times
        self.target_durs = fishes[0].target_durs
        self.ang_velocs = fishes[0].ang_velocs
        self.ang_periods = fishes[0].ang_periods
        self.red = fishes[0].red
        self.green = fishes[0].green

        # get roi data
        self.rois = np.array(fs.flatten([x.rois for x in fishes]))
        self.recs = np.array(fs.flatten([x.recs for x in fishes]))
        self.fish = np.array(fs.flatten([x.fish for x in fishes]))

        # get data
        all_dffs = [fish.dffs for fish in fishes]
        all_zscores = [fish.zscores for fish in fishes]
        all_eye_velocs = [fish.eye_velocs for fish in fishes if fish.behav]

        self.dffs = np.concatenate(all_dffs)
        self.zscores = np.concatenate(all_zscores)
        try:
            self.eye_velocs = np.concatenate(all_eye_velocs)
        except:
            pass

    def phase_means(self):

        print("")
        print(
            f"{tc.succ('[ MutliFish.phase_means ]')} Computing means across phases ...")

        # make new time
        self.times = self.stop_times-self.target_durs/2

        # make new dffs
        self.dffs = np.asarray([np.array([np.mean(x) for x in roi_dff])
                                for roi_dff in self.dffs])

        # make new zscores
        self.zscores = np.asarray([np.array([np.mean(x) for x in roi_zscore])
                                   for roi_zscore in self.zscores])
        try:
            self.eye_velocs = np.asarray([np.array([np.max(x) for x in eye_v])
                                      for eye_v in self.eye_velocs])
        except:
            pass

        self.type.append("phase_means")

    def repeat_means(self, nrepeats):

        print("")
        print(
            f"{tc.succ('[ MultiFish.repeat_means ]')} Computing means across repeats ...")

        # embed()

        repeats_on_time, repeats_on_stim = self.repeat_indices(nrepeats)

        rindex = repeats_on_time
        start_idxs = [x[0] for x in repeats_on_stim]
        stop_idxs = [x[1] for x in repeats_on_stim]

        # make actual repeat stack
        newtimes = self.times[rindex[0][0]: rindex[0][1]]
        newdffs = []
        newzscores = []

        for roi in range(len(self.dffs[:, 0])):

            # extract single dff track
            dff = self.dffs[roi, :]
            zscore = self.zscores[roi, :]

            # split into 3 repeats
            split_dff = np.asarray([dff[x:y+1]
                                    for x, y in zip(start_idxs, stop_idxs)])
            split_zscore = np.asarray([zscore[x:y+1]
                                       for x, y in zip(start_idxs, stop_idxs)])

            # compute mean of repeats
            mean_dff = np.mean(split_dff, axis=0)
            mean_zscore = np.mean(split_zscore, axis=0)

            # append into nan matrix
            newdffs.append(mean_dff)
            newzscores.append(mean_zscore)

        self.red = self.red[repeats_on_stim[0][0]:repeats_on_stim[0][1] + 1]
        self.green = self.green[repeats_on_stim[0]
                                [0]:repeats_on_stim[0][1] + 1]
        self.ang_velocs = self.ang_velocs[repeats_on_stim[0]
                                          [0]:repeats_on_stim[0][1] + 1]
        self.ang_periods = self.ang_periods[repeats_on_stim[0]
                                            [0]:repeats_on_stim[0][1] + 1]
        self.dffs = np.asanyarray(newdffs)
        self.zscores = np.asarray(newzscores)
        self.times = newtimes
        self.type.append("repeat_means")

    def responding_rois(self, dffs, nrepeats=3):

        def sorter(mean_dffs, inx):
            """calculate all active ROIs with a threshold.
            ROIs who have a high correlation with themselfs over time, are active rois
            Parameters
            ----------
            one_recording : list of vxtools.summarize.structure.Roi
                hdf5 SummaryFile with all rois of the same recording id

            inx : tupel
                index tupel, where the repeats of one recording starts and stops

            threshold : float, optional
                threshold of the correlation factor, by default 0.6

            Returns
            -------
            2d array
                1.dimension are the index fot the ROIs
                2.dimension are sorted correlation factors
            """

            spearmeans = []

            for i in range(len(mean_dffs[:, 0])):
                # start_time = time.time()
                means = mean_dffs[i, :]

                # start_time = time.time()
                spear_mean = fs.corr_repeats(means, inx)

                spearmeans.append(spear_mean)

            result = np.empty((len(mean_dffs[:, 0]), 2))
            result[:, 0] = np.arange(len(mean_dffs[:, 0]))
            result[:, 1] = spearmeans
            active_rois_sorted = np.array(
                sorted(result, key=lambda x: x[1], reverse=True))

            return active_rois_sorted

        repeats_on_time, repeats_on_stim = self.repeat_indices(nrepeats)

        if 'phase_means' in self.type:
            sort_indices = repeats_on_stim
        else:
            sort_indices = repeats_on_time

        # compute correlation coefficient of ROIs
        result = sorter(dffs, sort_indices)
        indices = np.asarray(result[:, 0], dtype=int)
        corrs = np.asarray(result[:, 1], dtype=float)

        return indices, corrs

    def filter_rois(self, sortindex):

        self.dffs = self.dffs[sortindex]
        self.zscores = self.zscores[sortindex]
        self.rois = self.rois[sortindex]
        self.recs = self.recs[sortindex]
        self.fish = self.fish[sortindex]

        self.type.append("filter_rois")

    def filter_phases(self, sortindex):

        # sort stimulus
        self.ang_velocs = self.ang_velocs[sortindex]
        self.ang_periods = self.ang_periods[sortindex]
        self.red = self.red[sortindex]
        self.green = self.green[sortindex]

        # sort data
        self.dffs = self.dffs[:, sortindex]
        self.zscores = self.zscores[:, sortindex]
        self.eye_velocs = self.eye_velocs[:, sortindex]

        self.type.append("filter_phases")

    def repeat_indices(self, nrepeats):

        # embed()

        indices = np.arange(len(self.start_times))
        frac = len(indices)/nrepeats

        # check if array lengths are dividable by nrepeats
        if frac % 1 != 0:
            raise ValueError(
                f'{tc.err("ERROR")} [ functions.repeats ] Cant divide by {nrepeats}!')

        # get starts and stops indices timestamps
        start_idxs = np.full(nrepeats, 0, dtype=int)
        stop_idxs = np.full(nrepeats, 0, dtype=int)
        start_ts = np.full(nrepeats, np.nan)
        stop_ts = np.full(nrepeats, np.nan)

        # get starts and stops timestamps from class start and stop times
        for i in range(nrepeats):
            start_idxs[i] = int(i*frac)
            stop_idxs[i] = int(np.arange(i*frac, i*frac+frac)[-1])
            start_ts[i] = self.start_times[start_idxs[i]]
            stop_ts[i] = self.stop_times[stop_idxs[i]]

        # find on time
        rindex = []
        for x, y in zip(start_ts, stop_ts):
            rindex.append([
                fs.find_on_time(self.times, x),
                fs.find_on_time(self.times, y)])

        repeats_on_time = rindex
        repeats_on_stim = [[x, y]
                           for x, y in zip(start_idxs, stop_idxs)]

        return repeats_on_time, repeats_on_stim
