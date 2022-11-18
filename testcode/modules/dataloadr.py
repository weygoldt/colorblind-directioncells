import os

import numpy as np
from IPython import embed
from scipy import interpolate
from tqdm.autonotebook import tqdm
from vxtools.summarize.structure import SummaryFile

from . import functions as fs
from .termcolors import TermColor as tc


class all_rois:

    def __init__(self, SummaryFile, recordings, overwrite=False):

        f = SummaryFile
        self.rec_nos = np.array(recordings)
        self.rec_nos_cached = np.array([])
        self.dataroot = os.path.dirname(f.file_path)

        # check if recordings in file are the same
        if os.path.exists(self.dataroot+'/rec_nos.npy'):
            self.rec_nos_cached = np.load(self.dataroot+'/rec_nos.npy')

        # check if the supplied recordings are already processed on disk
        if np.array_equal(self.rec_nos, self.rec_nos_cached) & (overwrite == False):

            # load all the data from files
            self.rec_nos = np.load(self.dataroot + '/rec_nos.npy')
            self.dffs = np.load(self.dataroot + '/dffs.npy')
            self.zscores = np.load(self.dataroot + '/zscores.npy')
            self.pmean_dffs = np.load(self.dataroot + '/pmean_dffs.npy')
            self.pmean_zscores = np.load(self.dataroot + '/pmean_zscores.npy')
            self.rois = np.load(self.dataroot + '/rois.npy')
            self.recs = np.load(self.dataroot + '/recs.npy')
            self.start_times = np.load(self.dataroot + '/start_times.npy')
            self.stop_times = np.load(self.dataroot + '/stop_times.npy')
            self.pmean_times = np.load(self.dataroot + '/pmean_times.npy')
            self.target_durs = np.load(self.dataroot + '/target_durs.npy')
            self.ang_velocs = np.load(self.dataroot + '/ang_velocs.npy')
            self.ang_periods = np.load(self.dataroot + '/ang_periods.npy')
            self.red = np.load(self.dataroot + '/red.npy')
            self.green = np.load(self.dataroot + '/green.npy')

        # if not the case, recompute and save the stuff
        else:

            interp_dt = 0.1

            all_dffs = []
            all_zscores = []

            self.rois = []
            self.recs = []
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
            self.stop_times = np.cumsum(self.target_durs)
            self.pmean_times = self.stop_times-self.start_times

            # compute dff and zscore means
            self.pmean_dffs = np.array([np.array([np.mean(x) for x in roi_dff])
                                       for roi_dff in self.dffs])

            self.pmean_zscores = np.array([np.array([np.mean(x) for x in roi_zscore])
                                          for roi_zscore in self.zscores])

            # load all the data from files
            np.save(self.dataroot + '/rec_nos.npy', self.rec_nos)
            np.save(self.dataroot + '/dffs.npy', self.dffs)
            np.save(self.dataroot + '/zscores.npy', self.zscores)
            np.save(self.dataroot + '/pmean_dffs.npy', self.pmean_dffs)
            np.save(self.dataroot + '/pmean_zscores.npy', self.pmean_zscores)
            np.save(self.dataroot + '/rois.npy', self.rois)
            np.save(self.dataroot + '/recs.npy', self.recs)
            np.save(self.dataroot + '/start_times.npy', self.start_times)
            np.save(self.dataroot + '/stop_times.npy', self.stop_times)
            np.save(self.dataroot + '/pmean_times.npy', self.pmean_times)
            np.save(self.dataroot + '/target_durs.npy', self.target_durs)
            np.save(self.dataroot + '/ang_velocs.npy', self.ang_velocs)
            np.save(self.dataroot + '/ang_periods.npy', self.ang_periods)
            np.save(self.dataroot + '/red.npy', self.red)
            np.save(self.dataroot + '/green.npy', self.green)

    def repeat_means(self):

        print("")
        print(
            f"{tc.succ('[ roimatrix.repeat_means ]')} Computing means across repeats...")

        self.pmean_rmean_times, \
            self.pmean_rmean_dffs = fs.meanstack(
                self.pmean_dffs, self.pmean_times, self.inx_pmean)

    def responding_rois(self):

        def sort_rois(mean_dffs, inx):
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
            print("")
            for i in tqdm(range(len(mean_dffs[:, 0])), desc=f"{tc.succ('[ roimatrix.sort_means_by_corr ]')} Computing autocorrelation for every dff ..."):

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

        # get indices for stimulus phase series repeats
        self.inx_pmean = fs.repeats(self.pmean_dffs)

        # compute correlation coefficient of ROIs
        self.corrs = sort_rois(self.pmean_dffs, self.inx_pmean)
