import numpy as np
from IPython import embed
from scipy import interpolate
from tqdm.autonotebook import tqdm
from vxtools.summarize.structure import SummaryFile

from . import functions as fs
from .termcolors import TermColor as tc


class all_rois:

    def __init__(self, SummaryFile, recordings):

        f = SummaryFile
        rec_nos = recordings

        index_rois = []
        index_recs = []
        all_dffs = []
        dff_times = []

        self.start_times = []
        self.stop_times = []
        self.ang_velocs = []
        self.ang_periods = []
        self.rgbs_1 = []
        self.rgbs_2 = []

        print("")
        for rec_no in tqdm(rec_nos, desc=f"{tc.succ('[ roimatrix.__init__ ]')} Loading data ..."):
            # get recording data
            one_rec = fs.data_one_rec_id(f, rec_no)  # extract one recording
            times = one_rec[0].times  # get the time axis

            start_time, stop_time, ang_veloc, ang_period, rgb_1, rgb_2 = fs.get_attributes(
                one_rec)

            # reformat time arrays
            start_time = start_time[1:-1]
            stop_time = stop_time[1:-1]
            ang_veloc = ang_veloc[1:-1]
            ang_period = ang_period[1:-1]
            rgb_1 = rgb_1[1:-1]
            rgb_2 = rgb_2[1:-1]

            # shift start and stop time
            new_end_time = stop_time-start_time[0]
            new_start_time = start_time-start_time[0]

            # create new time array
            new_times = np.arange(new_start_time[0], new_end_time[-1]+0.1, 0.1)

            # make empty matrix
            roi_dffs = np.ones((len(one_rec), len(new_times)))

            # collect all dffs for all ROIs in empty matrix ROI dffs interpoliert
            for i, roi in enumerate(one_rec):

                index_rois.append(i)
                index_recs.append(rec_no)

                # get rois from dataset
                dff = roi.dff

                # normalize
                dff = (dff - dff.min()) / (dff.max() - dff.min())

                # interpolate
                finterp = interpolate.interp1d(times-start_time[0], dff)
                dff_interp = finterp(new_times)

                # save into matrix
                roi_dffs[i, :] = dff_interp

            times = new_times
            start_time = new_start_time
            stop_time = new_end_time

            all_dffs.append(roi_dffs)
            dff_times.append(times)

            self.start_times.append(start_time)
            self.stop_times.append(stop_time)

        # get longest time array
        timelens = [len(x) for x in dff_times]
        idx = np.arange(len(timelens))
        times = dff_times[idx[timelens == np.max(timelens)][0]]

        # check stimulus data
        if len(np.unique([len(x) for x in self.start_times])) > 1:
            raise Exception(
                f"{tc.err('ERROR')} Start times mismatch across layers!")

        # concatenate numpy arrays in all_dffs to make one big matrix
        ydim = np.sum([len(x[:, 0]) for x in all_dffs])
        xdim = np.max([len(x[0, :]) for x in all_dffs])
        dffs = np.full((ydim, xdim), np.nan)

        ylen = 0
        for i, roi_dffs in enumerate(all_dffs):
            dims = np.shape(roi_dffs)
            dffs[ylen:ylen+dims[0], : dims[1]] = roi_dffs
            ylen += dims[0]

        self.times = times
        self.dffs = dffs
        self.index_rois = index_rois
        self.index_recs = index_recs
        self.metaindex = np.arange(len(self.dffs[:, 0]))
        self.ang_periods = ang_period
        self.ang_velocs = ang_veloc
        self.rgb_1 = rgb_1
        self.rgb_2 = rgb_2

    def stimulus_means(self):

        recordings = np.unique(self.index_recs)
        recordings_index = np.arange(len(recordings))

        self.mean_dffs = []
        self.mean_times = []

        for i1 in tqdm(recordings_index, desc=f"{tc.succ('[ roimatrix.stimulus_means ]')} Computing means in phases ..."):

            start_times = self.start_times[i1]
            stop_times = self.stop_times[i1]
            snippet_indices = []
            center_indices = []

            for st, end in zip(start_times, stop_times):

                start_inx = fs.find_on_time(self.times, st)
                stop_inx = fs.find_on_time(self.times, end)
                center_inx = fs.find_on_time(self.times, st + (end-st)/2)
                center_indices.append(center_inx)
                snippet_indices.append(np.arange(start_inx, stop_inx))

            dff_idx = self.index_recs == recordings[i1]

            for i2 in range(len(self.dffs[dff_idx, :])):
                roi = self.dffs[dff_idx, :][i2, :]
                mean_dff = np.array([np.mean(roi[snip])
                                     for snip in snippet_indices])
                self.mean_dffs.append(mean_dff)

            self.mean_times.append(self.times[center_indices])

        # convert to numpy array
        self.mean_dffs = np.array(self.mean_dffs)
        self.mean_times = np.mean(self.mean_times, axis=0)

        # print("")

        # for i in tqdm(range(len(self.dffs[:, 0])), desc=f"{tc.succ('[ roimatrix.stimulus_means ]')} Computing means in phases ..."):
        #     roi = self.dffs[i, :]
        #     snippet_indices = meta_snippet_indices[rec_idx]
        #     mean_dff = np.array([np.mean(roi[snip])
        #                         for snip in snippet_indices])
        #     self.mean_dffs[i, :] = mean_dff

        # self.mean_times = self.times[center_indices]

    def repeat_means(self):

        print("")
        print(
            f"{tc.succ('[ roimatrix.repeat_means ]')} Computing means across repeats...")

        self.meanstack_mean_times, \
            self.meanstack_mean_dffs = fs.meanstack(
                self.mean_dffs, self.mean_times, self.inx_mean)

    def sort_means_by_corr(self):

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
        inx_mean = fs.repeats(self.mean_dffs)
        self.inx_mean = inx_mean

        # compute correlation coefficient of ROIs
        self.corrs = sort_rois(self.mean_dffs, self.inx_mean)

        self.metaindex = self.corrs[:, 0]


if __name__ == "__main__":

    # import HDF5 file
    f = SummaryFile(
        '/mnt/archlinux/@home/weygoldt/Data/uni/neuro_gp/calciumimaging/data/Summary.hdf5')
    num_rec = len(f.recordings())
    rec_nos = np.arange(3, num_rec)
    d = all_rois(f, rec_nos)
    d.stimulus_means()
    d.repeat_means()
