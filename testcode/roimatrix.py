import numpy as np
from scipy import interpolate
from tqdm import tqdm
from vxtools.summarize.structure import SummaryFile

import functions as fs
from termcolors import TermColor as tc


class roimatrix:

    def __init__(self, SummaryFile, recordings):

        print(f"{tc.succ('[ roimatrix.__init__ ]')} Loading data ...")

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

        for rec_no in tqdm(rec_nos):

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

            # collect all dffs for all ROIs in empty matrix
            index_roi = []
            index_rec = []
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

            index_rois.extend(index_roi)
            index_recs.extend(index_rec)

            all_dffs.append(roi_dffs)
            dff_times.append(times)

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
        self.start_times = start_time
        self.stop_times = stop_time
        self.ang_velocs = ang_veloc
        self.ang_periods = ang_period
        self.rgb_1 = rgb_1
        self.rgb_2 = rgb_2

    def mean_matrix(self):

        snippet_indices = []
        center_indices = []
        for st, end in zip(self.start_times, self.stop_times):
            start_inx = fs.find_on_time(self.times, st)
            stop_inx = fs.find_on_time(self.times, end)
            center_inx = fs.find_on_time(self.times, st + (end-st)/2)
            center_indices.append(center_inx)
            snippet_indices.append(np.arange(start_inx, stop_inx))

        self.mean_dffs = np.full(
            (len(self.dffs[:, 0]), len(snippet_indices)), np.nan)

        print(f"{tc.succ('[ roimatrix.mean_matrix ]')} Computing means ...")
        for i in tqdm(range(len(self.dffs[:, 0]))):
            roi = self.dffs[i, :]
            mean_dff = np.array([np.mean(roi[snip])
                                 for snip in snippet_indices])
            self.mean_dffs[i, :] = mean_dff
        self.mean_times = self.times[center_indices]


if __name__ == "__main__":

    f = SummaryFile('../data/Summary.hdf5')   # import HDF5 file
    num_rec = len(f.recordings())
    rec_nos = np.arange(3, num_rec)
    d = roimatrix(f, rec_nos)
    d.mean_matrix()
