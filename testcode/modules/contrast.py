import numpy as np 

class selective_rois:
    def __init__(self, all_rois, corrs, thresh):

        # get active rois
        self.__all_dffs = all_rois.active_mean_dffs 
        self.__all_rois = all_rois.active_mean_rois
        self.__all_recs = all_rois.active_mean_recs
        self.__all_corrs = corrs
        self.thresh = thresh
        
        # do checks
        if len(corrs) != len(self.__all_dffs):
            raise ValueError('The correlations do not match the active dffs!')
        
        if (thresh > 0) & (len(self.__all_corrs[self.__all_corrs>thresh]) == 0):
            raise ValueError('The threshold is too large!')

        if (thresh < 0) & (len(self.__all_corrs[self.__all_corrs<thresh]) == 0):
            raise ValueError('The threshold is too small!')

        # get selective dff data
        self.dffs = self.__all_dffs[self.__all_corrs > self.thresh]
        self.rois = self.__all_rois[self.__all_corrs > self.thresh]
        self.recs = self.__all_recs[self.__all_corrs > self.thresh]
        self.corrs = self.__all_corrs[self.__all_corrs > self.thresh]


class rg_activity:
    def __init__(self, selective_rois, contr1, contr2):

        self.__rois = selective_rois
        self.__contr1 = contr1
        self.__contr2 = contr2
        self.__index = np.arange(len(self.__contr1))
        self.contr1 = np.unique(self.__contr1[~np.isnan(self.__contr1)])
        self.contr2 = np.unique(self.__contr2[~np.isnan(self.__contr2)])
        self.dffs = []
        self.mean_dffs = []
        self.contr1_index = []
        self.contr2_index = []
        self.rois = self.__rois.rois
        self.recs = self.__rois.recs
        
        for c1 in self.contr1:

            self.contr1_index.append(c1)
            idx = self.__index[self.__contr1 == c1]
            cat_dffs = self.__rois.dffs[:,idx]
            # mean_dffs = np.mean(cat_dffs, axis=1)
            self.contr2_index.append(self.__contr2[idx])
            self.dffs.append(cat_dffs)
            # self.mean_dffs.append(mean_dffs)

        self.mean_dffs = np.array(self.mean_dffs)
        self.contr1_index = np.array(self.contr1_index)
        self.contr2_index = np.array(self.contr2_index)