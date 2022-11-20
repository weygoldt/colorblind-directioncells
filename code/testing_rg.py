import numpy as np
import matplotlib.pyplot as plt
import modules.functions as fs
import matplotlib.gridspec as gridspec
from IPython import embed
from copy import deepcopy
from vxtools.summarize.structure import SummaryFile
from modules.dataloader import MultiFish, SingleFish
from modules.plotstyle import PlotStyle
from sklearn.metrics import auc
from scipy.stats import pearsonr
from modules.contrast import selective_rois_trash, rg_activity_trash, phase_activity
ps = PlotStyle()

# now load the data
data1 = '../data/data1/'
data2 = '../data/data2/'
data3 = '../data/data3/'

f1 = SummaryFile(data1 + 'Summary.hdf5')
f2 = SummaryFile(data2 + 'Summary.hdf5')
f3 = SummaryFile(data3 + 'Summary.hdf5')

good_recs1 = np.arange(3,15)
good_recs2 = [0, 1, 2, 4, 5]
good_recs3 = [0, 1, 2, 3, 4]

# load matrix of all rois of all layers with good rois
d1 = SingleFish(f1, good_recs1, overwrite=False)
d2 = SingleFish(f2, good_recs2, overwrite=False)
d3 = SingleFish(f3, good_recs3, overwrite=False)

# load all fish in multifish class
mf = MultiFish([
    d1, 
    d2, 
    d3
])

class rg_activity:
    def __init__(self, selective_rois, contr1, contr2):

        self.__rois = selective_rois
        self.__contr1 = contr1
        self.__contr2 = contr2
        self.__index = np.arange(len(self.__contr1))
        self.contr1 = np.unique(self.__contr1[~np.isnan(self.__contr1)])
        self.contr2 = np.unique(self.__contr2[~np.isnan(self.__contr2)])
        self.zscores = []
        # self.mean_dffs = []
        self.contr1_index = []
        self.contr2_index = []
        self.rois = self.__rois.rois
        self.recs = self.__rois.recs
        
        for c1 in self.contr1:
            embed()
            exit()
            self.contr1_index.append(c1)
            idx = self.__index[self.__contr1 == c1]
            cat_zscores = self.__rois.zscores[:,idx]
            # mean_dffs = np.mean(cat_dffs, axis=1)
            self.contr2_index.append(self.__contr2[idx])
            self.zscores.append(cat_zscores)
            # self.mean_dffs.append(mean_dffs)

        # self.mean_dffs = np.array(self.mean_dffs)
        self.contr1_index = np.array(self.contr1_index)
        self.contr2_index = np.array(self.contr2_index)


clock = np.array([1 if x > 0 else 0 for x in mf.ang_velocs])
# counterclockwise motion regressor
cclock = np.array([1 if x < 0 else 0 for x in mf.ang_velocs])



mfclock = deepcopy(mf)
mfcclock = deepcopy(mf)

clock = np.asarray([np.nan if x==0 else x for x in clock])
cclock = np.asarray([np.nan if x==0 else x for x in cclock])
red_clock = clock * mf.red
green_clock = clock * mf.green
red_cclock = cclock * mf.red
green_cclock = cclock * mf.green

rg_clock_data = rg_activity(mfclock, red_clock, green_clock)
rg_cclock_data = rg_activity(mfcclock, red_cclock, green_cclock)
gr_clock_data = rg_activity(mfclock, green_clock, red_clock)
gr_cclock_data = rg_activity(mfcclock, green_cclock, red_cclock)


#| code-fold: true

fig = plt.figure(figsize=(30*ps.cm, 20*ps.cm))

# build the subfigures
(subfig_l, subfig_r) = fig.subfigures(1, 2, hspace=0.05, width_ratios=[1, 1])

gs0 = gridspec.GridSpec(2, 3, figure=subfig_l)
stim_axs1 = [subfig_l.add_subplot(i) for i in gs0]

gs1 = gridspec.GridSpec(2, 3, figure=subfig_r)
stim_axs2 = [subfig_r.add_subplot(i) for i in gs1]

# plot the left side of the plot
colors = [ps.orange, ps.red]
labels = ['clockw.-selective', 'countercl.-selective']
for lab, color, rg_clock in zip(labels, colors, [rg_clock_data, rg_cclock_data]):
    for i1, rds in enumerate(rg_clock.contr1):
        zscores = []
        contr = []
        for dff in np.array(rg_clock.zscores)[rg_clock.contr1 == rds]:
            for i in range(len(dff[:,0])):
                
                # the data
                roi_dff = dff[i,:]
                contrsts2 = np.array(rg_clock.contr2_index)[rg_clock.contr1 == rds][0]

                # sort the data
                sort_roi_zscores = roi_dff[np.argsort(contrsts2)]
                sort_contrsts2 = contrsts2[np.argsort(contrsts2)]

                zscores.append(sort_roi_zscores)
                contr.append(sort_contrsts2)

        # compute means for each category
        contrs = np.unique(fs.flatten(contr))
        meanzscore = []
        stdzscore = []
        for c2 in contrs:
            meanzscore.append(np.mean(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))
            stdzscore.append(np.std(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))
        
        meanzscore = np.asarray(meanzscore)
        stdzscore = np.asarray(stdzscore)
        
        stim_axs1[i1].axhline(0, color="darkgray", lw=1, ls="dashed")
        stim_axs1[i1].plot(contrs, meanzscore, label=lab, c=color, lw=2.5, zorder = 100)
        stim_axs1[i1].fill_between(contrs, meanzscore-stdzscore, meanzscore+stdzscore, color=color, lw=1, alpha=0.2, label="_")
        stim_axs1[i1].set_title(f"red: {np.round(rds,2)}")

# plot the right side of the plot
colors = [ps.green, ps.blue]
labels = ['clockw.-selective', 'countercl.-selective']
for lab, color, gr_clock in zip(labels, colors, [gr_clock_data, gr_cclock_data]):
    for i1, rds in enumerate(gr_clock.contr1):
        zscores = []
        contr = []
        for dff in np.array(gr_clock.zscores)[gr_clock.contr1 == rds]:
            for i in range(len(dff[:,0])):
                
                # the data
                roi_dff = dff[i,:]
                contrsts2 = np.array(gr_clock.contr2_index)[gr_clock.contr1 == rds][0]

                # sort the data
                sort_roi_zscores = roi_dff[np.argsort(contrsts2)]
                sort_contrsts2 = contrsts2[np.argsort(contrsts2)]

                zscores.append(sort_roi_zscores)
                contr.append(sort_contrsts2)

        # compute means for each category
        contrs = np.unique(fs.flatten(contr))
        meanzscore = []
        stdzscore = []
        for c2 in contrs:
            meanzscore.append(np.mean(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))
            stdzscore.append(np.std(np.array(fs.flatten(zscores))[fs.flatten(contr) == c2]))
        
        meanzscore = np.asarray(meanzscore)
        stdzscore = np.asarray(stdzscore)
        
        stim_axs2[i1].axhline(0, color="darkgray", lw=1, ls="dashed")
        stim_axs2[i1].plot(contrs, meanzscore, label=lab, c=color, lw=2.5, zorder = 100)
        stim_axs2[i1].fill_between(contrs, meanzscore-stdzscore, meanzscore+stdzscore, color=color, lw=1, alpha=0.2, label="_")
        stim_axs2[i1].set_title(f"green: {np.round(rds,2)}")

# remove axes on right and top of plots
[x.spines["right"].set_visible(False) for x in np.asarray(stim_axs1)]
[x.spines["top"].set_visible(False) for x in np.asarray(stim_axs1)]
[x.spines["right"].set_visible(False) for x in np.asarray(stim_axs2)]
[x.spines["top"].set_visible(False) for x in np.asarray(stim_axs2)]

# turn y axes off where not needed
yaxes_off = [1,2,4,5]
[x.get_yaxis().set_visible(False) for x in np.asarray(stim_axs1)[yaxes_off]]
[x.get_yaxis().set_visible(False) for x in np.asarray(stim_axs2)[yaxes_off]]
[x.spines["left"].set_visible(False) for x in np.asarray(stim_axs1)[yaxes_off]]
[x.spines["left"].set_visible(False) for x in np.asarray(stim_axs2)[yaxes_off]]

# turn x axes off where they are not needed
xaxes_off = [0,1,2]
[x.get_xaxis().set_visible(False) for x in np.asarray(stim_axs1)[xaxes_off]]
[x.get_xaxis().set_visible(False) for x in np.asarray(stim_axs2)[xaxes_off]]
[x.spines["bottom"].set_visible(False) for x in np.asarray(stim_axs1)[xaxes_off]]
[x.spines["bottom"].set_visible(False) for x in np.asarray(stim_axs2)[xaxes_off]]

# set all axes to same tick ranges
x_range = np.arange(0,1.5,0.5)
y_range = np.arange(-1,2.1,1)
[x.set_yticks(y_range) for x in np.asarray(stim_axs1)]
[x.set_xticks(x_range) for x in np.asarray(stim_axs1)]
[x.set_yticks(y_range) for x in np.asarray(stim_axs2)]
[x.set_xticks(x_range) for x in np.asarray(stim_axs2)]

# set bounds to make axes nicer
x_bounds = (0,1)
y_bounds = (-1,2)
[x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs1)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs1)]
[x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs2)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs2)]

# make axes shared
stim_axs1[0].get_shared_x_axes().join(stim_axs1[0], *stim_axs1)
stim_axs1[0].get_shared_y_axes().join(stim_axs1[0], *stim_axs1)
stim_axs2[0].get_shared_x_axes().join(stim_axs2[0], *stim_axs2)
stim_axs2[0].get_shared_y_axes().join(stim_axs2[0], *stim_axs2)

# add labels
subfig_l.supylabel('z-score')
subfig_l.supxlabel('green contrast')
subfig_r.supxlabel('red contrast')

# get legend handles
handles1, labels1 = stim_axs1[i1].get_legend_handles_labels()
handles2, labels2 = stim_axs2[i1].get_legend_handles_labels()

# add legends
subfig_l.legend(handles1, labels1, loc='upper center', ncol=2)
subfig_r.legend(handles2, labels2, loc='upper center', ncol=2)

# adjust margins
plt.subplots_adjust(left=0.175, right=0.975, top=0.87, bottom=0.1, hspace=0.17, wspace=0.15)
#fs.doublesave("../plots/contrast_curves")

plt.show()