
import numpy as np
import matplotlib.pyplot as plt
import modules.functions as fs
import matplotlib.gridspec as gridspec
import scipy.stats
from IPython import embed

from copy import deepcopy
from vxtools.summarize.structure import SummaryFile
from modules.dataloader import MultiFish, SingleFish
from modules.plotstyle import PlotStyle
from sklearn.metrics import auc
from scipy.stats import pearsonr
from modules.contrast import selective_rois_trash, rg_activity_trash, phase_activity

def plot_lineplot(ax, mf):
    temp_zscores = np.asarray([fs.flatten(x) for x in mf.zscores])
    n = 10
    min = []
    max = []
    for i, roi in enumerate(np.arange(100, 100+n+1, 1)):
        dff = temp_zscores[roi, :]
        max.append(np.max(dff))
        min.append(np.min(dff))

    for i, roi in enumerate(range(100, 100+ n+1, 1)):
        dff = temp_zscores[roi, :]
        ax.plot(mf.times/60, i + (dff - np.min(min) ) / (np.max(max) - np.min(min)), color='black', linewidth='1.')
    ax.set_xlabel("Time [min]", fontsize= 15)
    ax.set_ylabel("ROIs",     fontsize= 15)
    ax.set_yticks(np.arange(0.3, n+1.6, 2))
    ax.set_xticks(np.round(np.arange((mf.times[0])/60, 30.1, 5), 1))
    #ax.vlines(9.631, -1, 11, ls='dashed', color='r')
    #ax.vlines(19.23, -1, 11, ls='dashed', color='r')
    ax.set_yticklabels(np.arange(0, n+0.1, 2), fontsize=13)

    ax.set_xticklabels(np.round(np.arange((mf.times[0])/60, 30.1, 5), 1), fontsize=13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylim(-1, n+1)

save_path = "../poster/figs/raw2"

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
d2 = SingleFish(f2, good_recs2, overwrite=False,)
d1 = SingleFish(f1, good_recs1, overwrite=False,)
d3 = SingleFish(f3, good_recs3, overwrite=False,)

# load all fish in multifish class
mf = MultiFish([
    d1, 
    d2, 
    d3
])

mfcopy1 = deepcopy(mf)
# Raw data plot 
fig, ax = plt.subplots(figsize=(ps.cm*17, ps.cm*12))
plot_lineplot(ax, mfcopy1)
fs.doublesave(f'{save_path}')
plt.show()
