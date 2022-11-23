
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

mfcopy = deepcopy(mf)
mfcopy1 = deepcopy(mf)


mf.phase_means()
target_auc = 0.2 # probability threshold

# compute the correlations and indices sorted by correlations
indices, corrs = mf.responding_rois(mf.dffs, nrepeats=3)

# make a histpgram
counts, edges = np.histogram(corrs, bins=50, range=(-1,1), density=True)

# compute a gaussian KDE
xkde, kde = fs.kde1d(corrs, 0.02, xlims=[edges.min(), edges.max()])

# create empty arrays for the gradient and index
gradient = np.zeros_like(xkde[:-1])
index = np.arange(len(xkde[:-1]))

# compute the gradient between the target and actual auc
for i in range(len(xkde[:-1])):
    area = auc(xkde[i:], kde[i:])
    gradient[i] = abs(area-target_auc)

# find the index where the gradient is smallest
idx = index[gradient == gradient.min()][0]

# get the threshold for correlation coefficients here
thresh = xkde[idx]
print(f"{thresh=}")

fig, ax = plt.subplots(figsize=(17*ps.cm, 12*ps.cm), constrained_layout=True)

ax.plot(xkde, kde, c=ps.c1, lw=2)

ax.bar(
    edges[:-1], 
    counts, 
    width=np.diff(edges),
    edgecolor="darkgray", 
    facecolor="white", 
    # alpha=0.2,
    linewidth=1, 
    align="edge", 
    zorder=-20
)

ax.axvline([0], lw=1, ls="dashed", c="k")

ax.fill_between(
    xkde[idx:], 
    np.zeros_like(xkde[idx:]), 
    kde[idx:], 
    zorder=-10, 
    alpha=0.5, 
    color=ps.c2
)
ax.set_xlim(-1.01, 1.01)
#ax[1].plot(xkde[:-1], gradient, c=ps.c1, lw=2)
#ax[1].scatter(xkde[idx], gradient[idx], zorder=10, color=ps.c2)
#ax[1].axvline(thresh, lw=1, ls="dashed", c=ps.black)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)  
ax.set_title("Area under PDF", fontsize=14)
ax.set_ylabel("PDF"         , fontsize=14)
#ax[1].set_title("Difference gradient")
#ax[1].set_ylabel("AUC-thresh")

fig.supxlabel("Pearson r")
fs.doublesave('../poster/figs/pcorrelation')

# only take the indices where correlation coefficients crossed the thresh
indices_thresh = indices[corrs > thresh]
corrs_thresh = corrs[corrs > thresh]

# build motion regressor and correlate
motion = np.array([1 if x != 0 else 0 for x in mf.ang_velocs])
corr_motion = np.array([pearsonr(x, motion)[0] for x in mf.dffs])

# clockwise motion regressor
clock = np.array([1 if x > 0 else 0 for x in mf.ang_velocs])
corr_clock = np.array([pearsonr(x, clock)[0] for x in mf.dffs])

# counterclockwise motion regressor
cclock = np.array([1 if x < 0 else 0 for x in mf.ang_velocs])
corr_cclock = np.array([pearsonr(x, cclock)[0] for x in mf.dffs])



fig, ax = plt.subplots(1,2, figsize=(17*ps.cm, 12*ps.cm), sharex=True, sharey=True, constrained_layout=True)

histr = (-0.55, 0.55)

#ax[0].hist(corr_motion, bins=40, range=histr, fc=ps.c1)
#ax[0].plot([0,0], [0, 600], lw=1, linestyle='dashed', c=ps.black)

ax[0].hist(corr_clock, bins=40, range=histr, fc=ps.c1)
ax[0].plot([0,0], [0, 1500], lw=1, linestyle='dashed', c=ps.black)

ax[1].hist(corr_cclock, bins=40, range=histr, fc=ps.c1)
ax[1].plot([0,0], [0, 1500], lw=1, linestyle='dashed', c=ps.black)
ax[0].set_ylim(0,1600)
ax[1].set_ylim(0,1600)
#ax[0].set_title("motion", y=0.9)
ax[0].set_title("clockw.",        fontsize=14)
ax[1].set_title("counterclockw.", fontsize=14)

# remove upper and right axis
[x.spines["right"].set_visible(False) for x in ax]
[x.spines["top"].set_visible(False) for x in ax]

# make axes nicer
[x.set_yticks(np.arange(0, 1500, 500)) for x in ax]
[x.spines.left.set_bounds((0, 1500)) for x in ax]

[x.set_xticks(np.arange(-0.6, 0.61, 0.3)) for x in ax]
[x.spines.bottom.set_bounds((-0.6, 0.6)) for x in ax]

plt.subplots_adjust(left=0.15, right=1, top=0.92, bottom=0.14, hspace=0)
fig.supxlabel("Pearson r")
fig.supylabel("Count")
fs.doublesave("../poster/figs/directioncorrelation")
embed()
exit()
plt.show()
