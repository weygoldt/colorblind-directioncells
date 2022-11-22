
import numpy as np
import matplotlib.pyplot as plt
import modules.functions as fs
import matplotlib.gridspec as gridspec
import scipy.stats

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


extent = (mf.times.min(), mf.times.max(), 0, len(mf.zscores[:,0]))
temp_zscores = np.asarray([fs.flatten(x) for x in mf.zscores])

for i, roi in enumerate(range(500)):
    dff = temp_zscores[i, :]
    plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()), color='black', linewidth='1.')
plt.show()