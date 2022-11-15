# load essential packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import auc
from vxtools.summarize.structure import SummaryFile

import modules.functions as fs
from modules.dataloader import all_rois
from modules.plotstyle import PlotStyle

ps = PlotStyle()

# now load the data
f = SummaryFile('../data/Summary.hdf5')

# now select the recordings that were good
good_recs = [5, 8, 9, 10, 11, 13, 14]
# good_recs = [14]

# load matrix of all rois of all layers with good rois
d = all_rois(f, good_recs)

d.stimulus_means()
