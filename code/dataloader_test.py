import matplotlib.pyplot as plt
import numpy as np
from vxtools.summarize.structure import SummaryFile

import modules.functions as fs
from modules.dataloadr import MultiFish, SingleFish

# now load the data
DATAROOT2 = '../data/data2/'
DATAROOT3 = '../data/data3/'

f2 = SummaryFile(DATAROOT2 + 'Summary.hdf5')
f3 = SummaryFile(DATAROOT3 + 'Summary.hdf5')

good_recs2 = [0, 1, 2, 4, 5]
good_recs3 = [0, 1, 2, 3, 4]

# load matrix of all rois of all layers with good rois
d2 = SingleFish(f2, good_recs2, overwrite=False)
d3 = SingleFish(f3, good_recs3, overwrite=False)

# test multifish class
mf = MultiFish([d2, d3])

# test phase mean computation
mf.phase_means()

# compute correlation coefficients
indices, corrs = mf.responding_rois(mf.dffs, nrepeats=3)

# sort by indices
mf.filter_rois(indices)

# test repeat mean computation
mf.repeat_means(nrepeats=3)
