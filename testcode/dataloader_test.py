import numpy as np
from vxtools.summarize.structure import SummaryFile

from modules.dataloader import all_rois

# now load the data
dataroot = '../data/'
f = SummaryFile(dataroot + 'Summary.hdf5')
good_recs = np.arange(len(f.recordings()))[:2]+1

# load matrix of all rois of all layers with good rois
d = all_rois(f, good_recs, recalc=True)

# test sorting
d.responding_rois()

# test repeat mean computation
d.repeat_means()
