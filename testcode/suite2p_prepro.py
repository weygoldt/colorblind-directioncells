import numpy as np 
import matplotlib.pyplot as plt
from  IPython import embed
import h5py as h5
from vxtools.summarize.structure import SummaryFile

f = SummaryFile('data/Summary.hdf5')
# get all rois 
rois = f.rois()

rois_0 = rois[0]

