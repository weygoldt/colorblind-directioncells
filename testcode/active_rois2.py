import matplotlib.pyplot as plt
import numpy as np
from vxtools.summarize.structure import SummaryFile

import functions as fs
from roimatrix import roimatrix

f = SummaryFile('../data/Summary.hdf5')   # import HDF5 file
num_rec = len(f.recordings())
rec_nos = [5, 8, 9, 10, 11, 13, 14]

d = roimatrix(f, rec_nos)

d.stimulus_means()

d.repeat_means()

d.sort_means_by_corr()
