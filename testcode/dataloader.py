import matplotlib.pyplot as plt
import numpy as np
from vxtools.summarize.structure import SummaryFile

f = SummaryFile("../data/Summary.hdf5")
roi = f.rois()[0]
