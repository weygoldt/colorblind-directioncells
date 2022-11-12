import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plotstyle import PlotStyle

ps = PlotStyle()

fig, ax = plt.subplots()
for rot in [0, 180]:

    marker, scale = ps.gen_arrow_head_marker(rot)
    markersize = 25
    ax.scatter(rot, 0, marker=marker, s=(markersize*scale)**2)

ax.set_xlabel('Rotation in degree')

plt.show()
