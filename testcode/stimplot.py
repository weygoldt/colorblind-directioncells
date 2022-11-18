import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from vxtools.summarize.structure import SummaryFile

from modules.dataloader import all_rois
from modules.plotstyle import PlotStyle

ps = PlotStyle()

# build the stimulus
reds = np.round(np.geomspace(0.1, 1, 5), 2)
reds = np.append(0, reds)
greens = np.round(np.geomspace(0.1, 1, 5), 2)
greens = np.append(0, greens)
r_rgb = [np.array([r, 0, 0]) for r in reds]
g_rgb = [np.array([0, g, 0]) for g in greens]
rgb_matrix = np.zeros((len(r_rgb), len(g_rgb)), dtype=((float, 3), 2))

for i1 in range(len(rgb_matrix[:, 0])):
    g = g_rgb[i1]

    for i2 in range(len(rgb_matrix[0, :])):
        r = r_rgb[i2]
        rg_rgb = [g, r]
        rgb_matrix[i1, i2] = rg_rgb

dim = np.shape(rgb_matrix)
rgb_matrix_flat = rgb_matrix.reshape(dim[0]*dim[1], dim[2], dim[3])
fill_r = [[0, 2], [1, 3]]
fill_g = [[1, 3], [2, 4]]

# gridspec inside gridspec
fig = plt.figure(figsize=(20*ps.cm, 10*ps.cm))
gs0 = gridspec.GridSpec(1, 2, figure=fig)

# axes for stimulus
gs00 = gs0[0].subgridspec(6, 6)
stim_axs = [fig.add_subplot(i) for i in gs00]

# axis for heatmap
gs01 = gs0[1].subgridspec(1, 1)
heatm_ax = fig.add_subplot(gs01[0])

# plot the stimulus
for axis, rgbs in zip(np.array(stim_axs).flat, rgb_matrix_flat):
    for i in range(len(fill_r)):
        axis.axvspan(fill_r[0][i], fill_r[1][i], color=rgbs[0])
        axis.axvspan(fill_g[0][i], fill_g[1][i], color=rgbs[1])
        axis.axis('off')

# plot the heatmap


plt.show()
