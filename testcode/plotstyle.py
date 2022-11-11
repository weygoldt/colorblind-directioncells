import cmocean
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def PlotStyle():
    class s:

        # units
        cm = 1 / 2.54
        mm = 1 / 25.4

        # standard colors
        black = "#1d1d1d"
        darkblue = "#264653"
        blue = "#2A7494"
        green = "#8AB17D"
        yellow = "#E9C46A"
        orange = "#F4A261"
        red = "#E86051"

        # gradient blue
        gblue1 = "#1a5b92"
        gblue2 = "#04a6c2"
        gblue3 = "#3fc1c0"

        # linear colormap
        lightcmap = cmocean.tools.lighten(cmocean.cm.haline, 0.8)

        @classmethod
        def lims(cls, data):
            """
            lims computes the axis limits for a list of data arrays.

            Parameters
            ----------
            data : array/list of arrays/lists
                All the data

            Returns
            -------
            touple
                Min and max
            """

            # flatten array
            data_flat = [item for sublist in data for item in sublist]

            return np.min(data_flat), np.max(data_flat)

        @classmethod
        def fancy_title(cls, axis, title):
            if " " in title:
                split_title = title.split(" ", 1)
                axis.set(
                    title=r"$\bf{{{}}}$".format(
                        split_title[0]) + f" {split_title[1]}"
                )
            else:
                axis.set_title(r"$\bf{{{}}}$".format(
                    title.replace(" ", r"\;")), pad=8)

        @classmethod
        def fancy_suptitle(cls, fig, title):
            split_title = title.split(" ", 1)
            fig.suptitle(
                r"$\bf{{{}}}$".format(split_title[0]) + f" {split_title[1]}",
                ha="left",
                x=0.078,
            )

        @classmethod
        def circled_annotation(cls, text, axis, xpos, ypos, padding=0.25):
            axis.text(
                xpos,
                ypos,
                text,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(
                    boxstyle=f"circle, pad={padding}", fc="white", ec="black", lw=1
                ),
            )

        @classmethod
        def fade_cmap(cls, cmap):

            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
            my_cmap = ListedColormap(my_cmap)

            return my_cmap

        @classmethod
        def label_subplots(cls, labels, axes, fig):
            for axis, label in zip(axes, labels):
                X = axis.get_position().x0
                Y = axis.get_position().y1
                fig.text(X, Y, label, weight="bold")

        @classmethod
        def hide_helper_xax(cls, ax):
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
            ax.patch.set_visible(False)

        @classmethod
        def set_boxplot_color(cls, bp, color):
            plt.setp(bp["boxes"], color=color)
            plt.setp(bp["whiskers"], color=color)
            plt.setp(bp["caps"], color=color)
            plt.setp(bp["medians"], color=color)

        @classmethod
        def letter_subplots(
            cls, axes=None, letters=None, xoffset=-0.1, yoffset=1.0, **kwargs
        ):
            """Add letters to the corners of subplots (panels). By default each axis is
            given an uppercase bold letter label placed in the upper-left corner.
            Args
                axes : list of pyplot ax objects. default plt.gcf().axes.
                letters : list of strings to use as labels, default ["A", "B", "C", ...]
                xoffset, yoffset : positions of each label relative to plot frame
                (default -0.1,1.0 = upper left margin). Can also be a list of
                offsets, in which case it should be the same length as the number of
                axes.
                Other keyword arguments will be passed to annotate() when panel letters
                are added.
            Returns:
                list of strings for each label added to the axes
            Examples:
                Defaults:
                    >>> fig, axes = plt.subplots(1,3)
                    >>> letter_subplots() # boldfaced A, B, C

                Common labeling schemes inferred from the first letter:
                    >>> fig, axes = plt.subplots(1,4)
                    >>> letter_subplots(letters='(a)') # panels labeled (a), (b), (c), (d)
                Fully custom lettering:
                    >>> fig, axes = plt.subplots(2,1)
                    >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
                Per-axis offsets:
                    >>> fig, axes = plt.subplots(1,2)
                    >>> letter_subplots(axes, xoffset=[-0.1, -0.15])

                Matrix of axes:
                    >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
                    >>> letter_subplots(fig.axes) # fig.axes is a list when axes is a 2x2 matrix
            """

            # get axes:
            if axes is None:
                axes = plt.gcf().axes
            # handle single axes:
            try:
                iter(axes)
            except TypeError:
                axes = [axes]

            # set up letter defaults (and corresponding fontweight):
            fontweight = "bold"
            ulets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(axes)])
            llets = list("abcdefghijklmnopqrstuvwxyz"[: len(axes)])
            if letters is None or letters == "A":
                letters = ulets
            elif letters == "(a)":
                letters = ["({})".format(lett) for lett in llets]
                fontweight = "normal"
            elif letters == "(A)":
                letters = ["({})".format(lett) for lett in ulets]
                fontweight = "normal"
            elif letters in ("lower", "lowercase", "a"):
                letters = llets

            # make sure there are x and y offsets for each ax in axes:
            if isinstance(xoffset, (int, float)):
                xoffset = [xoffset] * len(axes)
            else:
                assert len(xoffset) == len(axes)
            if isinstance(yoffset, (int, float)):
                yoffset = [yoffset] * len(axes)
            else:
                assert len(yoffset) == len(axes)

            # defaults for annotate (kwargs is second so it can overwrite these defaults):
            my_defaults = dict(
                fontweight=fontweight,
                fontsize="large",
                ha="center",
                va="center",
                xycoords="axes fraction",
                annotation_clip=False,
            )
            kwargs = dict(list(my_defaults.items()) + list(kwargs.items()))

            list_txts = []
            for ax, lbl, xoff, yoff in zip(axes, letters, xoffset, yoffset):
                t = ax.annotate(lbl, xy=(xoff, yoff), **kwargs)
                list_txts.append(t)
            return list_txts

        pass

    # rcparams text setup
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # rcparams
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["image.cmap"] = s.lightcmap
    plt.rcParams["axes.xmargin"] = 0.1
    plt.rcParams["axes.ymargin"] = 0.15
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["axes.titlesize"] = BIGGER_SIZE
    plt.rcParams["axes.titlepad"] = -10
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.borderpad"] = 0.4
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "white"
    plt.rcParams["legend.framealpha"] = 0.7
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.fancybox"] = False

    # specify the custom font to use
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Helvetica Now Text"

    return s


if __name__ == "__main__":
    s = PlotStyle()
