import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn
import matplotlib as mpl
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def initializeFigure(width="1col", height=None):
    """
    Initialize a single plot for publication.
     
    Creates a figure and an axis object that is set to be the 
    current working axis.
     
    @param width: Width of the figure in cm or either '1col' 
                  (default) or '2col' for single our double 
                  column usage. Single column equals 8.8cm and
                  double column 18cm.
    @type width: float or str (either '1col' or '2col')
    @param height: Height of the figure either in cm. If None
                   (default), will be calculated with an 
                   aspect ratio of 7/10 (~1/1.4).
    @type height: float or None
    @return: figure and axis objects.
    @rtype: tuple (figure, axis)
     
    """
    # Set up Matplotlib parameters for the figure.

    # make sure defaults are used
    mpl.rcParams.update(mpl.rcParamsDefault)

    # adjust parameters
    rcParams = {
        "text.latex.preamble": r"\usepackage{amsmath}",
        "text.usetex": False,
        "savefig.pad_inches": 0.0,
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.05,
        "figure.constrained_layout.w_pad": 0.05,
        "figure.constrained_layout.hspace": 0.0,
        "figure.constrained_layout.wspace": 0.0,
        "font.size": 6,
        "axes.labelsize": "small",
        "legend.fontsize": "small",
        "xtick.labelsize": "x-small",
        "ytick.labelsize": "x-small",
        "mathtext.default": "regular",
        "font.family": "sans-serif",
        "axes.labelpad": 1,
        #'xtick.direction': 'in',
        #'ytick.direction': 'in',
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.minor.pad": 2,
        "ytick.minor.pad": 2,
    }
    mpl.rcParams.update(rcParams)

    # Prepare figure width and height
    cm_to_inch = 0.393701  # [inch/cm]

    # Get figure width in inch
    if width == "1col":
        width = 8.8  # width [cm]
    elif width == "2col":
        width = 18.0  # width [cm]
    figWidth = width * cm_to_inch  # width [inch]

    # Get figure height in inch
    if height is None:
        fig_aspect_ratio = 7.0 / 10.0
        figHeight = figWidth * fig_aspect_ratio  # height [inch]
    else:
        figHeight = height * cm_to_inch  # height [inch]

    # Create figure with right resolution for publication
    fig = plt.figure(figsize=(figWidth, figHeight), dpi=300)

    # Add axis object and select as current axis for pyplot
    ax = fig.add_subplot(111)
    plt.sca(ax)

    return fig, ax


initializeFigure("1col")

dataset_labels = {
    "name": [
        "retraction",
        "prep",
        "pre-hold",
        "stroke",
        "post-hold",
        "deictic",
        "beat",
        "iconic",
        "discourse",
        "amount",
        "shape",
        "size",
        "direction",
    ],
    "count": [
        17528,
        38374,
        650,
        53021,
        16654,
        37806,
        20079,
        90222,
        16873,
        6474,
        16489,
        2447,
        17552,
    ],
    "type": ["phase"] * 5 + ["category"] * 4 + ["semantics"] * 4,
}

df = pd.DataFrame.from_dict(dataset_labels)

barplot = seaborn.barplot(x="name", y="count", data=df, hue="type", dodge=False)

# Show each number explicitly on top of the bars
for bar in barplot.patches:
    barplot.annotate(
        f"{bar.get_height():.0f}",
        (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
        ha="center",
        va="center",
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=5,
    )
total_number_of_frames = 127_581
bar = barplot.patches[0]
barplot.annotate(
    total_number_of_frames,
    (bar.get_x() + bar.get_width() / 2.0, total_number_of_frames),
    ha="center",
    va="center",
    xytext=(0, 5),
    textcoords="offset points",
)
plt.ylim(0, total_number_of_frames + 1500)
plt.gca().plot(
    plt.xlim(),
    [total_number_of_frames] * 2,
    "--",
    color="black",
    linewidth=1,
    label="total number of frames\nwith a gesture label",
)

# place legend at center right
# to avoid overlap with top line showing total nr. of frames
plt.legend(loc=7)
# plt.gca().legend().set_title('')
plt.xlabel(None)
plt.yticks([])
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment="center")
plt.gca().tick_params(axis="both", which="both", length=0)
spines = plt.gca().spines

spines["right"].set_visible(False)
spines["top"].set_visible(False)
spines["left"].set_visible(False)
spines["bottom"].set_visible(False)

plt.savefig("label_frequencies.pdf", dpi=600)
