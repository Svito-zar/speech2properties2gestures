from matplotlib import pyplot as plt
from PIL import Image
from os.path import join
from os import listdir
import matplotlib as mpl

def initializeFigure(width='1col', height=None):
    '''
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
     
    '''
    # Set up Matplotlib parameters for the figure.
 
    # make sure defaults are used
    mpl.rcParams.update(mpl.rcParamsDefault)
 
    # adjust parameters
    rcParams = {'text.latex.preamble':r'\usepackage{amsmath}',
                'text.usetex': False,
                'savefig.pad_inches': 0.0,
                'figure.autolayout': False,
                'figure.constrained_layout.use': True,
                'figure.constrained_layout.h_pad':  0.05,
                'figure.constrained_layout.w_pad':  0.05,
                'figure.constrained_layout.hspace': 0.0,
                'figure.constrained_layout.wspace': 0.0,
                'font.size':        6,
                'axes.labelsize':   'small',
                'legend.fontsize':  'small',
                'xtick.labelsize':  'x-small',
                'ytick.labelsize':  'x-small',
                'mathtext.default': 'regular',
                'font.family' :     'sans-serif',
                'axes.labelpad': 1,
                #'xtick.direction': 'in',
                #'ytick.direction': 'in',
                'xtick.major.pad': 2,
                'ytick.major.pad': 2,
                'xtick.minor.pad': 2,
                'ytick.minor.pad': 2
               }
    mpl.rcParams.update(rcParams)
     
 
    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]
     
    # Get figure width in inch
    if width == '1col':
        width = 8.8 # width [cm]
    elif width == '2col':
        width = 18.0 # width [cm]
    figWidth = width * cm_to_inch # width [inch]
     
 
    # Get figure height in inch
    if height is None:
        fig_aspect_ratio = 7./10.
        figHeight = figWidth * fig_aspect_ratio  # height [inch]
    else:
        figHeight = height * cm_to_inch # height [inch]
     
 
    # Create figure with right resolution for publication
    fig, axes = plt.subplots(2,2,sharex=True, figsize=(figWidth, figHeight), dpi=300)

 
    # Add axis object and select as current axis for pyplot
    #ax = fig.add_subplot(111)
    #plt.sca(ax)
     
    return fig, axes

fig, axes = initializeFigure('1col')

folder = "tarasplots"
files = listdir(folder)
for idx, axis in enumerate([axes[0,0], axes[0,1], axes[1,0], axes[1,1]]):
    axis.imshow(Image.open(join(folder, files[idx])))
    axis.axis('off')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(files[idx][:-4], fontsize=7, y=0, pad=-3, verticalalignment="top")
print("")
plt.savefig("example_predictions.pdf", dpi=300, bbox_inches="tight")