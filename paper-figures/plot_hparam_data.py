import matplotlib as mpl
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def initializeFigure(width='1col', height=None, width_multiplier=None, height_multiplier=None):
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
                'text.usetex': True,
                'savefig.pad_inches': 0.0,
                'figure.autolayout': False,
                'figure.constrained_layout.use': True,
                'figure.constrained_layout.h_pad':  0.05,
                'figure.constrained_layout.w_pad':  0.05,
                'figure.constrained_layout.hspace': 0.0,
                'figure.constrained_layout.wspace': 0.0,
                'font.size':        8,
                'axes.linewidth': 0.5,
                'axes.labelsize':   'small',
                'legend.fontsize':  'small',
                'xtick.labelsize':  'x-small',
                'ytick.labelsize':  'x-small',
                'mathtext.default': 'regular',
                'font.family' :     'sans-serif',
                'axes.labelpad': 1,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
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
    fig = plt.figure(figsize=(figWidth * width_multiplier, figHeight * height_multiplier), dpi=300)
     
 
    # Add axis object and select as current axis for pyplot
    ax = fig.add_subplot(111)
    plt.sca(ax)
     
    return fig, ax

initializeFigure('1col', width_multiplier=0.95, height_multiplier=0.6)

x_values = []
y_values = []
with open("svito_zar_new_cb_loss_normal_ge_semant_sp2prop_hparams_f1_feature_av_1_vs_step_chart_data", "r") as file:
    str = file.read()

# Get the data for every run in the hparam search
lines = str.split('{"x":[')[1:]

# Extract the x values for each run, which are at the beginning
xaxis_strings = [line.split('],"y":[')[0] for line in lines]
x_values = [list(literal_eval(x_string)) for x_string in xaxis_strings]

# Extract the y values for each run, which are between the "y":[ and the ],"type" strings
remaining_stuff = [line.split('],"y":[')[1] for line in lines]
yaxis_strings = [line.split('],"type"')[0] for line in remaining_stuff]
y_values = [list(literal_eval(y_string)) for y_string in yaxis_strings]


# Plot the runs
for x, y in zip(x_values, y_values):
    sns.lineplot(x=x, y=y, lw=0.5)

plt.xlabel("training iterations")
plt.ylabel(r"Macro $\text{F}_1$ score")

# Turn off top and right axis lines
spines = plt.gca().spines
spines['right'].set_visible(False)
spines['top'].set_visible(False)

# Remove the axis tick markers
#plt.gca().tick_params(axis=u'both', which=u'both',length=0)

plt.savefig("hparam_search_example.pdf", transparent=True)