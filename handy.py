import math as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
SPINE_COLOR = 'gray'

# From https://nipunbatra.github.io/blog/2014/latexify.html
def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    # assert(columns in [1,2])

    if fig_width is None:
        if type(columns) == int:
            fig_width = 3.39 if columns==1 else 6.9 # width in inches
        else:
            fig_width = 3.39*columns

    if fig_height is None:
        golden_mean = (m.sqrt(5)-1.0)/2.0    # Aesthetic ratio

        if type(columns) == int:
            fig_height = fig_width*golden_mean # height in inches
        else:
            fig_height = 3.39*golden_mean

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    # matplotlib.rcParams.update({"pgf.texsystem": "pdflatex"})

    params = {'backend': 'ps',
            "pgf.texsystem" : "pdflatex",
              # 'text.latex.preamble': [r'\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif' ,
              "text.latex.unicode": True,
              "axes.unicode_minus": True
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax
