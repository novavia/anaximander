#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting primitives for data objects.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import anaximander as nx

import numpy as np
import pandas as pd
if nx.INTERACTIVE:
    import matplotlib as mpl
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

from ..utilities import functions as fun
from .base import DataObject

__all__ = []

# =============================================================================
# Plotting constants
# =============================================================================

if nx.INTERACTIVE:
    rcparams = {'lines.solid_capstyle': 'butt',
                'lines.linewidth': 1,
                'legend.fancybox': True,
                'axes.facecolor': '#E8E8E8',
                'axes.edgecolor': '#E8E8E8',
                'axes.linewidth': 3.0,
                'axes.titlesize': 'x-large',
                'grid.color': '#D1D2D4',
                'savefig.edgecolor': '#E8E8E8',
                'savefig.facecolor': '#E8E8E8',
                'figure.facecolor': '#E8E8E8',
                }
    
    sns.set(font_scale=1.2, color_codes=True, rc=rcparams)
    
    sns.set_style({'axes.labelcolor': '.25',
                   'text.color': '0.25',
                   'xtick.color': '0.25',
                   'ytick.color': '0.25',
                   })
    
    CCV = mpl.colors.ColorConverter()
    
    PALETTE = sns.color_palette()
    DC1 = PALETTE[0]  # Data color #1
    DC2 = PALETTE[1]  # Data color #2
    NC1 = PALETTE[2]  # Annotation color #1
    NC2 = PALETTE[3]  # Annotation color #2
    NC3 = PALETTE[4]  # Annotation color #3
    NC4 = PALETTE[5]  # Annotation color #4
    GREY = CCV.to_rgba('#898989', 0.5)  # Medium grey
    LGF = (0.85, 0.85, 0.85, 0.75)  # Light grey fill
    
    # Ticker formatters
    THOSEP = mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))


def context(dataobject):
    """Fetches context recursively."""
    try:
        res = dataobject.context
    except AttributeError:
        return None
    else:
        if isinstance(res, DataObject):
            return context(res)
        else:
            return res

def titlemaker(dataobject):
    """Returns a plot title."""
    object_context = context(dataobject)
    return str(object_context) if object_context is not None else None

# =============================================================================
# Plotting function
# =============================================================================


def plot_series(series, ax='new', **kwargs):
    """Customized plot function.

    Params:
        ax: 'new' for new plot, otherwise see pandas series plot.
    """
    if ax == 'new':
        fig, ax = plt.subplots()
    kwargs.setdefault('color', DC1)
    kwargs.setdefault('title', titlemaker(series))
    ax = series.data.plot(ax=ax, **kwargs)
    ax.set_xlabel(series.index.name)
    ax.set_ylabel(series.data.name)
    return ax


def plot_marks(digest, y, ax='new', **kwargs):
    """Plots marks from a MarkDigest, at specified y-axis value."""
    if ax == 'new':
        fig, ax = plt.subplots()
    for i, dg in enumerate(digest.shadegroups()):
        shade = dg.shades().pop()
        marker = digest.marker_type(shade)
        color = marker.plargs.get('color', PALETTE[2 + i % 4])
        plot_data = pd.Series(y * np.ones(len(dg)), index=dg.data.index)
        plot_data.plot(ax=ax, marker='o', ls='none', color=color)
    title = kwargs.get('title', titlemaker(digest))
    ax.set_title(title)
    return ax


def plot_highlights(digest, y0=None, y1=None, ax='new', **kwargs):
    """Plots highlights from a HighlightDigest.

    y0 and y1 specify the lower and upper y-axis position of the highlights.
    """
    if ax == 'new':
        fig, ax = plt.subplots()
    if digest.empty:
        return
    legend_patches = []
    for i, dg in enumerate(digest.shadegroups()):
        if dg.empty:
            continue
        shade = dg.data['next_highlighter'][0]
        highlighter = digest.highlighter_type(shade)
        color = highlighter.plargs.get('color', PALETTE[2 + i % 4])
        if digest.domain == 'time':
            index = (mdates.date2num(i) for i in dg.data.index)
        else:
            index = iter(dg.data)
        intervals = [(p, n - p) for p, n in fun.pairwise(index, 2)]
        ax.broken_barh(intervals, (y0, y1 - y0), facecolor=color, zorder=-1)
        legend_patches.append(mpatches.Patch(color=color, label=shade))
    title = kwargs.get('title', titlemaker(digest))
    ax.set_title(title)
    return ax
