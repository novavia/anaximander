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

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from anaximander.utilities import functions as fun

__all__ = []

# =============================================================================
# Plotting constants
# =============================================================================


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
    title = str(series.context) if series.context else ''
    kwargs.setdefault('title', title)
    ax = series.data.plot(ax=ax, **kwargs)
    ax.set_xlabel(series.index.name)
    ax.set_ylabel(series.data.name)
    return ax


def plot_marks(digest, ax='new', y=None, **kwargs):
    """Plots marks."""
    if ax == 'new':
        fig, ax = plt.subplots()
    for i, dg in enumerate(digest.shadegroups()):
        shade = dg.shades().pop()
        marker = digest.marker_type(shade)
        color = marker.plargs.get('color', PALETTE[2 + i % 4])
        yval = fun.get(y, 0)
        plot_data = pd.Series(yval * np.ones(len(dg)), index=dg.data.index)
        plot_data.plot(ax=ax, marker='o', ls='none', color=color)
    return ax


def plot_highlights(digest, ax='new', ymin=None, ymax=None, **kwargs):
    """Plots marks."""
    if ax == 'new':
        fig, ax = plt.subplots()
    for i, dg in enumerate(digest.shadegroups()):
        shade = dg.shades().pop()
        highlighter = digest.highlighter_type(shade)
        color = highlighter.plargs.get('color', PALETTE[2 + i % 4])
        y = fun.get(ymin, 0)
        h = fun.get(ymax, 1) - y
        if digest.domain == 'time':
            index = (mdates.date2num(i) for i in dg.data.index)
        else:
            index = iter(dg.data)
        intervals = [(p, n - p) for p, n in fun.pairwise(index, 2)]
        ax.broken_barh(intervals, (y, h), facecolor=color, zorder=-1)
    return ax
