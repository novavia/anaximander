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

from collections import Mapping, defaultdict
from itertools import cycle

from dateutil import tz as dtz
import numpy as np
import pandas as pd

import anaximander3 as nx
from ..utilities import nxrange, xprops, functions as fun
from . import nxcolumns as cln
from .dataobject import DataObject

if nx.INTERACTIVE:
    import matplotlib as mpl
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns

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
# Plotting primitives
# =============================================================================


def line(ax, series, **kwargs):
    """Plot primitive for sample series."""
    if not series.empty:
        series.plot(ax=ax, **kwargs)


def vlines(ax, events, **kwargs):
    """Plot function for events."""
    for i, ix in enumerate(events.index):
        label = events.label[i] if i == 0 else '_'
        ax.axvline(mdates.date2num(ix), label=label, **kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)


def vspans(ax, spans, **kwargs):
    """Primitive for plot_sessions."""
    kwargs.setdefault('zorder', -1)
    kwargs.setdefault('alpha', 0.5)
    onsets = [mdates.date2num(i) for i in spans.index]
    offsets = spans.duration.dt.total_seconds() / 86400
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        label = spans.label[i] if i == 0 else '_'
        ax.axvspan(onset, onset + offset, label=label, **kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

# =============================================================================
# Staff class, wrapper for time series axes
# =============================================================================


class Staff:

    def __init__(self, score, ax):
        self.score = score
        self.ax = ax
        self._dt_range = nxrange.time_range()
        self._setup(score.tz)

    def _setup(self, tz='UTC'):
        now = pd.Timestamp.now(tz=tz)
        basedelta = pd.Timedelta('30min')
        baserange = [now - basedelta, now]
        baseline = pd.Series([0, 1], index=baserange)
        baseline.plot(ax=self.ax)
        self.ax.lines[0].remove()
        self.ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S.%f %Z')
        self._yrange = nxrange.float_range()
        self._twin_yrange = nxrange.float_range()

    @xprops.weakproperty
    def score(self):
        return None

    @xprops.cachedproperty
    def main(self):
        """The main sequence for the considered Staff."""
        return None

    @property
    def dt_range(self):
        return self._dt_range

    @dt_range.setter
    def dt_range(self, dt_range):
        self._dt_range = nxrange.TimeInterval.compact(self._dt_range, dt_range)
        if self._dt_range:
            xmin = mdates.date2num(self._dt_range.lower)
            xmax = mdates.date2num(self._dt_range.upper)
            self.ax.set_xlim(xmin, xmax)

    @property
    def yrange(self):
        return self._yrange

    @yrange.setter
    def yrange(self, yrange):
        self._yrange = nxrange.FloatInterval.compact(self._yrange, yrange)
        if self._yrange:
            self.ax.set_ylim(self._yrange.lower, self._yrange.upper)

    @property
    def twin_yrange(self):
        return self._twin_yrange

    @twin_yrange.setter
    def twin_yrange(self, yrange):
        self._twin_yrange = nxrange.FloatInterval.compact(self._twin_yrange,
                                                          yrange)
        if self._twin_yrange:
            self.twinx.set_ylim(self._twin_yrange.lower,
                                self._twin_yrange.upper)

    @property
    def tz(self):
        return self.ax.xaxis.get_major_formatter()._formatter.tz

    def _set_tz(self, tz_):
        tz_ = dtz.gettz(tz_)
        self.ax.xaxis.get_major_formatter()._tz = tz_
        if not self.dt_range:
            self._setup(tz_)
        else:
            self.dt_range = self.dt_range

    @property
    def empty(self):
        return not (self.ax.patches or self.ax.lines)

    @xprops.cachedproperty
    def twinx(self):
        return self.ax.twinx()

    def _plot_sequence(self, sequence, **kwargs):
        if self.main is None:
            self._main = sequence
            self.dt_range = sequence.dt_range
            cert = sequence.certification
            if not pd.isna(cert):
                cert = mdates.date2num(cert)
                self.ax.axvline(cert, c='r', lw=2, ls='--')
                self.ax.axvspan(cert, cert + 1e9, facecolor='0.5', alpha=0.15)

    def _label_styles(self, sequence, ysplit=False, **kwargs):
        label_styles = defaultdict(dict)
        categories = sequence.label.dtype.categories
        if ysplit and len(categories):
            splits = np.arange(0, 1, 1 / len(categories))
            yranges = list(fun.pairwise(list(splits) + [1.]))
            for l, r in zip(categories, yranges):
                yrange = {k: v for k, v in zip(('ymin', 'ymax'), r)}
                label_styles[l].update(yrange)
        for l, c in zip(categories, cycle(PALETTE)):
            label_styles[l]['color'] = c
        label_styles[np.nan]['color'] = GREY
        for k, v in kwargs.copy().items():
            if isinstance(v, Mapping):
                specs = []
                wildcard = False
                for label, val in v.items():
                    if label == '*':
                        wildcard = True
                    else:
                        label_styles[label][k] = val
                        specs.append(label)
                if wildcard:
                    wspec = v['*']
                    for l in label_styles.keys():
                        if l not in specs:
                            label_styles[l][k] = wspec
                del kwargs[k]
        for label, style in label_styles.items():
            style.update(kwargs)
        return label_styles

    def _label_plot(self, plotter, data, styles):
        for label, style in styles.items():
            if label is np.nan:
                seq = data[data.label.isna()]
            else:
                seq = data[data.label == label]
            plotter(self.ax, seq, **style)

    def plot_sample_sequence(self, samples, column=None, twinx=False,
                             **kwargs):
        if not column:
            columns = [k for k, v in samples.columns.items()
                       if isinstance(v, cln.Float)]
            try:
                column = columns[0]
            except IndexError:
                msg = "No plottable sample series."
                raise ValueError(msg)
        ax = self.twinx if twinx else self.ax
        series = getattr(samples, column)
        line(ax, series, **kwargs)
        if not series.empty:
            srange = series.max() - series.min()
            ymin = series.min() - 0.05 * srange
            ymax = series.max() + 0.05 * srange
            if twinx:
                self.twin_yrange = nxrange.float_range(ymin, ymax)
            else:
                self.yrange = nxrange.float_range(ymin, ymax)
        self._plot_sequence(samples)
        if not ax.get_ylabel():
            nxcol = samples.schema[column]
            ax.set_ylabel(nxcol.label)

    def plot_event_sequence(self, events, **kwargs):
        styles = self._label_styles(events, **kwargs)
        self._label_plot(vlines, events.data, styles)
        self._plot_sequence(events)

    def plot_multi_event_sequence(self, events, **kwargs):
        styles = self._label_styles(events, ysplit=True, **kwargs)
        self._label_plot(vlines, events.data.reset_index(1), styles)
        self._plot_sequence(events)

    def plot_session_sequence(self, sessions, **kwargs):
        styles = self._label_styles(sessions, **kwargs)
        self._label_plot(vspans, sessions.data, styles)
        self._plot_sequence(sessions)

    def plot_multi_session_sequence(self, sessions, **kwargs):
        styles = self._label_styles(sessions, ysplit=True, **kwargs)
        self._label_plot(vspans, sessions.data.reset_index(1), styles)
        self._plot_sequence(sessions)

    def plot_state_sequence(self, states, **kwargs):
        styles = self._label_styles(states, **kwargs)
        self._label_plot(vspans, states.spans, styles)
        self._plot_sequence(states)

# =============================================================================
# Score class, wrapper for figure
# =============================================================================


class Score:

    def __init__(self, tz=None):
        self.tz = tz
        self.fig, self.staves = self.__setup__()

    def __setup__(self):
        """Returns a figure and a set of staves."""
        fig, ax = plt.subplots()
        staves = [Staff(self, ax)]
        return fig, staves

    @property
    def tz(self):
        return self._tz

    @tz.setter
    def tz(self, tz_):
        tz_ = dtz.gettz(tz_)
        self._tz = tz_
        if hasattr(self, 'staves'):
            for staff in self.staves:
                staff._set_tz(tz_)


class SimpleScore(Score):
    """A Score with a single staff."""

    @property
    def staff(self):
        return self.staves[0]


class RowScore(Score):
    """A Score with staves arranged in rows sharing the x axis."""

    def __init__(self, nrows=1, tz=None):
        self.tz = tz
        self.fig, self.staves = self.__setup__(nrows)

    def __setup__(self, nrows):
        fig, axes = plt.subplots(nrows, sharex=True, squeeze=False)
        staves = [Staff(self, row[0]) for row in axes]
        return fig, staves
