#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines ...

This module is part of the Tzigane project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import numpy as np
import pandas as pd

from bokeh.document import Document
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot
from bokeh.models import LinearAxis, Range1d

from bokeh.palettes import Set1
import itertools

from dateutil import tz as dtz
from . import nxcolumns as cln

class Staff:

    def __init__(self, source=None, columns=None, tz=None, title=None):
        self.fig = figure(x_axis_type='datetime')
        self.title = title
        if source is not None:
            self.add_line(source, columns)

        self.colorIndex = 0

    def display(self):
        show(self.fig)

    def add_line(self, source, columns=None):
        if columns is None:
            columns = [k for k, v in source.columns.items()
                       if isinstance(v, cln.Float)]
        if not isinstance(columns, list):
            columns = [columns]

        for col in columns:
            minVal = min(getattr(source, col))
            maxVal = max(getattr(source, col))
            self.fig.y_range = Range1d(start=minVal, end=maxVal)
            self.fig.line(source.datetime, getattr(source, col),
                legend=col)

    def add_line_right(self, source, columns=None):
        self._add_right_axis()

        if columns is None:
            columns = [k for k, v in source.columns.items()
                       if isinstance(v, cln.Float)]
        if not isinstance(columns, list):
            columns = [columns]

        for col in columns:
            minVal = min(getattr(source, col))
            maxVal = max(getattr(source, col))
            self.fig.extra_y_ranges['right'] = Range1d(start=minVal, end=maxVal)
            self.fig.line(source.datetime, getattr(source, col),
                legend=col, y_range_name='right')

    def _add_right_axis(self):
        self.fig.add_layout(LinearAxis(y_range_name='right'), 'right')

    #Assumes there's a 'label' column
    def add_events(self, source, labels=None, colors=None):
        df = source._data
        if labels is None:
            labels = df['label'].cat.categories

        for label in labels:
            if colors is not None and label in colors.keys():
                color = colors[label]
            else:
                color = self._get_color()

            vals = df[df['label'] == label]
            self.fig.rect(x=vals.index, y=np.full(shape=len(vals), fill_value=0.5),
                width=1, height=1, width_units='screen', color=color)

        #NaN Labels Default to Gray
        vals = df[pd.isnull(df['label'])]
        self.fig.rect(x=vals.index, y=np.full(shape=len(vals), fill_value=0.5),
            width=1, height=1, width_units='screen', color='gray')


    def add_sessions(self, source, labels=None, colors=None):
        df = source._data
        df['start_time'] = source.datetime
        df['end_time'] = source.datetime + source.duration
        if labels is None:
            labels = df['label'].cat.categories

        for label in labels:
            if colors is not None and label in colors.keys():
                color = colors[label]
            else:
                color = self._get_color()

            vals = df[df['label'] == label]
            self.fig.quad(left=vals['start_time'], right=vals['end_time'],
                top=np.zeros(len(vals)), bottom=np.ones(len(vals)),
                color=color)
        vals = df[pd.isnull(df['label'])]
        self.fig.quad(left=vals['start_time'], right=vals['end_time'],
            top=np.zeros(len(vals)), bottom=np.ones(len(vals)),
            color='gray')

    def _get_color(self):
        rval = Set1[9][self.colorIndex % 10]
        self.colorIndex += 1
        return rval

class GridStaff(Staff):
    def __init__(self, tz=None, config=None):
        self.staffs = []
        self.titles = []
        self.config = None

    def display(self):
        figs = [[stf.fig] for stf in self.staffs]
        show(gridplot(figs))

    def add_line(self, source, columns=None, figureIndex=0):
        super().add_line(source, columns)

    def add_staff(self, staff):
        self.staffs.append(staff)

    def add_figs(self, source, columns=None):
        if columns is None:
            columns = [k for k, v in source.columns.items()
                       if isinstance(v, cln.Float)]
        if not isinstance(columns, list):
            columns = [columns]
        for col in columns:
            new_staff = Staff()
            new_staff.add_line(source, col)
            self.staffs.append(new_staff)

# class SimpleStaff(Staff):
#     ...
# class DoubleStaff(Staff):
#     ...
# class EmptyStaff(Staff):
#     ...
# class EventStaff(Staff):
#     ...
# class TimedEventStaff(Staff):
#     ...
# class StateStaff(Staff):
#     ...
