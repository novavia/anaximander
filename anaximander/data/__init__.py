#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This package defines Data, an abstraction that facilitate data handling.

A Data class serves first and foremost as a namespace for grouping
classes of data objects pertaining to a given type of data. For instance,
a class UserSession(Data) indicates that there is a type of data that
holds information about user sessions. That data may be accessed from files
or a database table, and consumed in tabular format using pandas dataframes
or as individual records. Anaximander offers to automate the creation of
classes to handle those use cases, and makes them accessible in the
UserSession namespace. Hence the programmer can bet on the existence of
classes UserSession.Schema, UserSession.Record, UserSession.Log, which
respectively provide access to a schema, a record class and a tabular data
class.

The purpose of the Data abstraction is to deal with information content
rather than typical application objects. Exemples of the latter would include
an Employee class in a payroll application or a Rectangle class in a plotting
framework. By contrast, Data could be subclassed into UserSession, which
is intended to capture historical information about actions taken by users of
a website, or Trajectory, which could serve to record time series of
moving object locations. There isn't a hard and fast rule that could
draw a clear distinction between the two, as this is purely a matter of
design. However, the primary purpose of this Data framework is to wrap
pandas dataframes with an interface that guarantees column names and types.
Hence it is intended for dealing with data that is most typically consumed
in tabular form -think logs as the most obvious exemple. Individual rows
can be pulled as 'Records' and these are objects, but as the name implies
they are thought of as mere data containers rather than fancy objects. That
being said, nothing prevents a programmer from adding attributes and methods
to an automatically generated record class. Further, by wrapping dataframes
into container objects that make interface guarantees, the Anaximander data
framework actually allows a much more object-oriented treatment of these
dataframes. For instance, a UserSessionLog class can define various
methods that are specific to the type of data it contains, something that
is not as elegantly possible with raw dataframes -i.e. subclassing is brittle
and the alternative is to resort to externally defined functions, losing
the benefits of object-oriented programming. Indeed the best justification
for the data package is probably the elevation of pandas' dataframe to the
role of a structural template that serves to define semantic hierarchies
of specialized data containers following an object-oriented paradigm; in
other words, a marriage between 'data' and 'object' -hence the DataObject.

Because the definition of a Data class hinges primarily on a schema,
the canonical mechanism for creating it is to decorate a Schema declaration.
Alternatively, the Schema declaration can be embedded in a Data class
declaration, or the two declarations can be written independently as long
as a reference to the Schema class is supplied to the Data class.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================
