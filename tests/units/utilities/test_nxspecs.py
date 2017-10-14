#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for xprops.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import datetime as dt
import io
import os
from pathlib import Path
import time

import pandas as pd
import pytest

from anaximander import TESTDIR
from anaximander.utilities import nxspecs as nxs

STORE_PATH = os.path.join(TESTDIR, 'data/specifications')
PROJECT = 'anaximander'
BUCKET = 'specifications'

# =============================================================================
# Test Constants
# =============================================================================

instrument_spec = nxs.Selection('Guitar',
                                'Lead Singing',
                                'Backup Singing',
                                'Drums',
                                'Bass',
                                'Keys')


class InstrumentList(nxs.SpecList, ispec=instrument_spec):
    pass


class BandSpec(nxs.SpecDict):
    __path__ = 'bands'
    __identifier__ = 'name'
    name = nxs.Str(required=True, key='Name')
    formation = nxs.Timestamp(key='Formation')
    musicians = nxs.Dict(InstrumentList, key='Musicians')


class Owner:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


JRDREADS = """# Jr Dreads 2017
              Name: Jr Dreads
              Formation: 2015-02-17
              Musicians:
                JD:
                  - Guitar
                  - Lead Singing
                Mike:
                  - Drums
                  - Backup Singing
                Greg:
                  - Bass
           """

JD = Owner("JD Margulici")

# =============================================================================
# Test Cases
# =============================================================================


def test_spec():
    with pytest.raises(TypeError):
        nxs.SpecContainer()


def test_spec_from_list():
    string = """# Jr Dreads 2017
                - JD  # Goes by Jr Dreads
                - Mike
                - Greg
             """
    jrdreads = nxs.SpecContainer.load(string)
    assert isinstance(jrdreads, nxs.SpecList)
    assert jrdreads[0] == "JD"
    jrdreads[0] = "Jr Dreads"
    output = io.StringIO()
    jrdreads.dump(output)
    printout = output.getvalue()
    assert printout.startswith("# Jr")
    i0 = printout.find('- ') + 2
    assert printout[i0:i0 + 2] == "Jr"


def test_spec_from_dict():
    string = """# Jr Dreads 2017
                Guitar: JD  # Goes by Jr Dreads
                Drums: Mike
                Bass: Greg
             """
    jrdreads = nxs.SpecContainer.load(string)
    assert isinstance(jrdreads, nxs.SpecDict)
    assert jrdreads["Guitar"] == "JD"
    jrdreads["Guitar"] = "Jr Dreads"
    output = io.StringIO()
    jrdreads.dump(output)
    printout = output.getvalue()
    assert printout.startswith("# Jr")
    ig = printout.find('Guitar: ') + len('Guitar: ')
    assert printout[ig:ig + 2] == "Jr"


def test_nested_list():
    string = """# Random number sequences
                -
                  - 4
                  - 5
                -
                  - 1
                  - 2
             """
    spec = nxs.SpecContainer.load(string)
    assert isinstance(spec[0], nxs.SpecList)
    assert spec[0][0] == 4
    output = io.StringIO()
    spec.dump(output)
    assert output.getvalue().startswith("#")


def test_nested_map():
    string = """# Jr Dreads 2017
                JD:  # Goes by Jr Dreads
                  - Guitar
                  - Lead Singer
                Mike:
                  - Drums
                  - Backup Singer
                Greg:
                  - Bass
             """
    spec = nxs.SpecContainer.load(string)
    assert isinstance(spec['JD'], nxs.SpecList)
    assert spec['JD'][0] == 'Guitar'
    output = io.StringIO()
    spec.dump(output)
    assert output.getvalue().startswith("#")


def test_list_instance():
    jrdreads = nxs.SpecList(['JD', 'Mike', 'Greg'])
    assert jrdreads[0] == 'JD'
    output = io.StringIO()
    jrdreads.dump(output)
    assert output.getvalue().startswith("- JD")


def test_dict_instance():
    jrdreads = nxs.SpecDict(Guitar='JD', Drums='Mike', Bass='Greg')
    assert jrdreads['Guitar'] == 'JD'
    output = io.StringIO()
    jrdreads.dump(output)
    assert output.getvalue().startswith("Guitar:")


def test_keys():

    class BandSpec(nxs.SpecDict):
        keyspecs = {'Guitar': nxs.Spec(str),
                    'Drums': nxs.Spec(str),
                    'Bass': nxs.Spec(str),
                    'Keys': nxs.Spec(str)}

        def __validator__(self):
            return sum(1 for v in self.values() if v is not None) >= 2

    assert len(BandSpec.__keyspecs__) == 4
    string = """# Jr Dreads 2017
                Guitar: JD  # Goes by Jr Dreads
                Drums: Mike
                Bass: Greg
             """
    jrdreads = BandSpec.load(string)
    assert jrdreads['Guitar'] == 'JD'
    assert jrdreads['Keys'] is None
    with pytest.raises(TypeError):
        jrdreads['Keys'] = 0
    del jrdreads['Bass']
    with pytest.raises(ValueError):
        del jrdreads['Drums']


def test_descriptors():

    class BandSpec(nxs.SpecDict):
        guitar = nxs.Str(key="Guitar")
        drums = nxs.Str(key="Drums")
        bass = nxs.Str(key="Bass")
        keys = nxs.Str(key="Keys")
    assert len(BandSpec.__keyspecs__) == 4
    string = """# Jr Dreads 2017
                Guitar: JD  # Goes by Jr Dreads
                Drums: Mike
                Bass: Greg
             """
    jrdreads = BandSpec.load(string)
    assert jrdreads['Guitar'] == 'JD'
    assert jrdreads.guitar == 'JD'
    # Checks that the empty key is merely an interface.
    assert jrdreads.keys is None
    assert jrdreads['Keys'] is None
    assert len(jrdreads._data) == 3
    assert len(jrdreads) == 4
    with pytest.raises(TypeError):
        jrdreads['Keys'] = 0


def test_datetime():

    class BandSpec(nxs.SpecDict):
        formation = nxs.Spec(dt.date, key='Formation')
        musicians = nxs.List(key='Musicians')
        last_show = nxs.DateTime('Last Show')

    string = """# Jr Dreads 2017
                Formation: 2015-02-17
                Musicians:
                  - JD
                  - Mike
                  - Greg
                Last Show: 2017-09-17 14:00:00
             """
    jrdreads = BandSpec.load(string)
    assert isinstance(jrdreads.formation, dt.date)
    assert isinstance(jrdreads['Last Show'], dt.datetime)


def test_full_spec():
    jrdreads = BandSpec.load(JRDREADS)
    assert isinstance(jrdreads.formation, pd.Timestamp)
    assert jrdreads.musicians['JD'][0] == 'Guitar'
    with pytest.raises(ValueError):
        jrdreads.musicians['JD'].append('Kazoo')
    with pytest.raises(TypeError):
        jrdreads.musicians['Paul'] = 0


def test_spec_directory():
    store = nxs.SpecDirectory(STORE_PATH)
    jrdreads = BandSpec.load(JRDREADS, owner=JD)
    store.store(jrdreads)
    path = Path(STORE_PATH) / 'JD Margulici' / 'bands' / 'Jr Dreads.yaml'
    assert path.exists()
    with path.open() as f:
        assert f.readline()[0:4] == "# Jr"
    retrieval = store.retrieve(JD, 'bands', 'Jr Dreads')
    assert str(retrieval) == str(jrdreads)
    assert store.list(JD, 'bands') == ['Jr Dreads']
    store.delete(JD, 'bands', 'Jr Dreads', confirm=False)
    assert store.list(JD, 'bands') == []
    store.store(jrdreads)
    assert store.list_ownership() == ['JD Margulici']
    assert store.list_paths(JD) == ['bands']
    store.drop_path('bands', confirm=False, force=True)
    assert store.list_paths(JD) == []
    store.store(jrdreads)
    store.drop_owner(JD, confirm=False, force=True)
    assert store.empty


@pytest.mark.online
def test_spec_bucket():
    store = nxs.SpecBucketGCP(PROJECT, BUCKET)
    jrdreads = BandSpec.load(JRDREADS, owner=JD)
    store.store(jrdreads)
    blob = store.blob('JD Margulici', 'bands', 'Jr Dreads')
    assert blob.exists()
    retrieval = store.retrieve(JD, 'bands', 'Jr Dreads')
    assert str(retrieval) == str(jrdreads)
    assert store.list(JD, 'bands') == ['Jr Dreads']
    store.delete(JD, 'bands', 'Jr Dreads', confirm=False)
    assert store.list(JD, 'bands') == []
    store.store(jrdreads)
    assert store.list_ownership() == ['JD Margulici']
    assert store.list_paths(JD) == ['bands']
    store.drop_path('bands', confirm=False, force=True)
    time.sleep(0.25)
    assert store.list_paths(JD) == []
    store.store(jrdreads)
    store.drop_owner(JD, confirm=False, force=True)
    time.sleep(0.25)
    assert store.empty


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
