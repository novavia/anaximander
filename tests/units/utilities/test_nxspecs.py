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

import io

import pytest

from anaximander.utilities import nxspecs as nxs

# =============================================================================
# Test Cases
# =============================================================================


def test_spec():
    with pytest.raises(TypeError):
        nxs.Spec()


def test_spec_from_list():
    string = """# Jr Dreads 2017
                - JD  # Goes by Jr Dreads
                - Mike
                - Greg
             """
    jrdreads = nxs.Spec.load(string)
    assert isinstance(jrdreads, nxs.SpecList)
    assert jrdreads[0] == "JD"
    output = io.StringIO()
    jrdreads.dump(output)
    assert output.getvalue().startswith("# Jr")


def test_spec_from_dict():
    string = """# Jr Dreads 2017
                Guitar: JD  # Goes by Jr Dreads
                Drums: Mike
                Bass: Greg
             """
    jrdreads = nxs.Spec.load(string)
    assert isinstance(jrdreads, nxs.SpecDict)
    assert jrdreads["Guitar"] == "JD"
    output = io.StringIO()
    jrdreads.dump(output)
    assert output.getvalue().startswith("# Jr")


def test_nested_list():
    string = """# Random number sequences
                -
                  - 4
                  - 5
                -
                  - 1
                  - 2
             """
    spec = nxs.Spec.load(string)
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
    spec = nxs.Spec.load(string)
    assert isinstance(spec['JD'], nxs.SpecList)
    assert spec['JD'][0] == 'Guitar'
    output = io.StringIO()
    spec.dump(output)
    assert output.getvalue().startswith("#")


def test_list_instance():
    jrdreads = nxs.SpecList('JD', 'Mike', 'Greg')
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
        keys = {'Guitar': nxs.SpecKey(stype=str),
                'Drums': nxs.SpecKey(stype=str),
                'Bass': nxs.SpecKey(stype=str),
                'Keyboards': nxs.SpecKey(stype=str)}
    assert len(BandSpec.__keys__) == 4
    string = """# Jr Dreads 2017
                Guitar: JD  # Goes by Jr Dreads
                Drums: Mike
                Bass: Greg
             """
    jrdreads = BandSpec.load(string)
    assert jrdreads['Guitar'] == 'JD'
    assert jrdreads['Keyboards'] is None


def test_descriptors():

    class BandSpec(nxs.SpecDict):
        guitar = nxs.spec(key="Guitar")
        drums = nxs.spec(key="Drums")
        bass = nxs.spec(key="Bass")
        keyboards = nxs.spec(key="Keyboards")
    assert len(BandSpec.__keys__) == 4
    string = """# Jr Dreads 2017
                Guitar: JD  # Goes by Jr Dreads
                Drums: Mike
                Bass: Greg
             """
    jrdreads = BandSpec.load(string)
    assert jrdreads['Guitar'] == 'JD'
    assert jrdreads.guitar == 'JD'
    # Checks that the empty key is merely an interface.
    assert jrdreads.keyboards is None
    assert jrdreads['Keyboards'] is None
    assert len(jrdreads._data) == 3
    assert len(jrdreads) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
