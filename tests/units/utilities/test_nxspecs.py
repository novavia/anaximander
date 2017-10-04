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
        keyspecs = {'Guitar': nxs.KeyedSpec(nxs.Spec(str)),
                    'Drums': nxs.KeyedSpec(nxs.Spec(str)),
                    'Bass': nxs.KeyedSpec(nxs.Spec(str)),
                    'Keys': nxs.KeyedSpec(nxs.Spec(str))}
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


def test_descriptors():

    class BandSpec(nxs.SpecDict):
        guitar = nxs.spec(str, key="Guitar")
        drums = nxs.spec(str, key="Drums")
        bass = nxs.spec(str, key="Bass")
        keys = nxs.spec(str, key="Keys")
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


def test_full_spec():

    instrument_spec = nxs.EnumerationSpec('Guitar',
                                          'Lead Singing',
                                          'Backup Singing',
                                          'Drums',
                                          'Bass',
                                          'Keys')

    class InstrumentList(nxs.SpecList, spec=instrument_spec):
        pass

    class BandSpec(nxs.SpecDict, spec='InstrumentList'):
        pass

    string = """# Jr Dreads 2017
                JD:
                  - Guitar
                  - Lead Singing
                Mike:
                  - Drums
                  - Backup Singing
                Greg:
                  - Bass
             """
    jrdreads = BandSpec.load(string)
    assert jrdreads['JD'][0] == 'Guitar'
    with pytest.raises(ValueError):
        jrdreads['JD'].append('Kazoo')
    with pytest.raises(TypeError):
        jrdreads['Paul'] = 0


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
