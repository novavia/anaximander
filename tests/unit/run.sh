#! /bin/bash

# Change to the directory where this script resides.
DIR=$(cd $(dirname "$0"); pwd)
cd $DIR

# Activate the virtual environment, with a silent output option.
source activate anaximander &> /dev/null

# Discover and run the tests.
python -W ignore::PendingDeprecationWarning -m unittest discover
