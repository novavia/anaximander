#! /bin/bash

# Change to the directory where this script resides.
DIR=$(cd $(dirname "$0"); pwd)
cd $DIR

sudo docker run -it --volume=$DIR:/Anaximander --workdir="/Anaximander" --memory=4g --entrypoint=/bin/bash python:3.6.0
