#! /bin/bash

# Change to the directory where this script resides.
DIR=$(cd $(dirname "$0"); pwd)
cd $DIR

# Google Cloud SDK is pinned for build reliability. Bump if the SDK complains about deprecation.
SDK_VERSION=127.0.0
SDK_FILENAME=google-cloud-sdk-${SDK_VERSION}-linux-x86_64.tar.gz
curl -O -J https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${SDK_FILENAME}
tar -zxvf ${SDK_FILENAME} --directory ${HOME}
export PATH=${PATH}:${HOME}/google-cloud-sdk/bin
# Create credentials
export CLOUDSDK_CORE_PROJECT=anaximander-tests
export GOOGLE_API_KEY=77aed24df7c3b3fc79e58deb6b382518a1753cac
gcloud auth activate-service-account --key-file client-secret.json
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/client-secret.json
gcloud components update
# Configure Python
pip install -U tox
pip --version
tox --version
pip install -r requirements.txt
# Run tests
find -type f -name '*.pyc' -delete
pytest
# Resets directory state
find -type f -name '*.pyc' -delete
