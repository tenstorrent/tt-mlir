#!/bin/bash

OS=$(uname)
TTMLIR_VENV=$TTMLIR_TOOLCHAIN/venv

python3 -m venv $TTMLIR_VENV
source $CURRENT_SOURCE_DIR/activate 
python -m pip install --upgrade pip
pip install -r $CURRENT_SOURCE_DIR/build-requirements.txt
pip install -r $CURRENT_SOURCE_DIR/../test/python/requirements.txt
