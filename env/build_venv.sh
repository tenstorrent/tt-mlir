#!/bin/bash

OS=$(uname)
TTMLIR_VENV=$TTMLIR_TOOLCHAIN/venv

echo $(pwd)
python3 -m venv $TTMLIR_VENV
source $CURRENT_SOURCE_DIR/activate 
python -m pip install --upgrade pip
#mkdir -p $(TTMLIR_VENV)/bin2 && ln -s ../bin/python3 $(TTMLIR_VENV)/bin2/Python3

#pip install -r $(MLIR_PYTHON_REQUIREMENTS)
pip install --pre -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu -r $CURRENT_SOURCE_DIR/../test/python_torch_examples/requirements$OS.txt
