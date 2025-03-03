#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

OS=$(uname)
TTMLIR_VENV=$TTMLIR_TOOLCHAIN_DIR/venv
TTMLIR_PYTHON_VERSION="${TTMLIR_PYTHON_VERSION:-python3.10}"

$TTMLIR_PYTHON_VERSION -m venv $TTMLIR_VENV
source $CURRENT_SOURCE_DIR/activate
python -m pip install --upgrade pip
# Requirements for third party projects are installed during their build in `CMakeLists.txt`
pip install -r $CURRENT_SOURCE_DIR/build-requirements.txt
pip install -r $CURRENT_SOURCE_DIR/../test/python/requirements.txt
