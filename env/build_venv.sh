#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

OS=$(uname)
TTMLIR_VENV=$TTMLIR_TOOLCHAIN_DIR/venv

python3 -m venv $TTMLIR_VENV
source $CURRENT_SOURCE_DIR/activate
python -m pip install --upgrade pip
pip install -r $CURRENT_SOURCE_DIR/build-requirements.txt
pip install -r $CURRENT_SOURCE_DIR/../test/python/requirements.txt
