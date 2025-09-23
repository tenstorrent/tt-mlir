#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

export _ACTIVATE_SUPPRESS_INIT_WARNING=1
ENV_DIR=${ENV_DIR:-$(git rev-parse --show-toplevel)/env}
TTMLIR_PYTHON_VERSION="${TTMLIR_PYTHON_VERSION:-python3.11}"

source $ENV_DIR/activate

$TTMLIR_PYTHON_VERSION -m venv $TTMLIR_VENV_DIR

if [ ! -d "$TTMLIR_TOOLCHAIN_DIR" ]; then
  echo "$TTMLIR_TOOLCHAIN_DIR does not exist. Creating dir..."
  mkdir -p $TTMLIR_TOOLCHAIN_DIR
  if [ $? -ne 0 ]; then
    echo "I don't have permission to create dir $TTMLIR_TOOLCHAIN_DIR, please run the following and then rerun this script:"
    echo "  sudo mkdir -p $TTMLIR_TOOLCHAIN_DIR"
    echo "  sudo chown -R $USER $TTMLIR_TOOLCHAIN_DIR"
    exit 1
  fi
  source $ENV_DIR/activate
fi

python -m pip install --upgrade pip
# Requirements for third party projects are installed during their build in `CMakeLists.txt`
pip install -r $ENV_DIR/build-requirements.txt
pip install -r $ENV_DIR/ttnn-requirements.txt
pip install -r $ENV_DIR/../test/python/requirements.txt
