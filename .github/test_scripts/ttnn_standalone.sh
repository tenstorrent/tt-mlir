#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

export TT_METAL_LIB="$INSTALL_DIR/lib"
cd $INSTALL_DIR/$tools/ttnn-standalone
./ttnn-standalone
