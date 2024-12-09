#!/usr/bin/env python
#
# # SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import model_explorer

# TODO(odjuricic): Hack to make our extension default for .mlir files.
# This can be handled better when we switch to our model-explorer fork.
model_explorer.extension_manager.ExtensionManager.BUILTIN_ADAPTER_MODULES = []
model_explorer.visualize_from_config(extensions=["tt_adapter"], no_open_in_browser=True)
