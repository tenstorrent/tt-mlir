# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Example d2m-jit pattern modules.

Each submodule registers one or more `@d2m.pattern`s on import. Load
them by importing the module — order matters only for `benefit` tie-
breaking.
"""
