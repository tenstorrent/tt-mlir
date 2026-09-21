# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`tt_crank::` custom torch ops: the graph form of kernels aten has no single op for, with their
fake (shape) functions and the compile-time rewrites that route torch onto them. Importing the package
registers every op. Lowerings live in `_compile`, DTensor sharding strategies in `_sharding`."""

from . import cross_entropy  # noqa: F401
