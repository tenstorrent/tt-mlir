# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Typed chisel exceptions.

Retained for historical reference by tests; emitters no longer raise these —
they catch native exceptions directly and record via the ChiselChecker.
"""


class ChiselError(Exception):
    """Base for all chisel-internal operational failures."""


class SkippableChiselError(ChiselError):
    """Failures that should be recorded as 'skipped' rather than 'error'."""


class NoGoldenImplementation(SkippableChiselError):
    """No golden mapping registered for this op type."""


class GoldenInputMissing(SkippableChiselError):
    """An input required by the golden function is not present in the pool."""


class GoldenExecutionError(ChiselError):
    """The golden function itself raised during execution."""


class TensorRetrievalError(ChiselError):
    """retrieve_torch_tensor failed to pull a tensor from the runtime pool."""


class TensorWriteError(ChiselError):
    """write_torch_tensor_to_pool failed to update a tensor in the runtime pool."""
