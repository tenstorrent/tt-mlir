# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ._ttkernel_ops_gen import *
from ._ttkernel_enum_gen import *
from .._mlir_libs._ttmlir import ttkernel_ir as ir


def TensorAccessor(
    args,
    bank_base_address_in,
    page_size_in=None,
    *,
    results=None,
    loc=None,
    ip=None,
):
    return TensorAccessorOp(
        args=args,
        bank_base_address_in=bank_base_address_in,
        page_size_in=page_size_in,
        results=results,
        loc=loc,
        ip=ip,
    ).result
