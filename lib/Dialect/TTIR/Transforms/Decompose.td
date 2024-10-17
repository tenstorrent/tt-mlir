// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDECOMPOSE_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDECOMPOSE_TD

include "mlir/IR/PatternBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROps.td"

def IdentityPatternForTTIRAdd : Pat<(TTIR_AddOp $x, $y, $c), (TTIR_AddOp (TTIR_AddOp $x, $y, $c), $y, $c)>;

#endif
