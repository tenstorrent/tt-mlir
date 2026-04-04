// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_LIB_DIALECT_D2M_TRANSFORMS_GENERICAFFINEUTILS_H
#define TTMLIR_LIB_DIALECT_D2M_TRANSFORMS_GENERICAFFINEUTILS_H

#include "mlir/IR/Operation.h"

namespace mlir::tt::d2m {

// Temporarily replace d2m.block_offset ops with tagged arith constants.
void convertBlockOffsetsToTaggedConstants(Operation *scope);

// Restore tagged constants back into d2m.block_offset ops.
void restoreTaggedConstantsToBlockOffsets(Operation *scope);

} // namespace mlir::tt::d2m

#endif
