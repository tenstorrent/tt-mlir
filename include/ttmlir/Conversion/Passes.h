// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES_H
#define TTMLIR_CONVERSION_PASSES_H

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace mlir::tt {

#define GEN_PASS_REGISTRATION
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_PASSES_H
