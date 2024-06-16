// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_PASSES_H
#define TTMLIR_DIALECT_TTIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include <memory>

namespace mlir::tt::ttir {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTIR/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTIR/Passes.h.inc"
} // namespace mlir::tt::ttir

#endif
