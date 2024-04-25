// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTPASSES_H
#define TTMLIR_TTMLIR_TTPASSES_H

#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "ttmlir/Dialect/TT/TTOps.h"
#include <memory>

namespace mlir {
namespace tt {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TT/TTPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TT/TTPasses.h.inc"
} // namespace tt
} // namespace mlir

#endif
