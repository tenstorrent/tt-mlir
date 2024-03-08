// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTPASSES_H
#define TTMLIR_TTMLIR_TTPASSES_H

#include "mlir/Pass/Pass.h"
#include "ttmlir/TTDialect.h"
#include "ttmlir/TTOps.h"
#include <memory>

namespace mlir {
namespace tt {
#define GEN_PASS_DECL
#include "ttmlir/TTPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/TTPasses.h.inc"
} // namespace tt
} // namespace mlir

#endif
