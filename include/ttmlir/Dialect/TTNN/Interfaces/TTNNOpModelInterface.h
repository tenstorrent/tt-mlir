// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_H
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_H

// Required for OpConfig definition to be used by tablegen'd
// TTNNOpModelInterface:
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
// Required for OpConstraints definition to be used by tablegen'd
// TTNNOpModelInterface:
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"
// This include is required for llvm::Expected in the tablegen'd
// TTNNOpModelInterface methods
#include "llvm/Support/Error.h"

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNOpModelInterface.h.inc"

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_H
