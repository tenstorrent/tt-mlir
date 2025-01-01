// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_PIPELINES_TTKERNELPIPELINES_H
#define TTMLIR_DIALECT_TTKERNEL_PIPELINES_TTKERNELPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttkernel {

void createTTKernelToEmitCPipeline(OpPassManager &pm);

void registerTTKernelPipelines();
} // namespace mlir::tt::ttkernel

#endif
