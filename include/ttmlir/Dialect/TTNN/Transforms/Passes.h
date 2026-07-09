// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/GreedyMemoryLayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/Transforms/Optimizer.h"
#include "ttmlir/Dialect/TTNN/Utils/CompositeResolution.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOptionParsers.h"

namespace mlir::tt::ttnn {

constexpr const char *kCacheDictAttr = "cache_dict";

enum class FileSplitTarget { EmitPy, EmitC };

// TTNN Passes
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Internal factory for the post-optimizer re-stamp of
// TTNNSetComputeKernelConfig. Builds the pass in "only-unconfigured-ops" mode:
// it restores compute_config on ops a rebuild dropped it from (e.g. reductions)
// but skips ops that still carry one, so it never augments configs the
// optimizer chose. This mode is not a pass option on purpose - there is no
// standalone use case, so it is not exposed on the command line; only the
// pipeline uses it.
std::unique_ptr<::mlir::Pass> createTTNNSetComputeKernelConfigRestamp(
    TTNNSetComputeKernelConfigOptions options);

} // namespace mlir::tt::ttnn

#endif
