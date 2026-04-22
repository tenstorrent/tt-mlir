// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUTPLAN_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUTPLAN_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <variant>

namespace mlir::tt::d2m {

// A Plan is a sequence of atomic Steps that transforms a tensor from a source
// layout+state into a target layout+state. Each Step changes exactly one
// aspect of the tensor (memory space, tile form, grid, VGM, or OOB region).
// Plans are produced by `canonicalize`, simplified by `minimize`, and lowered
// to IR by `emit`.

struct HostToDeviceStep {
  RankedTensorType outputType = {};
};

struct DeviceToHostStep {};

struct L1ToDRAMStep {
  AffineMap viewMap = {};
};

struct DRAMToL1Step {
  RankedTensorType outputType = {};
  AffineMap viewMap = {};
};

struct TilizeStep {
  llvm::SmallVector<int64_t> tileShape = {};
  RankedTensorType outputType = {};
};

struct UntilizeStep {
  RankedTensorType outputType = {};
};

struct ReshardStep {
  llvm::SmallVector<int64_t> gridShape = {};
  llvm::SmallVector<int64_t> dimAlignments = {};
  DenseIntElementsAttr collapsedIntervals = {};
  RankedTensorType outputType = {};
};

struct RemapStep {
  AffineMap fwdMap = {};
  AffineMap invMap = {};
};

struct MaskStep {
  ttcore::OOBVal oobVal = ttcore::OOBVal::Undef;
  llvm::SmallVector<int64_t> logicalShape = {};
  RankedTensorType outputType = {};
};

using Step =
    std::variant<HostToDeviceStep, DeviceToHostStep, L1ToDRAMStep, DRAMToL1Step,
                 TilizeStep, UntilizeStep, ReshardStep, RemapStep, MaskStep>;

using Plan = llvm::SmallVector<Step>;

// Apply cancellation, fusion, and commutation rules until fixpoint.
Plan minimize(Plan plan);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUTPLAN_H
