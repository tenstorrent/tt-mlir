// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUTPLAN_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUTPLAN_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <variant>

namespace mlir::tt::d2m {

// A Plan is a sequence of atomic Steps that transforms a tensor from a source
// PlanState into a target PlanState. Each Step changes exactly one aspect of
// the layout / format (memory space, tile form, grid, virtual-grid mapping,
// or OOB region). Plans are built by `canonicalize` and simplified by
// `minimize` before being emitted as IR by the LowerToLayout pass.

struct HostToDeviceStep {
  ttcore::MetalLayoutAttr targetLayout;
};

struct DeviceToHostStep {};

struct L1ToDRAMStep {
  ttcore::MetalLayoutAttr targetLayout;
  AffineMap viewMap;
};

struct DRAMToL1Step {
  ttcore::MetalLayoutAttr targetLayout;
  AffineMap viewMap;
};

struct TilizeStep {
  llvm::SmallVector<int64_t> tileShape;
};

struct UntilizeStep {};

struct ReshardStep {
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int64_t> dimAlignments;
  DenseIntElementsAttr collapsedIntervals;
};

struct RemapStep {
  AffineMap fwdMap;
  AffineMap invMap;
};

struct MaskStep {
  ttcore::OOBVal oobVal;
  llvm::SmallVector<int64_t> logicalShape;
};

using Step =
    std::variant<HostToDeviceStep, DeviceToHostStep, L1ToDRAMStep, DRAMToL1Step,
                 TilizeStep, UntilizeStep, ReshardStep, RemapStep, MaskStep>;

using Plan = llvm::SmallVector<Step>;

struct PlanState {
  RankedTensorType type;
  ttcore::OOBVal oobState;
};

// Emit the canonical (not yet minimized) sequence of atomic Steps that takes
// `source` to `target`. The result is deterministic and over-long by design:
// `minimize` is responsible for cancellation, fusion, and commutation.
Plan canonicalize(const PlanState &source, const PlanState &target,
                  llvm::ArrayRef<int64_t> targetGridShape);

// Apply cancellation, fusion, and commutation rules until fixpoint.
Plan minimize(Plan plan);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUTPLAN_H
