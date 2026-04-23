// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUT_PLAN_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUT_PLAN_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <variant>

namespace mlir::tt::d2m {

// Snapshot of the information the planner needs about a tensor. Captures
// everything that affects layout-change decisions but no IR-level detail.
// Populated by the pass at entry (see `extractPlanState` in LowerToLayout.cpp)
// and consumed by `canonicalize`.
struct PlanState {
  RankedTensorType type = {};
  AffineMap remapping = {};  // from a producing view op, if any
  AffineMap vgmForward = {}; // from a producing empty op with virtual grid
  AffineMap vgmInverse = {};

  std::optional<ttcore::MetalLayoutAttr> getLayout() const;
  bool hasLayout() const { return getLayout().has_value(); }
  bool isSystem() const;
  bool isL1() const;
  bool isDRAM() const;
  llvm::ArrayRef<int64_t> getGridShape() const; // requires hasLayout()
};

// A Plan is a sequence of atomic Steps that transforms a tensor from one
// PlanState into another. Each Step changes one aspect of the tensor (memory
// space, tile form, grid/alignment, VGM, or OOB region). Plans are produced by
// `canonicalize`, simplified by `minimize`, and lowered to IR by the pass's
// `emit`.

struct OutputBufferSpec {
  RankedTensorType type = {};
  AffineMap vgmForward = {};
  AffineMap vgmInverse = {};
};

struct HostToDeviceStep {
  OutputBufferSpec output = {};
};

struct HostToBounceBufferStep {
  OutputBufferSpec output = {};
};

struct DeviceToHostStep {
  RankedTensorType outputType = {};
};

struct L1ToDRAMStep {
  OutputBufferSpec output = {};
  AffineMap viewMap = {};
};

struct DRAMToL1Step {
  OutputBufferSpec output = {};
  AffineMap viewMap = {};
};

struct TilizeStep {
  llvm::SmallVector<int64_t> tileShape = {};
  OutputBufferSpec output = {};
};

struct UntilizeStep {
  OutputBufferSpec output = {};
};

struct RebufferStep {
  OutputBufferSpec output = {};
};

struct ReshardStep {
  llvm::SmallVector<int64_t> gridShape = {};
  llvm::SmallVector<int64_t> dimAlignments = {};
  DenseIntElementsAttr collapsedIntervals = {};
  OutputBufferSpec output = {};
};

struct RemapStep {
  RankedTensorType outputType = {};
  AffineMap remapping = {};
};

struct ReinterpretLayoutStep {
  RankedTensorType outputType = {};
};

struct MaskStep {
  ttcore::OOBVal oobVal = ttcore::OOBVal::Undef;
  llvm::SmallVector<int64_t> logicalShape = {};
  OutputBufferSpec output = {};
};

using Step = std::variant<HostToDeviceStep, HostToBounceBufferStep,
                          DeviceToHostStep, L1ToDRAMStep, DRAMToL1Step,
                          TilizeStep, UntilizeStep, RebufferStep, ReshardStep,
                          RemapStep, ReinterpretLayoutStep, MaskStep>;

using Plan = llvm::SmallVector<Step>;

// Produce the canonical sequence of atomic Steps that transforms `src` into
// `tgt`. Pure: no IR, no side effects. The result is deterministic and may
// contain redundant adjacent pairs which `minimize` is responsible for
// collapsing.
Plan canonicalize(const PlanState &src, const PlanState &tgt,
                  llvm::ArrayRef<int64_t> targetGridShape, MLIRContext *ctx);

// Apply cancellation, fusion, and commutation rules until fixpoint.
Plan minimize(Plan plan);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_LOWERTOLAYOUT_PLAN_H
