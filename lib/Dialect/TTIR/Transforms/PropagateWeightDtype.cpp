// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <optional>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRPROPAGATEWEIGHTDTYPE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kWeightDtypeAttrName = "ttcore.weight_dtype";

// Trace the weight operand backward to the originating func arg(s) and return
// the per-tensor weight dtype override they carry.
//
// The walk passes through ops that don't change which weight tensor this is
// (TM, CCL, Broadcast, MeshShard, MeshPartition, Typecast, RepeatInterleave).
// A ConcatOp (e.g. a fused QKV weight built by SharedLHSMatmulFusion) is a
// genuine combination of multiple weights: each branch is resolved
// independently and the results merged by keeping the highest-precision (most
// bits) dtype, so the fused weight never loses precision relative to any
// constituent. `sawMissing`/`sawDisagreement` record whether the branches were
// only partially annotated or disagreed, so the caller can warn.
static std::optional<ttcore::DataType>
resolveWeightDtype(mlir::Value weight, bool &sawMissing,
                   bool &sawDisagreement) {
  mlir::Value source = weight;
  while (auto *op = source.getDefiningOp()) {
    if (auto concatOp = mlir::dyn_cast<ConcatOp>(op)) {
      std::optional<ttcore::DataType> merged;
      for (mlir::Value operand : concatOp->getOperands()) {
        std::optional<ttcore::DataType> branch =
            resolveWeightDtype(operand, sawMissing, sawDisagreement);
        if (!branch) {
          sawMissing = true;
          continue;
        }
        if (!merged) {
          merged = branch;
        } else if (*merged != *branch) {
          sawDisagreement = true;
          // Keep the highest-precision (most bits) dtype.
          if (ttcore::getNumberOfBits(*branch) >
              ttcore::getNumberOfBits(*merged)) {
            merged = branch;
          }
        }
      }
      return merged;
    }

    if (op->hasTrait<TensorManipulation::Trait>() ||
        op->hasTrait<CCL::Trait>() ||
        mlir::isa<BroadcastOp, MeshShardOp, MeshPartitionOp, TypecastOp,
                  RepeatInterleaveOp>(op)) {
      source = op->getOperand(0);
      continue;
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::General,
                 "PropagateWeightDtype: unable to trace through op '{0}', "
                 "stopping backward walk",
                 op->getName().getStringRef());
    return std::nullopt;
  }

  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(source);
  if (!blockArg) {
    return std::nullopt;
  }

  auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      blockArg.getOwner()->getParentOp());
  if (!funcOp) {
    return std::nullopt;
  }

  auto dtypeAttr = funcOp.getArgAttrOfType<mlir::StringAttr>(
      blockArg.getArgNumber(), kWeightDtypeAttrName);
  if (!dtypeAttr) {
    return std::nullopt;
  }
  auto parsed = ttcore::DataTypeStringToEnum(dtypeAttr.getValue());
  if (!parsed) {
    funcOp.emitWarning() << "ignoring invalid " << kWeightDtypeAttrName << " \""
                         << dtypeAttr.getValue() << "\" on argument "
                         << blockArg.getArgNumber();
  }
  return parsed;
}

// Resolve the weight dtype override for a matmul/linear weight operand and, if
// found, set it on the consumer op for the TTNN weight dtype conversion pass.
static void resolveAndSetWeightDtype(mlir::Value weight,
                                     mlir::Operation *consumerOp) {
  bool sawMissing = false;
  bool sawDisagreement = false;
  std::optional<ttcore::DataType> dtype =
      resolveWeightDtype(weight, sawMissing, sawDisagreement);
  if (!dtype) {
    return;
  }

  if (sawDisagreement || sawMissing) {
    consumerOp->emitWarning()
        << "fusing matmuls with differing weight dtype overrides; using "
           "highest-precision dtype '"
        << ttcore::DataTypeEnumToString(*dtype)
        << "' for the fused weight to avoid precision loss";
  }

  consumerOp->setAttr(
      kWeightDtypeAttrName,
      mlir::StringAttr::get(consumerOp->getContext(),
                            ttcore::DataTypeEnumToString(*dtype)));
}

} // namespace

class TTIRPropagateWeightDtype
    : public impl::TTIRPropagateWeightDtypeBase<TTIRPropagateWeightDtype> {
public:
  using impl::TTIRPropagateWeightDtypeBase<
      TTIRPropagateWeightDtype>::TTIRPropagateWeightDtypeBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([](Operation *op) {
      if (auto matmulOp = mlir::dyn_cast<MatmulOp>(op)) {
        resolveAndSetWeightDtype(matmulOp.getB(), op);
      } else if (auto linearOp = mlir::dyn_cast<LinearOp>(op)) {
        resolveAndSetWeightDtype(linearOp.getB(), op);
      } else if (auto sparseMatmulOp = mlir::dyn_cast<SparseMatmulOp>(op)) {
        resolveAndSetWeightDtype(sparseMatmulOp.getB(), op);
      }
    });
  }
};

} // namespace mlir::tt::ttir
