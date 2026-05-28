// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Error.h"

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_APPLYARGUMENTSHARDSTATUSPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Returns the (possibly non-replicated) sdy sharding carried by a
// stablehlo.custom_call that encodes a sharding constraint (e.g. @Sharding from
// xs.mark_sharding, @tt.sharding_constraint, or @xla.sdy.* targets). Returns a
// null attr if the op does not carry a recoverable sdy sharding.
static mlir::sdy::TensorShardingAttr
getCustomCallSdySharding(MLIRContext *context,
                         mlir::stablehlo::CustomCallOp customCallOp) {
  llvm::StringRef callTargetName = customCallOp.getCallTargetName();
  if (callTargetName != gspmd_utils::kShardingCustomCallTargetName &&
      callTargetName != sharding_utils::kTTShardingConstraintTargetName &&
      callTargetName != shardy_utils::kFuncResultShardingTargetName) {
    return {};
  }

  mlir::DictionaryAttr newAttrDict = shardy_utils::convertXlaSdyToSdyDictionary(
      context, customCallOp->getAttrDictionary());
  return mlir::dyn_cast_or_null<mlir::sdy::TensorShardingAttr>(
      newAttrDict.get(mlir::sdy::TensorShardingAttr::name));
}

// Hardening heuristic for graphs where the frontend dropped the arg-level
// sdy.sharding/gspmd.sharding attribute but the body still constrains the
// argument to be sharded (e.g. torch_xla/Shardy graph-capture sometimes drops
// arg-level sharding on graphs after the first while keeping in-body
// sdy.sharding_constraint / @Sharding custom calls intact). If we can find a
// non-fully-replicated sharding applied to (a transitive forwarding use of) the
// argument inside the body, the argument is actually pre-sharded and must NOT be
// re-distributed at runtime. Returns true if such in-body sharding is found.
//
// The walk is intentionally conservative: it only follows uses through trivial
// single-operand "pass-through" ops so that we attribute a constraint to the arg
// only when it directly (or near-directly) applies to it.
static bool argHasInBodySharding(MLIRContext *context,
                                 mlir::BlockArgument arg) {
  llvm::SmallVector<mlir::Value> worklist;
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  worklist.push_back(arg);

  auto shardingIsSharded = [](mlir::sdy::TensorShardingAttr sharding) {
    return sharding && !shardy_utils::isFullyReplicatedTensor(sharding);
  };

  while (!worklist.empty()) {
    mlir::Value val = worklist.pop_back_val();
    if (!visited.insert(val).second) {
      continue;
    }

    for (mlir::OpOperand &use : val.getUses()) {
      mlir::Operation *user = use.getOwner();

      // Native sdy sharding constraint applied to this value.
      if (auto constraintOp =
              llvm::dyn_cast<mlir::sdy::ShardingConstraintOp>(user)) {
        if (shardingIsSharded(constraintOp.getSharding())) {
          return true;
        }
        continue;
      }

      // Sharding encoded as a stablehlo.custom_call (not yet lowered to a
      // sdy.sharding_constraint at this point in the pipeline).
      if (auto customCallOp =
              llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(user)) {
        if (shardingIsSharded(getCustomCallSdySharding(context, customCallOp))) {
          return true;
        }
        continue;
      }

      // Follow only trivial single-operand forwarding ops so a constraint on a
      // reshape/convert/transpose of the arg still counts toward the arg.
      if (user->getNumOperands() == 1 && user->getNumResults() == 1) {
        worklist.push_back(user->getResult(0));
      }
    }
  }

  return false;
}

static mlir::LogicalResult
updateArgumentShardStatus(MLIRContext *context, mlir::ModuleOp &module,
                          mlir::OpBuilder &builder, func::FuncOp &funcOp,
                          bool gspmdAnnotationsExist,
                          bool shardyAnnotationsExist) {
  llvm::StringRef annotationsKeyWord =
      shardyAnnotationsExist ? mlir::sdy::TensorShardingAttr::name
                             : mlir::tt::gspmd_utils::kXlaShardingAttr;

  // Iterate through all the arguments and determine whether they are
  // pre-sharded or not.
  for (auto arg : funcOp.getArguments()) {
    mlir::DictionaryAttr argAttrDict =
        funcOp.getArgAttrDict(arg.getArgNumber());
    llvm::SmallVector<mlir::NamedAttribute> newArgAttrs;
    mlir::tt::ttcore::ShardStatus shardStatus =
        mlir::tt::ttcore::ShardStatus::Unsharded;

    // If the argument contains a gspmd.sharding or sdy.sharding annotation, it
    // is already pre-sharded.
    if (argAttrDict) {
      newArgAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(argAttrDict.getValue());

      if (argAttrDict.contains(annotationsKeyWord)) {
        shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;
      }
    }

    // Hardening: if the arg-level sharding annotation is missing (frontend
    // dropped it) but the body still constrains this argument to be sharded,
    // treat it as pre-sharded. Otherwise the arg would default to Unsharded and
    // a spurious replicate DistributeTensorOp would be emitted, which crashes at
    // runtime when the input is already a multi-buffer mesh tensor.
    // This only applies to the Shardy path; the GSPMD path threads shard status
    // into @Sharding custom calls below and is left untouched.
    if (shardyAnnotationsExist &&
        shardStatus == mlir::tt::ttcore::ShardStatus::Unsharded &&
        argHasInBodySharding(context, arg)) {
      shardStatus = mlir::tt::ttcore::ShardStatus::Presharded;
    }

    mlir::NamedAttribute shardStatusNamedAttr = {
        mlir::tt::ttcore::ShardStatusAttr::name,
        mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus)};
    newArgAttrs.push_back(shardStatusNamedAttr);
    funcOp.setArgAttrs(arg.getArgNumber(),
                       mlir::DictionaryAttr::get(context, newArgAttrs));

    // Update the shard status for the @Sharding custom call if it exists for
    // this arg.
    if (!shardyAnnotationsExist) {
      gspmd_utils::updateShardStatusForArgument(context, arg,
                                                shardStatusNamedAttr);
    }
  }

  return mlir::success();
}

static mlir::LogicalResult
updateResultShardStatus(MLIRContext *context, mlir::ModuleOp &module,
                        mlir::OpBuilder &builder, func::FuncOp &funcOp,
                        llvm::SmallVector<int64_t> resultPreshardedRef) {
  // If user did not provide result shard status, we will assume all results are
  // unsharded.
  if (resultPreshardedRef.empty() && funcOp.getNumResults() > 0) {
    module.emitWarning("User did not provide result shard status, assuming all "
                       "results are unsharded.");
    resultPreshardedRef = llvm::SmallVector<int64_t>(funcOp.getNumResults(), 0);
  }

  // Check if the size of the result shard status matches the number of results
  // in the function.
  if (resultPreshardedRef.size() != funcOp.getNumResults()) {
    return module.emitError("The size of the result shard status does not "
                            "match the number of results in the function.");
  }

  // Iterate through all the results and set the shard status based on the user
  // provided pipeline option.
  mlir::FunctionType funcType = funcOp.getFunctionType();
  for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
    mlir::DictionaryAttr resultAttrDict =
        mlir::DictionaryAttr::get(context, funcOp.getResultAttrs(i));
    llvm::SmallVector<mlir::NamedAttribute> newResultAttrs;
    mlir::tt::ttcore::ShardStatus shardStatus =
        resultPreshardedRef[i] == 0 ? mlir::tt::ttcore::ShardStatus::Unsharded
                                    : mlir::tt::ttcore::ShardStatus::Presharded;

    if (resultAttrDict) {
      newResultAttrs =
          llvm::SmallVector<mlir::NamedAttribute>(resultAttrDict.getValue());
    }

    mlir::NamedAttribute shardStatusNamedAttr = {
        mlir::tt::ttcore::ShardStatusAttr::name,
        mlir::tt::ttcore::ShardStatusAttr::get(context, shardStatus)};
    newResultAttrs.push_back(shardStatusNamedAttr);
    funcOp.setResultAttrs(i,
                          mlir::DictionaryAttr::get(context, newResultAttrs));

    if (!shardy_utils::sdyAnnotationsExist(module)) {
      gspmd_utils::updateShardStatusForResult(context, funcOp, i,
                                              shardStatusNamedAttr);
    }
  }

  return mlir::success();
}

/*
In multi-device graphs jax/torch-xla will annotate arguments/results with a
sharding attribute. For arguments, the framework will either provide a list of
tensors or a single tensor for each argument and give it to ttmlir runtime. For
results, the framework will either expect a list of tensors or a single tensor
from ttmlir runtime.

The situations are as follows:
* jax
arguments
- if there is a sdy.sharding annotation, the framework will provide a list of
tensors, otherwise framework will provide a single tensor
results
- if there is a sdy.sharding annotation, framework expects a list of tensors,
otherwise framework expects a single tensor

* torchxla
arguments
- will always have a sdy.sharding annotation
- framework will always provide a list of tensors for each result
results
- framework will always expect a list of tensors for each result (regardless if
it has a sdy.sharding annotation or not)

Therefore, we can conclude the following for arguments:
- ttmlir compiler can infer the shard status of the arguments. If the argument
has a sdy.sharding annotation, the framework will provide a list of tensors. If
not, the framework provides only a single tensor For results:
- ttmlir compiler will not be able to infer the shard status of results based
solely on the shlo graph itself. The set of rules that govern torch_xla are
different from jax. Therefore, we need support from frontend teams to provide
this as a pipeline option.
*/

class ApplyArgumentShardStatusPass
    : public impl::ApplyArgumentShardStatusPassBase<
          ApplyArgumentShardStatusPass> {
public:
  using impl::ApplyArgumentShardStatusPassBase<
      ApplyArgumentShardStatusPass>::ApplyArgumentShardStatusPassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();
    mlir::OpBuilder builder(context);

    // Check what types of annotations exist in the module. If non exist, we
    // do nothing.
    bool gspmdAnnotationsExist = gspmd_utils::gspmdAnnotationsExist(rootModule);
    bool sdyAnnotationsExist = shardy_utils::sdyAnnotationsExist(rootModule);

    // Loop through all arguments and annotate it with its shard status.
    rootModule.walk([&](func::FuncOp funcOp) {
      if (failed(mlir::tt::stablehlo::updateArgumentShardStatus(
              context, rootModule, builder, funcOp, gspmdAnnotationsExist,
              sdyAnnotationsExist))) {
        rootModule.emitError("Failed to update shard status");
        signalPassFailure();
        return;
      }
    });

    // Loop through all results and annotate it with its shard status.
    llvm::SmallVector<int64_t> resultPreshardedRef = llvm::to_vector(
        llvm::map_range(resultPresharded, [](int64_t val) { return val; }));
    rootModule.walk([&](func::FuncOp funcOp) {
      if (!funcOp.isPublic()) {
        return;
      }
      if (failed(mlir::tt::stablehlo::updateResultShardStatus(
              context, rootModule, builder, funcOp, resultPreshardedRef))) {
        rootModule.emitError("Failed to update shard status");
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace mlir::tt::stablehlo
