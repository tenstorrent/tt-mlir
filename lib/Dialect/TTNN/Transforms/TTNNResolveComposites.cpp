// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "ttmlir/Asserts.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNRESOLVECOMPOSITES
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

using CompositeValidatorFn =
    std::function<OpValidationResult(ttcore::CompositeOp, OpBuilder &)>;
using CompositeBuilderFn =
    std::function<Operation *(ttcore::CompositeOp, OpBuilder &)>;
// A hard precondition checked before the inline/promote decision. When it
// returns failure (after emitting its own diagnostic), the pass fails instead
// of silently falling back to inlining the decomposition. Used for composites
// that are only valid on a specific architecture.
using CompositePreconditionFn =
    std::function<LogicalResult(ttcore::CompositeOp)>;

struct CompositeEntry {
  CompositeValidatorFn validate;
  CompositeBuilderFn build;
  CompositePreconditionFn precondition; // may be empty
};

static llvm::StringMap<CompositeEntry> &getCompositeRegistry() {
  static llvm::StringMap<CompositeEntry> registry;
  return registry;
}

// Operands and attributes recovered from a "flash_mla_prefill" composite,
// shared by its validate and build registry callbacks.
struct FlashMlaPrefillCompositeArgs {
  Value query;
  Value key;
  Value value;         // null when the latent form is used.
  Value attentionMask; // null when absent.
  IntegerAttr headDimV;
  BoolAttr isCausal;
  FloatAttr scale; // null when default is meant to be used
};

// Map the composite's variadic inputs back to (query, key, value, mask) using
// the has_value / has_attention_mask flags, and pull out the typed attributes.
static FlashMlaPrefillCompositeArgs
extractFlashMlaPrefillArgs(ttcore::CompositeOp compositeOp) {
  DictionaryAttr attrs = compositeOp.getCompositeAttributes().value_or(nullptr);
  TT_assert(attrs);

  auto readBool = [&](StringRef name) -> bool {
    auto a = attrs.getAs<BoolAttr>(name);
    return a && a.getValue();
  };
  bool hasValue = readBool("has_value");
  bool hasAttentionMask = readBool("has_attention_mask");

  auto inputs = compositeOp.getInputs();
  FlashMlaPrefillCompositeArgs args;
  args.query = inputs[0];
  args.key = inputs[1];
  unsigned idx = 2;
  args.value = hasValue ? inputs[idx++] : Value();
  args.attentionMask = hasAttentionMask ? inputs[idx++] : Value();
  args.headDimV = attrs.getAs<IntegerAttr>("head_dim_v");
  args.isCausal = attrs.getAs<BoolAttr>("is_causal");
  args.scale = attrs.getAs<FloatAttr>("scale");
  return args;
}

// Recover the chunk_start_idx attribute from an "indexer_score" composite,
// defaulting to 0 when absent. Shared by its validate and build callbacks.
static uint32_t getIndexerScoreChunkStartIdx(ttcore::CompositeOp compositeOp) {
  DictionaryAttr attrs = compositeOp.getCompositeAttributes().value_or(nullptr);
  if (!attrs) {
    return 0;
  }
  auto chunkStartIdxAttr = attrs.getAs<mlir::IntegerAttr>("chunk_start_idx");
  return chunkStartIdxAttr ? static_cast<uint32_t>(
                                 chunkStartIdxAttr.getValue().getZExtValue())
                           : 0;
}

static void registerBuiltinComposites() {
  auto &registry = getCompositeRegistry();
  if (!registry.empty()) {
    return;
  }

  registry["topk_router_gpt"] = CompositeEntry{
      // Validate
      [](ttcore::CompositeOp compositeOp,
         OpBuilder &builder) -> OpValidationResult {
        TT_assert(compositeOp.getInputs().size() == 3u);

        auto optAttrs = compositeOp.getCompositeAttributes();
        TT_assert(optAttrs);
        DictionaryAttr attrs = *optAttrs;

        auto kAttr = attrs.getAs<mlir::IntegerAttr>("k");
        auto numExpertsAttr = attrs.getAs<mlir::IntegerAttr>("num_experts");
        TT_assert(kAttr);
        TT_assert(numExpertsAttr);

        SmallVector<Type> resultTypes(compositeOp.getResultTypes());
        IsolatedIRValidationWrapper validator(compositeOp.getContext());
        return validator.validateOp<TopKRouterGptOp>(
            compositeOp.getOperation(), compositeOp.getLoc(), resultTypes,
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2],
            builder.getI32IntegerAttr(kAttr.getInt()),
            builder.getI32IntegerAttr(numExpertsAttr.getInt()));
      },
      // Build
      [](ttcore::CompositeOp compositeOp, OpBuilder &builder) -> Operation * {
        DictionaryAttr attrs = *compositeOp.getCompositeAttributes();
        auto kAttr = attrs.getAs<mlir::IntegerAttr>("k");
        auto numExpertsAttr = attrs.getAs<mlir::IntegerAttr>("num_experts");

        return builder.create<TopKRouterGptOp>(
            compositeOp.getLoc(), compositeOp.getResultTypes(),
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2],
            builder.getI32IntegerAttr(kAttr.getInt()),
            builder.getI32IntegerAttr(numExpertsAttr.getInt()));
      },
      /*precondition=*/nullptr};

  registry["rotary_embedding"] = CompositeEntry{
      // Validate
      [](ttcore::CompositeOp compositeOp,
         OpBuilder &builder) -> OpValidationResult {
        TT_assert(compositeOp.getInputs().size() == 3u);

        SmallVector<Type> resultTypes(compositeOp.getResultTypes());
        IsolatedIRValidationWrapper validator(compositeOp.getContext());
        return validator.validateOp<RotaryEmbeddingOp>(
            compositeOp.getOperation(), compositeOp.getLoc(), resultTypes,
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2],
            /*token_index=*/mlir::IntegerAttr(),
            /*compute_config=*/nullptr);
      },
      // Build
      [](ttcore::CompositeOp compositeOp, OpBuilder &builder) -> Operation * {
        return builder.create<RotaryEmbeddingOp>(
            compositeOp.getLoc(), compositeOp.getResultTypes(),
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2],
            /*token_index=*/mlir::IntegerAttr(),
            /*compute_config=*/nullptr);
      },
      /*precondition=*/nullptr};

  registry["flash_mla_prefill"] = CompositeEntry{
      // Validate
      [](ttcore::CompositeOp compositeOp,
         OpBuilder &builder) -> OpValidationResult {
        FlashMlaPrefillCompositeArgs args =
            extractFlashMlaPrefillArgs(compositeOp);
        TT_assert(args.headDimV);
        TT_assert(args.isCausal);

        SmallVector<Type> resultTypes(compositeOp.getResultTypes());
        IsolatedIRValidationWrapper validator(compositeOp.getContext());
        return validator.validateOp<FlashMlaPrefillOp>(
            compositeOp.getOperation(), compositeOp.getLoc(), resultTypes,
            args.query, args.key, args.value, args.attentionMask,
            static_cast<uint32_t>(args.headDimV.getValue().getZExtValue()),
            args.isCausal.getValue(), args.scale);
      },
      // Build
      [](ttcore::CompositeOp compositeOp, OpBuilder &builder) -> Operation * {
        FlashMlaPrefillCompositeArgs args =
            extractFlashMlaPrefillArgs(compositeOp);
        return builder.create<FlashMlaPrefillOp>(
            compositeOp.getLoc(), compositeOp.getResultTypes(), args.query,
            args.key, args.value, args.attentionMask,
            static_cast<uint32_t>(args.headDimV.getValue().getZExtValue()),
            args.isCausal.getValue(), args.scale);
      },
      /*precondition=*/nullptr};

  registry["indexer_score"] = CompositeEntry{
      // Validate
      [](ttcore::CompositeOp compositeOp,
         OpBuilder &builder) -> OpValidationResult {
        TT_assert(compositeOp.getInputs().size() == 3u);

        uint32_t chunkStartIdx = getIndexerScoreChunkStartIdx(compositeOp);
        SmallVector<Type> resultTypes(compositeOp.getResultTypes());
        IsolatedIRValidationWrapper validator(compositeOp.getContext());
        return validator.validateOp<IndexerScoreOp>(
            compositeOp.getOperation(), compositeOp.getLoc(), resultTypes,
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2], chunkStartIdx);
      },
      // Build
      [](ttcore::CompositeOp compositeOp, OpBuilder &builder) -> Operation * {
        uint32_t chunkStartIdx = getIndexerScoreChunkStartIdx(compositeOp);
        return builder.create<IndexerScoreOp>(
            compositeOp.getLoc(), compositeOp.getResultTypes(),
            compositeOp.getInputs()[0], compositeOp.getInputs()[1],
            compositeOp.getInputs()[2], chunkStartIdx);
      },
      // Precondition: ttnn.experimental.indexer_score is Blackhole-only.
      [](ttcore::CompositeOp compositeOp) -> LogicalResult {
        ModuleOp moduleOp = compositeOp->getParentOfType<ModuleOp>();
        auto sysDesc = moduleOp
                           ? moduleOp->getAttrOfType<ttcore::SystemDescAttr>(
                                 ttcore::SystemDescAttr::name)
                           : nullptr;
        // Without a system descriptor in scope (e.g. running the pass in
        // isolation) the architecture is unknown; defer the check to the
        // metal runtime, which fails on non-Blackhole devices.
        if (!sysDesc) {
          return success();
        }
        ttcore::Arch arch = sysDesc.getChipDesc(0).getArch().getValue();
        if (arch != ttcore::Arch::Blackhole) {
          return compositeOp.emitOpError()
                 << "ttnn.experimental.indexer_score is only supported on "
                    "Blackhole, but the target architecture is "
                 << ttcore::stringifyArch(arch) << ".";
        }
        return success();
      }};
}

// Inline the decomposition function body at the composite ops location,
// replacing the composites results with the inlined operations results.
static LogicalResult inlineDecomposition(ttcore::CompositeOp compositeOp,
                                         ModuleOp moduleOp) {
  auto decompName = compositeOp.getDecomposition();
  auto *symbolOp = SymbolTable::lookupSymbolIn(moduleOp, decompName);
  auto funcOp = dyn_cast_or_null<func::FuncOp>(symbolOp);
  if (!funcOp) {
    return compositeOp.emitOpError("decomposition function '")
           << decompName << "' not found";
  }

  OpBuilder builder(compositeOp);
  IRMapping mapping;

  for (auto [arg, input] :
       llvm::zip(funcOp.getArguments(), compositeOp.getInputs())) {
    mapping.map(arg, input);
  }

  Block &funcBody = funcOp.getBody().front();
  for (Operation &op : funcBody.without_terminator()) {
    builder.clone(op, mapping);
  }

  auto returnOp = dyn_cast<func::ReturnOp>(funcBody.getTerminator());
  if (!returnOp) {
    return compositeOp.emitOpError("decomposition function '")
           << decompName << "' must have func.return terminator";
  }

  for (auto [result, returnVal] :
       llvm::zip(compositeOp.getResults(), returnOp.getOperands())) {
    result.replaceAllUsesWith(mapping.lookup(returnVal));
  }

  compositeOp.erase();
  return success();
}

// Try to create the typed op for a registered composite.
//
// Returns nullptr when the composite should be inlined instead — either because
// the resolution mode is Inline, the composite is not in the registry, or
// validation failed (in Validate mode).
static Operation *tryCreateTypedOp(ttcore::CompositeOp compositeOp,
                                   OpBuilder &builder,
                                   CompositeResolution resolution) {
  if (resolution == CompositeResolution::Inline) {
    return nullptr;
  }

  auto &registry = getCompositeRegistry();
  auto it = registry.find(compositeOp.getCompositeName());
  if (it == registry.end()) {
    return nullptr;
  }

  auto &entry = it->second;

  if (resolution == CompositeResolution::Validate) {
    auto validationResult = entry.validate(compositeOp, builder);
    if (!validationResult.isSuccess()) {
      return nullptr;
    }
  }

  return entry.build(compositeOp, builder);
}

class TTNNResolveComposites
    : public impl::TTNNResolveCompositesBase<TTNNResolveComposites> {
public:
  using impl::TTNNResolveCompositesBase<
      TTNNResolveComposites>::TTNNResolveCompositesBase;

  void runOnOperation() final {
    registerBuiltinComposites();

    ModuleOp moduleOp = getOperation();
    llvm::DenseSet<func::FuncOp> decompositionFuncsToDelete;
    bool passFailed = false;

    moduleOp.walk([&](ttcore::CompositeOp compositeOp) {
      if (passFailed) {
        return;
      }

      auto decompName = compositeOp.getDecomposition();
      auto *symbolOp = SymbolTable::lookupSymbolIn(moduleOp, decompName);
      auto decompFunc = dyn_cast<func::FuncOp>(symbolOp);

      // Hard preconditions (e.g. architecture requirements) are enforced
      // regardless of the resolution mode, before deciding to promote or
      // inline. A failed precondition emits a diagnostic and fails the pass.
      auto &registry = getCompositeRegistry();
      auto regIt = registry.find(compositeOp.getCompositeName());
      if (regIt != registry.end() && regIt->second.precondition) {
        if (mlir::failed(regIt->second.precondition(compositeOp))) {
          passFailed = true;
          return;
        }
      }

      OpBuilder builder(compositeOp);
      Operation *typedOp =
          tryCreateTypedOp(compositeOp, builder, compositeResolution);

      if (typedOp) {
        for (auto [result, typedResult] :
             llvm::zip(compositeOp.getResults(), typedOp->getResults())) {
          result.replaceAllUsesWith(typedResult);
        }
        compositeOp.erase();
      } else {
        if (mlir::failed(inlineDecomposition(compositeOp, moduleOp))) {
          passFailed = true;
          return;
        }
      }

      if (decompFunc) {
        decompositionFuncsToDelete.insert(decompFunc);
      }
    });

    if (passFailed) {
      signalPassFailure();
      return;
    }

    // Clean up decomposition functions that are no longer referenced.
    for (func::FuncOp func : decompositionFuncsToDelete) {
      if (func && SymbolTable::symbolKnownUseEmpty(func, moduleOp)) {
        func.erase();
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
