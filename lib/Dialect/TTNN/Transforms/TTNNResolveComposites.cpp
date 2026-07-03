// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
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

struct CompositeEntry {
  CompositeValidatorFn validate;
  CompositeBuilderFn build;
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

// Operands and attributes recovered from a "moe_decode" composite. moe_decode
// fuses all_to_all_dispatch_metadata + weight-prep + moe_compute; resolving it
// here is the single place the a2a->moe boundary is wired. The device-derived
// tilize-drain core is finalized later by TTNNDeduceMoEComputeLayouts, so this
// build is device-free (works in force-promote without a device).
struct MoeDecodeCompositeArgs {
  Value tokens;
  Value expertIndices;
  Value expertScores;
  Value expertMapping;
  Value w0, w1, w2;
  Value bias0, bias1, bias2; // null when has_bias is false
  bool hasBias = false;
  int64_t numDevices = 0;
  int64_t clusterAxis = 0;
  int64_t layerId = 0;
  int64_t outputHeightShardDim = 0;
  int64_t intermediateSize = 0;
  std::optional<int64_t> bhRingSize;
  ttcore::MoEActivationFunction activation =
      ttcore::MoEActivationFunction::Silu;
};

// inputs: [tokens, expert_indices, expert_scores, expert_mapping, w0, w1, w2,
//          (bias_0, bias_1, bias_2)?]  (7 without bias, 10 with).
static MoeDecodeCompositeArgs
extractMoeDecodeArgs(ttcore::CompositeOp compositeOp) {
  auto inputs = compositeOp.getInputs();
  TT_assertv((inputs.size() == 7u || inputs.size() == 10u),
             "moe_decode expects 7 (no bias) or 10 (bias) inputs, got {}",
             inputs.size());

  MoeDecodeCompositeArgs a;
  a.tokens = inputs[0];
  a.expertIndices = inputs[1];
  a.expertScores = inputs[2];
  a.expertMapping = inputs[3];
  a.w0 = inputs[4];
  a.w1 = inputs[5];
  a.w2 = inputs[6];
  a.hasBias = inputs.size() == 10u;
  if (a.hasBias) {
    a.bias0 = inputs[7];
    a.bias1 = inputs[8];
    a.bias2 = inputs[9];
  }

  DictionaryAttr attrs = compositeOp.getCompositeAttributes().value_or(nullptr);
  TT_assert(attrs);
  auto readInt = [&](StringRef name) -> int64_t {
    auto v = attrs.getAs<mlir::IntegerAttr>(name);
    TT_assertv(v, "moe_decode composite missing integer attribute '{}'",
               name.str());
    return v.getInt();
  };
  a.numDevices = readInt("num_devices");
  a.clusterAxis = readInt("cluster_axis");
  a.layerId = readInt("layer_id");
  a.outputHeightShardDim = readInt("output_height_shard_dim");
  a.intermediateSize = readInt("intermediate_size");
  if (auto br = attrs.getAs<mlir::IntegerAttr>("bh_ring_size")) {
    a.bhRingSize = br.getInt();
  }
  if (auto act = attrs.getAs<mlir::StringAttr>("activation_function")) {
    if (auto sym = ttcore::symbolizeMoEActivationFunction(act.getValue())) {
      a.activation = *sym;
    }
  }
  return a;
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
      }};

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
      }};

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
      }};

  registry["moe_decode"] = CompositeEntry{
      // Validate: moe_compute / all_to_all_dispatch_metadata are OpModelExempt,
      // so the contract is enforced by extractMoeDecodeArgs + the typed ops'
      // verifiers after build. Always promotable.
      [](ttcore::CompositeOp compositeOp, OpBuilder &) -> OpValidationResult {
        (void)extractMoeDecodeArgs(compositeOp);
        return OpValidationResult::success();
      },
      // Build all_to_all_dispatch_metadata -> weight-prep -> moe_compute,
      // wiring
      // moe's dispatched/indices/scores straight from a2a's results so
      // TTNNDeduceMoEComputeLayouts can pin both ops onto one drain core.
      // Result
      // types are placeholders finalized by workarounds + the deduce pass.
      [](ttcore::CompositeOp compositeOp, OpBuilder &builder) -> Operation * {
        MoeDecodeCompositeArgs a = extractMoeDecodeArgs(compositeOp);
        Location loc = compositeOp.getLoc();
        MLIRContext *ctx = compositeOp.getContext();

        mlir::IRRewriter rewriter(builder);
        rewriter.setInsertionPoint(compositeOp);

        auto tokensTy = cast<RankedTensorType>(a.tokens.getType());
        auto idxTy = cast<RankedTensorType>(a.expertIndices.getType());
        auto scrTy = cast<RankedTensorType>(a.expertScores.getType());
        int64_t M = tokensTy.getShape()[tokensTy.getRank() - 2];
        int64_t H = tokensTy.getShape().back();
        int64_t K = idxTy.getShape().back();
        int64_t totalTokens = a.numDevices * M;

        // a2a output placeholder types ([1, tokens_global, C]); deduce +
        // workarounds finalize their layouts (indices/scores onto the drain
        // core, dispatched to DRAM interleaved).
        auto dispatchedTy = ttnn::utils::RankedTensorTypeFactory::create(
            tokensTy, llvm::SmallVector<int64_t>{1, totalTokens, H});
        auto outIdxTy = ttnn::utils::RankedTensorTypeFactory::create(
            idxTy, llvm::SmallVector<int64_t>{1, totalTokens, K});
        auto outScrTy = ttnn::utils::RankedTensorTypeFactory::create(
            scrTy, llvm::SmallVector<int64_t>{1, totalTokens, K});

        // Persistent-mode a2a: the dispatched/indices/scores buffers and the
        // cross-device semaphore are left unbound (the DistributedOpInterface
        // prelude hooks materialize them); the drain core is internal (the
        // kernel derives it from the metadata shard spec that
        // TTNNDeduceMoEComputeLayouts pins). Mirrors the TTIRToTTNN a2a build.
        auto a2a = rewriter.create<ttnn::AllToAllDispatchMetadataOp>(
            loc, dispatchedTy, outIdxTy, outScrTy, a.tokens, a.expertIndices,
            a.expertScores, a.expertMapping, /*dispatched_buffer=*/Value(),
            /*indices_buffer=*/Value(), /*scores_buffer=*/Value(),
            /*cross_device_semaphore=*/Value(),
            rewriter.getI64IntegerAttr(a.numDevices),
            rewriter.getI64IntegerAttr(a.clusterAxis));

        Value device =
            ttnn::utils::getOrInsertDevice(rewriter, compositeOp).getResult();

        // Prepack weights. Result types are placeholders refined by the deduce
        // pass via OpModel (mirrors MoeComputeOpConversionPattern). hidden_size
        // = w0 K dim (logical shape (L, E, K, N)).
        auto w0Ty = cast<RankedTensorType>(a.w0.getType());
        auto w2Ty = cast<RankedTensorType>(a.w2.getType());
        auto hiddenSizeAttr = rewriter.getUI32IntegerAttr(w0Ty.getShape()[2]);
        auto intermediateSizeAttr =
            rewriter.getUI32IntegerAttr(a.intermediateSize);

        auto w0w1Prepared =
            rewriter.create<ttnn::PrepareMoEComputeW0W1WeightsOp>(
                loc, /*placeholder=*/w0Ty, a.w0, a.w1, a.bias0, a.bias1, device,
                hiddenSizeAttr, intermediateSizeAttr);
        auto w2Prepared = rewriter.create<ttnn::PrepareMoEComputeW2WeightsOp>(
            loc, /*placeholder=*/w2Ty, a.w2, a.bias2, device, hiddenSizeAttr,
            intermediateSizeAttr);

        // Fabric-mux cores: same 3x3 default at (1,1) as the standalone
        // MoeComputeOpConversionPattern.
        ttnn::CoreRangeSetAttr muxCoreRangeSet = ttnn::CoreRangeSetAttr::get(
            ctx,
            ttnn::CoreRangeAttr::get(ctx, ttnn::CoreCoordAttr::get(ctx, 1, 1),
                                     ttnn::CoreCoordAttr::get(ctx, 3, 3)));

        auto activationAttr =
            ttcore::MoEActivationFunctionAttr::get(ctx, a.activation);

        // optional_output_tensor + cross_device_semaphore are left unbound; the
        // MoeComputeOp DistributedOpInterface hooks bind them in the prelude.
        return rewriter.create<ttnn::MoeComputeOp>(
            loc, compositeOp.getResultTypes()[0], a2a.getDispatched(),
            a2a.getIndices(), a2a.getScores(), a.expertMapping,
            w0w1Prepared.getResult(), w2Prepared.getResult(),
            /*optional_output_tensor=*/Value(),
            /*cross_device_semaphore=*/Value(), device,
            rewriter.getUI32IntegerAttr(a.layerId),
            rewriter.getUI32IntegerAttr(a.outputHeightShardDim),
            intermediateSizeAttr, rewriter.getBoolAttr(a.hasBias),
            activationAttr, rewriter.getUI32IntegerAttr(a.clusterAxis),
            /*num_links=*/mlir::IntegerAttr(),
            /*topology=*/ttcore::TopologyAttr(), muxCoreRangeSet);
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
