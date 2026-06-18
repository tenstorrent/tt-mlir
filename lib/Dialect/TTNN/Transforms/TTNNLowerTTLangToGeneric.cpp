// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Pass: --ttnn-lower-tt-lang-to-generic
//
// Rewrite each `ttnn.tt_lang_op` whose `kernel_artifact` attribute has
// been populated (by `--ttnn-resolve-tt-lang-kernels`) into an
// equivalent `ttnn.generic` op carrying a `#ttnn.program` descriptor.
//
// The `kernel_artifact` is the UTF-8 JSON payload produced by tt-xla's
// `tt_torch.tt_lang._serialize_compiled_operation`:
//
//   {
//     "format_version": 1,
//     "kernels": [
//       {"thread_type": "compute"|"noc",
//        "cpp_source":  "<embedded C++ source>",
//        "tensor_indices": [<int>, ...],
//        "kernel_config": {...}},
//       ...
//     ],
//     "core_range": {"start": [x, y], "end": [x, y]},
//     "cb_configs": [{buffer_index, data_format, page_size, total_size}, ...],
//     ...
//   }
//
// Each kernel becomes a `#ttnn.compute_kernel` / `#ttnn.read_kernel` /
// `#ttnn.write_kernel` attribute carrying the kernel's C++ `source`
// inline; `cb_configs` become `#ttnn.kernel_cb` descriptors. The
// resulting `ttnn.generic` runs through the same flatbuffer / runtime
// path as hand-written generic kernels, so the TTNN-to-flatbuffer
// emitter needs no `tt_lang_op`-specific code.

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNLOWERTTLANGTOGENERIC
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Schema version of the `kernel_artifact` JSON this pass understands. The
// emitter inserts tensor-accessor markers into every NOC kernel's
// compile-time args and the runtime expands each at launch time, so the
// artifact carries no pre-baked TensorAccessor values. Anything other than
// this version is fatal so the build fails loudly on schema drift. Bump
// this when the artifact schema changes.
constexpr int64_t EXPECTED_FORMAT_VERSION = 1;

// Map the JSON `data_format` spelling (e.g. "BFloat16") to a
// `ttcore::DataType`. The names match the enum's C++ symbol spelling 1:1.
std::optional<ttcore::DataType> parseDataType(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<ttcore::DataType>>(name)
      .Case("Float32", ttcore::DataType::Float32)
      .Case("Float16", ttcore::DataType::Float16)
      .Case("BFloat16", ttcore::DataType::BFloat16)
      .Case("BFP_Float8", ttcore::DataType::BFP_Float8)
      .Case("BFP_BFloat8", ttcore::DataType::BFP_BFloat8)
      .Case("BFP_Float4", ttcore::DataType::BFP_Float4)
      .Case("BFP_BFloat4", ttcore::DataType::BFP_BFloat4)
      .Case("UInt32", ttcore::DataType::UInt32)
      .Case("UInt16", ttcore::DataType::UInt16)
      .Case("UInt8", ttcore::DataType::UInt8)
      .Case("Int32", ttcore::DataType::Int32)
      .Default(std::nullopt);
}

std::optional<ComputeKernelMathFidelity>
parseMathFidelity(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<ComputeKernelMathFidelity>>(name)
      .Case("LoFi", ComputeKernelMathFidelity::LoFi)
      .Case("HiFi2", ComputeKernelMathFidelity::HiFi2)
      .Case("HiFi3", ComputeKernelMathFidelity::HiFi3)
      .Case("HiFi4", ComputeKernelMathFidelity::HiFi4)
      .Default(std::nullopt);
}

// Parse `core_range`: {"start": [x, y], "end": [x, y]} into a
// CoreRangeSetAttr covering the inclusive rectangle. tt-lang currently
// emits a single rectangle, so we wrap it in a one-element range set.
CoreRangeSetAttr parseCoreRange(TTLangOp op, MLIRContext *ctx,
                                const llvm::json::Object *root) {
  const llvm::json::Object *cr = root->getObject("core_range");
  if (!cr) {
    op.emitError("kernel_artifact is missing required `core_range` object.");
    return {};
  }
  const llvm::json::Array *startArr = cr->getArray("start");
  const llvm::json::Array *endArr = cr->getArray("end");
  if (!startArr || !endArr || startArr->size() != 2 || endArr->size() != 2) {
    op.emitError("kernel_artifact `core_range` must contain `start` and "
                 "`end` arrays of length 2.");
    return {};
  }
  auto sx = (*startArr)[0].getAsInteger();
  auto sy = (*startArr)[1].getAsInteger();
  auto ex = (*endArr)[0].getAsInteger();
  auto ey = (*endArr)[1].getAsInteger();
  if (!sx || !sy || !ex || !ey) {
    op.emitError("kernel_artifact `core_range.start`/`end` entries must be "
                 "integers.");
    return {};
  }
  // Casting straight to a coordinate would silently wrap negative values
  // to gigantic core coordinates. Validate non-negativity and start <=
  // end so a malformed artifact is rejected here with a diagnostic
  // pointing at the offending entry.
  if (*sx < 0 || *sy < 0 || *ex < 0 || *ey < 0) {
    op.emitError("kernel_artifact `core_range.start`/`end` coordinates must "
                 "be non-negative; got start=(")
        << *sx << "," << *sy << "), end=(" << *ex << "," << *ey << ").";
    return {};
  }
  if (*sx > *ex || *sy > *ey) {
    op.emitError("kernel_artifact `core_range` is empty: start=(")
        << *sx << "," << *sy << ") must be component-wise <= end=(" << *ex
        << "," << *ey << ").";
    return {};
  }
  auto start = CoreCoordAttr::get(ctx, static_cast<uint64_t>(*sx),
                                  static_cast<uint64_t>(*sy));
  auto end = CoreCoordAttr::get(ctx, static_cast<uint64_t>(*ex),
                                static_cast<uint64_t>(*ey));
  llvm::SmallVector<CoreRangeAttr> ranges{CoreRangeAttr::get(ctx, start, end)};
  return CoreRangeSetAttr::get(ctx, ranges);
}

// Parse each entry of the `cb_configs` array into a KernelCBAttr scoped
// to the kernel's core range. Page/total sizes pass through verbatim --
// the Python emitter already did the byte arithmetic.
std::optional<llvm::SmallVector<KernelCBAttr>>
parseCbConfigs(TTLangOp op, MLIRContext *ctx, const llvm::json::Array *cbs,
               CoreRangeSetAttr coreRanges) {
  if (!cbs) {
    op.emitError("kernel_artifact is missing required `cb_configs` array.");
    return std::nullopt;
  }
  llvm::SmallVector<KernelCBAttr> out;
  out.reserve(cbs->size());
  for (size_t i = 0; i < cbs->size(); ++i) {
    const llvm::json::Object *cb = (*cbs)[i].getAsObject();
    if (!cb) {
      op.emitError("cb_configs[") << i << "] is not a JSON object.";
      return std::nullopt;
    }
    auto bufferIndex = cb->getInteger("buffer_index");
    auto pageSize = cb->getInteger("page_size");
    auto totalSize = cb->getInteger("total_size");
    auto dtName = cb->getString("data_format");
    if (!bufferIndex || !pageSize || !totalSize || !dtName) {
      op.emitError("cb_configs[")
          << i
          << "] is missing one of "
             "buffer_index/page_size/total_size/data_format.";
      return std::nullopt;
    }
    // `llvm::json::Value::getAsInteger` returns int64_t but the attr
    // schema (and tt-metal's CB descriptor) takes uint32_t. Bound-check
    // each field so a malformed artifact (negative, or > UINT32_MAX) is
    // rejected here rather than wrapping into a bad CB layout late inside
    // tt-metal.
    auto checkU32 = [&](int64_t v, llvm::StringRef field) -> bool {
      if (v < 0 ||
          static_cast<uint64_t>(v) > std::numeric_limits<uint32_t>::max()) {
        op.emitError("cb_configs[") << i << "]." << field << " = " << v
                                    << " is out of range for uint32_t.";
        return false;
      }
      return true;
    };
    if (!checkU32(*bufferIndex, "buffer_index") ||
        !checkU32(*pageSize, "page_size") ||
        !checkU32(*totalSize, "total_size")) {
      return std::nullopt;
    }
    std::optional<ttcore::DataType> dt = parseDataType(*dtName);
    if (!dt) {
      op.emitError("cb_configs[")
          << i << "] has unknown data_format " << *dtName << ".";
      return std::nullopt;
    }
    llvm::SmallVector<KernelCBFormatAttr> formats{
        KernelCBFormatAttr::get(ctx, static_cast<uint32_t>(*bufferIndex), *dt,
                                static_cast<uint32_t>(*pageSize))};
    out.push_back(KernelCBAttr::get(
        ctx, static_cast<uint32_t>(*totalSize), coreRanges, formats,
        /*buffer=*/KernelCBGlobalBufferAddressOfTensorAttr()));
  }
  return out;
}

// Build the per-kernel `ct_args` and `common_rt_args` lists.
//
// * `ct_args` starts with the CB-index prefix (one entry per CB). For NOC
//   kernels we then append one `kernel_arg_tensor_accessor_args` marker
//   per operand (declaration order == io-tensors order, so the marker's
//   operand_index is the identity index). Compute kernels read/write
//   through CBs and so get the CB-index prefix only.
// * `common_rt_args` maps each kernel's `tensor_indices` to
//   `kernel_arg_address_of_tensor` records.
mlir::LogicalResult
buildKernelArgs(TTLangOp op, MLIRContext *ctx,
                const llvm::json::Object *kernelObj, llvm::StringRef threadType,
                uint32_t numCbs, unsigned numOperands,
                llvm::SmallVectorImpl<mlir::Attribute> &ctArgs,
                llvm::SmallVectorImpl<mlir::Attribute> &commonRtArgs) {
  for (uint32_t cb = 0; cb < numCbs; ++cb) {
    ctArgs.push_back(KernelArgCBBufferIndexAttr::get(ctx, cb));
  }

  if (threadType == "noc") {
    for (unsigned i = 0; i < numOperands; ++i) {
      ctArgs.push_back(KernelArgTensorAccessorArgsAttr::get(ctx, i));
    }
  }

  if (const llvm::json::Array *idxs = kernelObj->getArray("tensor_indices")) {
    for (size_t i = 0; i < idxs->size(); ++i) {
      auto idx = (*idxs)[i].getAsInteger();
      if (!idx) {
        op.emitError("kernel_artifact tensor_indices[")
            << i << "] is not an integer.";
        return mlir::failure();
      }
      if (*idx < 0) {
        op.emitError("kernel_artifact tensor_indices[")
            << i << "] is negative (" << *idx << ").";
        return mlir::failure();
      }
      commonRtArgs.push_back(
          KernelArgAddressOfTensorAttr::get(ctx, static_cast<uint64_t>(*idx)));
    }
  }
  return mlir::success();
}

// Build the kernel descriptor attribute (compute/read/write) for one
// `kernels[i]` JSON entry.
mlir::Attribute buildKernelAttr(TTLangOp op, MLIRContext *ctx,
                                const llvm::json::Object *kobj,
                                CoreRangeSetAttr coreRanges, uint32_t numCbs,
                                unsigned numOperands, size_t i) {
  std::optional<llvm::StringRef> src = kobj->getString("cpp_source");
  std::optional<llvm::StringRef> thr = kobj->getString("thread_type");
  if (!src || !thr) {
    op.emitError("kernels[")
        << i << "] is missing required `cpp_source` / `thread_type`.";
    return {};
  }
  // Whitelist `thread_type`: only "compute" and "noc" map to kernel
  // descriptor kinds. Reject anything else with a diagnostic pointing at
  // the offending entry.
  if (*thr != "compute" && *thr != "noc") {
    op.emitError("kernels[")
        << i << "].thread_type=\"" << *thr
        << "\" is not supported (expected \"compute\" or \"noc\").";
    return {};
  }

  const llvm::json::Object *cfg = kobj->getObject("kernel_config");
  if (!cfg) {
    op.emitError("kernels[") << i << "] is missing `kernel_config`.";
    return {};
  }
  std::optional<llvm::StringRef> type = cfg->getString("type");
  if (!type) {
    op.emitError("kernels[") << i << "].kernel_config is missing `type`.";
    return {};
  }

  // Cross-check thread_type against kernel_config.type: a compute thread
  // can only host a ComputeKernelConfig, a NOC thread only a
  // Reader/WriterKernelConfig. A mismatch would wire the wrong NOC /
  // fidelity into tt-metal at launch with no clear failure mode, so
  // reject it here.
  const bool isCompute = *type == "ComputeKernelConfig";
  const bool isReader = *type == "ReaderKernelConfig";
  const bool isWriter = *type == "WriterKernelConfig";
  if (*thr == "compute" && !isCompute) {
    op.emitError("kernels[")
        << i
        << "]: thread_type=\"compute\" requires "
           "kernel_config.type=\"ComputeKernelConfig\", got \""
        << *type << "\".";
    return {};
  }
  if (*thr == "noc" && !(isReader || isWriter)) {
    op.emitError("kernels[")
        << i
        << "]: thread_type=\"noc\" requires kernel_config.type in "
           "{\"ReaderKernelConfig\", \"WriterKernelConfig\"}, got \""
        << *type << "\".";
    return {};
  }

  auto source = mlir::StringAttr::get(ctx, *src);

  llvm::SmallVector<mlir::Attribute> ctArgs;
  llvm::SmallVector<mlir::Attribute> commonRtArgs;
  if (mlir::failed(buildKernelArgs(op, ctx, kobj, *thr, numCbs, numOperands,
                                   ctArgs, commonRtArgs))) {
    return {};
  }

  if (isCompute) {
    auto fidelity =
        parseMathFidelity(cfg->getString("math_fidelity").value_or("HiFi4"))
            .value_or(ComputeKernelMathFidelity::HiFi4);
    bool fp32DestAccEn = cfg->getBoolean("fp32_dest_acc_en").value_or(false);
    bool dstFullSyncEn = cfg->getBoolean("dst_full_sync_en").value_or(false);
    bool bfp8PackPrecise = cfg->getBoolean("bfp8_pack_precise").value_or(false);
    bool mathApproxMode = cfg->getBoolean("math_approx_mode").value_or(false);
    return SourceComputeKernelAttr::get(
        ctx, source, coreRanges, fidelity, fp32DestAccEn, dstFullSyncEn,
        /*unpack_to_dest_modes=*/
        llvm::ArrayRef<ComputeKernelUnpackToDestMode>{}, bfp8PackPrecise,
        mathApproxMode, commonRtArgs,
        /*rt_args=*/llvm::ArrayRef<CoreRuntimeArgsAttr>{}, ctArgs);
  }
  if (isReader) {
    return SourceReadKernelAttr::get(
        ctx, source, coreRanges, commonRtArgs,
        /*rt_args=*/llvm::ArrayRef<CoreRuntimeArgsAttr>{}, ctArgs);
  }

  // isWriter
  return SourceWriteKernelAttr::get(
      ctx, source, coreRanges, commonRtArgs,
      /*rt_args=*/llvm::ArrayRef<CoreRuntimeArgsAttr>{}, ctArgs);
}

// Build a `#ttnn.program` from the `kernel_artifact` JSON.
ProgramAttr buildProgramAttr(TTLangOp op, MLIRContext *ctx,
                             llvm::StringRef artifactJson,
                             unsigned numOperands) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(artifactJson);
  if (!parsed) {
    op.emitError("kernel_artifact is not valid JSON: ")
        << llvm::toString(parsed.takeError());
    return {};
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    op.emitError("kernel_artifact root is not a JSON object.");
    return {};
  }

  // Schema version gate: anything other than the expected version is fatal.
  auto fv = root->getInteger("format_version");
  if (!fv || *fv != EXPECTED_FORMAT_VERSION) {
    op.emitError("kernel_artifact has unsupported format_version (expected ")
        << EXPECTED_FORMAT_VERSION << ").";
    return {};
  }

  CoreRangeSetAttr coreRanges = parseCoreRange(op, ctx, root);
  if (!coreRanges) {
    return {};
  }

  std::optional<llvm::SmallVector<KernelCBAttr>> cbs =
      parseCbConfigs(op, ctx, root->getArray("cb_configs"), coreRanges);
  if (!cbs) {
    return {};
  }
  const uint32_t numCbs = static_cast<uint32_t>(cbs->size());

  const llvm::json::Array *kernelArr = root->getArray("kernels");
  if (!kernelArr || kernelArr->empty()) {
    op.emitError("kernel_artifact is missing required non-empty `kernels` "
                 "array.");
    return {};
  }

  llvm::SmallVector<mlir::Attribute> kernels;
  kernels.reserve(kernelArr->size());
  for (size_t i = 0; i < kernelArr->size(); ++i) {
    const llvm::json::Object *kobj = (*kernelArr)[i].getAsObject();
    if (!kobj) {
      op.emitError("kernels[") << i << "] is not a JSON object.";
      return {};
    }
    mlir::Attribute kernel =
        buildKernelAttr(op, ctx, kobj, coreRanges, numCbs, numOperands, i);
    if (!kernel) {
      return {};
    }
    kernels.push_back(kernel);
  }

  // PipeNet semaphores are not yet plumbed through the artifact; emit an
  // empty list. A future schema bump will carry the structured semaphore
  // layout (`num_pipe_nets`, IDs, core ranges).
  return ProgramAttr::get(ctx, kernels, *cbs,
                          /*semaphores=*/llvm::ArrayRef<KernelSemaphoreAttr>{});
}

// Lower one `ttnn.tt_lang_op` to a `ttnn.generic`. Returns failure (with
// a diagnostic already emitted) on a malformed/unset artifact.
mlir::LogicalResult lowerTTLangOpToGeneric(TTLangOp op) {
  MLIRContext *ctx = op.getContext();

  mlir::StringAttr artifactAttr = op.getKernelArtifactAttr();
  if (!artifactAttr || artifactAttr.getValue().empty()) {
    return op.emitError(
        "ttnn.tt_lang_op has an empty/unset `kernel_artifact`; run "
        "`--ttnn-resolve-tt-lang-kernels` before "
        "`--ttnn-lower-tt-lang-to-generic`.");
  }

  // `ttnn.tt_lang_op` is destination-passing style with `arg_roles ==
  // in* out+`: operand declaration order already matches the io-tensors
  // order the runtime/generic emitter expects (ins first, then outs), and
  // result `i` ties to operand `numIns + i`.
  unsigned numOperands = op.getInputs().size();
  unsigned numResults = op.getResults().size();
  assert(numResults > 0 && numResults <= numOperands &&
         "tt_lang_op verifier guarantees 1..=numOperands DPS results");
  unsigned numIns = numOperands - numResults;

  ProgramAttr program =
      buildProgramAttr(op, ctx, artifactAttr.getValue(), numOperands);
  if (!program) {
    return mlir::failure();
  }

  OpBuilder builder(op);
  // `ttnn.generic` writes in place into its "out" operands and has no
  // results; downstream IR references the out operands directly. Replace
  // each tt_lang_op result with its tied "out" operand before erasing.
  builder.create<GenericOp>(op.getLoc(), op.getInputs(),
                            /*additional_args=*/ValueRange{}, program);

  for (unsigned r = 0; r < numResults; ++r) {
    op.getResult(r).replaceAllUsesWith(op.getInputs()[numIns + r]);
  }
  op.erase();
  return mlir::success();
}

class TTNNLowerTTLangToGeneric
    : public impl::TTNNLowerTTLangToGenericBase<TTNNLowerTTLangToGeneric> {
public:
  using impl::TTNNLowerTTLangToGenericBase<
      TTNNLowerTTLangToGeneric>::TTNNLowerTTLangToGenericBase;

  void runOnOperation() final {
    llvm::SmallVector<TTLangOp> ops;
    getOperation().walk([&](TTLangOp op) { ops.push_back(op); });

    for (TTLangOp op : ops) {
      if (mlir::failed(lowerTTLangOpToGeneric(op))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
