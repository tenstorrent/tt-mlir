// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_FUNCOPTOPROGRAM_H
#define TTMLIR_TARGET_UTILS_FUNCOPTOPROGRAM_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "flatbuffers/flatbuffers.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::ttnn {

template <typename OpT>
struct Program {
  ::flatbuffers::FlatBufferBuilder *fbb;
  std::string name;
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> inputs;
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> outputs;
  std::vector<::flatbuffers::Offset<OpT>> ops;
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::GlobalSemaphoreRef>>
      semaphoreInputs;
};

inline std::string getOpDebugString(mlir::Operation *op,
                                    mlir::AsmState &printState) {
  std::string str;
  llvm::raw_string_ostream os(str);
  op->print(os, printState);
  return str;
};

inline std::string getOpLocInfo(mlir::Operation *op) {
  std::string str;
  llvm::raw_string_ostream os(str);
  op->getLoc().print(os);
  return str;
}

// Scalar args are serialized in 32-bit unsigned integers
inline bool isSupportedScalarArgType(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type) &&
         type.getIntOrFloatBitWidth() <= 32;
}

inline Value getOperandThroughDPSOps(Value value) {
  auto *op = value.getDefiningOp();
  while (isa_and_nonnull<DestinationStyleOpInterface>(op)) {
    auto dps = cast<DestinationStyleOpInterface>(op);
    OpOperand *opOperand = dps.getTiedOpOperand(cast<OpResult>(value));
    assert(opOperand &&
           "DPS op result must be tied to a destination init operand");
    value = opOperand->get();
    op = value.getDefiningOp();
  }
  return value;
}

inline OpPrintingFlags getProgramDebugPrintingFlags() {
  OpPrintingFlags printFlags;
  return printFlags.elideLargeElementsAttrs()
      .elideLargeResourceString()
      .skipRegions()
      .enableDebugInfo()
      .assumeVerified();
}

/// Serializes `region` as an additional top-level flatbuffer program and
/// returns its index.
///
/// Regions of control-flow ops (today only `ttnn.while`) are not flattened into
/// the enclosing program: each one is executed by its own nested
/// `ProgramExecutor` with its own tensor pool, so it needs to be its own
/// `Program` that the parent references by index. The callback runs in the
/// middle of building the parent program, so appending to the binary's program
/// vector is the caller's business.
using RegionProgramEmitterFn = std::function<uint32_t(
    FlatbufferObjectCache &, mlir::Region &, llvm::StringRef name)>;

/// Emits the operations of `block` into `program.ops`, skipping the terminator.
///
/// Only the operations directly in `block` are emitted; anything nested in a
/// region is reached through `emitRegionProgram` instead. A recursive walk here
/// would flatten those ops into the parent program and, being post-order, emit
/// them before the op that owns them.
template <typename OpT, typename FnT>
void blockOpsToProgram(Program<OpT> &program, FlatbufferObjectCache &cache,
                       mlir::Block &block, FnT fn, mlir::AsmState &printState,
                       const llvm::StringMap<uint32_t> &programIndexMap,
                       const llvm::StringMap<std::string> &constEvalFuncHashes,
                       const RegionProgramEmitterFn &emitRegionProgram) {
  for (mlir::Operation &op : block.without_terminator()) {
    std::string debugStr = getOpDebugString(&op, printState);
    std::string locInfo = getOpLocInfo(&op);
    program.ops.push_back(fn(cache, &op, programIndexMap, debugStr, locInfo,
                             constEvalFuncHashes, emitRegionProgram));
  }
}

template <typename OpT, typename FnT, typename TensorFnT>
Program<OpT>
regionToProgram(FlatbufferObjectCache &cache, mlir::Region &region,
                llvm::StringRef name, FnT fn, TensorFnT tensorValueToFlatbuffer,
                mlir::AsmState &printState,
                const llvm::StringMap<uint32_t> &programIndexMap,
                const llvm::StringMap<std::string> &constEvalFuncHashes,
                const RegionProgramEmitterFn &emitRegionProgram) {
  assert(region.hasOneBlock() &&
         "region programs are emitted from a single block");

  Program<OpT> program;
  program.name = name.str();

  mlir::Block &block = region.front();

  // Region arguments carry no shard-status or local-shape attributes, so they
  // are described like any other intermediate value in a program.
  for (mlir::BlockArgument arg : block.getArguments()) {
    program.inputs.push_back(cache.getOrCreateNoSharding(
        arg, tensorValueToFlatbuffer, /*local_shape=*/std::nullopt));
  }

  blockOpsToProgram(program, cache, block, fn, printState, programIndexMap,
                    constEvalFuncHashes, emitRegionProgram);

  for (mlir::Value yielded : block.getTerminator()->getOperands()) {
    program.outputs.push_back(cache.getOrCreateNoSharding(
        getOperandThroughDPSOps(yielded), tensorValueToFlatbuffer,
        /*local_shape=*/std::nullopt));
  }

  return program;
}

template <typename OpT, typename FnT, typename TensorFnT>
Program<OpT>
funcOpToProgram(FlatbufferObjectCache &cache, func::FuncOp entry, FnT fn,
                TensorFnT tensorValueToFlatbuffer,
                const llvm::StringMap<uint32_t> &programIndexMap,
                const llvm::StringMap<std::string> &constEvalFuncHashes,
                const RegionProgramEmitterFn &emitRegionProgram) {
  OpPrintingFlags printFlags = getProgramDebugPrintingFlags();

  Program<OpT> program;
  program.name = entry.getSymName().str();

  for (auto &input : entry.getBody().getArguments()) {
    if (mlir::isa<mlir::tt::ttnn::GlobalSemaphoreType>(input.getType())) {
      program.semaphoreInputs.push_back(
          cache.getOrCreate(input, [](FlatbufferObjectCache &c, mlir::Value) {
            return ::tt::target::ttnn::CreateGlobalSemaphoreRef(
                *c.fbb, c.nextGlobalId());
          }));
      continue;
    }

    if (!isa<RankedTensorType>(input.getType())) {
      if (!isSupportedScalarArgType(input.getType())) {
        llvm::report_fatal_error(
            "Unsupported non-tensor program argument type in "
            "TTNN-to-flatbuffer "
            "lowering; only integer/float scalars up to 32 bits are supported");
      }

      program.inputs.push_back(
          cache.getOrCreate(input, [](FlatbufferObjectCache &c, mlir::Value) {
            // Scalars are represented as 1-element UInt32 tensors at the
            // runtime layer regardless of their original type (see
            // isSupportedScalarArgType above and runtime/lib/ttnn/runtime.cpp
            // createScalarTensorImpl).
            ttcore::DataType dtype = ttcore::DataType::UInt32;
            std::vector<int32_t> shape = {1};
            std::vector<int32_t> meshShape = {1, 1};

            ::tt::target::Dim2d tileShape(1, 1);
            auto memoryDesc = ::tt::target::ttnn::CreateMemoryDesc(
                *c.fbb, ::tt::target::ttnn::StorageType::Host, &tileShape,
                toFlatbuffer(c, dtype),
                /* memory_config=*/0);
            auto layoutDesc = ::tt::target::ttnn::CreateLayoutDesc(
                *c.fbb, ::tt::target::OOBVal::Undef, memoryDesc);
            auto tensorDesc = ::tt::target::ttnn::CreateTensorDescDirect(
                *c.fbb, &shape, &meshShape, layoutDesc,
                ::tt::target::ttnn::ShardStatus::Unsharded,
                /* local_shape */ nullptr);
            return ::tt::target::ttnn::CreateTensorRef(*c.fbb, c.nextGlobalId(),
                                                       tensorDesc);
          }));
      continue;
    }

    // Get argument encoding to determine sharding status.
    mlir::DictionaryAttr argAttrDict =
        entry.getArgAttrDict(input.getArgNumber());
    ttcore::ShardStatus shardStatus = ttcore::ShardStatus::Unsharded;
    mlir::RankedTensorType localShape =
        mlir::cast<mlir::RankedTensorType>(input.getType());

    if (argAttrDict) {
      auto shardStatusAttr =
          argAttrDict.get(mlir::tt::ttcore::ShardStatusAttr::name);
      if (shardStatusAttr) {
        auto ssAttr =
            mlir::cast<mlir::tt::ttcore::ShardStatusAttr>(shardStatusAttr);
        shardStatus = ssAttr.getValue();
      }

      auto localShapeAttr =
          argAttrDict.get(mlir::tt::ttcore::LocalShapeAttr::name);
      if (localShapeAttr) {
        auto lsAttr =
            mlir::cast<mlir::tt::ttcore::LocalShapeAttr>(localShapeAttr);
        localShape = mlir::cast<mlir::RankedTensorType>(lsAttr.getLocalShape());
      }
    }

    program.inputs.push_back(cache.getOrCreate(input, tensorValueToFlatbuffer,
                                               shardStatus, localShape));
  }

  assert(entry.getBody().hasOneBlock() &&
         "programs are emitted from a single block");

  mlir::AsmState printState(entry, printFlags);
  mlir::Block &entryBlock = entry.getBody().front();
  blockOpsToProgram(program, cache, entryBlock, fn, printState, programIndexMap,
                    constEvalFuncHashes, emitRegionProgram);

  if (auto returnOp = dyn_cast<func::ReturnOp>(entryBlock.getTerminator())) {
    for (auto [i, output] : llvm::enumerate(returnOp.getOperands())) {
      ttcore::ShardStatus shardStatus = ttcore::ShardStatus::Unsharded;
      mlir::RankedTensorType localShape =
          mlir::cast<mlir::RankedTensorType>(output.getType());

      auto resultAttrs = mlir::DictionaryAttr::get(entry.getContext(),
                                                   entry.getResultAttrs(i));
      if (resultAttrs) {
        auto shardStatusAttr =
            resultAttrs.get(mlir::tt::ttcore::ShardStatusAttr::name);
        if (shardStatusAttr) {
          auto ssAttr =
              mlir::cast<mlir::tt::ttcore::ShardStatusAttr>(shardStatusAttr);
          shardStatus = ssAttr.getValue();
        }

        auto localShapeAttr =
            resultAttrs.get(mlir::tt::ttcore::LocalShapeAttr::name);
        if (localShapeAttr) {
          auto lsAttr =
              mlir::cast<mlir::tt::ttcore::LocalShapeAttr>(localShapeAttr);
          localShape =
              mlir::cast<mlir::RankedTensorType>(lsAttr.getLocalShape());
        }
      }

      auto tensorRefResult =
          cache.getOrCreate(getOperandThroughDPSOps(output),
                            tensorValueToFlatbuffer, shardStatus, localShape);
      program.outputs.push_back(tensorRefResult);
    }
  }

  return program;
}

} // namespace mlir::tt::ttnn

#endif
