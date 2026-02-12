// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_FUNCOPTOPROGRAM_H
#define TTMLIR_TARGET_UTILS_FUNCOPTOPROGRAM_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

#include "flatbuffers/flatbuffers.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::ttnn {

template <typename OpT>
struct Program {
  ::flatbuffers::FlatBufferBuilder *fbb;
  const char *name;
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> inputs;
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> outputs;
  std::vector<::flatbuffers::Offset<OpT>> ops;
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

inline Value getOperandThroughDPSOps(Value value) {
  auto *op = value.getDefiningOp();
  if (!op) {
    return value;
  }
  while (isa<DestinationStyleOpInterface>(op)) {
    assert(op->getResults().size() == 1);
    auto dps = cast<DestinationStyleOpInterface>(op);
    assert(dps.getNumDpsInits() == 1);
    auto *opOperand = dps.getDpsInitOperand(0);
    value = opOperand->get();
    op = value.getDefiningOp();
  }
  return value;
}

template <typename OpT, typename FnT, typename TensorFnT>
Program<OpT>
funcOpToProgram(FlatbufferObjectCache &cache, func::FuncOp entry, FnT fn,
                TensorFnT tensorValueToFlatbuffer,
                const llvm::StringMap<uint32_t> &programIndexMap,
                const llvm::StringMap<std::string> &constEvalFuncHashes) {
  OpPrintingFlags printFlags;
  printFlags = printFlags.elideLargeElementsAttrs()
                   .elideLargeResourceString()
                   .skipRegions()
                   .enableDebugInfo()
                   .assumeVerified();

  Program<OpT> program;
  program.name = entry.getSymName().data();

  for (auto &input : entry.getBody().getArguments()) {
    // Get argument encoding to determine sharding status.
    mlir::DictionaryAttr argAttrDict =
        entry.getArgAttrDict(input.getArgNumber());
    ttcore::ShardStatus shardStatus = ttcore::ShardStatus::Unsharded;
    mlir::RankedTensorType localShape =
        mlir::cast<mlir::RankedTensorType>(input.getType());

    if (argAttrDict) {
      auto runtimeTensorShardingAttr =
          argAttrDict.get(mlir::tt::ttcore::RuntimeTensorShardingAttr::name);
      if (runtimeTensorShardingAttr) {
        auto rtsAttr = mlir::cast<mlir::tt::ttcore::RuntimeTensorShardingAttr>(
            runtimeTensorShardingAttr);
        shardStatus = rtsAttr.getShardStatus().getValue();
        localShape = rtsAttr.getLocalShape();
      }
    }

    program.inputs.push_back(cache.getOrCreate(input, tensorValueToFlatbuffer,
                                               shardStatus, localShape));
  }

  mlir::AsmState printState(entry, printFlags);
  entry.getBody().walk([&](mlir::Operation *op) {
    if (auto returnOp = dyn_cast_if_present<func::ReturnOp>(op); returnOp) {
      for (auto [i, output] : llvm::enumerate(returnOp.getOperands())) {
        ttcore::ShardStatus shardStatus = ttcore::ShardStatus::Unsharded;
        mlir::RankedTensorType localShape =
            mlir::cast<mlir::RankedTensorType>(output.getType());

        auto resultAttrs = mlir::DictionaryAttr::get(op->getContext(),
                                                     entry.getResultAttrs(i));
        if (resultAttrs) {
          auto runtimeTensorShardingAttr = resultAttrs.get(
              mlir::tt::ttcore::RuntimeTensorShardingAttr::name);
          if (runtimeTensorShardingAttr) {
            auto rtsAttr =
                mlir::cast<mlir::tt::ttcore::RuntimeTensorShardingAttr>(
                    runtimeTensorShardingAttr);
            shardStatus = rtsAttr.getShardStatus().getValue();
            localShape = rtsAttr.getLocalShape();
          }
        }

        auto tensorRefResult =
            cache.getOrCreate(getOperandThroughDPSOps(output),
                              tensorValueToFlatbuffer, shardStatus, localShape);
        program.outputs.push_back(tensorRefResult);
      }
    } else {
      std::string debugStr = getOpDebugString(op, printState);
      std::string locInfo = getOpLocInfo(op);
      program.ops.push_back(fn(cache, op, programIndexMap, debugStr, locInfo,
                               constEvalFuncHashes));
    }
  });

  return program;
}

} // namespace mlir::tt::ttnn

#endif
