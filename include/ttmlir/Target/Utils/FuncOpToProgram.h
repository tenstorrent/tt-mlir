// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_FUNCOPTOPROGRAM_H
#define TTMLIR_TARGET_UTILS_FUNCOPTOPROGRAM_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  const char *name;
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
    if (mlir::isa<mlir::tt::ttnn::GlobalSemaphoreType>(input.getType())) {
      program.semaphoreInputs.push_back(
          cache.getOrCreate(input, [](FlatbufferObjectCache &c, mlir::Value) {
            return ::tt::target::ttnn::CreateGlobalSemaphoreRef(
                *c.fbb, c.nextGlobalId());
          }));
      continue;
    }

    if (!isa<RankedTensorType>(input.getType())) {
      assert((mlir::isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType>(input.getType())) && "expected supported scalar type");

      program.inputs.push_back(cache.getOrCreate(input, [](FlatbufferObjectCache &c, mlir::Value value) {
        // Scalars are always represented as 1-element UInt32 tensors at the
        // runtime layer (see runtime/lib/ttnn/runtime.cpp createScalarTensor
        // and the UINT32 assert in runtime/lib/ttnn/types/types.cpp:128).
        // The original scalar type doesn't matter — the runtime memcpys the
        // raw bytes into a uint32 buffer regardless.
        ttcore::DataType dtype = ttcore::DataType::UInt32;
        std::vector<int32_t> shape = {1};
        std::vector<int32_t> meshShape = {1, 1};

        ::tt::target::Dim2d tileShape(1, 1);
        auto memoryDesc = ::tt::target::ttnn::CreateMemoryDesc(
            *c.fbb,
            ::tt::target::ttnn::StorageType::Host,    
            &tileShape,
            toFlatbuffer(c, dtype),                   
            /* memory_config=*/0);                   
        auto layoutDesc = ::tt::target::ttnn::CreateLayoutDesc(
            *c.fbb,
            ::tt::target::OOBVal::Undef,        
            memoryDesc);
        auto tensorDesc = ::tt::target::ttnn::CreateTensorDescDirect(
            *c.fbb, &shape, &meshShape, layoutDesc,
            ::tt::target::ttnn::ShardStatus::Unsharded,
            /* local_shape */ nullptr);
        return ::tt::target::ttnn::CreateTensorRef(
            *c.fbb, c.nextGlobalId(), tensorDesc);
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
