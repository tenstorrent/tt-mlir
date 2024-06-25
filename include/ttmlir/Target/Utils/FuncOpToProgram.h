// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_UTILS_PROGRAMBUILDER_H
#define TTMLIR_TARGET_UTILS_PROGRAMBUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

#include "flatbuffers/flatbuffers.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt {

template <typename OpT> struct Program {
  ::flatbuffers::FlatBufferBuilder *fbb;
  const char *name;
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> inputs;
  std::vector<::flatbuffers::Offset<::tt::target::TensorRef>> outputs;
  std::vector<::flatbuffers::Offset<OpT>> ops;
};

inline std::string getOpDebugString(mlir::Operation *op,
                                    OpPrintingFlags printFlags) {
  std::string str;
  llvm::raw_string_ostream os(str);
  op->print(os, printFlags);
  return str;
};

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

template <typename OpT, typename FnT>
Program<OpT> funcOpToProgram(FlatbufferObjectCache &cache, func::FuncOp entry,
                             FnT fn) {
  constexpr uint64_t kHostAllocatedAddress = 0;
  constexpr uint64_t kHostAllocatedSize = 0;

  OpPrintingFlags printFlags;
  printFlags = printFlags.elideLargeElementsAttrs()
                   .elideLargeResourceString()
                   .skipRegions()
                   .enableDebugInfo();

  Program<OpT> program;
  program.name = entry.getSymName().data();

  for (auto &input : entry.getBody().getArguments()) {
    program.inputs.push_back(cache.getOrCreate(input, tensorValueToFlatbuffer,
                                               kHostAllocatedAddress,
                                               kHostAllocatedSize));
  }

  entry.getBody().walk([&](mlir::Operation *op) {
    if (auto returnOp = dyn_cast_or_null<func::ReturnOp>(op); returnOp) {
      for (auto output : returnOp.getOperands()) {
        program.outputs.push_back(
            cache.at<::tt::target::TensorRef>(getOperandThroughDPSOps(output)));
      }
    } else {
      std::string debugStr = getOpDebugString(op, printFlags);
      program.ops.push_back(fn(cache, op, debugStr));
    }
  });

  return program;
}

} // namespace mlir::tt

#endif
