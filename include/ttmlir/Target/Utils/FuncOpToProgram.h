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
  // Debug: Log first few location infos during compilation
  static int loc_count = 0;
  if (loc_count < 5) {
    llvm::errs() << "DEBUG [Compilation]: Op #" << loc_count 
                 << " loc_info: " << str << "\n";
    loc_count++;
  }
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
Program<OpT> funcOpToProgram(FlatbufferObjectCache &cache, func::FuncOp entry,
                             FnT fn, TensorFnT tensorValueToFlatbuffer,
                             const llvm::StringMap<uint32_t> &programIndexMap) {
  OpPrintingFlags printFlags;
  printFlags = printFlags.elideLargeElementsAttrs()
                   .elideLargeResourceString()
                   .skipRegions()
                   .enableDebugInfo()
                   .assumeVerified();

  Program<OpT> program;
  program.name = entry.getSymName().data();

  for (auto &input : entry.getBody().getArguments()) {
    program.inputs.push_back(cache.getOrCreate(input, tensorValueToFlatbuffer));
  }

  mlir::AsmState printState(entry, printFlags);
  entry.getBody().walk([&](mlir::Operation *op) {
    if (auto returnOp = dyn_cast_if_present<func::ReturnOp>(op); returnOp) {
      for (auto output : returnOp.getOperands()) {
        program.outputs.push_back(cache.at<::tt::target::ttnn::TensorRef>(
            getOperandThroughDPSOps(output)));
      }
    } else {
      std::string debugStr = getOpDebugString(op, printState);
      std::string locInfo = getOpLocInfo(op);
      program.ops.push_back(fn(cache, op, programIndexMap, debugStr, locInfo));
    }
  });

  return program;
}

} // namespace mlir::tt::ttnn

#endif
