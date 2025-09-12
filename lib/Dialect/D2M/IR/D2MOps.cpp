// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;

void mlir::tt::d2m::GenericOp::getAsmBlockArgumentNames(
    Region &region, function_ref<void(Value, StringRef)> setNameFn) {
  int cbIndex = 0;
  int semIndex = 0;
  for (BlockArgument arg : region.getArguments()) {
    if (mlir::isa<MemRefType>(arg.getType())) {
      setNameFn(arg, "cb" + std::to_string(cbIndex++));
    } else if (mlir::isa<RankedTensorType>(arg.getType())) {
      setNameFn(arg, "t" + std::to_string(cbIndex++));
    } else if (mlir::isa<SemaphoreType>(arg.getType())) {
      setNameFn(arg, "sem" + std::to_string(semIndex++));
    } else {
      llvm_unreachable("Unexpected region argument type");
    }
  }
}

void mlir::tt::d2m::GenericOp::getAsmBlockNames(
    function_ref<void(Block *, StringRef)> setNameFn) {
  std::array<int, getMaxEnumValForThreadType() + 1> threadTypeCounts{};
  for (Region &region : getRegions()) {
    auto type = getRegionThreadType(region.getRegionNumber());
    setNameFn(&region.front(),
              stringifyEnum(type).str() +
                  Twine(threadTypeCounts[llvm::to_underlying(type)]++).str());
  }
}
