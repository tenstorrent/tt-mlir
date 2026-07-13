// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.cpp.inc"

namespace mlir::tt::ttkernel {

ArgSpecAttr mlir::tt::ttkernel::ArgSpecAttr::setArgSpec(func::FuncOp op,
                                                        ArgSpecAttr argSpec) {
  op->setAttr(ArgSpecAttr::name, argSpec);
  return argSpec;
}

static std::pair<SmallVector<ArgAttr>, SmallVector<ArgAttr>>
getOrCreateArgSpec(mlir::func::FuncOp op) {
  SmallVector<ArgAttr> rtArgSpecVector;
  SmallVector<ArgAttr> ctArgSpecVector;
  if (auto argSpec = op->getAttrOfType<ArgSpecAttr>(ArgSpecAttr::name)) {
    rtArgSpecVector = llvm::to_vector(argSpec.getRtArgs());
    ctArgSpecVector = llvm::to_vector(argSpec.getCtArgs());
  }
  return {rtArgSpecVector, ctArgSpecVector};
}

bool mlir::tt::ttkernel::ArgSpecAttr::isSameArg(ArgAttr lhs, ArgAttr rhs) {
  return lhs.getArgType() == rhs.getArgType() &&
         lhs.getOperandIndex() == rhs.getOperandIndex() &&
         lhs.getIsUniform() == rhs.getIsUniform() &&
         lhs.getArgumentName() == rhs.getArgumentName();
}

bool mlir::tt::ttkernel::ArgSpecAttr::isLessArg(ArgAttr lhs, ArgAttr rhs) {
  if (lhs.getArgType() != rhs.getArgType()) {
    return static_cast<uint32_t>(lhs.getArgType()) <
           static_cast<uint32_t>(rhs.getArgType());
  }
  if (lhs.getOperandIndex() != rhs.getOperandIndex()) {
    return lhs.getOperandIndex() < rhs.getOperandIndex();
  }
  if (lhs.getIsUniform() != rhs.getIsUniform()) {
    return lhs.getIsUniform() < rhs.getIsUniform();
  }
  return lhs.getArgumentName().getValue() < rhs.getArgumentName().getValue();
}

static void incrementCompileArgUsers(func::FuncOp op, size_t firstIndex,
                                     size_t increment = 1) {
  op->walk([&](GetCompileArgValOp getArgOp) {
    uint32_t index = getArgOp.getArgIndex();
    if (index >= firstIndex) {
      getArgOp.setArgIndex(index + increment);
    }
  });
}

static size_t appendArgImpl(func::FuncOp op, ArgAttr arg, bool isCompileTime,
                            size_t numSlots) {
  auto [rtArgSpecVector, ctArgSpecVector] = getOrCreateArgSpec(op);
  auto &argSpecVector = isCompileTime ? ctArgSpecVector : rtArgSpecVector;

  // Dedup: if an identical arg was already appended, reuse its index.
  auto *it = std::find_if(argSpecVector.begin(), argSpecVector.end(),
                          [&](ArgAttr existingArg) {
                            return ArgSpecAttr::isSameArg(arg, existingArg);
                          });
  if (it != argSpecVector.end()) {
    return static_cast<size_t>(std::distance(argSpecVector.begin(), it));
  }

  // Find the insertion point.  Compile-time args are kept sorted (by isLessArg)
  // so the CRTA layout is deterministic; runtime args append at the end.
  // Multi-slot args (numSlots > 1) reserve trailing ArgType::Reserved
  // placeholders so vector position equals the flat uint32 arg offset; those
  // placeholders are not sort keys, so the scan skips over them rather than
  // using std::lower_bound (which would require a fully sorted range).
  size_t insertIndex;
  if (isCompileTime) {
    insertIndex = 0;
    for (ArgAttr existing : argSpecVector) {
      if (existing.getArgType() != ArgType::Reserved &&
          !ArgSpecAttr::isLessArg(existing, arg)) {
        break;
      }
      ++insertIndex;
    }
  } else {
    insertIndex = argSpecVector.size();
  }

  // Shift the indices of compile-time arg users at or after the insertion point
  // by the number of slots we are about to insert.
  if (isCompileTime && insertIndex < argSpecVector.size()) {
    incrementCompileArgUsers(op, insertIndex, numSlots);
  }

  argSpecVector.insert(argSpecVector.begin() + insertIndex, arg);
  // Reserve additional placeholder slots so that subsequent appends get correct
  // indices.  The runtime fills all numSlots positions from a single KernelArg
  // entry.
  if (numSlots > 1) {
    ArgAttr reserved = ArgAttr::get(op.getContext(), ArgType::Reserved,
                                    arg.getOperandIndex());
    argSpecVector.insert(argSpecVector.begin() + insertIndex + 1, numSlots - 1,
                         reserved);
  }
  auto argSpec =
      ArgSpecAttr::get(arg.getContext(), rtArgSpecVector, ctArgSpecVector);
  op->setAttr(ArgSpecAttr::name, argSpec);
  return insertIndex;
}

size_t mlir::tt::ttkernel::ArgSpecAttr::appendCompileTimeArg(func::FuncOp op,
                                                             ArgAttr arg,
                                                             size_t numSlots) {
  return appendArgImpl(op, arg, /*isCompileTime=*/true, numSlots);
}

size_t mlir::tt::ttkernel::ArgSpecAttr::appendRuntimeArg(func::FuncOp op,
                                                         ArgAttr arg,
                                                         size_t numSlots) {
  return appendArgImpl(op, arg, /*isCompileTime=*/false, numSlots);
}

void TTKernelDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::tt::ttkernel
