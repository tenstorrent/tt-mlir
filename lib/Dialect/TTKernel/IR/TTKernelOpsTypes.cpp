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

static void incrementCompileArgUsers(func::FuncOp op, size_t firstIndex) {
  op->walk([&](GetCompileArgValOp getArgOp) {
    uint32_t index = getArgOp.getArgIndex();
    if (index >= firstIndex) {
      getArgOp.setArgIndex(index + 1);
    }
  });
}

static size_t appendArgImpl(func::FuncOp op, ArgAttr arg, bool isCompileTime) {
  auto [rtArgSpecVector, ctArgSpecVector] = getOrCreateArgSpec(op);
  auto &argSpecVector = isCompileTime ? ctArgSpecVector : rtArgSpecVector;
  auto *it = std::find_if(argSpecVector.begin(), argSpecVector.end(),
                          [&](ArgAttr existingArg) {
                            return ArgSpecAttr::isSameArg(arg, existingArg);
                          });
  if (it != argSpecVector.end()) {
    return static_cast<size_t>(std::distance(argSpecVector.begin(), it));
  }

  auto *insertionIt =
      isCompileTime
          ? std::lower_bound(argSpecVector.begin(), argSpecVector.end(), arg,
                             [](ArgAttr lhs, ArgAttr rhs) {
                               return ArgSpecAttr::isLessArg(lhs, rhs);
                             })
          : argSpecVector.end();
  size_t nextIndex =
      static_cast<size_t>(std::distance(argSpecVector.begin(), insertionIt));
  if (isCompileTime && insertionIt != argSpecVector.end()) {
    incrementCompileArgUsers(op, nextIndex);
  }
  argSpecVector.insert(insertionIt, arg);
  auto argSpec =
      ArgSpecAttr::get(arg.getContext(), rtArgSpecVector, ctArgSpecVector);
  op->setAttr(ArgSpecAttr::name, argSpec);
  return nextIndex;
}

size_t mlir::tt::ttkernel::ArgSpecAttr::appendCompileTimeArg(func::FuncOp op,
                                                             ArgAttr arg) {
  return appendArgImpl(op, arg, /*isCompileTime=*/true);
}

size_t mlir::tt::ttkernel::ArgSpecAttr::appendRuntimeArg(func::FuncOp op,
                                                         ArgAttr arg) {
  return appendArgImpl(op, arg, /*isCompileTime=*/false);
}

void TTKernelDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::tt::ttkernel
