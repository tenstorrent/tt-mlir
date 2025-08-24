// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

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

static size_t appendArgImpl(func::FuncOp op, ArgAttr arg, bool isCompileTime) {
  auto [rtArgSpecVector, ctArgSpecVector] = getOrCreateArgSpec(op);
  auto &argSpecVector = isCompileTime ? ctArgSpecVector : rtArgSpecVector;
  size_t nextIndex = argSpecVector.size();
  argSpecVector.push_back(arg);
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
