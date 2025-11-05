// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_UTILS_H
#define TTMLIR_DIALECT_TTCORE_IR_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::tt::ttcore {

class DeviceOp;
class DeviceAttr;
class SystemDescAttr;

inline constexpr llvm::StringRef getDefaultDeviceName() {
  return "default_device";
}

SystemDescAttr getCurrentScopeSystemDesc(Operation *op);

DeviceOp lookupDeviceOp(Operation *op,
                        llvm::StringRef deviceName = getDefaultDeviceName());

DeviceAttr lookupDevice(Operation *op, SymbolRefAttr deviceName);

DeviceAttr lookupDevice(Operation *op,
                        llvm::StringRef deviceName = getDefaultDeviceName());

ChipDescAttr getOpChipDescAttr(Operation *op);

// Create a global memref in the top-level module's symbol table.
mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, StringRef name,
                                    MemRefType type, ElementsAttr value,
                                    bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

// Overload auto-generating the name for the above.
mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, MemRefType type,
                                    ElementsAttr value, bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

// Helper function to check if a block argument is consteval-able (Parameter or
// Constant)
inline bool isConstOrParamArg(mlir::BlockArgument blockArg,
                              mlir::func::FuncOp funcOp) {
  if (auto typeAttr = funcOp.getArgAttrOfType<ArgumentTypeAttr>(
          blockArg.getArgNumber(), ArgumentTypeAttr::name)) {
    auto argTypeValue = typeAttr.getValue();
    return argTypeValue == ArgumentType::Parameter ||
           argTypeValue == ArgumentType::Constant;
  }
  return false;
}

// Filters out the constant parameters from the function signature.
inline llvm::SmallPtrSet<mlir::BlockArgument, 4>
getConstsAndParams(mlir::func::FuncOp funcOp) {
  llvm::SmallPtrSet<mlir::BlockArgument, 4> constsAndParams;

  for (auto arg : funcOp.getArguments()) {
    if (isConstOrParamArg(arg, funcOp)) {
      constsAndParams.insert(arg);
    }
  }

  return constsAndParams;
}

// This function will return true if a given Value is the result of operations
// performed only between  block arguments in which have been marked as
// consteval-able (Parameter or Constant ArgumentType).
inline bool valueTracesToConstantArgs(const mlir::Value &value) {
  auto useDefChain = ttmlir::utils::getUseDefChain(value);
  auto subgraphBlockArgs =
      ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());
  mlir::func::FuncOp funcOp = nullptr;

  if (!subgraphBlockArgs.empty()) {
    mlir::Block *argOwner = subgraphBlockArgs.front().getOwner();
    funcOp =
        mlir::dyn_cast_or_null<mlir::func::FuncOp>(argOwner->getParentOp());
  }
  if (!funcOp) {
    return false;
  }

  for (auto blockArg : subgraphBlockArgs) {
    if (!isConstOrParamArg(blockArg, funcOp)) {
      return false;
    }
  }

  return true;
}

bool isTiled(RankedTensorType tensorType);

ArrayRef<int64_t> getTensorTileShape(RankedTensorType tensorType);

ArrayRef<int64_t> getTensorTileShapeOrEmpty(RankedTensorType tensorType);

llvm::SmallVector<int64_t, 2> collapseGridTo2D(ArrayRef<int64_t> gridShape);

// Retrieve the layout from the shaped type (ie. getEncoding for tensors and
// getLayout for memrefs).
inline DeviceLayoutInterface getDeviceLayout(ShapedType shapedType) {
  if (auto tensor = mlir::dyn_cast_if_present<RankedTensorType>(shapedType)) {
    return mlir::dyn_cast_if_present<DeviceLayoutInterface>(
        tensor.getEncoding());
  }

  if (auto memref = mlir::dyn_cast_if_present<MemRefType>(shapedType)) {
    return mlir::dyn_cast_if_present<DeviceLayoutInterface>(memref.getLayout());
  }

  return nullptr;
}

// Convenience overload that extracts the shaped type from a value.
inline DeviceLayoutInterface getDeviceLayout(Value value) {
  return getDeviceLayout(mlir::cast<ShapedType>(value.getType()));
}

inline bool hasDeviceLayout(ShapedType shapedType) {
  return getDeviceLayout(shapedType) != nullptr;
}

inline bool hasDeviceLayout(Value value) {
  return hasDeviceLayout(mlir::cast<ShapedType>(value.getType()));
}

Type getOperandInnerElementType(const mlir::Value operand);

// Convert a TensorType with MetalLayoutAttr encoding into a MemRefType with
// appropriate layout attributes (Shard/View/Host/Interleaved).
bufferization::BufferLikeType
getBufferType(Type type, bool isView,
              std::optional<MetalLayoutAttr> hostInfo = std::nullopt);

} // namespace mlir::tt::ttcore

#endif // TTMLIR_DIALECT_TTCORE_IR_UTILS_H
