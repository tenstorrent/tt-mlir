// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_OPERANDCONSTRAINTS_H
#define TTMLIR_DIALECT_TT_UTILS_OPERANDCONSTRAINTS_H

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir::tt {

inline OperandConstraint
memorySpaceAsOperandConstraint(MemorySpace memorySpace) {
  switch (memorySpace) {
  case MemorySpace::System:
  case MemorySpace::SystemMMIO:
    return OperandConstraint::System;
  case MemorySpace::DeviceDRAM:
    return OperandConstraint::DRAM;
  case MemorySpace::DeviceL1:
    return OperandConstraint::L1;
  }
}

inline OperandConstraint
memoryLayoutAsOperandConstraint(TensorMemoryLayout memoryLayout) {
  switch (memoryLayout) {
  case TensorMemoryLayout::None:
    return OperandConstraint::None;
  case TensorMemoryLayout::Interleaved:
    return OperandConstraint::Interleaved;
  case TensorMemoryLayout::SingleBank:
    return OperandConstraint::SingleBank;
  case TensorMemoryLayout::HeightSharded:
    return OperandConstraint::HeightSharded;
  case TensorMemoryLayout::WidthSharded:
    return OperandConstraint::WidthSharded;
  case TensorMemoryLayout::BlockSharded:
    return OperandConstraint::BlockSharded;
  }
}

inline MemorySpace getLegalMemorySpace(OperandConstraint operandConstraint,
                                       MemorySpace defaultMemorySpace) {
  if (bitEnumContainsAny(operandConstraint,
                         memorySpaceAsOperandConstraint(defaultMemorySpace))) {
    return defaultMemorySpace;
  }
  if (bitEnumContainsAny(operandConstraint, OperandConstraint::DRAM)) {
    return MemorySpace::DeviceDRAM;
  }
  if (bitEnumContainsAny(operandConstraint, OperandConstraint::L1)) {
    return MemorySpace::DeviceL1;
  }
  return MemorySpace::System;
}

inline TensorMemoryLayout
getLegalTensorMemoryLayout(OperandConstraint operandConstraint,
                           MemorySpace targetMemorySpace,
                           TensorMemoryLayout defaultDeviceMemLayout) {
  if (defaultDeviceMemLayout == TensorMemoryLayout::None) {
    return TensorMemoryLayout::None;
  }

  if (isSystemMemorySpace(targetMemorySpace)) {
    return TensorMemoryLayout::None;
  }

  assert(isDeviceMemorySpace(targetMemorySpace));
  if (bitEnumContainsAny(operandConstraint, memoryLayoutAsOperandConstraint(
                                                defaultDeviceMemLayout))) {
    return defaultDeviceMemLayout;
  }

  std::map<OperandConstraint, TensorMemoryLayout> validLayoutsMap = {
      {OperandConstraint::Interleaved, TensorMemoryLayout::Interleaved},
      {OperandConstraint::SingleBank, TensorMemoryLayout::SingleBank},
      {OperandConstraint::HeightSharded, TensorMemoryLayout::HeightSharded},
      {OperandConstraint::WidthSharded, TensorMemoryLayout::WidthSharded},
      {OperandConstraint::BlockSharded, TensorMemoryLayout::BlockSharded}};

  for (const auto &[constraintLayout, memLayout] : validLayoutsMap) {
    if (bitEnumContainsAny(operandConstraint, constraintLayout)) {
      return memLayout;
    }
  }

  return TensorMemoryLayout::None;
}

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TT_UTILS_OPERANDCONSTRAINTS_H
