// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt;

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"

mlir::tt::SystemDescAttr
mlir::tt::SystemDescAttr::getDefault(MLIRContext *context) {
  return tt::SystemDescAttr::get(
      context,
      // Chip Descriptors
      {
          tt::ChipDescAttr::get(
              context, tt::ArchAttr::get(context, tt::Arch::WormholeB0),
              tt::GridAttr::get(context, {8, 8}), (1 << 20), 12, (1 << 20)),
      },
      // Chip Descriptor Indices
      {
          0,
      },
      // Chip IDs
      {
          0,
      },
      // Chip capabilities
      {
          tt::ChipCapabilityAttr::get(context,
                                      // NOLINTNEXTLINE
                                      tt::ChipCapability::PCIE |
                                          tt::ChipCapability::HostMMIO),
      },
      // Chip Mesh Coordinates
      {
          tt::ChipCoordAttr::get(context, 0, 0, 0, 0),
      },
      // Chip Channel Connections
      {});
}

MemorySpace LayoutAttr::getMemorySpace() const {
  return getMemref()
      .getMemorySpace()
      .template cast<mlir::tt::MemorySpaceAttr>()
      .getValue();
}

void TTDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"
      >();
}
