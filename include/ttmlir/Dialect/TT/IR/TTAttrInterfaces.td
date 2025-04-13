// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TT_TTATTRINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TT_TTATTRINTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// TT attribute interface definitions
//===----------------------------------------------------------------------===//

def TT_DeviceLayoutInterface : AttrInterface<"DeviceLayoutInterface", [MemRefLayoutAttrInterface]> {
  let summary = "TT DeviceLayoutInterface";
  let cppNamespace = "::mlir::tt";
  let description = [{
  }];
  let methods = [
    InterfaceMethod<
      "Get the grid shape part of the memref shape (aka the first half)",
      "llvm::ArrayRef<int64_t>", "getGridShape", (ins "MemRefType":$memrefType),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(memrefType.getRank() % 2 == 0 && "rank must be even");
        return memrefType.getShape().take_front(memrefType.getRank() / 2);
      }]
    >,
    InterfaceMethod<
      "Get the shard shape part of the memref shape (aka the second half)",
      "llvm::ArrayRef<int64_t>", "getShardShape", (ins "MemRefType":$memrefType),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(memrefType.getRank() % 2 == 0 && "rank must be even");
        return memrefType.getShape().drop_front(memrefType.getRank() / 2);
      }]
    >,
  ];
}

#endif
