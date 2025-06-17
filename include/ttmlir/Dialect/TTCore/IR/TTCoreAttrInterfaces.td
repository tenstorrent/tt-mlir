// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_TTCOREATTRINTERFACES_TD
#define TTMLIR_DIALECT_TTCORE_IR_TTCOREATTRINTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// TT attribute interface definitions
//===----------------------------------------------------------------------===//

def TTCore_DeviceLayoutInterface : AttrInterface<"DeviceLayoutInterface", [MemRefLayoutAttrInterface]> {
  let summary = "TT DeviceLayoutInterface";
  let cppNamespace = "::mlir::tt";
  let description = [{
    This interface implies a MemRefLayoutAttrInterface that conforms to Tenstorrent device layout.
    Tenstorrent devices are divided into a grid of cores, onto which we map tensors that have been
    chunked into shards.  This device layout attribute semantically means exactly this, i.e. we have
    some tensors that are distributed amongst the device core grid. This means that memref shapes
    should be reinterpreted as a concatenation of a grid shape and a shard shape, for example,
    memref<1x2x3x4xttcore.tile> should be interpreted as a grid of 1x2 and a shard of 3x4. This interface
    provides some convenience methods for working with this interpretation of memref shape.
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
    InterfaceMethod<
      "Get the number of elements for the shard shape part of the memref shape (aka the second half)",
      "int64_t", "getShardNumElements", (ins "MemRefType":$memrefType),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto shardShape = getShardShape(memrefType);
        return std::accumulate(shardShape.begin(), shardShape.end(), 1LL, std::multiplies<int64_t>());
      }]
    >,
  ];
}

#endif // TTMLIR_DIALECT_TTCORE_IR_TTCOREATTRINTERFACES_TD
