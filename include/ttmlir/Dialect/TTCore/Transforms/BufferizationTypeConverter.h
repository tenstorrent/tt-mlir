// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_TRANSFORMS_BUFFERIZATIONTYPECONVERTER_H
#define TTMLIR_DIALECT_TTCORE_TRANSFORMS_BUFFERIZATIONTYPECONVERTER_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

namespace mlir::tt::ttcore {

/// Configure bufferization options to properly convert tensor types with
/// MetalLayoutAttr encoding to memref types with appropriate layout attributes
/// (ShardLayoutAttr, ViewLayoutAttr, InterleavedLayoutAttr) and memory space.
inline void setTTCoreBufferizationTypeConverter(
    bufferization::OneShotBufferizationOptions &options) {
  options.functionArgTypeConverterFn =
      [](bufferization::TensorLikeType tensorType, Attribute memorySpace, func::FuncOp funcOp,
         const bufferization::BufferizationOptions &options) -> bufferization::BufferLikeType {
    auto rankedTensorType = mlir::dyn_cast<RankedTensorType>(tensorType);
    if (!rankedTensorType) {
      // Fallback to default conversion for unranked tensors.
      return mlir::cast<bufferization::BufferLikeType>(
          UnrankedMemRefType::get(rankedTensorType.getElementType(), memorySpace));
    }

    // Check if this tensor has a MetalLayoutAttr encoding.
    if (auto metalLayout = mlir::dyn_cast_if_present<MetalLayoutAttr>(
            rankedTensorType.getEncoding())) {
      // Function arguments are not views.
      return ttcore::getBufferType(rankedTensorType, /*isView=*/false);
    }

    // For tensors without MetalLayoutAttr, use default identity layout.
    return mlir::cast<bufferization::BufferLikeType>(
        bufferization::getMemRefTypeWithStaticIdentityLayout(rankedTensorType,
                                                              memorySpace));
  };
}

} // namespace mlir::tt::ttcore

#endif // TTMLIR_DIALECT_TTCORE_TRANSFORMS_BUFFERIZATIONTYPECONVERTER_H
