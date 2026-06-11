// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H
#define TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::ttnn {

// Convert a TTNNIR operation to a flatbuffer
std::shared_ptr<void> ttnnToFlatbuffer(
    Operation *op,
    /* goldenMap has following structure
    {
      loc: {
        device_id: GoldenTensor
      }
    }
    */
    const std::unordered_map<std::string,
                             std::unordered_map<std::uint32_t, GoldenTensor>>
        &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {});

// Convert a TTNNIR operation to a flatbuffer
// This function signature is required in order to register the conversion in
// mlir translation framework
LogicalResult translateTTNNToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    const std::unordered_map<std::string,
                             std::unordered_map<std::uint32_t, GoldenTensor>>
        /* goldenMap has following structure
        {
          loc: {
            device_id: GoldenTensor
          }
        }
        */
        &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {});

::flatbuffers::Offset<::tt::target::ttnn::MatmulOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, MatmulOp op);

::flatbuffers::Offset<::tt::target::ttnn::LinearOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, LinearOp op);

::flatbuffers::Offset<::tt::target::ttnn::Conv2dOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, Conv2dOp op);

::flatbuffers::Offset<::tt::target::ttnn::Conv3dOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, Conv3dOp op);

::flatbuffers::Offset<::tt::target::ttnn::ConvTranspose2dOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, ConvTranspose2dOp op);

::flatbuffers::Offset<::tt::target::ttnn::PrepareConv2dWeightsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, PrepareConv2dWeightsOp op);

::flatbuffers::Offset<::tt::target::ttnn::PrepareConv2dBiasOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, PrepareConv2dBiasOp op);

::flatbuffers::Offset<::tt::target::ttnn::PrepareConvTranspose2dWeightsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         PrepareConvTranspose2dWeightsOp op);

::flatbuffers::Offset<::tt::target::ttnn::PrepareConvTranspose2dBiasOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         PrepareConvTranspose2dBiasOp op);

::flatbuffers::Offset<::tt::target::DeviceRef>
createDeviceRef(::mlir::tt::FlatbufferObjectCache &cache, ::mlir::Value device);

template <typename EltwiseUnaryOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseUnaryOp>
createEltwiseUnaryOp(::mlir::tt::FlatbufferObjectCache &cache,
                     EltwiseUnaryOp op);

template <typename EltwiseUnaryCompositeOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseUnaryCompositeOp>
createEltwiseUnaryCompositeOp(::mlir::tt::FlatbufferObjectCache &cache,
                              EltwiseUnaryCompositeOp op);

template <typename EltwiseBinaryOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryOp>
createEltwiseBinaryOp(::mlir::tt::FlatbufferObjectCache &cache,
                      EltwiseBinaryOp op);

template <typename EltwiseBinaryCompositeOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryCompositeOp>
createEltwiseBinaryCompositeOp(::mlir::tt::FlatbufferObjectCache &cache,
                               EltwiseBinaryCompositeOp op);

template <typename EltwiseBinaryCompositeScalarOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryCompositeScalarOp>
createEltwiseBinaryCompositeScalarOp(::mlir::tt::FlatbufferObjectCache &cache,
                                     EltwiseBinaryCompositeScalarOp op);

::flatbuffers::Offset<::tt::target::ttnn::EltwiseTernaryWhereOp>
createEltwiseTernaryWhereOp(::mlir::tt::FlatbufferObjectCache &cache,
                            WhereOp op);

template <typename EltwiseQuantizationOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseQuantizationOp>
createEltwiseQuantizationOp(::mlir::tt::FlatbufferObjectCache &cache,
                            EltwiseQuantizationOp op);
} // namespace mlir::tt::ttnn

#endif // TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H
