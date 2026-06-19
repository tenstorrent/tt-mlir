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

::flatbuffers::Offset<::tt::target::ttnn::ConcatenateHeadsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, ConcatenateHeadsOp op);

::flatbuffers::Offset<::tt::target::ttnn::NLPConcatHeadsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, NLPConcatHeadsOp op);

::flatbuffers::Offset<::tt::target::ttnn::NLPCreateQKVHeadsDecodeOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         NLPCreateQKVHeadsDecodeOp op);

::flatbuffers::Offset<
    ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         PagedFlashMultiLatentAttentionDecodeOp op);

::flatbuffers::Offset<
    ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         PagedScaledDotProductAttentionDecodeOp op);

::flatbuffers::Offset<::tt::target::ttnn::RotaryEmbeddingLlamaOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, RotaryEmbeddingLlamaOp op);

::flatbuffers::Offset<::tt::target::ttnn::RotaryEmbeddingOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, RotaryEmbeddingOp op);

::flatbuffers::Offset<::tt::target::ttnn::ScaledDotProductAttentionDecodeOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         ScaledDotProductAttentionDecodeOp op);

::flatbuffers::Offset<::tt::target::ttnn::ScaledDotProductAttentionOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         ScaledDotProductAttentionOp op);

::flatbuffers::Offset<::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         SplitQueryKeyValueAndSplitHeadsOp op);

::flatbuffers::Offset<::tt::target::ttnn::NLPConcatHeadsDecodeOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, NLPConcatHeadsDecodeOp op);

::flatbuffers::Offset<::tt::target::ttnn::BatchNormInferenceOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, BatchNormInferenceOp op);

::flatbuffers::Offset<::tt::target::ttnn::BatchNormTrainingOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, BatchNormTrainingOp op);

::flatbuffers::Offset<::tt::target::ttnn::GroupNormOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, GroupNormOp op);

::flatbuffers::Offset<::tt::target::ttnn::LayerNormPostAllGatherOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, LayerNormPostAllGatherOp op);

::flatbuffers::Offset<::tt::target::ttnn::LayerNormPreAllGatherOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, LayerNormPreAllGatherOp op);

::flatbuffers::Offset<::tt::target::ttnn::LayerNormOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, LayerNormOp op);

::flatbuffers::Offset<::tt::target::ttnn::RMSNormPreAllGatherOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, RMSNormPreAllGatherOp op);

::flatbuffers::Offset<::tt::target::ttnn::RMSNormOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, RMSNormOp op);

::flatbuffers::Offset<::tt::target::ttnn::SoftmaxOp>
createSoftmaxOp(::mlir::tt::FlatbufferObjectCache &cache, SoftmaxOp op);

::flatbuffers::Offset<::tt::target::ttnn::AssignOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, AssignOp op);

::flatbuffers::Offset<::tt::target::ttnn::ConcatOp>
createConcatOp(::mlir::tt::FlatbufferObjectCache &cache, ConcatOp op);

::flatbuffers::Offset<::tt::target::ttnn::GatherOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, GatherOp op);

::flatbuffers::Offset<::tt::target::ttnn::PadOp>
createPadOp(::mlir::tt::FlatbufferObjectCache &cache, PadOp op);

::flatbuffers::Offset<::tt::target::ttnn::PermuteOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, PermuteOp op);

::flatbuffers::Offset<::tt::target::ttnn::RepeatInterleaveOp>
createRepeatInterleaveOp(::mlir::tt::FlatbufferObjectCache &cache,
                         RepeatInterleaveOp op);

template <typename RepeatOp>
::flatbuffers::Offset<::tt::target::ttnn::RepeatOp>
createRepeatOp(::mlir::tt::FlatbufferObjectCache &cache, RepeatOp op);

::flatbuffers::Offset<::tt::target::ttnn::ReshapeOp>
createReshapeOp(::mlir::tt::FlatbufferObjectCache &cache, ReshapeOp op);

::flatbuffers::Offset<::tt::target::ttnn::ScatterOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, ScatterOp op);

template <typename SliceOpT>
::flatbuffers::Offset<::tt::target::ttnn::SliceOp>
createSliceOp(::mlir::tt::FlatbufferObjectCache &cache, SliceOpT op);

::flatbuffers::Offset<::tt::target::ttnn::SortOp>
createSortOp(::mlir::tt::FlatbufferObjectCache &cache, SortOp op);

::flatbuffers::Offset<::tt::target::ttnn::TransposeOp>
createTransposeOp(::mlir::tt::FlatbufferObjectCache &cache, TransposeOp op);

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionArgMaxOp>
createReductionArgMaxOp(::mlir::tt::FlatbufferObjectCache &cache,
                        ReductionOp op);

::flatbuffers::Offset<::tt::target::ttnn::CumSumOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, CumSumOp op);

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionProdOp>
createReductionProdOp(::mlir::tt::FlatbufferObjectCache &cache, ReductionOp op);

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionOp>
createReductionOp(::mlir::tt::FlatbufferObjectCache &cache, ReductionOp op);

::flatbuffers::Offset<::tt::target::ttnn::TopKRouterGptOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, TopKRouterGptOp op);

::flatbuffers::Offset<::tt::target::ttnn::TopKOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, TopKOp op);

::flatbuffers::Offset<::tt::target::ttnn::FillCacheOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, FillCacheOp op);

::flatbuffers::Offset<::tt::target::ttnn::PagedFillCacheOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, PagedFillCacheOp op);

::flatbuffers::Offset<::tt::target::ttnn::PagedUpdateCacheOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, PagedUpdateCacheOp op);

template <typename EmbeddingBackwardOpTy>
::flatbuffers::Offset<::tt::target::ttnn::EmbeddingBackwardOp>
createEmbeddingBackwardOp(::mlir::tt::FlatbufferObjectCache &cache,
                          EmbeddingBackwardOpTy op);

::flatbuffers::Offset<::tt::target::ttnn::DropoutOp>
createDropoutOp(::mlir::tt::FlatbufferObjectCache &cache, DropoutOp op);

::flatbuffers::Offset<::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOp>
createExperimentalEltwiseBinaryBackwardOp(
    ::mlir::tt::FlatbufferObjectCache &cache, GeluBackwardOp op);

template <typename Pool2dOpTy>
::flatbuffers::Offset<::tt::target::ttnn::Pool2dOp>
createPool2dOp(::mlir::tt::FlatbufferObjectCache &cache, Pool2dOpTy op);

::flatbuffers::Offset<::tt::target::ttnn::MaxPool2dWithIndicesOp>
createMaxPool2dWithIndicesOp(::mlir::tt::FlatbufferObjectCache &cache,
                             MaxPool2dWithIndicesOp op);

::flatbuffers::Offset<::tt::target::ttnn::UpsampleOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, UpsampleOp op);

::flatbuffers::Offset<::tt::target::ttnn::RandOp>
createRandOp(::mlir::tt::FlatbufferObjectCache &cache, RandOp op);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H
