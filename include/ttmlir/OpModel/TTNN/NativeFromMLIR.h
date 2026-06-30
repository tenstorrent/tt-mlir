// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_NATIVEFROMMLIR_H
#define TTMLIR_OPMODEL_TTNN_NATIVEFROMMLIR_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Target/TTNN/Target.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <variant>

#ifdef TTMLIR_ENABLE_OPMODEL
namespace mlir::tt::ttnn::op_model {

namespace detail {
std::optional<::tt::target::ttnn::MemoryConfigT>
getNullableMemoryConfigT(TTNNLayoutAttr layout);

std::unique_ptr<::tt::target::ttnn::TensorRefT>
getOutputTensorRefT(TTNNLayoutAttr layout);

std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>
reorderPool2dPadding(llvm::ArrayRef<int32_t> padding);
} // namespace detail

template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryOpT
buildEltwiseUnaryOpTFromMLIR(TTNNLayoutAttr outputLayout,
                             std::optional<llvm::APFloat> slope = std::nullopt);

template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout);

template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryOpT
buildEltwiseBinaryOpTFromMLIR(TTNNLayoutAttr outputLayout,
                              ttcore::DataTypeAttr opDtypeAttr = nullptr);

template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryCompositeOpT
buildEltwiseBinaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout);

::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
buildEltwiseBinaryCompositeScalarOpTFromMLIR(mlir::Attribute exponent,
                                             TTNNLayoutAttr outputLayout);

template <typename OpTy>
::tt::target::ttnn::EltwiseTernaryWhereOpT
buildEltwiseTernaryOpTFromMLIR(TTNNLayoutAttr outputLayout);

template <typename OpTy>
::tt::target::ttnn::EltwiseQuantizationOpT
buildEltwiseQuantizationOpTFromMLIR(std::optional<int32_t> axis,
                                    std::optional<ttcore::DataType> outputDtype,
                                    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::LinearOpT buildLinearOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::MatmulOpT buildMatmulOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::Conv2dOpT buildConv2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(mlir::Attribute min,
                                                 mlir::Attribute max,
                                                 TTNNLayoutAttr outputLayout);

::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(TTNNLayoutAttr outputLayout);

::tt::target::ttnn::Conv3dOpT buildConv3dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_depth, uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::StringRef padding_mode,
    uint32_t groups, std::optional<ttcore::DataTypeAttr> outputDtype,
    std::optional<Conv3dConfigAttr> conv3dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::ConvTranspose2dOpT buildConvTranspose2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> output_padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PrepareConv2dWeightsOpT
buildPrepareConv2dWeightsOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool hasBias, int32_t groups, ttcore::DataType inputDtype,
    std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PrepareConv2dBiasOpT buildPrepareConv2dBiasOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT
buildPrepareConvTranspose2dWeightsOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> outputPadding,
    llvm::ArrayRef<int32_t> dilation, bool hasBias, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool mirrorKernel,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PrepareConvTranspose2dBiasOpT
buildPrepareConvTranspose2dBiasOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::ConcatenateHeadsOpT
buildConcatenateHeadsOpTFromMLIR(TTNNLayoutAttr outputLayout);

::tt::target::ttnn::NLPConcatHeadsOpT
buildNLPConcatHeadsOpTFromMLIR(TTNNLayoutAttr outputLayout);

::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT
buildNLPCreateQKVHeadsDecodeOpTFromMLIR(uint32_t numHeads,
                                        std::optional<uint32_t> numKVHeads,
                                        std::optional<bool> overlapQKCoregrid,
                                        std::optional<uint32_t> sliceSize,
                                        TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT
buildPagedFlashMultiLatentAttentionDecodeOpTFromMLIR(
    uint32_t headDimV, bool isCausal, std::optional<llvm::APFloat> scale,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT
buildPagedScaledDotProductAttentionDecodeOpTFromMLIR(
    bool isCausal, std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::RotaryEmbeddingLlamaOpT
buildRotaryEmbeddingLlamaOpTFromMLIR(
    bool isDecodeMode,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::RotaryEmbeddingOpT buildRotaryEmbeddingOpTFromMLIR(
    std::optional<uint32_t> tokenIndex,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT
buildScaledDotProductAttentionDecodeOpTFromMLIR(
    bool isCausal, std::optional<llvm::APFloat> scale,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::ScaledDotProductAttentionOpT
buildScaledDotProductAttentionOpTFromMLIR(
    bool isCausal, std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize, TTNNLayoutAttr outputLayout);

::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT
buildSplitQueryKeyValueAndSplitHeadsOpTFromMLIR(
    uint32_t numHeads, std::optional<uint32_t> numKVHeads, bool transposeKey,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::NLPConcatHeadsDecodeOpT
buildNLPConcatHeadsDecodeOpTFromMLIR(uint32_t numHeads,
                                     TTNNLayoutAttr outputLayout);

::tt::target::ttnn::BatchNormInferenceOpT buildBatchNormInferenceOpTFromMLIR(
    llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::BatchNormTrainingOpT buildBatchNormTrainingOpTFromMLIR(
    llvm::APFloat epsilon, llvm::APFloat momentum,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::GroupNormOpT
buildGroupNormOpTFromMLIR(int64_t numGroups, llvm::APFloat epsilon,
                          TTNNLayoutAttr outputLayout);

::tt::target::ttnn::LayerNormPostAllGatherOpT
buildLayerNormPostAllGatherOpTFromMLIR(
    llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::LayerNormPreAllGatherOpT
buildLayerNormPreAllGatherOpTFromMLIR(
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::LayerNormOpT
buildLayerNormOpTFromMLIR(llvm::APFloat epsilon, TTNNLayoutAttr outputLayout);

::tt::target::ttnn::RMSNormPreAllGatherOpT buildRMSNormPreAllGatherOpTFromMLIR(
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    std::optional<bool> use2DCoreGrid, TTNNLayoutAttr outputLayout);

::tt::target::ttnn::RMSNormOpT buildRMSNormOpTFromMLIR(
    llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::SoftmaxOpT buildSoftmaxOpTFromMLIR(
    int32_t dimension, bool numericStable,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);

::tt::target::ttnn::AssignOpT
buildAssignOpTFromMLIR(TTNNLayoutAttr outputLayout);

::tt::target::ttnn::ConcatOpT
buildConcatOpTFromMLIR(int32_t dim, TTNNLayoutAttr outputLayout);

::tt::target::ttnn::GatherOpT
buildGatherOpTFromMLIR(int32_t dim, TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PadOpT buildPadOpTFromMLIR(llvm::ArrayRef<int32_t> padding,
                                               llvm::APFloat padValue,
                                               bool useMulticore,
                                               TTNNLayoutAttr outputLayout);

::tt::target::ttnn::PermuteOpT
buildPermuteOpTFromMLIR(llvm::ArrayRef<int64_t> permutation,
                        llvm::APFloat padValue, TTNNLayoutAttr outputLayout);

::tt::target::ttnn::RepeatInterleaveOpT
buildRepeatInterleaveOpTFromMLIR(const unsigned int repeats, const int dim,
                                 TTNNLayoutAttr outputLayout);

::tt::target::ttnn::RepeatOpT
buildRepeatOpTFromMLIR(llvm::ArrayRef<int64_t> repeatDims,
                       TTNNLayoutAttr outputLayout);

::tt::target::ttnn::ReshapeOpT
buildReshapeOpTFromMLIR(llvm::ArrayRef<int64_t> outputShape,
                        TTNNLayoutAttr outputLayout);

::tt::target::ttnn::ScatterOpT
buildScatterOpTFromMLIR(int32_t dim, mlir::tt::ttcore::ReduceType reduceType,
                        TTNNLayoutAttr outputLayout);

::tt::target::ttnn::SliceOpT buildSliceStaticOpTFromMLIR(
    llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
    llvm::ArrayRef<int64_t> step, TTNNLayoutAttr outputLayout);

::tt::target::ttnn::SortOpT buildSortOpTFromMLIR(int dim, bool descending,
                                                 bool stable,
                                                 TTNNLayoutAttr outputLayout);

::tt::target::ttnn::TransposeOpT
buildTransposeOpTFromMLIR(int32_t dim0, int32_t dim1,
                          TTNNLayoutAttr outputLayout);

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL

#endif // TTMLIR_OPMODEL_TTNN_NATIVEFROMMLIR_H
