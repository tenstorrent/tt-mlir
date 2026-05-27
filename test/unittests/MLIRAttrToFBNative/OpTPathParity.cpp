// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpTPathParity.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

//===----------------------------------------------------------------------===//
// MatmulOpTPathParity
//===----------------------------------------------------------------------===//
constexpr std::array<int64_t, 2> shapeArray = {32, 32};
const llvm::ArrayRef<int64_t> defaultShape = shapeArray;

const mlir::tt::ttcore::DataTypeAttr bf16DtypeAttr =
    mlir::tt::ttcore::DataTypeAttr::get(getContext(),
                                        mlir::tt::ttcore::DataType::BFloat16);

const mlir::tt::ttnn::Conv2dConfigAttr nonDefaultConv2dConfigAttr =
    mlir::tt::ttnn::Conv2dConfigAttr::get(
        getContext(),
        /*weights_dtype=*/mlir::tt::ttcore::DataType::BFloat16,
        /*activation=*/mlir::tt::ttnn::UnaryWithParamAttr(),
        /*deallocate_activation=*/mlir::BoolAttr::get(getContext(), false),
        /*reallocate_halo_output=*/mlir::BoolAttr::get(getContext(), true),
        /*act_block_h_override=*/0, /*act_block_w_div=*/1,
        /*reshard_if_not_optimal=*/mlir::BoolAttr::get(getContext(), false),
        /*override_sharding_config=*/mlir::BoolAttr::get(getContext(), false),
        /*shard_layout=*/std::nullopt,
        /*core_grid=*/mlir::tt::ttnn::CoreRangeSetAttr(),
        /*transpose_shards=*/mlir::BoolAttr::get(getContext(), false),
        /*output_layout=*/mlir::tt::ttnn::Layout::Tile,
        /*enable_act_double_buffer=*/mlir::BoolAttr::get(getContext(), true),
        /*enable_weights_double_buffer=*/
        mlir::BoolAttr::get(getContext(), true),
        /*enable_kernel_stride_folding=*/
        mlir::BoolAttr::get(getContext(), false),
        /*config_tensors_in_dram=*/mlir::BoolAttr());

const mlir::tt::ttnn::DeviceComputeKernelConfigAttr
    nonDefaultDeviceComputeKernelConfigAttr =
        mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
            getContext(), mlir::tt::ttnn::MathFidelity::HiFi2,
            mlir::BoolAttr::get(getContext(), false),
            mlir::BoolAttr::get(getContext(), true),
            mlir::BoolAttr::get(getContext(), true),
            mlir::BoolAttr::get(getContext(), false));

const mlir::tt::ttnn::Conv2dSliceConfigAttr nonDefaultConv2dSliceConfigAttr =
    mlir::tt::ttnn::Conv2dSliceConfigAttr::get(
        getContext(), mlir::tt::ttnn::Conv2dSliceType::DramHeight, 4);

const mlir::tt::ttnn::MemoryConfigAttr nonDefaultInputMemoryConfigAttr =
    mlir::tt::ttnn::MemoryConfigAttr::Builder(
        mlir::tt::ttnn::MemoryConfigAttr::get(
            createTiledL1InterleavedLayout({32, 32})))
        .setBufferType(mlir::tt::ttnn::BufferType::DRAM);

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::MatmulOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, MatmulOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::MatmulOpT buildMatmulOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::MatmulOpT &opTOpModel,
                       ::tt::target::ttnn::MatmulOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::MatmulOpT &opT) {
    opT.a.reset();
    opT.b.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::MatmulOp buildTestMatmulOp(
    bool transposeA = false, bool transposeB = false,
    mlir::StringAttr activation = {}, mlir::Attribute programConfigAttr = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {}) {
  auto &e = env();

  auto typeA = tiledL1BF16Type(defaultShape);
  auto typeB = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  auto loc = e.builder.getUnknownLoc();
  mlir::Value a = e.builder
                      .create<mlir::tt::ttnn::OnesOp>(
                          loc, mlir::TypeRange{typeA}, mlir::ValueRange{})
                      .getResult();
  mlir::Value b = e.builder
                      .create<mlir::tt::ttnn::OnesOp>(
                          loc, mlir::TypeRange{typeB}, mlir::ValueRange{})
                      .getResult();

  return e.builder.create<mlir::tt::ttnn::MatmulOp>(
      loc, outputType, a, b, transposeA, transposeB, programConfigAttr,
      activation, computeKernelConfig);
}

} // namespace

using MatmulOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::MatmulOp>;

TEST_P(MatmulOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::MatmulOp matmulOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::MatmulOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildMatmulOpTFromMLIR(
          matmulOp.getTransposeA(), matmulOp.getTransposeB(),
          matmulOp.getActivation(), matmulOp.getMatmulProgramConfig(),
          matmulOp.getComputeConfig(), resolveOutputLayout(matmulOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, matmulOp.getA(), matmulOp.getB());

  auto fbOffset = mlir::tt::ttnn::createOp(cache, matmulOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::MatmulOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::MatmulOp> matmulOpList = {
    buildTestMatmulOp(),
    buildTestMatmulOp(/*transposeA=*/true),
    buildTestMatmulOp(/*transposeA=*/false, /*transposeB=*/true),
    buildTestMatmulOp(
        /*transposeA=*/false, /*transposeB=*/false,
        /*activation=*/mlir::StringAttr::get(getContext(), "relu")),
    buildTestMatmulOp(
        /*transposeA=*/false, /*transposeB=*/false, /*activation=*/{},
        /*programConfigAttr=*/
        mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::get(
            getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(getContext(), 8, 8), 2, 4, 4, 8,
            8)),
    buildTestMatmulOp(
        /*transposeA=*/false, /*transposeB=*/false, /*activation=*/{},
        /*programConfigAttr=*/
        mlir::tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
            getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(getContext(), 8, 8), 2, 4, 4, 8,
            8, 8, 8, false, mlir::tt::ttnn::UnaryWithParamAttr(), false)),
    buildTestMatmulOp(
        /*transposeA=*/false, /*transposeB=*/false, /*activation=*/{},
        /*programConfigAttr=*/
        mlir::tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
            getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(getContext(), 4, 4), 1, 2, 2, 4,
            4, 4, 4, true, mlir::tt::ttnn::UnaryWithParamAttr(), false, true,
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                getContext(), llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>()),
            0, false)),
    buildTestMatmulOp(
        /*transposeA=*/false, /*transposeB=*/false, /*activation=*/{},
        /*programConfigAttr=*/
        mlir::tt::ttnn::
            MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
                getContext(), 4, 8, 8, mlir::tt::ttnn::UnaryWithParamAttr())),
    buildTestMatmulOp(/*transposeA=*/false, /*transposeB=*/false,
                      /*activation=*/{}, /*programConfigAttr=*/{},
                      /*computeKernelConfig=*/
                      mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
                          getContext(), mlir::tt::ttnn::MathFidelity::HiFi2,
                          mlir::BoolAttr::get(getContext(), false),
                          mlir::BoolAttr::get(getContext(), true),
                          mlir::BoolAttr::get(getContext(), true),
                          mlir::BoolAttr::get(getContext(), false))),
    buildTestMatmulOp(
        /*transposeA=*/true, /*transposeB=*/true,
        /*activation=*/mlir::StringAttr::get(getContext(), "relu"),
        /*programConfigAttr=*/
        mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::get(
            getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(getContext(), 8, 8), 2, 4, 4, 8,
            8),
        /*computeKernelConfig=*/
        mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
            getContext(), mlir::tt::ttnn::MathFidelity::HiFi4,
            mlir::BoolAttr::get(getContext(), true),
            mlir::BoolAttr::get(getContext(), false),
            mlir::BoolAttr::get(getContext(), true),
            mlir::BoolAttr::get(getContext(), true))),
};

INSTANTIATE_TEST_SUITE_P(MatmulOpTPathParityTest, MatmulOpTPathParityTest,
                         ::testing::ValuesIn(matmulOpList));

//===----------------------------------------------------------------------===//
// Conv2dOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::Conv2dOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, Conv2dOp op);

::flatbuffers::Offset<::tt::target::DeviceRef>
createDeviceRef(::mlir::tt::FlatbufferObjectCache &cache, ::mlir::Value device);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::Conv2dOpT buildConv2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::Conv2dOpT &opTOpModel,
                       ::tt::target::ttnn::Conv2dOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::Conv2dOpT &opT) {
    opT.input.reset();
    opT.weight.reset();
    opT.bias.reset();
    opT.device.reset();
    resetOutputTensorRefT(opT.out);
    opT.output_dtype.reset();
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::Conv2dOp buildTestConv2dOp(
    bool withBias = false, mlir::tt::ttcore::DataTypeAttr outputDtype = {},
    mlir::tt::ttnn::Conv2dConfigAttr conv2dConfig = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {},
    mlir::tt::ttnn::Conv2dSliceConfigAttr conv2dSliceConfig = {},
    uint32_t inChannels = 3, uint32_t outChannels = 64, uint32_t batchSize = 1,
    uint32_t inputHeight = 224, uint32_t inputWidth = 224,
    llvm::ArrayRef<int32_t> kernelSize = {7, 7},
    llvm::ArrayRef<int32_t> stride = {2, 2},
    llvm::ArrayRef<int32_t> padding = {3, 3},
    llvm::ArrayRef<int32_t> dilation = {1, 1}, uint32_t groups = 1) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto weightType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value weight =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{weightType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value bias = nullptr;
  if (withBias) {
    auto biasType = tiledL1BF16Type(defaultShape);
    bias = e.builder
               .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{biasType},
                                               mlir::ValueRange{})
               .getResult();
  }

  mlir::Value device =
      e.builder
          .create<mlir::tt::ttnn::GetDeviceOp>(
              loc, e.builder.getType<mlir::tt::ttnn::DeviceType>(),
              mlir::tt::ttnn::MeshShapeAttr::get(&e.context, 1, 1),
              mlir::tt::ttnn::MeshOffsetAttr::get(&e.context, 0, 0))
          .getResult();

  return e.builder.create<mlir::tt::ttnn::Conv2dOp>(
      loc, outputType, input, weight, bias, device, inChannels, outChannels,
      batchSize, inputHeight, inputWidth, kernelSize, stride, padding, dilation,
      groups, outputDtype, conv2dConfig, computeKernelConfig,
      conv2dSliceConfig);
}

} // namespace

using Conv2dOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::Conv2dOp>;

TEST_P(Conv2dOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::Conv2dOp conv2dOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::Conv2dOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildConv2dOpTFromMLIR(
          conv2dOp.getInChannels(), conv2dOp.getOutChannels(),
          conv2dOp.getBatchSize(), conv2dOp.getInputHeight(),
          conv2dOp.getInputWidth(), conv2dOp.getKernelSize(),
          conv2dOp.getStride(), conv2dOp.getPadding(), conv2dOp.getDilation(),
          conv2dOp.getGroups(), conv2dOp.getConv2dConfig(),
          conv2dOp.getComputeConfig(), conv2dOp.getConv2dSliceConfig(),
          resolveOutputLayout(conv2dOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, conv2dOp.getInput(),
                               conv2dOp.getWeight());
  if (conv2dOp.getBias()) {
    prepopulateOperandTensorRefs(cache, conv2dOp.getBias());
  }
  cache.getOrCreate(conv2dOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, conv2dOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::Conv2dOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::Conv2dOp> conv2dOpList = {
    buildTestConv2dOp(),
    buildTestConv2dOp(/*withBias=*/true),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/bf16DtypeAttr),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/nonDefaultConv2dConfigAttr),
    buildTestConv2dOp(
        /*withBias=*/false, /*outputDtype=*/{}, /*conv2dConfig=*/{},
        /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/64u),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/128u),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/8u),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/1u,
                      /*inputHeight=*/56u),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/1u,
                      /*inputHeight=*/224u, /*inputWidth=*/56u),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/1u,
                      /*inputHeight=*/224u, /*inputWidth=*/224u,
                      /*kernelSize=*/{3, 3}),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/1u,
                      /*inputHeight=*/224u, /*inputWidth=*/224u,
                      /*kernelSize=*/{7, 7}, /*stride=*/{1, 1}),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/1u,
                      /*inputHeight=*/224u, /*inputWidth=*/224u,
                      /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
                      /*padding=*/{1, 1}),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/1u,
                      /*inputHeight=*/224u, /*inputWidth=*/224u,
                      /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
                      /*padding=*/{3, 3}, /*dilation=*/{2, 2}),
    buildTestConv2dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*conv2dSliceConfig=*/{}, /*inChannels=*/3u,
                      /*outChannels=*/64u, /*batchSize=*/1u,
                      /*inputHeight=*/224u, /*inputWidth=*/224u,
                      /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
                      /*padding=*/{3, 3}, /*dilation=*/{1, 1},
                      /*groups=*/3u),
    buildTestConv2dOp(/*withBias=*/true, /*outputDtype=*/bf16DtypeAttr,
                      /*conv2dConfig=*/nonDefaultConv2dConfigAttr,
                      /*computeKernelConfig=*/
                      nonDefaultDeviceComputeKernelConfigAttr,
                      /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr,
                      /*inChannels=*/64u, /*outChannels=*/128u,
                      /*batchSize=*/8u, /*inputHeight=*/56u,
                      /*inputWidth=*/56u, /*kernelSize=*/{3, 3},
                      /*stride=*/{1, 1}, /*padding=*/{1, 1},
                      /*dilation=*/{2, 2}, /*groups=*/2u),
};

INSTANTIATE_TEST_SUITE_P(Conv2dOpTPathParityTest, Conv2dOpTPathParityTest,
                         ::testing::ValuesIn(conv2dOpList));

//===----------------------------------------------------------------------===//
// Conv3dOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::Conv3dOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, Conv3dOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::Conv3dOpT buildConv3dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_depth, uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::StringRef padding_mode,
    uint32_t groups, std::optional<ttcore::DataTypeAttr> outputDtype,
    std::optional<Conv3dConfigAttr> conv3dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

const mlir::tt::ttnn::Conv3dConfigAttr nonDefaultConv3dConfigAttr =
    mlir::tt::ttnn::Conv3dConfigAttr::get(
        getContext(),
        /*weights_dtype=*/mlir::tt::ttcore::DataType::BFloat16,
        /*t_out_block=*/1u,
        /*w_out_block=*/2u,
        /*h_out_block=*/2u,
        /*c_out_block=*/16u,
        /*c_in_block=*/16u,
        /*compute_with_storage_grid_size=*/
        mlir::tt::ttcore::GridAttr::get(getContext(),
                                        llvm::ArrayRef<int64_t>{8, 8}));

void resetUnusedFields(::tt::target::ttnn::Conv3dOpT &opTOpModel,
                       ::tt::target::ttnn::Conv3dOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::Conv3dOpT &opT) {
    opT.input.reset();
    opT.weight.reset();
    opT.bias.reset();
    opT.device.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::Conv3dOp buildTestConv3dOp(
    bool withBias = false, mlir::tt::ttcore::DataTypeAttr outputDtype = {},
    mlir::tt::ttnn::Conv3dConfigAttr conv3dConfig = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {},
    uint32_t inChannels = 32, uint32_t outChannels = 64, uint32_t batchSize = 1,
    uint32_t inputDepth = 8, uint32_t inputHeight = 8, uint32_t inputWidth = 8,
    llvm::ArrayRef<int32_t> kernelSize = {3, 3, 3},
    llvm::ArrayRef<int32_t> stride = {1, 1, 1},
    llvm::ArrayRef<int32_t> padding = {0, 0, 0},
    llvm::StringRef paddingMode = "zeros", uint32_t groups = 1) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto weightType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value weight =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{weightType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value bias = nullptr;
  if (withBias) {
    auto biasType = tiledL1BF16Type(defaultShape);
    bias = e.builder
               .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{biasType},
                                               mlir::ValueRange{})
               .getResult();
  }

  mlir::Value device =
      e.builder
          .create<mlir::tt::ttnn::GetDeviceOp>(
              loc, e.builder.getType<mlir::tt::ttnn::DeviceType>(),
              mlir::tt::ttnn::MeshShapeAttr::get(&e.context, 1, 1),
              mlir::tt::ttnn::MeshOffsetAttr::get(&e.context, 0, 0))
          .getResult();

  return e.builder.create<mlir::tt::ttnn::Conv3dOp>(
      loc, outputType, input, weight, bias, device, inChannels, outChannels,
      batchSize, inputDepth, inputHeight, inputWidth, kernelSize, stride,
      padding, paddingMode, groups, outputDtype, conv3dConfig,
      computeKernelConfig);
}

} // namespace

using Conv3dOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::Conv3dOp>;

TEST_P(Conv3dOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::Conv3dOp conv3dOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::Conv3dOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildConv3dOpTFromMLIR(
          conv3dOp.getInChannels(), conv3dOp.getOutChannels(),
          conv3dOp.getBatchSize(), conv3dOp.getInputDepth(),
          conv3dOp.getInputHeight(), conv3dOp.getInputWidth(),
          conv3dOp.getKernelSize(), conv3dOp.getStride(), conv3dOp.getPadding(),
          conv3dOp.getPaddingMode(), conv3dOp.getGroups(),
          conv3dOp.getDtypeAttr(), conv3dOp.getConv3dConfig(),
          conv3dOp.getComputeConfig(), resolveOutputLayout(conv3dOp));

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, conv3dOp.getInput(),
                               conv3dOp.getWeight());
  if (conv3dOp.getBias()) {
    prepopulateOperandTensorRefs(cache, conv3dOp.getBias());
  }
  cache.getOrCreate(conv3dOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, conv3dOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::Conv3dOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::Conv3dOp> conv3dOpList = {
    buildTestConv3dOp(),
    buildTestConv3dOp(/*withBias=*/true),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/bf16DtypeAttr),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/nonDefaultConv3dConfigAttr),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{},
                      /*computeKernelConfig=*/
                      nonDefaultDeviceComputeKernelConfigAttr),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/64u),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/128u),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/4u),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/16u),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/8u,
                      /*inputHeight=*/16u),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/8u,
                      /*inputHeight=*/8u, /*inputWidth=*/16u),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/8u,
                      /*inputHeight=*/8u, /*inputWidth=*/8u,
                      /*kernelSize=*/{2, 2, 2}),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/8u,
                      /*inputHeight=*/8u, /*inputWidth=*/8u,
                      /*kernelSize=*/{3, 3, 3}, /*stride=*/{2, 2, 2}),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/8u,
                      /*inputHeight=*/8u, /*inputWidth=*/8u,
                      /*kernelSize=*/{3, 3, 3}, /*stride=*/{1, 1, 1},
                      /*padding=*/{1, 1, 1}),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/8u,
                      /*inputHeight=*/8u, /*inputWidth=*/8u,
                      /*kernelSize=*/{3, 3, 3}, /*stride=*/{1, 1, 1},
                      /*padding=*/{0, 0, 0},
                      /*paddingMode=*/"replicate"),
    buildTestConv3dOp(/*withBias=*/false, /*outputDtype=*/{},
                      /*conv3dConfig=*/{}, /*computeKernelConfig=*/{},
                      /*inChannels=*/32u, /*outChannels=*/64u,
                      /*batchSize=*/1u, /*inputDepth=*/8u,
                      /*inputHeight=*/8u, /*inputWidth=*/8u,
                      /*kernelSize=*/{3, 3, 3}, /*stride=*/{1, 1, 1},
                      /*padding=*/{0, 0, 0}, /*paddingMode=*/"zeros",
                      /*groups=*/4u),
    buildTestConv3dOp(/*withBias=*/true, /*outputDtype=*/bf16DtypeAttr,
                      /*conv3dConfig=*/nonDefaultConv3dConfigAttr,
                      /*computeKernelConfig=*/
                      nonDefaultDeviceComputeKernelConfigAttr,
                      /*inChannels=*/64u, /*outChannels=*/128u,
                      /*batchSize=*/4u, /*inputDepth=*/16u,
                      /*inputHeight=*/16u, /*inputWidth=*/16u,
                      /*kernelSize=*/{2, 2, 2}, /*stride=*/{2, 2, 2},
                      /*padding=*/{1, 1, 1}, /*paddingMode=*/"replicate",
                      /*groups=*/2u),
};

INSTANTIATE_TEST_SUITE_P(Conv3dOpTPathParityTest, Conv3dOpTPathParityTest,
                         ::testing::ValuesIn(conv3dOpList));

//===----------------------------------------------------------------------===//
// ConvTranspose2dOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::ConvTranspose2dOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, ConvTranspose2dOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ConvTranspose2dOpT buildConvTranspose2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> output_padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::ConvTranspose2dOpT &opTOpModel,
                       ::tt::target::ttnn::ConvTranspose2dOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::ConvTranspose2dOpT &opT) {
    opT.input.reset();
    opT.weight.reset();
    opT.bias.reset();
    opT.device.reset();
    resetOutputTensorRefT(opT.out);
    opT.output_dtype.reset();
    opT.memory_config.reset();
    opT.compute_config.reset();
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::ConvTranspose2dOp buildTestConvTranspose2dOp(
    bool withBias = false, mlir::tt::ttcore::DataTypeAttr outputDtype = {},
    mlir::tt::ttnn::Conv2dConfigAttr conv2dConfig = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {},
    mlir::tt::ttnn::Conv2dSliceConfigAttr conv2dSliceConfig = {},
    mlir::tt::ttnn::MemoryConfigAttr memoryConfig = {},
    uint32_t inChannels = 64, uint32_t outChannels = 32, uint32_t batchSize = 1,
    uint32_t inputHeight = 28, uint32_t inputWidth = 28,
    llvm::ArrayRef<int32_t> kernelSize = {3, 3},
    llvm::ArrayRef<int32_t> stride = {2, 2},
    llvm::ArrayRef<int32_t> padding = {1, 1},
    llvm::ArrayRef<int32_t> outputPadding = {0, 0},
    llvm::ArrayRef<int32_t> dilation = {1, 1}, uint32_t groups = 1) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto weightType = tiledL1BF16Type(defaultShape);
  auto outputType =
      memoryConfig ? tiledBF16TypeFromMemoryConfig(defaultShape, memoryConfig)
                   : tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value weight =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{weightType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value bias = nullptr;
  if (withBias) {
    auto biasType = tiledL1BF16Type(defaultShape);
    bias = e.builder
               .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{biasType},
                                               mlir::ValueRange{})
               .getResult();
  }

  mlir::Value device =
      e.builder
          .create<mlir::tt::ttnn::GetDeviceOp>(
              loc, e.builder.getType<mlir::tt::ttnn::DeviceType>(),
              mlir::tt::ttnn::MeshShapeAttr::get(&e.context, 1, 1),
              mlir::tt::ttnn::MeshOffsetAttr::get(&e.context, 0, 0))
          .getResult();

  return e.builder.create<mlir::tt::ttnn::ConvTranspose2dOp>(
      loc, outputType, input, weight, bias, device, inChannels, outChannels,
      batchSize, inputHeight, inputWidth, kernelSize, stride, padding,
      outputPadding, dilation, groups, outputDtype, conv2dConfig,
      computeKernelConfig, conv2dSliceConfig);
}

} // namespace

using ConvTranspose2dOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::ConvTranspose2dOp>;

TEST_P(ConvTranspose2dOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::ConvTranspose2dOp convTranspose2dOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::ConvTranspose2dOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildConvTranspose2dOpTFromMLIR(
          convTranspose2dOp.getInChannels(), convTranspose2dOp.getOutChannels(),
          convTranspose2dOp.getBatchSize(), convTranspose2dOp.getInputHeight(),
          convTranspose2dOp.getInputWidth(), convTranspose2dOp.getKernelSize(),
          convTranspose2dOp.getStride(), convTranspose2dOp.getPadding(),
          convTranspose2dOp.getOutputPadding(), convTranspose2dOp.getDilation(),
          convTranspose2dOp.getGroups(), convTranspose2dOp.getConv2dConfig(),
          convTranspose2dOp.getConv2dSliceConfig(),
          resolveOutputLayout(convTranspose2dOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, convTranspose2dOp.getInput(),
                               convTranspose2dOp.getWeight());
  if (convTranspose2dOp.getBias()) {
    prepopulateOperandTensorRefs(cache, convTranspose2dOp.getBias());
  }
  cache.getOrCreate(convTranspose2dOp.getDevice(),
                    mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, convTranspose2dOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::ConvTranspose2dOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::ConvTranspose2dOp>
    convTranspose2dOpList = {
        buildTestConvTranspose2dOp(),
        buildTestConvTranspose2dOp(/*withBias=*/true),
        buildTestConvTranspose2dOp(/*withBias=*/false,
                                   /*outputDtype=*/bf16DtypeAttr),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/nonDefaultConv2dConfigAttr),
        buildTestConvTranspose2dOp(
            /*withBias=*/false, /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
        buildTestConvTranspose2dOp(
            /*withBias=*/false, /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/
                                   nonDefaultInputMemoryConfigAttr),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/128u),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/64u),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/4u),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/14u),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/28u,
                                   /*inputWidth=*/14u),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/28u,
                                   /*inputWidth=*/28u, /*kernelSize=*/{4, 4}),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/28u,
                                   /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
                                   /*stride=*/{1, 1}),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/28u,
                                   /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
                                   /*stride=*/{2, 2}, /*padding=*/{0, 0}),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/28u,
                                   /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
                                   /*stride=*/{2, 2}, /*padding=*/{1, 1},
                                   /*outputPadding=*/{1, 1}),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/28u,
                                   /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
                                   /*stride=*/{2, 2}, /*padding=*/{1, 1},
                                   /*outputPadding=*/{0, 0},
                                   /*dilation=*/{2, 2}),
        buildTestConvTranspose2dOp(/*withBias=*/false, /*outputDtype=*/{},
                                   /*conv2dConfig=*/{},
                                   /*computeKernelConfig=*/{},
                                   /*conv2dSliceConfig=*/{},
                                   /*memoryConfig=*/{},
                                   /*inChannels=*/64u, /*outChannels=*/32u,
                                   /*batchSize=*/1u, /*inputHeight=*/28u,
                                   /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
                                   /*stride=*/{2, 2}, /*padding=*/{1, 1},
                                   /*outputPadding=*/{0, 0},
                                   /*dilation=*/{1, 1}, /*groups=*/4u),
        buildTestConvTranspose2dOp(
            /*withBias=*/true, /*outputDtype=*/bf16DtypeAttr,
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr,
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr,
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr,
            /*memoryConfig=*/nonDefaultInputMemoryConfigAttr,
            /*inChannels=*/128u, /*outChannels=*/64u, /*batchSize=*/4u,
            /*inputHeight=*/14u, /*inputWidth=*/14u, /*kernelSize=*/{4, 4},
            /*stride=*/{1, 1}, /*padding=*/{0, 0}, /*outputPadding=*/{1, 1},
            /*dilation=*/{2, 2}, /*groups=*/2u),
};

INSTANTIATE_TEST_SUITE_P(ConvTranspose2dOpTPathParityTest,
                         ConvTranspose2dOpTPathParityTest,
                         ::testing::ValuesIn(convTranspose2dOpList));

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::PrepareConv2dBiasOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, PrepareConv2dBiasOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PrepareConv2dBiasOpT buildPrepareConv2dBiasOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::PrepareConv2dBiasOpT &opTOpModel,
                       ::tt::target::ttnn::PrepareConv2dBiasOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::PrepareConv2dBiasOpT &opT) {
    opT.bias_tensor.reset();
    opT.device.reset();
    resetOutputTensorRefT(opT.out);
    opT.conv2d_slice_config.reset();
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::PrepareConv2dBiasOp buildTestPrepareConv2dBiasOp(
    mlir::tt::ttcore::DataTypeAttr outputDtype = {},
    mlir::tt::ttnn::Conv2dConfigAttr conv2dConfig = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {},
    mlir::tt::ttnn::Conv2dSliceConfigAttr conv2dSliceConfig = {},
    uint32_t inChannels = 3, uint32_t outChannels = 64, uint32_t batchSize = 1,
    uint32_t inputHeight = 224, uint32_t inputWidth = 224,
    llvm::ArrayRef<int32_t> kernelSize = {7, 7},
    llvm::ArrayRef<int32_t> stride = {2, 2},
    llvm::ArrayRef<int32_t> padding = {3, 3},
    llvm::ArrayRef<int32_t> dilation = {1, 1}, uint32_t groups = 1,
    mlir::tt::ttnn::MemoryConfigAttr inputMemoryConfig = {},
    mlir::tt::ttnn::Layout inputTensorLayout = mlir::tt::ttnn::Layout::Tile,
    mlir::tt::ttcore::DataType inputDtype =
        mlir::tt::ttcore::DataType::BFloat16) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  llvm::SmallVector<int64_t, 4> biasShape = {1, 1, 1,
                                             static_cast<int64_t>(outChannels)};
  auto biasType = tiledL1BF16Type(biasShape);
  auto outputType = tiledL1BF16Type(biasShape);

  mlir::Value bias = e.builder
                         .create<mlir::tt::ttnn::OnesOp>(
                             loc, mlir::TypeRange{biasType}, mlir::ValueRange{})
                         .getResult();

  mlir::Value device =
      e.builder
          .create<mlir::tt::ttnn::GetDeviceOp>(
              loc, e.builder.getType<mlir::tt::ttnn::DeviceType>(),
              mlir::tt::ttnn::MeshShapeAttr::get(&e.context, 1, 1),
              mlir::tt::ttnn::MeshOffsetAttr::get(&e.context, 0, 0))
          .getResult();

  if (!inputMemoryConfig) {
    auto biasLayout = mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(biasType).getEncoding());
    inputMemoryConfig = mlir::tt::ttnn::MemoryConfigAttr::get(biasLayout);
  }

  return e.builder.create<mlir::tt::ttnn::PrepareConv2dBiasOp>(
      loc, outputType, bias, inputMemoryConfig, inputTensorLayout, inChannels,
      outChannels, batchSize, inputHeight, inputWidth, kernelSize, stride,
      padding, dilation, groups, device, inputDtype, outputDtype, conv2dConfig,
      computeKernelConfig, conv2dSliceConfig);
}

} // namespace

using PrepareConv2dBiasOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::PrepareConv2dBiasOp>;

TEST_P(PrepareConv2dBiasOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::PrepareConv2dBiasOp prepareOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::PrepareConv2dBiasOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildPrepareConv2dBiasOpTFromMLIR(
          prepareOp.getInputMemoryConfig(), prepareOp.getInputTensorLayout(),
          prepareOp.getInChannels(), prepareOp.getOutChannels(),
          prepareOp.getBatchSize(), prepareOp.getInputHeight(),
          prepareOp.getInputWidth(), prepareOp.getKernelSize(),
          prepareOp.getStride(), prepareOp.getPadding(),
          prepareOp.getDilation(), prepareOp.getGroups(),
          prepareOp.getInputDtype(), prepareOp.getOutputDtype(),
          prepareOp.getConv2dConfig(), prepareOp.getComputeConfig(),
          resolveOutputLayout(prepareOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, prepareOp.getBiasTensor());
  cache.getOrCreate(prepareOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, prepareOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PrepareConv2dBiasOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::PrepareConv2dBiasOp>
    prepareConv2dBiasOpList = {
        buildTestPrepareConv2dBiasOp(),
        buildTestPrepareConv2dBiasOp(/*outputDtype=*/bf16DtypeAttr),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/nonDefaultConv2dConfigAttr),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr),
        buildTestPrepareConv2dBiasOp(/*outputDtype=*/{}, /*conv2dConfig=*/{},
                                     /*computeKernelConfig=*/{},
                                     /*conv2dSliceConfig=*/{},
                                     /*inChannels=*/64u),
        buildTestPrepareConv2dBiasOp(/*outputDtype=*/{}, /*conv2dConfig=*/{},
                                     /*computeKernelConfig=*/{},
                                     /*conv2dSliceConfig=*/{},
                                     /*inChannels=*/3u,
                                     /*outChannels=*/128u),
        buildTestPrepareConv2dBiasOp(/*outputDtype=*/{}, /*conv2dConfig=*/{},
                                     /*computeKernelConfig=*/{},
                                     /*conv2dSliceConfig=*/{},
                                     /*inChannels=*/3u, /*outChannels=*/64u,
                                     /*batchSize=*/8u),
        buildTestPrepareConv2dBiasOp(/*outputDtype=*/{}, /*conv2dConfig=*/{},
                                     /*computeKernelConfig=*/{},
                                     /*conv2dSliceConfig=*/{},
                                     /*inChannels=*/3u, /*outChannels=*/64u,
                                     /*batchSize=*/1u, /*inputHeight=*/56u),
        buildTestPrepareConv2dBiasOp(/*outputDtype=*/{}, /*conv2dConfig=*/{},
                                     /*computeKernelConfig=*/{},
                                     /*conv2dSliceConfig=*/{},
                                     /*inChannels=*/3u, /*outChannels=*/64u,
                                     /*batchSize=*/1u, /*inputHeight=*/224u,
                                     /*inputWidth=*/56u),
        buildTestPrepareConv2dBiasOp(/*outputDtype=*/{}, /*conv2dConfig=*/{},
                                     /*computeKernelConfig=*/{},
                                     /*conv2dSliceConfig=*/{},
                                     /*inChannels=*/3u, /*outChannels=*/64u,
                                     /*batchSize=*/1u, /*inputHeight=*/224u,
                                     /*inputWidth=*/224u,
                                     /*kernelSize=*/{3, 3}),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/{}, /*inChannels=*/3u, /*outChannels=*/64u,
            /*batchSize=*/1u, /*inputHeight=*/224u, /*inputWidth=*/224u,
            /*kernelSize=*/{7, 7}, /*stride=*/{1, 1}),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/{}, /*inChannels=*/3u, /*outChannels=*/64u,
            /*batchSize=*/1u, /*inputHeight=*/224u, /*inputWidth=*/224u,
            /*kernelSize=*/{7, 7}, /*stride=*/{2, 2}, /*padding=*/{1, 1}),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/{}, /*inChannels=*/3u, /*outChannels=*/64u,
            /*batchSize=*/1u, /*inputHeight=*/224u, /*inputWidth=*/224u,
            /*kernelSize=*/{7, 7}, /*stride=*/{2, 2}, /*padding=*/{3, 3},
            /*dilation=*/{2, 2}),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/{}, /*inChannels=*/3u, /*outChannels=*/64u,
            /*batchSize=*/1u, /*inputHeight=*/224u, /*inputWidth=*/224u,
            /*kernelSize=*/{7, 7}, /*stride=*/{2, 2}, /*padding=*/{3, 3},
            /*dilation=*/{1, 1}, /*groups=*/3u),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/{}, /*inChannels=*/3u, /*outChannels=*/64u,
            /*batchSize=*/1u, /*inputHeight=*/224u, /*inputWidth=*/224u,
            /*kernelSize=*/{7, 7}, /*stride=*/{2, 2}, /*padding=*/{3, 3},
            /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/{}, /*inChannels=*/3u, /*outChannels=*/64u,
            /*batchSize=*/1u, /*inputHeight=*/224u, /*inputWidth=*/224u,
            /*kernelSize=*/{7, 7}, /*stride=*/{2, 2}, /*padding=*/{3, 3},
            /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{}, /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/{}, /*inChannels=*/3u, /*outChannels=*/64u,
            /*batchSize=*/1u, /*inputHeight=*/224u, /*inputWidth=*/224u,
            /*kernelSize=*/{7, 7}, /*stride=*/{2, 2}, /*padding=*/{3, 3},
            /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::Tile,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
        buildTestPrepareConv2dBiasOp(
            /*outputDtype=*/bf16DtypeAttr,
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr,
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr,
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr,
            /*inChannels=*/64u, /*outChannels=*/128u, /*batchSize=*/8u,
            /*inputHeight=*/56u, /*inputWidth=*/56u, /*kernelSize=*/{3, 3},
            /*stride=*/{1, 1}, /*padding=*/{1, 1}, /*dilation=*/{2, 2},
            /*groups=*/2u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr,
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
};

INSTANTIATE_TEST_SUITE_P(PrepareConv2dBiasOpTPathParityTest,
                         PrepareConv2dBiasOpTPathParityTest,
                         ::testing::ValuesIn(prepareConv2dBiasOpList));

//===----------------------------------------------------------------------===//
// PrepareConv2dWeightsOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::PrepareConv2dWeightsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, PrepareConv2dWeightsOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
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
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::PrepareConv2dWeightsOpT &opTOpModel,
                       ::tt::target::ttnn::PrepareConv2dWeightsOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::PrepareConv2dWeightsOpT &opT) {
    opT.weight_tensor.reset();
    opT.device.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::PrepareConv2dWeightsOp buildTestPrepareConv2dWeightsOp(
    mlir::tt::ttcore::DataTypeAttr outputDtype = {},
    mlir::tt::ttnn::Conv2dConfigAttr conv2dConfig = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {},
    mlir::tt::ttnn::Conv2dSliceConfigAttr conv2dSliceConfig = {},
    llvm::StringRef weightsFormat = "OIHW", bool hasBias = false,
    uint32_t inChannels = 3, uint32_t outChannels = 64, uint32_t batchSize = 1,
    uint32_t inputHeight = 224, uint32_t inputWidth = 224,
    llvm::ArrayRef<int32_t> kernelSize = {7, 7},
    llvm::ArrayRef<int32_t> stride = {2, 2},
    llvm::ArrayRef<int32_t> padding = {3, 3},
    llvm::ArrayRef<int32_t> dilation = {1, 1}, uint32_t groups = 1,
    mlir::tt::ttnn::MemoryConfigAttr inputMemoryConfig = {},
    mlir::tt::ttnn::Layout inputTensorLayout = mlir::tt::ttnn::Layout::Tile,
    mlir::tt::ttcore::DataType inputDtype =
        mlir::tt::ttcore::DataType::BFloat16) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  llvm::SmallVector<int64_t, 4> weightShape = {
      static_cast<int64_t>(outChannels),
      static_cast<int64_t>(inChannels / groups),
      static_cast<int64_t>(kernelSize[0]), static_cast<int64_t>(kernelSize[1])};
  auto weightType = tiledL1BF16Type(weightShape);
  auto outputType = tiledL1BF16Type(weightShape);

  mlir::Value weight =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{weightType},
                                          mlir::ValueRange{})
          .getResult();

  mlir::Value device =
      e.builder
          .create<mlir::tt::ttnn::GetDeviceOp>(
              loc, e.builder.getType<mlir::tt::ttnn::DeviceType>(),
              mlir::tt::ttnn::MeshShapeAttr::get(&e.context, 1, 1),
              mlir::tt::ttnn::MeshOffsetAttr::get(&e.context, 0, 0))
          .getResult();

  if (!inputMemoryConfig) {
    auto weightLayout = mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(weightType).getEncoding());
    inputMemoryConfig = mlir::tt::ttnn::MemoryConfigAttr::get(weightLayout);
  }

  return e.builder.create<mlir::tt::ttnn::PrepareConv2dWeightsOp>(
      loc, outputType, weight, inputMemoryConfig, inputTensorLayout,
      weightsFormat, inChannels, outChannels, batchSize, inputHeight,
      inputWidth, kernelSize, stride, padding, dilation, hasBias, groups,
      device, inputDtype, outputDtype, conv2dConfig, computeKernelConfig,
      conv2dSliceConfig);
}

} // namespace

using PrepareConv2dWeightsOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::PrepareConv2dWeightsOp>;

TEST_P(PrepareConv2dWeightsOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::PrepareConv2dWeightsOp prepareOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::PrepareConv2dWeightsOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildPrepareConv2dWeightsOpTFromMLIR(
          prepareOp.getInputMemoryConfig(), prepareOp.getInputTensorLayout(),
          prepareOp.getWeightsFormat(), prepareOp.getInChannels(),
          prepareOp.getOutChannels(), prepareOp.getBatchSize(),
          prepareOp.getInputHeight(), prepareOp.getInputWidth(),
          prepareOp.getKernelSize(), prepareOp.getStride(),
          prepareOp.getPadding(), prepareOp.getDilation(),
          prepareOp.getHasBias(), prepareOp.getGroups(),
          prepareOp.getInputDtype(), prepareOp.getOutputDtype(),
          prepareOp.getConv2dConfig(), prepareOp.getComputeConfig(),
          prepareOp.getConv2dSliceConfig(), resolveOutputLayout(prepareOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, prepareOp.getWeightTensor());
  cache.getOrCreate(prepareOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, prepareOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PrepareConv2dWeightsOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::PrepareConv2dWeightsOp>
    prepareConv2dWeightsOpList = {
        buildTestPrepareConv2dWeightsOp(),
        buildTestPrepareConv2dWeightsOp(/*outputDtype=*/bf16DtypeAttr),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{},
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"IOHW"),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/true),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/64u),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/128u),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/8u),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/56u),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/56u),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{3, 3}),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{7, 7}, /*stride=*/{1, 1}),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
            /*padding=*/{1, 1}),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
            /*padding=*/{3, 3}, /*dilation=*/{2, 2}),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
            /*padding=*/{3, 3}, /*dilation=*/{1, 1}, /*groups=*/3u),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
            /*padding=*/{3, 3}, /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
            /*padding=*/{3, 3}, /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false, /*inChannels=*/3u,
            /*outChannels=*/64u, /*batchSize=*/1u, /*inputHeight=*/224u,
            /*inputWidth=*/224u, /*kernelSize=*/{7, 7}, /*stride=*/{2, 2},
            /*padding=*/{3, 3}, /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::Tile,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
        buildTestPrepareConv2dWeightsOp(
            /*outputDtype=*/bf16DtypeAttr,
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr,
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr,
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr,
            /*weightsFormat=*/"IOHW", /*hasBias=*/true,
            /*inChannels=*/64u, /*outChannels=*/128u, /*batchSize=*/8u,
            /*inputHeight=*/56u, /*inputWidth=*/56u, /*kernelSize=*/{3, 3},
            /*stride=*/{1, 1}, /*padding=*/{1, 1}, /*dilation=*/{2, 2},
            /*groups=*/2u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr,
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
};

INSTANTIATE_TEST_SUITE_P(PrepareConv2dWeightsOpTPathParityTest,
                         PrepareConv2dWeightsOpTPathParityTest,
                         ::testing::ValuesIn(prepareConv2dWeightsOpList));

//===----------------------------------------------------------------------===//
// PrepareConvTranspose2dBiasOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::PrepareConvTranspose2dBiasOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         PrepareConvTranspose2dBiasOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
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
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opTOpModel,
    ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opT) {
    opT.bias_tensor.reset();
    opT.device.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::PrepareConvTranspose2dBiasOp
buildTestPrepareConvTranspose2dBiasOp(
    mlir::tt::ttcore::DataTypeAttr outputDtype = {},
    mlir::tt::ttnn::Conv2dConfigAttr conv2dConfig = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {},
    mlir::tt::ttnn::Conv2dSliceConfigAttr conv2dSliceConfig = {},
    uint32_t inChannels = 64, uint32_t outChannels = 32, uint32_t batchSize = 1,
    uint32_t inputHeight = 28, uint32_t inputWidth = 28,
    llvm::ArrayRef<int32_t> kernelSize = {3, 3},
    llvm::ArrayRef<int32_t> stride = {2, 2},
    llvm::ArrayRef<int32_t> padding = {1, 1},
    llvm::ArrayRef<int32_t> dilation = {1, 1}, uint32_t groups = 1,
    mlir::tt::ttnn::MemoryConfigAttr inputMemoryConfig = {},
    mlir::tt::ttnn::Layout inputTensorLayout = mlir::tt::ttnn::Layout::Tile,
    mlir::tt::ttcore::DataType inputDtype =
        mlir::tt::ttcore::DataType::BFloat16) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  llvm::SmallVector<int64_t, 4> biasShape = {1, 1, 1,
                                             static_cast<int64_t>(outChannels)};
  auto biasType = tiledL1BF16Type(biasShape);
  auto outputType = tiledL1BF16Type(biasShape);

  mlir::Value bias = e.builder
                         .create<mlir::tt::ttnn::OnesOp>(
                             loc, mlir::TypeRange{biasType}, mlir::ValueRange{})
                         .getResult();

  mlir::Value device =
      e.builder
          .create<mlir::tt::ttnn::GetDeviceOp>(
              loc, e.builder.getType<mlir::tt::ttnn::DeviceType>(),
              mlir::tt::ttnn::MeshShapeAttr::get(&e.context, 1, 1),
              mlir::tt::ttnn::MeshOffsetAttr::get(&e.context, 0, 0))
          .getResult();

  if (!inputMemoryConfig) {
    auto biasLayout = mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(biasType).getEncoding());
    inputMemoryConfig = mlir::tt::ttnn::MemoryConfigAttr::get(biasLayout);
  }

  return e.builder.create<mlir::tt::ttnn::PrepareConvTranspose2dBiasOp>(
      loc, outputType, bias, inputMemoryConfig, inputTensorLayout, inChannels,
      outChannels, batchSize, inputHeight, inputWidth, kernelSize, stride,
      padding, dilation, groups, device, inputDtype, outputDtype, conv2dConfig,
      computeKernelConfig, conv2dSliceConfig);
}

} // namespace

using PrepareConvTranspose2dBiasOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::PrepareConvTranspose2dBiasOp>;

TEST_P(PrepareConvTranspose2dBiasOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::PrepareConvTranspose2dBiasOp prepareOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildPrepareConvTranspose2dBiasOpTFromMLIR(
          prepareOp.getInputMemoryConfig(), prepareOp.getInputTensorLayout(),
          prepareOp.getInChannels(), prepareOp.getOutChannels(),
          prepareOp.getBatchSize(), prepareOp.getInputHeight(),
          prepareOp.getInputWidth(), prepareOp.getKernelSize(),
          prepareOp.getStride(), prepareOp.getPadding(),
          prepareOp.getDilation(), prepareOp.getGroups(),
          prepareOp.getInputDtype(), prepareOp.getOutputDtype(),
          prepareOp.getConv2dConfig(), prepareOp.getComputeConfig(),
          prepareOp.getConv2dSliceConfig(), resolveOutputLayout(prepareOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, prepareOp.getBiasTensor());
  cache.getOrCreate(prepareOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, prepareOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::PrepareConvTranspose2dBiasOp>
    prepareConvTranspose2dBiasOpList = {
        buildTestPrepareConvTranspose2dBiasOp(),
        buildTestPrepareConvTranspose2dBiasOp(/*outputDtype=*/bf16DtypeAttr),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{},
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/128u),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/64u),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/4u),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/14u),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/14u),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{4, 4}),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
            /*stride=*/{1, 1}),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
            /*stride=*/{2, 2}, /*padding=*/{0, 0}),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
            /*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{2, 2}),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
            /*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
            /*groups=*/4u),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
            /*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
            /*groups=*/1u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
            /*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
            /*groups=*/1u, /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*inChannels=*/64u, /*outChannels=*/32u, /*batchSize=*/1u,
            /*inputHeight=*/28u, /*inputWidth=*/28u, /*kernelSize=*/{3, 3},
            /*stride=*/{2, 2}, /*padding=*/{1, 1}, /*dilation=*/{1, 1},
            /*groups=*/1u, /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::Tile,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
        buildTestPrepareConvTranspose2dBiasOp(
            /*outputDtype=*/bf16DtypeAttr,
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr,
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr,
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr,
            /*inChannels=*/128u, /*outChannels=*/64u, /*batchSize=*/4u,
            /*inputHeight=*/14u, /*inputWidth=*/14u, /*kernelSize=*/{4, 4},
            /*stride=*/{1, 1}, /*padding=*/{0, 0}, /*dilation=*/{2, 2},
            /*groups=*/2u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr,
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
};

INSTANTIATE_TEST_SUITE_P(PrepareConvTranspose2dBiasOpTPathParityTest,
                         PrepareConvTranspose2dBiasOpTPathParityTest,
                         ::testing::ValuesIn(prepareConvTranspose2dBiasOpList));

//===----------------------------------------------------------------------===//
// PrepareConvTranspose2dWeightsOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::PrepareConvTranspose2dWeightsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache,
         PrepareConvTranspose2dWeightsOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT
buildPrepareConvTranspose2dWeightsOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool hasBias, int32_t groups, ttcore::DataType inputDtype,
    std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool mirrorKernel,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &opTOpModel,
    ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &opT) {
    opT.weight_tensor.reset();
    opT.device.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::PrepareConvTranspose2dWeightsOp
buildTestPrepareConvTranspose2dWeightsOp(
    mlir::tt::ttcore::DataTypeAttr outputDtype = {},
    mlir::tt::ttnn::Conv2dConfigAttr conv2dConfig = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {},
    mlir::tt::ttnn::Conv2dSliceConfigAttr conv2dSliceConfig = {},
    llvm::StringRef weightsFormat = "OIHW", bool hasBias = false,
    bool mirrorKernel = true, uint32_t inChannels = 64,
    uint32_t outChannels = 32, uint32_t batchSize = 1,
    uint32_t inputHeight = 28, uint32_t inputWidth = 28,
    llvm::ArrayRef<int32_t> kernelSize = {3, 3},
    llvm::ArrayRef<int32_t> stride = {2, 2},
    llvm::ArrayRef<int32_t> padding = {1, 1},
    llvm::ArrayRef<int32_t> dilation = {1, 1}, uint32_t groups = 1,
    mlir::tt::ttnn::MemoryConfigAttr inputMemoryConfig = {},
    mlir::tt::ttnn::Layout inputTensorLayout = mlir::tt::ttnn::Layout::Tile,
    mlir::tt::ttcore::DataType inputDtype =
        mlir::tt::ttcore::DataType::BFloat16) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  llvm::SmallVector<int64_t, 4> weightShape = {
      static_cast<int64_t>(inChannels),
      static_cast<int64_t>(outChannels / groups),
      static_cast<int64_t>(kernelSize[0]), static_cast<int64_t>(kernelSize[1])};
  auto weightType = tiledL1BF16Type(weightShape);
  auto outputType = tiledL1BF16Type(weightShape);

  mlir::Value weight =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{weightType},
                                          mlir::ValueRange{})
          .getResult();

  mlir::Value device =
      e.builder
          .create<mlir::tt::ttnn::GetDeviceOp>(
              loc, e.builder.getType<mlir::tt::ttnn::DeviceType>(),
              mlir::tt::ttnn::MeshShapeAttr::get(&e.context, 1, 1),
              mlir::tt::ttnn::MeshOffsetAttr::get(&e.context, 0, 0))
          .getResult();

  if (!inputMemoryConfig) {
    auto weightLayout = mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(weightType).getEncoding());
    inputMemoryConfig = mlir::tt::ttnn::MemoryConfigAttr::get(weightLayout);
  }

  return e.builder.create<mlir::tt::ttnn::PrepareConvTranspose2dWeightsOp>(
      loc, outputType, weight, inputMemoryConfig, inputTensorLayout,
      weightsFormat, inChannels, outChannels, batchSize, inputHeight,
      inputWidth, kernelSize, stride, padding, dilation, hasBias, groups,
      device, inputDtype, outputDtype, conv2dConfig, computeKernelConfig,
      conv2dSliceConfig, mirrorKernel);
}

} // namespace

using PrepareConvTranspose2dWeightsOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::PrepareConvTranspose2dWeightsOp>;

TEST_P(PrepareConvTranspose2dWeightsOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::PrepareConvTranspose2dWeightsOp prepareOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildPrepareConvTranspose2dWeightsOpTFromMLIR(
          prepareOp.getInputMemoryConfig(), prepareOp.getInputTensorLayout(),
          prepareOp.getWeightsFormat(), prepareOp.getInChannels(),
          prepareOp.getOutChannels(), prepareOp.getBatchSize(),
          prepareOp.getInputHeight(), prepareOp.getInputWidth(),
          prepareOp.getKernelSize(), prepareOp.getStride(),
          prepareOp.getPadding(), prepareOp.getDilation(),
          prepareOp.getHasBias(), prepareOp.getGroups(),
          prepareOp.getInputDtype(), prepareOp.getOutputDtype(),
          prepareOp.getConv2dConfig(), prepareOp.getComputeConfig(),
          prepareOp.getConv2dSliceConfig(), prepareOp.getMirrorKernel(),
          resolveOutputLayout(prepareOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, prepareOp.getWeightTensor());
  cache.getOrCreate(prepareOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, prepareOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::PrepareConvTranspose2dWeightsOp>
    prepareConvTranspose2dWeightsOpList = {
        buildTestPrepareConvTranspose2dWeightsOp(),
        buildTestPrepareConvTranspose2dWeightsOp(/*outputDtype=*/bf16DtypeAttr),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{},
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{},
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"IOHW"),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/true),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/false),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/128u),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/64u),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/4u),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/14u),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/14u),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{4, 4}),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{1, 1}),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{0, 0}),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*dilation=*/{2, 2}),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*dilation=*/{1, 1}, /*groups=*/4u),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*dilation=*/{1, 1}, /*groups=*/1u, /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*dilation=*/{1, 1}, /*groups=*/1u, /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::Tile,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/bf16DtypeAttr,
            /*conv2dConfig=*/nonDefaultConv2dConfigAttr,
            /*computeKernelConfig=*/nonDefaultDeviceComputeKernelConfigAttr,
            /*conv2dSliceConfig=*/nonDefaultConv2dSliceConfigAttr,
            /*weightsFormat=*/"IOHW", /*hasBias=*/true, /*mirrorKernel=*/false,
            /*inChannels=*/128u, /*outChannels=*/64u, /*batchSize=*/4u,
            /*inputHeight=*/14u, /*inputWidth=*/14u, /*kernelSize=*/{4, 4},
            /*stride=*/{1, 1}, /*padding=*/{0, 0}, /*dilation=*/{2, 2},
            /*groups=*/2u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr,
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor,
            /*inputDtype=*/mlir::tt::ttcore::DataType::Float32),
};

INSTANTIATE_TEST_SUITE_P(
    PrepareConvTranspose2dWeightsOpTPathParityTest,
    PrepareConvTranspose2dWeightsOpTPathParityTest,
    ::testing::ValuesIn(prepareConvTranspose2dWeightsOpList));

//===----------------------------------------------------------------------===//
// LinearOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::LinearOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, LinearOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::LinearOpT buildLinearOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::LinearOpT &opTOpModel,
                       ::tt::target::ttnn::LinearOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::LinearOpT &opT) {
    opT.a.reset();
    opT.b.reset();
    opT.bias.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::LinearOp buildTestLinearOp(
    bool withBias = false, bool transposeA = false, bool transposeB = false,
    mlir::StringAttr activation = {}, mlir::Attribute programConfigAttr = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeKernelConfig = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto typeA = tiledL1BF16Type(defaultShape);
  auto typeB = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  mlir::Value a = e.builder
                      .create<mlir::tt::ttnn::OnesOp>(
                          loc, mlir::TypeRange{typeA}, mlir::ValueRange{})
                      .getResult();
  mlir::Value b = e.builder
                      .create<mlir::tt::ttnn::OnesOp>(
                          loc, mlir::TypeRange{typeB}, mlir::ValueRange{})
                      .getResult();
  mlir::Value bias = nullptr;
  if (withBias) {
    auto biasType = tiledL1BF16Type(defaultShape);
    bias = e.builder
               .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{biasType},
                                               mlir::ValueRange{})
               .getResult();
  }

  return e.builder.create<mlir::tt::ttnn::LinearOp>(
      loc, outputType, a, b, bias, transposeA, transposeB, programConfigAttr,
      activation, computeKernelConfig);
}

} // namespace

using LinearOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::LinearOp>;

TEST_P(LinearOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::LinearOp linearOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::LinearOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildLinearOpTFromMLIR(
          linearOp.getTransposeA(), linearOp.getTransposeB(),
          linearOp.getActivation(), linearOp.getMatmulProgramConfig(),
          linearOp.getComputeConfig(), resolveOutputLayout(linearOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, linearOp.getA(), linearOp.getB());
  if (linearOp.getBias()) {
    prepopulateOperandTensorRefs(cache, linearOp.getBias());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, linearOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::LinearOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::LinearOp> linearOpList = {
    buildTestLinearOp(),
    buildTestLinearOp(/*withBias=*/true),
    buildTestLinearOp(/*withBias=*/false, /*transposeA=*/true),
    buildTestLinearOp(/*withBias=*/false, /*transposeA=*/false,
                      /*transposeB=*/true),
    buildTestLinearOp(/*withBias=*/false, /*transposeA=*/false,
                      /*transposeB=*/false,
                      mlir::StringAttr::get(getContext(), "relu")),
    buildTestLinearOp(
        /*withBias=*/false, /*transposeA=*/false, /*transposeB=*/false,
        /*activation=*/{},
        mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::get(
            getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(getContext(), 8, 8), 2, 4, 4, 8,
            8)),
    buildTestLinearOp(/*withBias=*/false, /*transposeA=*/false,
                      /*transposeB=*/false,
                      /*activation=*/{}, /*programConfigAttr=*/{},
                      mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
                          getContext(), mlir::tt::ttnn::MathFidelity::HiFi2,
                          mlir::BoolAttr::get(getContext(), false),
                          mlir::BoolAttr::get(getContext(), true),
                          mlir::BoolAttr::get(getContext(), true),
                          mlir::BoolAttr::get(getContext(), false))),
    buildTestLinearOp(
        /*withBias=*/true, /*transposeA=*/true, /*transposeB=*/true,
        mlir::StringAttr::get(getContext(), "relu"),
        mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::get(
            getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(getContext(), 8, 8), 2, 4, 4, 8,
            8),
        mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
            getContext(), mlir::tt::ttnn::MathFidelity::HiFi2,
            mlir::BoolAttr::get(getContext(), false),
            mlir::BoolAttr::get(getContext(), true),
            mlir::BoolAttr::get(getContext(), true),
            mlir::BoolAttr::get(getContext(), false))),
};

INSTANTIATE_TEST_SUITE_P(LinearOpTPathParityTest, LinearOpTPathParityTest,
                         ::testing::ValuesIn(linearOpList));

//===----------------------------------------------------------------------===//
// EltwiseUnaryOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
template <typename EltwiseUnaryOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseUnaryOp>
createEltwiseUnaryOp(::mlir::tt::FlatbufferObjectCache &cache,
                     EltwiseUnaryOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryOpT
buildEltwiseUnaryOpTFromMLIR(TTNNLayoutAttr outputLayout,
                             std::optional<llvm::APFloat> slope = std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::EltwiseUnaryOpT &opTOpModel,
                       ::tt::target::ttnn::EltwiseUnaryOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::EltwiseUnaryOpT &opT) {
    opT.in.reset();
    opT.memory_config.reset();
    resetOutputTensorRefT(opT.out);
    if (opT.type != ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu &&
        opT.type != ::tt::target::ttnn::EltwiseUnaryOpType::Tanh &&
        opT.type != ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid) {
      opT.type = tt::target::ttnn::EltwiseUnaryOpType::Abs;
    }
  };

  helper(opTOpModel);
  helper(opTFB);
}

template <typename OpTy>
OpTy buildTestEltwiseUnaryOp(float slope = 0.01f) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto inputType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LeakyReluOp>) {
    return e.builder.create<OpTy>(loc, outputType, input,
                                  e.builder.getF32FloatAttr(slope));
  } else {
    return e.builder.create<OpTy>(loc, outputType, input);
  }
}

} // namespace

template <typename OpTy>
static void runEltwiseUnaryParityCheck(OpTy op) {
  ::tt::target::ttnn::EltwiseUnaryOpT opTOpModel;
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LeakyReluOp>) {
    opTOpModel = mlir::tt::ttnn::op_model::buildEltwiseUnaryOpTFromMLIR<OpTy>(
        resolveOutputLayout(op), op.getParameter());
  } else {
    opTOpModel = mlir::tt::ttnn::op_model::buildEltwiseUnaryOpTFromMLIR<OpTy>(
        resolveOutputLayout(op));
  }

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getInput());
  auto fbOffset = mlir::tt::ttnn::createEltwiseUnaryOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseUnaryOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);
  EXPECT_EQ(opTOpModel, opTFB);
}

#define ELTWISE_UNARY_PARITY_SUITE(OpTy)                                       \
  using OpTy##TPathParityTest =                                                \
      ::testing::TestWithParam<mlir::tt::ttnn::OpTy>;                          \
  TEST_P(OpTy##TPathParityTest, BuildEqualsFlatbufferRoundTrip) {              \
    runEltwiseUnaryParityCheck<mlir::tt::ttnn::OpTy>(GetParam());              \
  }                                                                            \
  const std::initializer_list<mlir::tt::ttnn::OpTy> OpTy##List = {             \
      buildTestEltwiseUnaryOp<mlir::tt::ttnn::OpTy>(),                         \
  };                                                                           \
  INSTANTIATE_TEST_SUITE_P(OpTy##TPathParityTest, OpTy##TPathParityTest,       \
                           ::testing::ValuesIn(OpTy##List))

ELTWISE_UNARY_PARITY_SUITE(ReluOp);
ELTWISE_UNARY_PARITY_SUITE(Relu6Op);
ELTWISE_UNARY_PARITY_SUITE(HardsigmoidOp);
ELTWISE_UNARY_PARITY_SUITE(SqrtOp);
ELTWISE_UNARY_PARITY_SUITE(SinOp);
ELTWISE_UNARY_PARITY_SUITE(AbsOp);
ELTWISE_UNARY_PARITY_SUITE(CosOp);
ELTWISE_UNARY_PARITY_SUITE(LogOp);
ELTWISE_UNARY_PARITY_SUITE(CeilOp);
ELTWISE_UNARY_PARITY_SUITE(SignOp);
ELTWISE_UNARY_PARITY_SUITE(FloorOp);
ELTWISE_UNARY_PARITY_SUITE(IsFiniteOp);
ELTWISE_UNARY_PARITY_SUITE(LogicalNotOp);
ELTWISE_UNARY_PARITY_SUITE(NegOp);
ELTWISE_UNARY_PARITY_SUITE(TanOp);
ELTWISE_UNARY_PARITY_SUITE(AtanOp);
ELTWISE_UNARY_PARITY_SUITE(AsinOp);
ELTWISE_UNARY_PARITY_SUITE(AsinhOp);
ELTWISE_UNARY_PARITY_SUITE(AcosOp);
ELTWISE_UNARY_PARITY_SUITE(ReciprocalOp);
ELTWISE_UNARY_PARITY_SUITE(BitwiseNotOp);
ELTWISE_UNARY_PARITY_SUITE(SiluOp);
ELTWISE_UNARY_PARITY_SUITE(MishOp);
ELTWISE_UNARY_PARITY_SUITE(Expm1Op);
ELTWISE_UNARY_PARITY_SUITE(RsqrtOp);
ELTWISE_UNARY_PARITY_SUITE(ErfOp);
ELTWISE_UNARY_PARITY_SUITE(ErfcOp);
ELTWISE_UNARY_PARITY_SUITE(ExpOp);
ELTWISE_UNARY_PARITY_SUITE(GeluOp);
ELTWISE_UNARY_PARITY_SUITE(TanhOp);
ELTWISE_UNARY_PARITY_SUITE(SigmoidOp);

#undef ELTWISE_UNARY_PARITY_SUITE

using LeakyReluOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::LeakyReluOp>;

TEST_P(LeakyReluOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseUnaryParityCheck<mlir::tt::ttnn::LeakyReluOp>(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::LeakyReluOp> leakyReluOpList = {
    buildTestEltwiseUnaryOp<mlir::tt::ttnn::LeakyReluOp>(),
    buildTestEltwiseUnaryOp<mlir::tt::ttnn::LeakyReluOp>(/*slope=*/0.2f),
};

INSTANTIATE_TEST_SUITE_P(LeakyReluOpTPathParityTest, LeakyReluOpTPathParityTest,
                         ::testing::ValuesIn(leakyReluOpList));

//===----------------------------------------------------------------------===//
// EltwiseUnaryCompositeOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
template <typename EltwiseUnaryCompositeOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseUnaryCompositeOp>
createEltwiseUnaryCompositeOp(::mlir::tt::FlatbufferObjectCache &cache,
                              EltwiseUnaryCompositeOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout);

::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(mlir::Attribute min,
                                                 mlir::Attribute max,
                                                 TTNNLayoutAttr outputLayout);

::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::EltwiseUnaryCompositeOpT &opTOpModel,
                       ::tt::target::ttnn::EltwiseUnaryCompositeOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::EltwiseUnaryCompositeOpT &opT) {
    opT.in.reset();
    opT.memory_config.reset();
    resetOutputTensorRefT(opT.out);
    if (opT.type ==
        ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor) {
      opT.params.Reset();
    }

    if (opT.type !=
            ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampScalar &&
        opT.type !=
            ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor) {
      opT.type = ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Cbrt;
    }
  };

  helper(opTOpModel);
  helper(opTFB);
}

template <typename OpTy>
OpTy buildTestEltwiseUnaryCompositeOp() {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto inputType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  return e.builder.create<OpTy>(loc, outputType, input);
}

mlir::tt::ttnn::ClampScalarOp buildTestClampScalarOp(mlir::Attribute min,
                                                     mlir::Attribute max) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto inputType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  return e.builder.create<mlir::tt::ttnn::ClampScalarOp>(loc, outputType, input,
                                                         min, max);
}

mlir::tt::ttnn::ClampTensorOp buildTestClampTensorOp() {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto inputType = tiledL1BF16Type(defaultShape);
  auto minType = tiledL1BF16Type(defaultShape);
  auto maxType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value min = e.builder
                        .create<mlir::tt::ttnn::OnesOp>(
                            loc, mlir::TypeRange{minType}, mlir::ValueRange{})
                        .getResult();
  mlir::Value max = e.builder
                        .create<mlir::tt::ttnn::OnesOp>(
                            loc, mlir::TypeRange{maxType}, mlir::ValueRange{})
                        .getResult();
  return e.builder.create<mlir::tt::ttnn::ClampTensorOp>(loc, outputType, input,
                                                         min, max);
}

template <typename OpTy>
void runEltwiseUnaryCompositeParityCheck(OpTy op) {
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT opTOpModel;
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ClampScalarOp>) {
    opTOpModel = mlir::tt::ttnn::op_model::
        buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(
            op.getMin(), op.getMax(), resolveOutputLayout(op));
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ClampTensorOp>) {
    opTOpModel = mlir::tt::ttnn::op_model::
        buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(
            resolveOutputLayout(op));
  } else {
    opTOpModel =
        mlir::tt::ttnn::op_model::buildEltwiseUnaryCompositeOpTFromMLIR<OpTy>(
            resolveOutputLayout(op));
  }

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getInput());
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ClampTensorOp>) {
    prepopulateOperandTensorRefs(cache, op.getMin(), op.getMax());
  }

  auto fbOffset = mlir::tt::ttnn::createEltwiseUnaryCompositeOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);
  EXPECT_EQ(opTOpModel, opTFB);
}

} // namespace

using CbrtOpTPathParityTest = ::testing::TestWithParam<mlir::tt::ttnn::CbrtOp>;

TEST_P(CbrtOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseUnaryCompositeParityCheck<mlir::tt::ttnn::CbrtOp>(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::CbrtOp> cbrtOpList = {
    buildTestEltwiseUnaryCompositeOp<mlir::tt::ttnn::CbrtOp>(),
};

INSTANTIATE_TEST_SUITE_P(CbrtOpTPathParityTest, CbrtOpTPathParityTest,
                         ::testing::ValuesIn(cbrtOpList));

using Log1pOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::Log1pOp>;

TEST_P(Log1pOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseUnaryCompositeParityCheck<mlir::tt::ttnn::Log1pOp>(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::Log1pOp> log1pOpList = {
    buildTestEltwiseUnaryCompositeOp<mlir::tt::ttnn::Log1pOp>(),
};

INSTANTIATE_TEST_SUITE_P(Log1pOpTPathParityTest, Log1pOpTPathParityTest,
                         ::testing::ValuesIn(log1pOpList));

using ClampScalarOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::ClampScalarOp>;

TEST_P(ClampScalarOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseUnaryCompositeParityCheck<mlir::tt::ttnn::ClampScalarOp>(
      GetParam());
}

const std::initializer_list<mlir::tt::ttnn::ClampScalarOp> clampScalarOpList = {
    buildTestClampScalarOp(
        /*min=*/mlir::Builder(getContext()).getF32FloatAttr(0.0f),
        /*max=*/mlir::Builder(getContext()).getF32FloatAttr(1.0f)),
    buildTestClampScalarOp(
        /*min=*/mlir::Builder(getContext()).getF32FloatAttr(-1.5f),
        /*max=*/mlir::Builder(getContext()).getF32FloatAttr(1.0f)),
    buildTestClampScalarOp(
        /*min=*/mlir::Builder(getContext()).getF32FloatAttr(0.0f),
        /*max=*/mlir::Builder(getContext()).getF32FloatAttr(5.5f)),
    buildTestClampScalarOp(
        /*min=*/mlir::Builder(getContext()).getI32IntegerAttr(-3),
        /*max=*/mlir::Builder(getContext()).getI32IntegerAttr(7)),
};

INSTANTIATE_TEST_SUITE_P(ClampScalarOpTPathParityTest,
                         ClampScalarOpTPathParityTest,
                         ::testing::ValuesIn(clampScalarOpList));

using ClampTensorOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::ClampTensorOp>;

TEST_P(ClampTensorOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseUnaryCompositeParityCheck<mlir::tt::ttnn::ClampTensorOp>(
      GetParam());
}

const std::initializer_list<mlir::tt::ttnn::ClampTensorOp> clampTensorOpList = {
    buildTestClampTensorOp(),
};

INSTANTIATE_TEST_SUITE_P(ClampTensorOpTPathParityTest,
                         ClampTensorOpTPathParityTest,
                         ::testing::ValuesIn(clampTensorOpList));

//===----------------------------------------------------------------------===//
// EltwiseBinaryOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
template <typename EltwiseBinaryOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryOp>
createEltwiseBinaryOp(::mlir::tt::FlatbufferObjectCache &cache,
                      EltwiseBinaryOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryOpT buildEltwiseBinaryOpTFromMLIR(
    TTNNLayoutAttr outputLayout,
    mlir::tt::ttcore::DataTypeAttr opDtypeAttr = nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::EltwiseBinaryOpT &opTOpModel,
                       ::tt::target::ttnn::EltwiseBinaryOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::EltwiseBinaryOpT &opT) {
    opT.lhs.reset();
    opT.rhs.reset();
    opT.memory_config.reset();
    opT.output_dtype.reset();
    resetOutputTensorRefT(opT.out);
    opT.type = ::tt::target::ttnn::EltwiseBinaryOpType::Add;
  };
  helper(opTOpModel);
  helper(opTFB);
}

template <typename OpTy>
OpTy buildTestEltwiseBinaryOp(mlir::tt::ttcore::DataTypeAttr dtype = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto lhsType = tiledL1BF16Type(defaultShape);
  auto rhsType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value lhs = e.builder
                        .create<mlir::tt::ttnn::OnesOp>(
                            loc, mlir::TypeRange{lhsType}, mlir::ValueRange{})
                        .getResult();
  mlir::Value rhs = e.builder
                        .create<mlir::tt::ttnn::OnesOp>(
                            loc, mlir::TypeRange{rhsType}, mlir::ValueRange{})
                        .getResult();
  return e.builder.create<OpTy>(loc, outputType, lhs, rhs, dtype);
}

template <typename OpTy>
void runEltwiseBinaryParityCheck(OpTy op) {
  ::tt::target::ttnn::EltwiseBinaryOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseBinaryOpTFromMLIR<OpTy>(
          resolveOutputLayout(op), op.getDtypeAttr());

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getLhs(), op.getRhs());
  auto fbOffset = mlir::tt::ttnn::createEltwiseBinaryOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseBinaryOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);
  EXPECT_EQ(opTOpModel, opTFB);
}

} // namespace

#define ELTWISE_BINARY_PARITY_SUITE(OpTy)                                      \
  using OpTy##TPathParityTest =                                                \
      ::testing::TestWithParam<mlir::tt::ttnn::OpTy>;                          \
  TEST_P(OpTy##TPathParityTest, BuildEqualsFlatbufferRoundTrip) {              \
    runEltwiseBinaryParityCheck<mlir::tt::ttnn::OpTy>(GetParam());             \
  }                                                                            \
  const std::initializer_list<mlir::tt::ttnn::OpTy> OpTy##List = {             \
      buildTestEltwiseBinaryOp<mlir::tt::ttnn::OpTy>(),                        \
      buildTestEltwiseBinaryOp<mlir::tt::ttnn::OpTy>(bf16DtypeAttr),           \
  };                                                                           \
  INSTANTIATE_TEST_SUITE_P(OpTy##TPathParityTest, OpTy##TPathParityTest,       \
                           ::testing::ValuesIn(OpTy##List))

ELTWISE_BINARY_PARITY_SUITE(AddOp);
ELTWISE_BINARY_PARITY_SUITE(MultiplyOp);
ELTWISE_BINARY_PARITY_SUITE(LogicalRightShiftOp);
ELTWISE_BINARY_PARITY_SUITE(SubtractOp);
ELTWISE_BINARY_PARITY_SUITE(DivideOp);
ELTWISE_BINARY_PARITY_SUITE(EqualOp);
ELTWISE_BINARY_PARITY_SUITE(NotEqualOp);
ELTWISE_BINARY_PARITY_SUITE(GreaterEqualOp);
ELTWISE_BINARY_PARITY_SUITE(GreaterThanOp);
ELTWISE_BINARY_PARITY_SUITE(LessEqualOp);
ELTWISE_BINARY_PARITY_SUITE(LessThanOp);
ELTWISE_BINARY_PARITY_SUITE(LogicalAndOp);
ELTWISE_BINARY_PARITY_SUITE(LogicalOrOp);
ELTWISE_BINARY_PARITY_SUITE(LogicalXorOp);

#undef ELTWISE_BINARY_PARITY_SUITE

//===----------------------------------------------------------------------===//
// EltwiseBinaryCompositeOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
template <typename EltwiseBinaryCompositeOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryCompositeOp>
createEltwiseBinaryCompositeOp(::mlir::tt::FlatbufferObjectCache &cache,
                               EltwiseBinaryCompositeOp op);

template <typename EltwiseBinaryCompositeScalarOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryCompositeScalarOp>
createEltwiseBinaryCompositeScalarOp(::mlir::tt::FlatbufferObjectCache &cache,
                                     EltwiseBinaryCompositeScalarOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryCompositeOpT
buildEltwiseBinaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout);

::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
buildEltwiseBinaryCompositeScalarOpTFromMLIR(mlir::Attribute exponent,
                                             TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::EltwiseBinaryCompositeOpT &opTOpModel,
    ::tt::target::ttnn::EltwiseBinaryCompositeOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::EltwiseBinaryCompositeOpT &opT) {
    opT.lhs.reset();
    opT.rhs.reset();
    opT.memory_config.reset();
    resetOutputTensorRefT(opT.out);
    opT.type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Maximum;
  };

  helper(opTOpModel);
  helper(opTFB);
}

void resetUnusedFields(
    ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT &opTOpModel,
    ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT &opT) {
    opT.lhs.reset();
    opT.memory_config.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

template <typename OpTy>
OpTy buildTestEltwiseBinaryCompositeOp() {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto lhsType = tiledL1BF16Type(defaultShape);
  auto rhsType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value lhs = e.builder
                        .create<mlir::tt::ttnn::OnesOp>(
                            loc, mlir::TypeRange{lhsType}, mlir::ValueRange{})
                        .getResult();
  mlir::Value rhs = e.builder
                        .create<mlir::tt::ttnn::OnesOp>(
                            loc, mlir::TypeRange{rhsType}, mlir::ValueRange{})
                        .getResult();
  return e.builder.create<OpTy>(loc, outputType, lhs, rhs);
}

mlir::tt::ttnn::PowScalarOp buildTestPowScalarOp(mlir::Attribute exponent) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto lhsType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value lhs = e.builder
                        .create<mlir::tt::ttnn::OnesOp>(
                            loc, mlir::TypeRange{lhsType}, mlir::ValueRange{})
                        .getResult();
  return e.builder.create<mlir::tt::ttnn::PowScalarOp>(loc, outputType, lhs,
                                                       exponent);
}

template <typename OpTy>
void runEltwiseBinaryCompositeParityCheck(OpTy op) {
  ::tt::target::ttnn::EltwiseBinaryCompositeOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseBinaryCompositeOpTFromMLIR<OpTy>(
          resolveOutputLayout(op));

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getLhs(), op.getRhs());
  auto fbOffset = mlir::tt::ttnn::createEltwiseBinaryCompositeOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseBinaryCompositeOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);
  EXPECT_EQ(opTOpModel, opTFB);
}

void runPowScalarParityCheck(mlir::tt::ttnn::PowScalarOp op) {
  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseBinaryCompositeScalarOpTFromMLIR(
          op.getRhs(), resolveOutputLayout(op));

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getLhs());
  auto fbOffset =
      mlir::tt::ttnn::createEltwiseBinaryCompositeScalarOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);
  EXPECT_EQ(opTOpModel, opTFB);
}

} // namespace

#define ELTWISE_BINARY_COMPOSITE_PARITY_SUITE(OpTy)                            \
  using OpTy##TPathParityTest =                                                \
      ::testing::TestWithParam<mlir::tt::ttnn::OpTy>;                          \
  TEST_P(OpTy##TPathParityTest, BuildEqualsFlatbufferRoundTrip) {              \
    runEltwiseBinaryCompositeParityCheck<mlir::tt::ttnn::OpTy>(GetParam());    \
  }                                                                            \
  const std::initializer_list<mlir::tt::ttnn::OpTy> OpTy##List = {             \
      buildTestEltwiseBinaryCompositeOp<mlir::tt::ttnn::OpTy>(),               \
  };                                                                           \
  INSTANTIATE_TEST_SUITE_P(OpTy##TPathParityTest, OpTy##TPathParityTest,       \
                           ::testing::ValuesIn(OpTy##List))

ELTWISE_BINARY_COMPOSITE_PARITY_SUITE(BitwiseAndOp);
ELTWISE_BINARY_COMPOSITE_PARITY_SUITE(BitwiseOrOp);
ELTWISE_BINARY_COMPOSITE_PARITY_SUITE(BitwiseXorOp);
ELTWISE_BINARY_COMPOSITE_PARITY_SUITE(LogicalLeftShiftOp);
ELTWISE_BINARY_COMPOSITE_PARITY_SUITE(Atan2Op);

#undef ELTWISE_BINARY_COMPOSITE_PARITY_SUITE

using PowScalarOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::PowScalarOp>;

TEST_P(PowScalarOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runPowScalarParityCheck(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::PowScalarOp> powScalarOpList = {
    buildTestPowScalarOp(env().builder.getF32FloatAttr(2.0f)),
    buildTestPowScalarOp(env().builder.getI32IntegerAttr(3)),
};

INSTANTIATE_TEST_SUITE_P(PowScalarOpTPathParityTest, PowScalarOpTPathParityTest,
                         ::testing::ValuesIn(powScalarOpList));

//===----------------------------------------------------------------------===//
// EltwiseTernaryOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::EltwiseTernaryWhereOp>
createEltwiseTernaryWhereOp(::mlir::tt::FlatbufferObjectCache &cache,
                            WhereOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseTernaryWhereOpT
buildEltwiseTernaryOpTFromMLIR(TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::EltwiseTernaryWhereOpT &opTOpModel,
                       ::tt::target::ttnn::EltwiseTernaryWhereOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::EltwiseTernaryWhereOpT &opT) {
    opT.first.reset();
    opT.second.reset();
    opT.third.reset();
    opT.memory_config.reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::WhereOp buildTestWhereOp() {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto firstType = tiledL1BF16Type(defaultShape);
  auto secondType = tiledL1BF16Type(defaultShape);
  auto thirdType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value first =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{firstType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value second =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{secondType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value third =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{thirdType},
                                          mlir::ValueRange{})
          .getResult();
  return e.builder.create<mlir::tt::ttnn::WhereOp>(loc, outputType, first,
                                                   second, third);
}

void runEltwiseTernaryParityCheck(mlir::tt::ttnn::WhereOp op) {
  ::tt::target::ttnn::EltwiseTernaryWhereOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseTernaryOpTFromMLIR<
          mlir::tt::ttnn::WhereOp>(resolveOutputLayout(op));

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getFirst(), op.getSecond(),
                               op.getThird());
  auto fbOffset = mlir::tt::ttnn::createEltwiseTernaryWhereOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseTernaryWhereOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);
  EXPECT_EQ(opTOpModel, opTFB);
}

} // namespace

using WhereOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::WhereOp>;

TEST_P(WhereOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseTernaryParityCheck(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::WhereOp> whereOpList = {
    buildTestWhereOp(),
};

INSTANTIATE_TEST_SUITE_P(WhereOpTPathParityTest, WhereOpTPathParityTest,
                         ::testing::ValuesIn(whereOpList));

//===----------------------------------------------------------------------===//
// EltwiseQuantizationOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
template <typename EltwiseQuantizationOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseQuantizationOp>
createEltwiseQuantizationOp(::mlir::tt::FlatbufferObjectCache &cache,
                            EltwiseQuantizationOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseQuantizationOpT buildEltwiseQuantizationOpTFromMLIR(
    std::optional<int32_t> axis,
    std::optional<mlir::tt::ttcore::DataType> outputDtype,
    TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::EltwiseQuantizationOpT &opTOpModel,
                       ::tt::target::ttnn::EltwiseQuantizationOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::EltwiseQuantizationOpT &opT) {
    opT.in.reset();
    opT.memory_config.reset();
    opT.output_dtype.reset();
    opT.params.Reset();
    resetOutputTensorRefT(opT.out);
  };

  helper(opTOpModel);
  helper(opTFB);
}

template <typename OpTy>
OpTy buildTestQuantDequantOp(mlir::IntegerAttr axis = {},
                             mlir::tt::ttcore::DataTypeAttr outputDtype = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto inputType = tiledL1BF16Type(defaultShape);
  auto scaleType = tiledL1BF16Type(defaultShape);
  auto zeroPointType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value scale =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{scaleType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value zeroPoint =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{zeroPointType},
                                          mlir::ValueRange{})
          .getResult();
  return e.builder.create<OpTy>(loc, outputType, input, scale, zeroPoint, axis,
                                outputDtype);
}

mlir::tt::ttnn::RequantizeOp
buildTestRequantizeOp(mlir::IntegerAttr axis = {},
                      mlir::tt::ttcore::DataTypeAttr outputDtype = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();
  auto inputType = tiledL1BF16Type(defaultShape);
  auto inScaleType = tiledL1BF16Type(defaultShape);
  auto inZeroPointType = tiledL1BF16Type(defaultShape);
  auto outScaleType = tiledL1BF16Type(defaultShape);
  auto outZeroPointType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);
  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value inScale =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inScaleType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value inZeroPoint =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inZeroPointType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value outScale =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{outScaleType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value outZeroPoint =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(
              loc, mlir::TypeRange{outZeroPointType}, mlir::ValueRange{})
          .getResult();
  return e.builder.create<mlir::tt::ttnn::RequantizeOp>(
      loc, outputType, input, inScale, inZeroPoint, outScale, outZeroPoint,
      axis, outputDtype);
}

template <typename OpTy>
void runEltwiseQuantizationParityCheck(OpTy op) {
  ::tt::target::ttnn::EltwiseQuantizationOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseQuantizationOpTFromMLIR<OpTy>(
          op.getAxis(), op.getOutputDtype(), resolveOutputLayout(op));

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::RequantizeOp>) {
    prepopulateOperandTensorRefs(cache, op.getInput(), op.getInScale(),
                                 op.getInZeroPoint(), op.getOutScale(),
                                 op.getOutZeroPoint());
  } else {
    prepopulateOperandTensorRefs(cache, op.getInput(), op.getScale(),
                                 op.getZeroPoint());
  }
  auto fbOffset = mlir::tt::ttnn::createEltwiseQuantizationOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseQuantizationOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);
  EXPECT_EQ(opTOpModel, opTFB);
}

} // namespace

using QuantizeOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::QuantizeOp>;

TEST_P(QuantizeOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseQuantizationParityCheck<mlir::tt::ttnn::QuantizeOp>(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::QuantizeOp> quantizeOpList = {
    buildTestQuantDequantOp<mlir::tt::ttnn::QuantizeOp>(),
    buildTestQuantDequantOp<mlir::tt::ttnn::QuantizeOp>(
        mlir::Builder(getContext()).getI32IntegerAttr(0)),
    buildTestQuantDequantOp<mlir::tt::ttnn::QuantizeOp>(
        mlir::Builder(getContext()).getI32IntegerAttr(1)),
    buildTestQuantDequantOp<mlir::tt::ttnn::QuantizeOp>(
        /*axis=*/{}, bf16DtypeAttr),
};

INSTANTIATE_TEST_SUITE_P(QuantizeOpTPathParityTest, QuantizeOpTPathParityTest,
                         ::testing::ValuesIn(quantizeOpList));

using DequantizeOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::DequantizeOp>;

TEST_P(DequantizeOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseQuantizationParityCheck<mlir::tt::ttnn::DequantizeOp>(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::DequantizeOp> dequantizeOpList = {
    buildTestQuantDequantOp<mlir::tt::ttnn::DequantizeOp>(),
    buildTestQuantDequantOp<mlir::tt::ttnn::DequantizeOp>(
        mlir::Builder(getContext()).getI32IntegerAttr(0)),
    buildTestQuantDequantOp<mlir::tt::ttnn::DequantizeOp>(
        mlir::Builder(getContext()).getI32IntegerAttr(1)),
    buildTestQuantDequantOp<mlir::tt::ttnn::DequantizeOp>(
        /*axis=*/{}, bf16DtypeAttr),
};

INSTANTIATE_TEST_SUITE_P(DequantizeOpTPathParityTest,
                         DequantizeOpTPathParityTest,
                         ::testing::ValuesIn(dequantizeOpList));

using RequantizeOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::RequantizeOp>;

TEST_P(RequantizeOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  runEltwiseQuantizationParityCheck<mlir::tt::ttnn::RequantizeOp>(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::RequantizeOp> requantizeOpList = {
    buildTestRequantizeOp(),
    buildTestRequantizeOp(mlir::Builder(getContext()).getI32IntegerAttr(0)),
    buildTestRequantizeOp(mlir::Builder(getContext()).getI32IntegerAttr(1)),
    buildTestRequantizeOp(/*axis=*/{}, bf16DtypeAttr),
};

INSTANTIATE_TEST_SUITE_P(RequantizeOpTPathParityTest,
                         RequantizeOpTPathParityTest,
                         ::testing::ValuesIn(requantizeOpList));

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOpTPathParity
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttnn {
::flatbuffers::Offset<::tt::target::ttnn::ConcatenateHeadsOp>
createOp(::mlir::tt::FlatbufferObjectCache &cache, ConcatenateHeadsOp op);
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttnn::op_model {
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ConcatenateHeadsOpT
buildConcatenateHeadsOpTFromMLIR(TTNNLayoutAttr outputLayout);
#endif // TTMLIR_ENABLE_OPMODEL
} // namespace mlir::tt::ttnn::op_model

namespace {

void resetUnusedFields(::tt::target::ttnn::ConcatenateHeadsOpT &opTOpModel,
                       ::tt::target::ttnn::ConcatenateHeadsOpT &opTFB) {
  auto helper = [](::tt::target::ttnn::ConcatenateHeadsOpT &opT) {
    opT.in.reset();
    resetOutputTensorRefT(opT.out);
    opT.memcfg.reset();
  };

  helper(opTOpModel);
  helper(opTFB);
}

mlir::tt::ttnn::ConcatenateHeadsOp buildTestConcatenateHeadsOp(
    mlir::tt::ttnn::MemoryConfigAttr outputMemoryConfig = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto outputType =
      outputMemoryConfig
          ? tiledBF16TypeFromMemoryConfig(defaultShape, outputMemoryConfig)
          : tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();

  return e.builder.create<mlir::tt::ttnn::ConcatenateHeadsOp>(loc, outputType,
                                                              input);
}

} // namespace

using ConcatenateHeadsOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::ConcatenateHeadsOp>;

TEST_P(ConcatenateHeadsOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::ConcatenateHeadsOp concatOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::ConcatenateHeadsOpT opTOpModel =
      mlir::tt::ttnn::op_model::buildConcatenateHeadsOpTFromMLIR(
          resolveOutputLayout(concatOp));

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, concatOp.getInput());

  auto fbOffset = mlir::tt::ttnn::createOp(cache, concatOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::ConcatenateHeadsOpT opTFB;
  r->UnPackTo(&opTFB);

  resetUnusedFields(opTOpModel, opTFB);

  EXPECT_EQ(opTOpModel, opTFB);
}

const std::initializer_list<mlir::tt::ttnn::ConcatenateHeadsOp>
    concatenateHeadsOpList = {
        buildTestConcatenateHeadsOp(),
        buildTestConcatenateHeadsOp(
            /*outputMemoryConfig=*/nonDefaultInputMemoryConfigAttr),
};

INSTANTIATE_TEST_SUITE_P(ConcatenateHeadsOpTPathParityTest,
                         ConcatenateHeadsOpTPathParityTest,
                         ::testing::ValuesIn(concatenateHeadsOpList));

#endif // TTMLIR_ENABLE_OPMODEL
