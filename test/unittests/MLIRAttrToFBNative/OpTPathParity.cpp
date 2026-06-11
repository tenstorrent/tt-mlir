// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpTPathParity.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttmlir/OpModel/TTNN/NativeFromMLIR.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

void compareOutputTensorRefT(
    const std::unique_ptr<::tt::target::ttnn::TensorRefT> &opNativeOpModelOut,
    const std::unique_ptr<::tt::target::ttnn::TensorRefT> &opNativeFBOut) {
  if (!opNativeOpModelOut && !opNativeFBOut) {
    return;
  }
  ASSERT_TRUE(opNativeOpModelOut && opNativeFBOut);
  EXPECT_EQ(ttnn_op_invoke::operations::utils::getTensorRefMemoryConfig(
                *opNativeOpModelOut),
            ttnn_op_invoke::operations::utils::getTensorRefMemoryConfig(
                *opNativeFBOut));
  EXPECT_EQ(ttnn_op_invoke::operations::utils::getDataType(*opNativeOpModelOut),
            ttnn_op_invoke::operations::utils::getDataType(*opNativeFBOut));
}

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

namespace {

void resetUnusedFields(::tt::target::ttnn::MatmulOpT &opNativeOpModel,
                       ::tt::target::ttnn::MatmulOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::MatmulOpT &op) {
    op.a.reset();
    op.b.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::MatmulOpT opNativeOpModel =
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
  ::tt::target::ttnn::MatmulOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(::tt::target::ttnn::Conv2dOpT &opNativeOpModel,
                       ::tt::target::ttnn::Conv2dOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::Conv2dOpT &op) {
    op.input.reset();
    op.weight.reset();
    op.bias.reset();
    op.device.reset();
    op.out.reset();
    op.output_dtype.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
      groups, conv2dConfig, computeKernelConfig, conv2dSliceConfig);
}

} // namespace

using Conv2dOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::Conv2dOp>;

TEST_P(Conv2dOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::Conv2dOp conv2dOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::Conv2dOpT opNativeOpModel =
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
  ::tt::target::ttnn::Conv2dOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

void resetUnusedFields(::tt::target::ttnn::Conv3dOpT &opNativeOpModel,
                       ::tt::target::ttnn::Conv3dOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::Conv3dOpT &op) {
    op.input.reset();
    op.weight.reset();
    op.bias.reset();
    op.device.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
      padding, paddingMode, groups, conv3dConfig, computeKernelConfig);
}

} // namespace

using Conv3dOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::Conv3dOp>;

TEST_P(Conv3dOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::Conv3dOp conv3dOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::Conv3dOpT opNativeOpModel =
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
  ::tt::target::ttnn::Conv3dOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(::tt::target::ttnn::ConvTranspose2dOpT &opNativeOpModel,
                       ::tt::target::ttnn::ConvTranspose2dOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::ConvTranspose2dOpT &op) {
    op.input.reset();
    op.weight.reset();
    op.bias.reset();
    op.device.reset();
    op.out.reset();
    op.output_dtype.reset();
    op.memory_config.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
      outputPadding, dilation, groups, conv2dConfig, computeKernelConfig,
      conv2dSliceConfig);
}

} // namespace

using ConvTranspose2dOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::ConvTranspose2dOp>;

TEST_P(ConvTranspose2dOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::ConvTranspose2dOp convTranspose2dOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::ConvTranspose2dOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildConvTranspose2dOpTFromMLIR(
          convTranspose2dOp.getInChannels(), convTranspose2dOp.getOutChannels(),
          convTranspose2dOp.getBatchSize(), convTranspose2dOp.getInputHeight(),
          convTranspose2dOp.getInputWidth(), convTranspose2dOp.getKernelSize(),
          convTranspose2dOp.getStride(), convTranspose2dOp.getPadding(),
          convTranspose2dOp.getOutputPadding(), convTranspose2dOp.getDilation(),
          convTranspose2dOp.getGroups(), convTranspose2dOp.getConv2dConfig(),
          convTranspose2dOp.getConv2dSliceConfig(),
          convTranspose2dOp.getComputeConfig(),
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
  ::tt::target::ttnn::ConvTranspose2dOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::PrepareConv2dBiasOpT &opNativeOpModel,
    ::tt::target::ttnn::PrepareConv2dBiasOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::PrepareConv2dBiasOpT &op) {
    op.bias_tensor.reset();
    op.device.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::PrepareConv2dBiasOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildPrepareConv2dBiasOpTFromMLIR(
          prepareOp.getInputMemoryConfig(), prepareOp.getInputTensorLayout(),
          prepareOp.getInChannels(), prepareOp.getOutChannels(),
          prepareOp.getBatchSize(), prepareOp.getInputHeight(),
          prepareOp.getInputWidth(), prepareOp.getKernelSize(),
          prepareOp.getStride(), prepareOp.getPadding(),
          prepareOp.getDilation(), prepareOp.getGroups(),
          prepareOp.getInputDtype(), prepareOp.getOutputDtype(),
          prepareOp.getConv2dConfig(), prepareOp.getConv2dSliceConfig(),
          prepareOp.getComputeConfig(), resolveOutputLayout(prepareOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, prepareOp.getBiasTensor());
  cache.getOrCreate(prepareOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, prepareOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PrepareConv2dBiasOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::PrepareConv2dWeightsOpT &opNativeOpModel,
    ::tt::target::ttnn::PrepareConv2dWeightsOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::PrepareConv2dWeightsOpT &op) {
    op.weight_tensor.reset();
    op.device.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::PrepareConv2dWeightsOpT opNativeOpModel =
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
  ::tt::target::ttnn::PrepareConv2dWeightsOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opNativeOpModel,
    ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::PrepareConvTranspose2dBiasOpT &op) {
    op.bias_tensor.reset();
    op.device.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT opNativeOpModel =
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
  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &opNativeOpModel,
    ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT &op) {
    op.weight_tensor.reset();
    op.device.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
    llvm::ArrayRef<int32_t> outputPadding = {1, 1},
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
      inputWidth, kernelSize, stride, padding, outputPadding, dilation, hasBias,
      groups, device, inputDtype, outputDtype, conv2dConfig,
      computeKernelConfig, conv2dSliceConfig, mirrorKernel);
}

} // namespace

using PrepareConvTranspose2dWeightsOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::PrepareConvTranspose2dWeightsOp>;

TEST_P(PrepareConvTranspose2dWeightsOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::PrepareConvTranspose2dWeightsOp prepareOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildPrepareConvTranspose2dWeightsOpTFromMLIR(
          prepareOp.getInputMemoryConfig(), prepareOp.getInputTensorLayout(),
          prepareOp.getWeightsFormat(), prepareOp.getInChannels(),
          prepareOp.getOutChannels(), prepareOp.getBatchSize(),
          prepareOp.getInputHeight(), prepareOp.getInputWidth(),
          prepareOp.getKernelSize(), prepareOp.getStride(),
          prepareOp.getPadding(), prepareOp.getOutputPadding(),
          prepareOp.getDilation(), prepareOp.getHasBias(),
          prepareOp.getGroups(), prepareOp.getInputDtype(),
          prepareOp.getOutputDtype(), prepareOp.getConv2dConfig(),
          prepareOp.getComputeConfig(), prepareOp.getConv2dSliceConfig(),
          prepareOp.getMirrorKernel(), resolveOutputLayout(prepareOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, prepareOp.getWeightTensor());
  cache.getOrCreate(prepareOp.getDevice(), mlir::tt::ttnn::createDeviceRef);

  auto fbOffset = mlir::tt::ttnn::createOp(cache, prepareOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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
            /*outputPadding=*/
            {1, 1}),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*outputPadding=*/
            {1, 1},
            /*dilation=*/{2, 2}),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2},
            /*padding=*/{1, 1}, /*outputPadding=*/{1, 1},
            /*dilation=*/{1, 1}, /*groups=*/4u),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*outputPadding=*/{1, 1},
            /*dilation=*/{1, 1}, /*groups=*/1u,
            /*inputMemoryConfig=*/nonDefaultInputMemoryConfigAttr),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*outputPadding=*/{1, 1},
            /*dilation=*/{1, 1}, /*groups=*/1u, /*inputMemoryConfig=*/{},
            /*inputTensorLayout=*/mlir::tt::ttnn::Layout::RowMajor),
        buildTestPrepareConvTranspose2dWeightsOp(
            /*outputDtype=*/{}, /*conv2dConfig=*/{},
            /*computeKernelConfig=*/{}, /*conv2dSliceConfig=*/{},
            /*weightsFormat=*/"OIHW", /*hasBias=*/false,
            /*mirrorKernel=*/true, /*inChannels=*/64u, /*outChannels=*/32u,
            /*batchSize=*/1u, /*inputHeight=*/28u, /*inputWidth=*/28u,
            /*kernelSize=*/{3, 3}, /*stride=*/{2, 2}, /*padding=*/{1, 1},
            /*outputPadding=*/{1, 1},
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
            /*stride=*/{1, 1}, /*padding=*/{0, 0}, /*outputPadding=*/{0, 0},
            /*dilation=*/{2, 2},
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

namespace {

void resetUnusedFields(::tt::target::ttnn::LinearOpT &opNativeOpModel,
                       ::tt::target::ttnn::LinearOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::LinearOpT &op) {
    op.a.reset();
    op.b.reset();
    op.bias.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::LinearOpT opNativeOpModel =
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
  ::tt::target::ttnn::LinearOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(::tt::target::ttnn::EltwiseUnaryOpT &opNativeOpModel,
                       ::tt::target::ttnn::EltwiseUnaryOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::EltwiseUnaryOpT &op) {
    op.in.reset();
    op.memory_config.reset();
    op.out.reset();
    if (op.type != ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu &&
        op.type != ::tt::target::ttnn::EltwiseUnaryOpType::Tanh &&
        op.type != ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid) {
      op.type = tt::target::ttnn::EltwiseUnaryOpType::Abs;
    }
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::EltwiseUnaryOpT opNativeOpModel;
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LeakyReluOp>) {
    opNativeOpModel =
        mlir::tt::ttnn::op_model::buildEltwiseUnaryOpTFromMLIR<OpTy>(
            resolveOutputLayout(op), op.getParameter());
  } else {
    opNativeOpModel =
        mlir::tt::ttnn::op_model::buildEltwiseUnaryOpTFromMLIR<OpTy>(
            resolveOutputLayout(op));
  }

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getInput());
  auto fbOffset = mlir::tt::ttnn::createEltwiseUnaryOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseUnaryOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);
  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::EltwiseUnaryCompositeOpT &opNativeOpModel,
    ::tt::target::ttnn::EltwiseUnaryCompositeOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::EltwiseUnaryCompositeOpT &op) {
    op.in.reset();
    op.memory_config.reset();
    op.out.reset();
    if (op.type ==
        ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor) {
      op.params.Reset();
    }

    if (op.type !=
            ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampScalar &&
        op.type !=
            ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor) {
      op.type = ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Cbrt;
    }
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT opNativeOpModel;
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ClampScalarOp>) {
    opNativeOpModel = mlir::tt::ttnn::op_model::
        buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(
            op.getMin(), op.getMax(), resolveOutputLayout(op));
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ClampTensorOp>) {
    opNativeOpModel = mlir::tt::ttnn::op_model::
        buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(
            resolveOutputLayout(op));
  } else {
    opNativeOpModel =
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
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);
  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(::tt::target::ttnn::EltwiseBinaryOpT &opNativeOpModel,
                       ::tt::target::ttnn::EltwiseBinaryOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::EltwiseBinaryOpT &op) {
    op.lhs.reset();
    op.rhs.reset();
    op.memory_config.reset();
    op.output_dtype.reset();
    op.out.reset();
    op.type = ::tt::target::ttnn::EltwiseBinaryOpType::Add;
  };
  helper(opNativeOpModel);
  helper(opNativeFB);
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
  return e.builder.create<OpTy>(loc, outputType, lhs, rhs);
}

template <typename OpTy>
void runEltwiseBinaryParityCheck(OpTy op) {
  ::tt::target::ttnn::EltwiseBinaryOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseBinaryOpTFromMLIR<OpTy>(
          resolveOutputLayout(op), op.getDtypeAttr());

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getLhs(), op.getRhs());
  auto fbOffset = mlir::tt::ttnn::createEltwiseBinaryOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseBinaryOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);
  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::EltwiseBinaryCompositeOpT &opNativeOpModel,
    ::tt::target::ttnn::EltwiseBinaryCompositeOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::EltwiseBinaryCompositeOpT &op) {
    op.lhs.reset();
    op.rhs.reset();
    op.memory_config.reset();
    op.out.reset();
    op.type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Maximum;
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

void resetUnusedFields(
    ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT &opNativeOpModel,
    ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT &op) {
    op.lhs.reset();
    op.memory_config.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::EltwiseBinaryCompositeOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseBinaryCompositeOpTFromMLIR<OpTy>(
          resolveOutputLayout(op));

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getLhs(), op.getRhs());
  auto fbOffset = mlir::tt::ttnn::createEltwiseBinaryCompositeOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseBinaryCompositeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);
  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

void runPowScalarParityCheck(mlir::tt::ttnn::PowScalarOp op) {
  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseBinaryCompositeScalarOpTFromMLIR(
          op.getRhs(), resolveOutputLayout(op));

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getLhs());
  auto fbOffset =
      mlir::tt::ttnn::createEltwiseBinaryCompositeScalarOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);
  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::EltwiseTernaryWhereOpT &opNativeOpModel,
    ::tt::target::ttnn::EltwiseTernaryWhereOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::EltwiseTernaryWhereOpT &op) {
    op.first.reset();
    op.second.reset();
    op.third.reset();
    op.memory_config.reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::EltwiseTernaryWhereOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildEltwiseTernaryOpTFromMLIR<
          mlir::tt::ttnn::WhereOp>(resolveOutputLayout(op));

  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, op.getFirst(), op.getSecond(),
                               op.getThird());
  auto fbOffset = mlir::tt::ttnn::createEltwiseTernaryWhereOp(cache, op);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::EltwiseTernaryWhereOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);
  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::EltwiseQuantizationOpT &opNativeOpModel,
    ::tt::target::ttnn::EltwiseQuantizationOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::EltwiseQuantizationOpT &op) {
    op.in.reset();
    op.memory_config.reset();
    op.output_dtype.reset();
    op.params.Reset();
    op.out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::EltwiseQuantizationOpT opNativeOpModel =
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
  ::tt::target::ttnn::EltwiseQuantizationOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);
  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

// ConcatenateHeadsOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(::tt::target::ttnn::ConcatenateHeadsOpT &opNativeOpModel,
                       ::tt::target::ttnn::ConcatenateHeadsOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::ConcatenateHeadsOpT &op) {
    op.in.reset();
    op.out.reset();
    op.memcfg.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
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
  ::tt::target::ttnn::ConcatenateHeadsOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildConcatenateHeadsOpTFromMLIR(
          resolveOutputLayout(concatOp));

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, concatOp.getInput());

  auto fbOffset = mlir::tt::ttnn::createOp(cache, concatOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::ConcatenateHeadsOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
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

//===----------------------------------------------------------------------===//
// NLPConcatHeadsOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(::tt::target::ttnn::NLPConcatHeadsOpT &opNativeOpModel,
                       ::tt::target::ttnn::NLPConcatHeadsOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::NLPConcatHeadsOpT &op) {
    op.in.reset();
    op.out.reset();
    op.memcfg.reset();
  };
  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::NLPConcatHeadsOp buildTestNLPConcatHeadsOp(
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

  return e.builder.create<mlir::tt::ttnn::NLPConcatHeadsOp>(loc, outputType,
                                                            input);
}

} // namespace

using NLPConcatHeadsOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::NLPConcatHeadsOp>;

TEST_P(NLPConcatHeadsOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::NLPConcatHeadsOp nlpConcatOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::NLPConcatHeadsOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildNLPConcatHeadsOpTFromMLIR(
          resolveOutputLayout(nlpConcatOp));

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, nlpConcatOp.getInput());

  auto fbOffset = mlir::tt::ttnn::createOp(cache, nlpConcatOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::NLPConcatHeadsOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<mlir::tt::ttnn::NLPConcatHeadsOp>
    nlpConcatHeadsOpList = {
        buildTestNLPConcatHeadsOp(),
        buildTestNLPConcatHeadsOp(
            /*outputMemoryConfig=*/nonDefaultInputMemoryConfigAttr),
};

INSTANTIATE_TEST_SUITE_P(NLPConcatHeadsOpTPathParityTest,
                         NLPConcatHeadsOpTPathParityTest,
                         ::testing::ValuesIn(nlpConcatHeadsOpList));

//===----------------------------------------------------------------------===//
// NLPCreateQKVHeadsDecodeOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &opNativeOpModel,
    ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT &op) {
    op.input.reset();
    op.batch_offset.reset();
    op.q_out.reset();
    op.k_out.reset();
    op.v_out.reset();
  };
  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp buildTestNLPCreateQKVHeadsDecodeOp(
    bool withBatchOffset = false, uint32_t numHeads = 8,
    std::optional<uint32_t> numKVHeads = std::nullopt,
    std::optional<bool> overlapQKCoregrid = std::nullopt,
    std::optional<uint32_t> sliceSize = std::nullopt) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value batchOffset = nullptr;
  if (withBatchOffset) {
    auto batchOffsetType = tiledL1BF16Type(defaultShape);
    batchOffset =
        e.builder
            .create<mlir::tt::ttnn::OnesOp>(
                loc, mlir::TypeRange{batchOffsetType}, mlir::ValueRange{})
            .getResult();
  }

  mlir::IntegerAttr numKVHeadsAttr;
  if (numKVHeads.has_value()) {
    numKVHeadsAttr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Unsigned),
        *numKVHeads);
  }
  mlir::BoolAttr overlapQKCoregridAttr;
  if (overlapQKCoregrid.has_value()) {
    overlapQKCoregridAttr =
        mlir::BoolAttr::get(getContext(), *overlapQKCoregrid);
  }
  mlir::IntegerAttr sliceSizeAttr;
  if (sliceSize.has_value()) {
    sliceSizeAttr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Unsigned),
        *sliceSize);
  }

  return e.builder.create<mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp>(
      loc, /*query=*/outputType, /*key=*/outputType, /*value=*/outputType,
      input, batchOffset, numHeads, numKVHeadsAttr, overlapQKCoregridAttr,
      sliceSizeAttr);
}

} // namespace

using NLPCreateQKVHeadsDecodeOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp>;

TEST_P(NLPCreateQKVHeadsDecodeOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp qkvOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildNLPCreateQKVHeadsDecodeOpTFromMLIR(
          qkvOp.getNumHeads(), qkvOp.getNumKvHeads(),
          qkvOp.getOverlapQkCoregrid(), qkvOp.getSliceSize(), nullptr);

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, qkvOp.getInput());
  if (qkvOp.getBatchOffset()) {
    prepopulateOperandTensorRefs(cache, qkvOp.getBatchOffset());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, qkvOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.q_out, opNativeFB.q_out);
}

const std::initializer_list<mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp>
    nlpCreateQKVHeadsDecodeOpList = {
        buildTestNLPCreateQKVHeadsDecodeOp(),
        buildTestNLPCreateQKVHeadsDecodeOp(/*withBatchOffset=*/true),
        buildTestNLPCreateQKVHeadsDecodeOp(/*withBatchOffset=*/false,
                                           /*numHeads=*/16u),
        buildTestNLPCreateQKVHeadsDecodeOp(
            /*withBatchOffset=*/false, /*numHeads=*/8u,
            /*numKVHeads=*/std::optional<uint32_t>(4u)),
        buildTestNLPCreateQKVHeadsDecodeOp(
            /*withBatchOffset=*/false, /*numHeads=*/8u,
            /*numKVHeads=*/std::nullopt,
            /*overlapQKCoregrid=*/std::optional<bool>(true)),
        buildTestNLPCreateQKVHeadsDecodeOp(
            /*withBatchOffset=*/false, /*numHeads=*/8u,
            /*numKVHeads=*/std::nullopt,
            /*overlapQKCoregrid=*/std::nullopt,
            /*sliceSize=*/std::optional<uint32_t>(32u)),
        buildTestNLPCreateQKVHeadsDecodeOp(
            /*withBatchOffset=*/true, /*numHeads=*/16u,
            /*numKVHeads=*/std::optional<uint32_t>(4u),
            /*overlapQKCoregrid=*/std::optional<bool>(false),
            /*sliceSize=*/std::optional<uint32_t>(64u)),
};

INSTANTIATE_TEST_SUITE_P(NLPCreateQKVHeadsDecodeOpTPathParityTest,
                         NLPCreateQKVHeadsDecodeOpTPathParityTest,
                         ::testing::ValuesIn(nlpCreateQKVHeadsDecodeOpList));

TEST_F(NLPCreateQKVHeadsDecodeOpTPathParityTest, NonDefaultMemoryConfig) {
  mlir::tt::ttnn::NLPCreateQKVHeadsDecodeOp qkvOp =
      buildTestNLPCreateQKVHeadsDecodeOp(
          /*withBatchOffset=*/true, /*numHeads=*/16u,
          /*numKVHeads=*/std::optional<uint32_t>(4u),
          /*overlapQKCoregrid=*/std::optional<bool>(false),
          /*sliceSize=*/std::optional<uint32_t>(64u));

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildNLPCreateQKVHeadsDecodeOpTFromMLIR(
          qkvOp.getNumHeads(), qkvOp.getNumKvHeads(),
          qkvOp.getOverlapQkCoregrid(), qkvOp.getSliceSize(),
          mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
              mlir::cast<mlir::RankedTensorType>(qkvOp.getQuery().getType())
                  .getEncoding()));

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, qkvOp.getInput());
  if (qkvOp.getBatchOffset()) {
    prepopulateOperandTensorRefs(cache, qkvOp.getBatchOffset());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, qkvOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_NE(opNativeOpModel, opNativeFB);
  EXPECT_NE(opNativeOpModel.memcfg, opNativeFB.memcfg);
  EXPECT_NE(opNativeOpModel.memcfg, nullptr);
  EXPECT_EQ(opNativeFB.memcfg, nullptr);
  opNativeOpModel.memcfg.reset();
  opNativeFB.memcfg.reset();
  EXPECT_EQ(opNativeOpModel, opNativeFB);
}

//===----------------------------------------------------------------------===//
// PagedFlashMultiLatentAttentionDecodeOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT
        &opNativeOpModel,
    ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &opNativeFB) {
  auto helper =
      [](::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT &op) {
        op.query.reset();
        op.key.reset();
        op.value.reset();
        op.page_table.reset();
        op.attention_mask.reset();
        op.cur_pos_tensor.reset();
        op.attention_sink.reset();
        op.out.reset();
        op.memcfg.reset();
      };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::PagedFlashMultiLatentAttentionDecodeOp
buildTestPagedFlashMultiLatentAttentionDecodeOp(
    bool withValue = false, uint32_t headDimV = 64, bool isCausal = true,
    bool withAttentionMask = false, bool withCurPosTensor = false,
    bool withAttentionSink = false, mlir::FloatAttr scale = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto tensorType = tiledL1BF16Type(defaultShape);

  auto makeOnes = [&]() {
    return e.builder
        .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{tensorType},
                                        mlir::ValueRange{})
        .getResult();
  };

  mlir::Value query = makeOnes();
  mlir::Value key = makeOnes();
  mlir::Value value = withValue ? makeOnes() : mlir::Value();
  mlir::Value pageTable = makeOnes();
  mlir::Value attentionMask = withAttentionMask ? makeOnes() : mlir::Value();
  mlir::Value curPosTensor = withCurPosTensor ? makeOnes() : mlir::Value();
  mlir::Value attentionSink = withAttentionSink ? makeOnes() : mlir::Value();

  return e.builder
      .create<mlir::tt::ttnn::PagedFlashMultiLatentAttentionDecodeOp>(
          loc, tensorType, query, key, value, headDimV, pageTable, isCausal,
          attentionMask, curPosTensor, attentionSink, scale);
}

} // namespace

using PagedFlashMultiLatentAttentionDecodeOpTPathParityTest =
    ::testing::TestWithParam<
        mlir::tt::ttnn::PagedFlashMultiLatentAttentionDecodeOp>;

TEST_P(PagedFlashMultiLatentAttentionDecodeOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::PagedFlashMultiLatentAttentionDecodeOp mlaOp = GetParam();

  std::optional<llvm::APFloat> scaleOpt;
  if (auto scaleAttr = mlaOp.getScaleAttr()) {
    scaleOpt = scaleAttr.getValue();
  }

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::
          buildPagedFlashMultiLatentAttentionDecodeOpTFromMLIR(
              mlaOp.getHeadDimV(), mlaOp.getIsCausal(), scaleOpt,
              resolveOutputLayout(mlaOp));

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, mlaOp.getQuery(), mlaOp.getKey(),
                               mlaOp.getPageTable());
  if (mlaOp.getValue()) {
    prepopulateOperandTensorRefs(cache, mlaOp.getValue());
  }
  if (mlaOp.getAttentionMask()) {
    prepopulateOperandTensorRefs(cache, mlaOp.getAttentionMask());
  }
  if (mlaOp.getCurPosTensor()) {
    prepopulateOperandTensorRefs(cache, mlaOp.getCurPosTensor());
  }
  if (mlaOp.getAttentionSink()) {
    prepopulateOperandTensorRefs(cache, mlaOp.getAttentionSink());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, mlaOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<
    mlir::tt::ttnn::PagedFlashMultiLatentAttentionDecodeOp>
    pagedFlashMlaDecodeOpList = {
        buildTestPagedFlashMultiLatentAttentionDecodeOp(),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(/*withValue=*/true),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(/*withValue=*/false,
                                                        /*headDimV=*/128u),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(/*withValue=*/false,
                                                        /*headDimV=*/64u,
                                                        /*isCausal=*/false),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(
            /*withValue=*/false, /*headDimV=*/64u, /*isCausal=*/true,
            /*withAttentionMask=*/true),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(
            /*withValue=*/false, /*headDimV=*/64u, /*isCausal=*/true,
            /*withAttentionMask=*/false, /*withCurPosTensor=*/true),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(
            /*withValue=*/false, /*headDimV=*/64u, /*isCausal=*/true,
            /*withAttentionMask=*/false, /*withCurPosTensor=*/false,
            /*withAttentionSink=*/true),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(
            /*withValue=*/false, /*headDimV=*/64u, /*isCausal=*/true,
            /*withAttentionMask=*/false, /*withCurPosTensor=*/false,
            /*withAttentionSink=*/false,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.125f)),
        buildTestPagedFlashMultiLatentAttentionDecodeOp(
            /*withValue=*/true, /*headDimV=*/128u, /*isCausal=*/false,
            /*withAttentionMask=*/true, /*withCurPosTensor=*/true,
            /*withAttentionSink=*/true,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.25f)),
};

INSTANTIATE_TEST_SUITE_P(PagedFlashMultiLatentAttentionDecodeOpTPathParityTest,
                         PagedFlashMultiLatentAttentionDecodeOpTPathParityTest,
                         ::testing::ValuesIn(pagedFlashMlaDecodeOpList));

//===----------------------------------------------------------------------===//
// PagedScaledDotProductAttentionDecodeOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

const mlir::tt::ttnn::SDPAProgramConfigAttr nonDefaultSDPAProgramConfigAttr =
    mlir::tt::ttnn::SDPAProgramConfigAttr::get(
        getContext(),
        /*computeWithStorageGridSize=*/
        mlir::tt::ttnn::CoreCoordAttr::get(getContext(), 8, 8),
        /*subCoreGrids=*/mlir::tt::ttnn::CoreRangeSetAttr(),
        /*qChunkSize=*/64, /*kChunkSize=*/64,
        /*expApproxMode=*/mlir::BoolAttr::get(getContext(), false),
        /*maxCoresPerHeadBatch=*/64);

void resetUnusedFields(
    ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT
        &opNativeOpModel,
    ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT &opNativeFB) {
  auto helper =
      [](::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT &op) {
        op.query.reset();
        op.key.reset();
        op.value.reset();
        op.page_table.reset();
        op.attention_mask.reset();
        op.cur_pos_tensor.reset();
        op.attention_sink.reset();
        op.out.reset();
        op.memcfg.reset();
      };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::PagedScaledDotProductAttentionDecodeOp
buildTestPagedScaledDotProductAttentionDecodeOp(
    bool isCausal = true, bool withAttentionMask = false,
    bool withCurPosTensor = false, bool withAttentionSink = false,
    mlir::FloatAttr scale = {}, mlir::IntegerAttr slidingWindowSize = {},
    mlir::tt::ttnn::SDPAProgramConfigAttr programConfig = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto tensorType = tiledL1BF16Type(defaultShape);

  auto makeOnes = [&]() {
    return e.builder
        .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{tensorType},
                                        mlir::ValueRange{})
        .getResult();
  };

  mlir::Value query = makeOnes();
  mlir::Value key = makeOnes();
  mlir::Value value = makeOnes();
  mlir::Value pageTable = makeOnes();
  mlir::Value attentionMask = withAttentionMask ? makeOnes() : mlir::Value();
  mlir::Value curPosTensor = withCurPosTensor ? makeOnes() : mlir::Value();
  mlir::Value attentionSink = withAttentionSink ? makeOnes() : mlir::Value();

  return e.builder
      .create<mlir::tt::ttnn::PagedScaledDotProductAttentionDecodeOp>(
          loc, tensorType, query, key, value, pageTable, isCausal,
          attentionMask, curPosTensor, attentionSink, scale, slidingWindowSize,
          programConfig);
}

} // namespace

using PagedScaledDotProductAttentionDecodeOpTPathParityTest =
    ::testing::TestWithParam<
        mlir::tt::ttnn::PagedScaledDotProductAttentionDecodeOp>;

TEST_P(PagedScaledDotProductAttentionDecodeOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::PagedScaledDotProductAttentionDecodeOp sdpaOp = GetParam();

  std::optional<llvm::APFloat> scaleOpt;
  if (auto scaleAttr = sdpaOp.getScaleAttr()) {
    scaleOpt = scaleAttr.getValue();
  }

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::
          buildPagedScaledDotProductAttentionDecodeOpTFromMLIR(
              sdpaOp.getIsCausal(), scaleOpt, sdpaOp.getSlidingWindowSize(),
              sdpaOp.getProgramConfig(), resolveOutputLayout(sdpaOp));

  // Path B: FB serialization round-trip.
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, sdpaOp.getQuery(), sdpaOp.getKey(),
                               sdpaOp.getValue(), sdpaOp.getPageTable());
  if (sdpaOp.getAttentionMask()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getAttentionMask());
  }
  if (sdpaOp.getCurPosTensor()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getCurPosTensor());
  }
  if (sdpaOp.getAttentionSink()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getAttentionSink());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, sdpaOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<
    mlir::tt::ttnn::PagedScaledDotProductAttentionDecodeOp>
    pagedScaledDotProductAttentionDecodeOpList = {
        buildTestPagedScaledDotProductAttentionDecodeOp(),
        buildTestPagedScaledDotProductAttentionDecodeOp(/*isCausal=*/false),
        buildTestPagedScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/true),
        buildTestPagedScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/true),
        buildTestPagedScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/false, /*withAttentionSink=*/true),
        buildTestPagedScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/false, /*withAttentionSink=*/false,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.125f)),
        buildTestPagedScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/false, /*withAttentionSink=*/false,
            /*scale=*/{},
            /*slidingWindowSize=*/
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                128u)),
        buildTestPagedScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/false, /*withAttentionSink=*/false,
            /*scale=*/{}, /*slidingWindowSize=*/{},
            /*programConfig=*/nonDefaultSDPAProgramConfigAttr),
        buildTestPagedScaledDotProductAttentionDecodeOp(
            /*isCausal=*/false, /*withAttentionMask=*/true,
            /*withCurPosTensor=*/true, /*withAttentionSink=*/true,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.25f),
            /*slidingWindowSize=*/
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                256u),
            /*programConfig=*/nonDefaultSDPAProgramConfigAttr),
};

INSTANTIATE_TEST_SUITE_P(
    PagedScaledDotProductAttentionDecodeOpTPathParityTest,
    PagedScaledDotProductAttentionDecodeOpTPathParityTest,
    ::testing::ValuesIn(pagedScaledDotProductAttentionDecodeOpList));

//===----------------------------------------------------------------------===//
// RotaryEmbeddingLlamaOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &opNativeOpModel,
    ::tt::target::ttnn::RotaryEmbeddingLlamaOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::RotaryEmbeddingLlamaOpT &op) {
    op.input.reset();
    op.cos_cache.reset();
    op.sin_cache.reset();
    op.trans_mat.reset();
    op.out.reset();
    op.memcfg.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::RotaryEmbeddingLlamaOp buildTestRotaryEmbeddingLlamaOp(
    bool isDecodeMode = false,
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfig = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto cosCacheType = tiledL1BF16Type(defaultShape);
  auto sinCacheType = tiledL1BF16Type(defaultShape);
  auto transMatType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value cosCache =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{cosCacheType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value sinCache =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{sinCacheType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value transMat =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{transMatType},
                                          mlir::ValueRange{})
          .getResult();

  return e.builder.create<mlir::tt::ttnn::RotaryEmbeddingLlamaOp>(
      loc, outputType, input, cosCache, sinCache, transMat, isDecodeMode,
      computeConfig);
}

} // namespace

using RotaryEmbeddingLlamaOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::RotaryEmbeddingLlamaOp>;

TEST_P(RotaryEmbeddingLlamaOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::RotaryEmbeddingLlamaOp rotaryOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::RotaryEmbeddingLlamaOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildRotaryEmbeddingLlamaOpTFromMLIR(
          rotaryOp.getIsDecodeMode(), rotaryOp.getComputeConfig(),
          resolveOutputLayout(rotaryOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, rotaryOp.getInput(),
                               rotaryOp.getCosCache(), rotaryOp.getSinCache(),
                               rotaryOp.getTransMat());

  auto fbOffset = mlir::tt::ttnn::createOp(cache, rotaryOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::RotaryEmbeddingLlamaOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<mlir::tt::ttnn::RotaryEmbeddingLlamaOp>
    rotaryEmbeddingLlamaOpList = {
        buildTestRotaryEmbeddingLlamaOp(),
        buildTestRotaryEmbeddingLlamaOp(/*isDecodeMode=*/true),
        buildTestRotaryEmbeddingLlamaOp(
            /*isDecodeMode=*/false,
            /*computeConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
        buildTestRotaryEmbeddingLlamaOp(
            /*isDecodeMode=*/true,
            /*computeConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
};

INSTANTIATE_TEST_SUITE_P(RotaryEmbeddingLlamaOpTPathParityTest,
                         RotaryEmbeddingLlamaOpTPathParityTest,
                         ::testing::ValuesIn(rotaryEmbeddingLlamaOpList));

//===----------------------------------------------------------------------===//
// RotaryEmbeddingOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(::tt::target::ttnn::RotaryEmbeddingOpT &opNativeOpModel,
                       ::tt::target::ttnn::RotaryEmbeddingOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::RotaryEmbeddingOpT &op) {
    op.input.reset();
    op.cos_cache.reset();
    op.sin_cache.reset();
    op.out.reset();
    op.memcfg.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::RotaryEmbeddingOp buildTestRotaryEmbeddingOp(
    mlir::IntegerAttr tokenIndex = {},
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfig = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto cosCacheType = tiledL1BF16Type(defaultShape);
  auto sinCacheType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value cosCache =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{cosCacheType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value sinCache =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{sinCacheType},
                                          mlir::ValueRange{})
          .getResult();

  return e.builder.create<mlir::tt::ttnn::RotaryEmbeddingOp>(
      loc, outputType, input, cosCache, sinCache, tokenIndex, computeConfig);
}

} // namespace

using RotaryEmbeddingOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::RotaryEmbeddingOp>;

TEST_P(RotaryEmbeddingOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::RotaryEmbeddingOp rotaryOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::RotaryEmbeddingOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildRotaryEmbeddingOpTFromMLIR(
          rotaryOp.getTokenIndex(), rotaryOp.getComputeConfig(),
          resolveOutputLayout(rotaryOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, rotaryOp.getInput(),
                               rotaryOp.getCosCache(), rotaryOp.getSinCache());

  auto fbOffset = mlir::tt::ttnn::createOp(cache, rotaryOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::RotaryEmbeddingOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<mlir::tt::ttnn::RotaryEmbeddingOp>
    rotaryEmbeddingOpList = {
        buildTestRotaryEmbeddingOp(),
        buildTestRotaryEmbeddingOp(
            /*tokenIndex=*/mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                42u)),
        buildTestRotaryEmbeddingOp(
            /*tokenIndex=*/{},
            /*computeConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
        buildTestRotaryEmbeddingOp(
            /*tokenIndex=*/
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                42u),
            /*computeConfig=*/nonDefaultDeviceComputeKernelConfigAttr),
};

INSTANTIATE_TEST_SUITE_P(RotaryEmbeddingOpTPathParityTest,
                         RotaryEmbeddingOpTPathParityTest,
                         ::testing::ValuesIn(rotaryEmbeddingOpList));

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &opNativeOpModel,
    ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT &op) {
    op.query.reset();
    op.key.reset();
    op.value.reset();
    op.attention_mask.reset();
    op.cur_pos_tensor.reset();
    op.attention_sink.reset();
    op.out.reset();
    op.memcfg.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp
buildTestScaledDotProductAttentionDecodeOp(
    bool isCausal = true, bool withAttentionMask = false,
    bool withCurPosTensor = false, bool withAttentionSink = false,
    mlir::FloatAttr scale = {},
    mlir::tt::ttnn::SDPAProgramConfigAttr programConfig = {}) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto tensorType = tiledL1BF16Type(defaultShape);

  auto makeOnes = [&]() {
    return e.builder
        .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{tensorType},
                                        mlir::ValueRange{})
        .getResult();
  };

  mlir::Value query = makeOnes();
  mlir::Value key = makeOnes();
  mlir::Value value = makeOnes();
  mlir::Value attentionMask = withAttentionMask ? makeOnes() : mlir::Value();
  mlir::Value curPosTensor = withCurPosTensor ? makeOnes() : mlir::Value();
  mlir::Value attentionSink = withAttentionSink ? makeOnes() : mlir::Value();

  return e.builder.create<mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp>(
      loc, tensorType, query, key, value, isCausal, attentionMask, curPosTensor,
      attentionSink, scale, programConfig);
}

} // namespace

using ScaledDotProductAttentionDecodeOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp>;

TEST_P(ScaledDotProductAttentionDecodeOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp sdpaOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildScaledDotProductAttentionDecodeOpTFromMLIR(
          sdpaOp.getIsCausal(), sdpaOp.getScale(),
          sdpaOp.getProgramConfigAttr(), resolveOutputLayout(sdpaOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, sdpaOp.getQuery(), sdpaOp.getKey(),
                               sdpaOp.getValue());
  if (sdpaOp.getAttentionMask()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getAttentionMask());
  }
  if (sdpaOp.getCurPosTensor()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getCurPosTensor());
  }
  if (sdpaOp.getAttentionSink()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getAttentionSink());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, sdpaOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<mlir::tt::ttnn::ScaledDotProductAttentionDecodeOp>
    scaledDotProductAttentionDecodeOpList = {
        buildTestScaledDotProductAttentionDecodeOp(),
        buildTestScaledDotProductAttentionDecodeOp(/*isCausal=*/false),
        buildTestScaledDotProductAttentionDecodeOp(/*isCausal=*/true,
                                                   /*withAttentionMask=*/true),
        buildTestScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/true),
        buildTestScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/false, /*withAttentionSink=*/true),
        buildTestScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/false, /*withAttentionSink=*/false,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.125f)),
        buildTestScaledDotProductAttentionDecodeOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*withCurPosTensor=*/false, /*withAttentionSink=*/false,
            /*scale=*/{},
            /*programConfig=*/nonDefaultSDPAProgramConfigAttr),
        buildTestScaledDotProductAttentionDecodeOp(
            /*isCausal=*/false, /*withAttentionMask=*/true,
            /*withCurPosTensor=*/true, /*withAttentionSink=*/true,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.125f),
            /*programConfig=*/nonDefaultSDPAProgramConfigAttr),
};

INSTANTIATE_TEST_SUITE_P(
    ScaledDotProductAttentionDecodeOpTPathParityTest,
    ScaledDotProductAttentionDecodeOpTPathParityTest,
    ::testing::ValuesIn(scaledDotProductAttentionDecodeOpList));

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::ScaledDotProductAttentionOpT &opNativeOpModel,
    ::tt::target::ttnn::ScaledDotProductAttentionOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::ScaledDotProductAttentionOpT &op) {
    op.query.reset();
    op.key.reset();
    op.value.reset();
    op.attention_mask.reset();
    op.attention_sink.reset();
    op.out.reset();
    op.memcfg.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::ScaledDotProductAttentionOp
buildTestScaledDotProductAttentionOp(bool isCausal = true,
                                     bool withAttentionMask = false,
                                     mlir::FloatAttr scale = {},
                                     mlir::IntegerAttr slidingWindowSize = {},
                                     bool withAttentionSink = false) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto tensorType = tiledL1BF16Type(defaultShape);

  auto makeOnes = [&]() {
    return e.builder
        .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{tensorType},
                                        mlir::ValueRange{})
        .getResult();
  };

  mlir::Value query = makeOnes();
  mlir::Value key = makeOnes();
  mlir::Value value = makeOnes();
  mlir::Value attentionMask = withAttentionMask ? makeOnes() : mlir::Value();
  mlir::Value attentionSink = withAttentionSink ? makeOnes() : mlir::Value();

  return e.builder.create<mlir::tt::ttnn::ScaledDotProductAttentionOp>(
      loc, tensorType, query, key, value, attentionMask, isCausal, scale,
      slidingWindowSize, attentionSink);
}

} // namespace

using ScaledDotProductAttentionOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::ScaledDotProductAttentionOp>;

TEST_P(ScaledDotProductAttentionOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::ScaledDotProductAttentionOp sdpaOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::ScaledDotProductAttentionOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildScaledDotProductAttentionOpTFromMLIR(
          sdpaOp.getIsCausal(), sdpaOp.getScale(),
          sdpaOp.getSlidingWindowSize(), resolveOutputLayout(sdpaOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, sdpaOp.getQuery(), sdpaOp.getKey(),
                               sdpaOp.getValue());
  if (sdpaOp.getAttentionMask()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getAttentionMask());
  }
  if (sdpaOp.getAttentionSink()) {
    prepopulateOperandTensorRefs(cache, sdpaOp.getAttentionSink());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, sdpaOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::ScaledDotProductAttentionOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<mlir::tt::ttnn::ScaledDotProductAttentionOp>
    scaledDotProductAttentionOpList = {
        buildTestScaledDotProductAttentionOp(),
        buildTestScaledDotProductAttentionOp(/*isCausal=*/false),
        buildTestScaledDotProductAttentionOp(/*isCausal=*/true,
                                             /*withAttentionMask=*/true),
        buildTestScaledDotProductAttentionOp(
            /*isCausal=*/true, /*withAttentionMask=*/false,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.125f)),
        buildTestScaledDotProductAttentionOp(
            /*isCausal=*/true, /*withAttentionMask=*/false, /*scale=*/{},
            /*slidingWindowSize=*/
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                128u)),
        buildTestScaledDotProductAttentionOp(
            /*isCausal=*/true, /*withAttentionMask=*/false, /*scale=*/{},
            /*slidingWindowSize=*/{},
            /*withAttentionSink=*/true),
        buildTestScaledDotProductAttentionOp(
            /*isCausal=*/false, /*withAttentionMask=*/true,
            /*scale=*/mlir::Builder(getContext()).getF32FloatAttr(0.125f),
            /*slidingWindowSize=*/
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                128u),
            /*withAttentionSink=*/true),
};

INSTANTIATE_TEST_SUITE_P(ScaledDotProductAttentionOpTPathParityTest,
                         ScaledDotProductAttentionOpTPathParityTest,
                         ::testing::ValuesIn(scaledDotProductAttentionOpList));

//===----------------------------------------------------------------------===//
// SplitQueryKeyValueAndSplitHeadsOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &opNativeOpModel,
    ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &op) {
    op.in.reset();
    op.kv_input.reset();
    op.memcfg.reset();
    op.q_out.reset();
    op.k_out.reset();
    op.v_out.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp
buildTestSplitQueryKeyValueAndSplitHeadsOp(bool withKvInputTensor = false,
                                           uint32_t numHeads = 4u,
                                           mlir::IntegerAttr numKvHeads = {},
                                           bool transposeKey = false) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto tensorType = tiledL1BF16Type(defaultShape);

  mlir::Value inputTensor =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{tensorType},
                                          mlir::ValueRange{})
          .getResult();
  mlir::Value kvInputTensor =
      withKvInputTensor
          ? e.builder
                .create<mlir::tt::ttnn::OnesOp>(
                    loc, mlir::TypeRange{tensorType}, mlir::ValueRange{})
                .getResult()
          : mlir::Value();

  return e.builder.create<mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp>(
      loc, mlir::TypeRange{tensorType, tensorType, tensorType}, inputTensor,
      kvInputTensor, numHeads, numKvHeads, transposeKey);
}

} // namespace

using SplitQueryKeyValueAndSplitHeadsOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp>;

TEST_P(SplitQueryKeyValueAndSplitHeadsOpTPathParityTest,
       BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp sqkvOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildSplitQueryKeyValueAndSplitHeadsOpTFromMLIR(
          sqkvOp.getNumHeads(), sqkvOp.getNumKvHeads(),
          sqkvOp.getTransposeKey(),
          mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
              mlir::cast<mlir::RankedTensorType>(sqkvOp.getQuery().getType())
                  .getEncoding()));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, sqkvOp.getInputTensor());
  if (sqkvOp.getKvInputTensor()) {
    prepopulateOperandTensorRefs(cache, sqkvOp.getKvInputTensor());
  }

  auto fbOffset = mlir::tt::ttnn::createOp(cache, sqkvOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.q_out, opNativeFB.q_out);
}

const std::initializer_list<mlir::tt::ttnn::SplitQueryKeyValueAndSplitHeadsOp>
    splitQueryKeyValueAndSplitHeadsOpList = {
        buildTestSplitQueryKeyValueAndSplitHeadsOp(),
        buildTestSplitQueryKeyValueAndSplitHeadsOp(/*withKvInputTensor=*/true),
        buildTestSplitQueryKeyValueAndSplitHeadsOp(/*withKvInputTensor=*/false,
                                                   /*numHeads=*/8u),
        buildTestSplitQueryKeyValueAndSplitHeadsOp(
            /*withKvInputTensor=*/false, /*numHeads=*/4u,
            /*numKvHeads=*/
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                2u)),
        buildTestSplitQueryKeyValueAndSplitHeadsOp(
            /*withKvInputTensor=*/false, /*numHeads=*/4u, /*numKvHeads=*/{},
            /*transposeKey=*/true),
        buildTestSplitQueryKeyValueAndSplitHeadsOp(
            /*withKvInputTensor=*/true, /*numHeads=*/8u,
            /*numKvHeads=*/
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(getContext(), 32,
                                       mlir::IntegerType::Unsigned),
                2u),
            /*transposeKey=*/true),
};

INSTANTIATE_TEST_SUITE_P(
    SplitQueryKeyValueAndSplitHeadsOpTPathParityTest,
    SplitQueryKeyValueAndSplitHeadsOpTPathParityTest,
    ::testing::ValuesIn(splitQueryKeyValueAndSplitHeadsOpList));

//===----------------------------------------------------------------------===//
// NLPConcatHeadsDecodeOpTPathParity
//===----------------------------------------------------------------------===//

namespace {

void resetUnusedFields(
    ::tt::target::ttnn::NLPConcatHeadsDecodeOpT &opNativeOpModel,
    ::tt::target::ttnn::NLPConcatHeadsDecodeOpT &opNativeFB) {
  auto helper = [](::tt::target::ttnn::NLPConcatHeadsDecodeOpT &op) {
    op.in.reset();
    op.out.reset();
    op.memcfg.reset();
  };

  helper(opNativeOpModel);
  helper(opNativeFB);
}

mlir::tt::ttnn::NLPConcatHeadsDecodeOp
buildTestNLPConcatHeadsDecodeOp(uint32_t numHeads = 4u) {
  auto &e = env();
  auto loc = e.builder.getUnknownLoc();

  auto inputType = tiledL1BF16Type(defaultShape);
  auto outputType = tiledL1BF16Type(defaultShape);

  mlir::Value input =
      e.builder
          .create<mlir::tt::ttnn::OnesOp>(loc, mlir::TypeRange{inputType},
                                          mlir::ValueRange{})
          .getResult();

  return e.builder.create<mlir::tt::ttnn::NLPConcatHeadsDecodeOp>(
      loc, outputType, input, numHeads);
}

} // namespace

using NLPConcatHeadsDecodeOpTPathParityTest =
    ::testing::TestWithParam<mlir::tt::ttnn::NLPConcatHeadsDecodeOp>;

TEST_P(NLPConcatHeadsDecodeOpTPathParityTest, BuildEqualsFlatbufferRoundTrip) {
  mlir::tt::ttnn::NLPConcatHeadsDecodeOp nlpOp = GetParam();

  // Path A: OpModel-style construction.
  ::tt::target::ttnn::NLPConcatHeadsDecodeOpT opNativeOpModel =
      mlir::tt::ttnn::op_model::buildNLPConcatHeadsDecodeOpTFromMLIR(
          nlpOp.getNumHeads(), resolveOutputLayout(nlpOp));

  // Path B: FB serialization round-trip (what runtime sees).
  ::flatbuffers::FlatBufferBuilder fbb;
  mlir::tt::FlatbufferObjectCache cache(&fbb);
  prepopulateOperandTensorRefs(cache, nlpOp.getInput());

  auto fbOffset = mlir::tt::ttnn::createOp(cache, nlpOp);
  fbb.Finish(fbOffset);
  auto *r = ::flatbuffers::GetTemporaryPointer(fbb, fbOffset);
  ::tt::target::ttnn::NLPConcatHeadsDecodeOpT opNativeFB;
  r->UnPackTo(&opNativeFB);

  resetUnusedFields(opNativeOpModel, opNativeFB);

  EXPECT_EQ(opNativeOpModel, opNativeFB);
  compareOutputTensorRefT(opNativeOpModel.out, opNativeFB.out);
}

const std::initializer_list<mlir::tt::ttnn::NLPConcatHeadsDecodeOp>
    nlpConcatHeadsDecodeOpList = {
        buildTestNLPConcatHeadsDecodeOp(),
        buildTestNLPConcatHeadsDecodeOp(/*numHeads=*/8u),
};

INSTANTIATE_TEST_SUITE_P(NLPConcatHeadsDecodeOpTPathParityTest,
                         NLPConcatHeadsDecodeOpTPathParityTest,
                         ::testing::ValuesIn(nlpConcatHeadsDecodeOpList));

#endif // TTMLIR_ENABLE_OPMODEL
