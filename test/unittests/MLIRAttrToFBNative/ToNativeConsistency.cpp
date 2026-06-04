// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "gtest/gtest.h"

template <typename Attr, typename NativeT>
class ToNativeConsistencyTest : public ::testing::Test,
                                public ::testing::WithParamInterface<Attr> {
protected:
  static mlir::MLIRContext context;
  static bool initialized;

public:
  static mlir::MLIRContext *getContext() {
    if (!initialized) {
      context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
      initialized = true;
    }
    return &context;
  }

  void SetUp() override { getContext(); }

protected:
  void RunTest(Attr attr) {
    NativeT native = mlir::tt::toNative(attr);

    flatbuffers::FlatBufferBuilder fbb;
    mlir::tt::FlatbufferObjectCache cache(&fbb);
    auto offset = mlir::tt::toFlatbuffer(cache, attr);
    auto *table = flatbuffers::GetTemporaryPointer(fbb, offset);
    NativeT unpacked;
    table->UnPackTo(&unpacked);

    EXPECT_EQ(native, unpacked);
  }
};

template <typename Attr, typename NativeT>
mlir::MLIRContext ToNativeConsistencyTest<Attr, NativeT>::context;

template <typename Attr, typename NativeT>
bool ToNativeConsistencyTest<Attr, NativeT>::initialized = false;

//===----------------------------------------------------------------------===//
// UnaryWithParam
//===----------------------------------------------------------------------===//

class UnaryWithParamTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::UnaryWithParamAttr,
                                     ::tt::target::ttnn::UnaryWithParamT> {};

TEST_P(UnaryWithParamTest, UnaryWithParam) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::UnaryWithParamAttr>
    unaryWithParamAttrList = {
        mlir::tt::ttnn::UnaryWithParamAttr::get(
            UnaryWithParamTest::getContext(), mlir::tt::ttnn::UnaryOpType::Relu,
            llvm::ArrayRef<mlir::FloatAttr>()),
        mlir::tt::ttnn::UnaryWithParamAttr::get(
            UnaryWithParamTest::getContext(),
            mlir::tt::ttnn::UnaryOpType::LeakyRelu,
            llvm::ArrayRef<mlir::FloatAttr>({mlir::FloatAttr::get(
                mlir::Float64Type::get(UnaryWithParamTest::getContext()),
                0.01)})),
        mlir::tt::ttnn::UnaryWithParamAttr::get(
            UnaryWithParamTest::getContext(),
            mlir::tt::ttnn::UnaryOpType::Power,
            llvm::ArrayRef<mlir::FloatAttr>(
                {mlir::FloatAttr::get(
                     mlir::Float64Type::get(UnaryWithParamTest::getContext()),
                     2.0),
                 mlir::FloatAttr::get(
                     mlir::Float64Type::get(UnaryWithParamTest::getContext()),
                     3.5)}))};

INSTANTIATE_TEST_SUITE_P(UnaryWithParamTest, UnaryWithParamTest,
                         ::testing::ValuesIn(unaryWithParamAttrList));

//===----------------------------------------------------------------------===//
// CoreRangeSet
//===----------------------------------------------------------------------===//

class CoreRangeSetTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::CoreRangeSetAttr,
                                     ::tt::target::ttnn::CoreRangeSetT> {};

TEST_P(CoreRangeSetTest, CoreRangeSet) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::CoreRangeSetAttr>
    coreRangeSetAttrList = {
        mlir::tt::ttnn::CoreRangeSetAttr::get(
            CoreRangeSetTest::getContext(),
            llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>()),
        mlir::tt::ttnn::CoreRangeSetAttr::get(
            CoreRangeSetTest::getContext(),
            llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                {mlir::tt::ttnn::CoreRangeAttr::get(
                    CoreRangeSetTest::getContext(),
                    mlir::tt::ttnn::CoreCoordAttr::get(
                        CoreRangeSetTest::getContext(), 0, 0),
                    mlir::tt::ttnn::CoreCoordAttr::get(
                        CoreRangeSetTest::getContext(), 7, 7))})),
        mlir::tt::ttnn::CoreRangeSetAttr::get(
            CoreRangeSetTest::getContext(),
            llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                {mlir::tt::ttnn::CoreRangeAttr::get(
                     CoreRangeSetTest::getContext(),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         CoreRangeSetTest::getContext(), 0, 0),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         CoreRangeSetTest::getContext(), 3, 3)),
                 mlir::tt::ttnn::CoreRangeAttr::get(
                     CoreRangeSetTest::getContext(),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         CoreRangeSetTest::getContext(), 4, 0),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         CoreRangeSetTest::getContext(), 7, 3))}))};

INSTANTIATE_TEST_SUITE_P(CoreRangeSetTest, CoreRangeSetTest,
                         ::testing::ValuesIn(coreRangeSetAttrList));

//===----------------------------------------------------------------------===//
// DeviceComputeKernelConfig
//===----------------------------------------------------------------------===//

class DeviceComputeKernelConfigTest
    : public ToNativeConsistencyTest<
          mlir::tt::ttnn::DeviceComputeKernelConfigAttr,
          ::tt::target::ttnn::DeviceComputeKernelConfigT> {};

TEST_P(DeviceComputeKernelConfigTest, DeviceComputeKernelConfig) {
  RunTest(GetParam());
}

const std::initializer_list<mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
    deviceComputeKernelConfigAttrList = {
        mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
            DeviceComputeKernelConfigTest::getContext()),
        mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
            DeviceComputeKernelConfigTest::getContext(),
            mlir::tt::ttnn::MathFidelity::HiFi2,
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                false),
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                true),
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                true),
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                false)),
        mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
            DeviceComputeKernelConfigTest::getContext(), std::nullopt,
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                true),
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                false),
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                false),
            mlir::BoolAttr::get(DeviceComputeKernelConfigTest::getContext(),
                                true))};

INSTANTIATE_TEST_SUITE_P(
    DeviceComputeKernelConfigTest, DeviceComputeKernelConfigTest,
    ::testing::ValuesIn(deviceComputeKernelConfigAttrList));

//===----------------------------------------------------------------------===//
// Conv2dConfig
//===----------------------------------------------------------------------===//

class Conv2dConfigTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::Conv2dConfigAttr,
                                     ::tt::target::ttnn::Conv2dConfigT> {};

TEST_P(Conv2dConfigTest, Conv2dConfig) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::Conv2dConfigAttr>
    conv2dConfigAttrList = {
        mlir::tt::ttnn::Conv2dConfigAttr::get(Conv2dConfigTest::getContext()),
        mlir::tt::ttnn::Conv2dConfigAttr::get(
            Conv2dConfigTest::getContext(),
            mlir::tt::ttcore::DataType::BFP_BFloat4,
            mlir::tt::ttnn::UnaryWithParamAttr::get(
                Conv2dConfigTest::getContext(),
                mlir::tt::ttnn::UnaryOpType::Relu,
                ::llvm::ArrayRef<mlir::FloatAttr>()),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false), 4, 7,
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                Conv2dConfigTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                    {mlir::tt::ttnn::CoreRangeAttr::get(
                        Conv2dConfigTest::getContext(),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            Conv2dConfigTest::getContext(), 0, 0),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            Conv2dConfigTest::getContext(), 7, 0))})),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::tt::ttnn::Layout::Tile,
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true)),
        mlir::tt::ttnn::Conv2dConfigAttr::get(
            Conv2dConfigTest::getContext(), std::nullopt,
            mlir::tt::ttnn::UnaryWithParamAttr::get(
                Conv2dConfigTest::getContext(),
                mlir::tt::ttnn::UnaryOpType::UnaryGt,
                llvm::ArrayRef<mlir::FloatAttr>()),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            std::nullopt, 7,
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                Conv2dConfigTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                    {mlir::tt::ttnn::CoreRangeAttr::get(
                         Conv2dConfigTest::getContext(),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 1, 2),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 3, 5)),
                     mlir::tt::ttnn::CoreRangeAttr::get(
                         Conv2dConfigTest::getContext(),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 35, 442),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 333, 555)),
                     mlir::tt::ttnn::CoreRangeAttr::get(
                         Conv2dConfigTest::getContext(),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 822, 772),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 865, 5456)),
                     mlir::tt::ttnn::CoreRangeAttr::get(
                         Conv2dConfigTest::getContext(),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 10000, 2100),
                         mlir::tt::ttnn::CoreCoordAttr::get(
                             Conv2dConfigTest::getContext(), 11111, 2111))})),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            std::nullopt,
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false)),
        mlir::tt::ttnn::Conv2dConfigAttr::get(
            Conv2dConfigTest::getContext(),
            mlir::tt::ttcore::DataType::BFP_BFloat4,
            mlir::tt::ttnn::UnaryWithParamAttr::get(
                Conv2dConfigTest::getContext(),
                mlir::tt::ttnn::UnaryOpType::Relu,
                ::llvm::ArrayRef<mlir::FloatAttr>()),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false), 4, 7,
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                Conv2dConfigTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>()),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::tt::ttnn::Layout::Tile,
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), false),
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true))};

INSTANTIATE_TEST_SUITE_P(Conv2dConfigTest, Conv2dConfigTest,
                         ::testing::ValuesIn(conv2dConfigAttrList));

//===----------------------------------------------------------------------===//
// Conv2dSliceConfig
//===----------------------------------------------------------------------===//

class Conv2dSliceConfigTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::Conv2dSliceConfigAttr,
                                     ::tt::target::ttnn::Conv2dSliceConfigT> {};

TEST_P(Conv2dSliceConfigTest, Conv2dSliceConfig) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::Conv2dSliceConfigAttr>
    conv2dSliceConfigAttrList = {
        mlir::tt::ttnn::Conv2dSliceConfigAttr::get(
            Conv2dSliceConfigTest::getContext(),
            mlir::tt::ttnn::Conv2dSliceType::DramHeight, 4),
        mlir::tt::ttnn::Conv2dSliceConfigAttr::get(
            Conv2dSliceConfigTest::getContext(),
            mlir::tt::ttnn::Conv2dSliceType::DramWidth, 8),
        mlir::tt::ttnn::Conv2dSliceConfigAttr::get(
            Conv2dSliceConfigTest::getContext(),
            mlir::tt::ttnn::Conv2dSliceType::L1Full, 1)};

INSTANTIATE_TEST_SUITE_P(Conv2dSliceConfigTest, Conv2dSliceConfigTest,
                         ::testing::ValuesIn(conv2dSliceConfigAttrList));

//===----------------------------------------------------------------------===//
// Conv3dConfig
//===----------------------------------------------------------------------===//

class Conv3dConfigTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::Conv3dConfigAttr,
                                     ::tt::target::ttnn::Conv3dConfigT> {};

TEST_P(Conv3dConfigTest, Conv3dConfig) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::Conv3dConfigAttr>
    conv3dConfigAttrList = {
        mlir::tt::ttnn::Conv3dConfigAttr::get(
            Conv3dConfigTest::getContext(), std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt),
        mlir::tt::ttnn::Conv3dConfigAttr::get(
            Conv3dConfigTest::getContext(),
            mlir::tt::ttcore::DataType::BFloat16, 2u, 3u, 4u, 8u, 16u,
            std::nullopt),
        mlir::tt::ttnn::Conv3dConfigAttr::get(
            Conv3dConfigTest::getContext(), mlir::tt::ttcore::DataType::Float32,
            std::nullopt, std::nullopt, std::nullopt, 4u, 8u, std::nullopt)};

INSTANTIATE_TEST_SUITE_P(Conv3dConfigTest, Conv3dConfigTest,
                         ::testing::ValuesIn(conv3dConfigAttrList));

//===----------------------------------------------------------------------===//
// ShardSpec
//===----------------------------------------------------------------------===//

class ShardSpecTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::ShardSpecAttr,
                                     ::tt::target::ttnn::ShardSpecT> {};

TEST_P(ShardSpecTest, ShardSpec) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::ShardSpecAttr> shardSpecAttrList = {
    mlir::tt::ttnn::ShardSpecAttr::get(
        ShardSpecTest::getContext(),
        mlir::tt::ttnn::CoreRangeSetAttr::get(
            ShardSpecTest::getContext(),
            llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                {mlir::tt::ttnn::CoreRangeAttr::get(
                    ShardSpecTest::getContext(),
                    mlir::tt::ttnn::CoreCoordAttr::get(
                        ShardSpecTest::getContext(), 0, 0),
                    mlir::tt::ttnn::CoreCoordAttr::get(
                        ShardSpecTest::getContext(), 7, 0))})),
        mlir::tt::ttnn::ShapeAttr::get(ShardSpecTest::getContext(),
                                       {64LL, 128LL}),
        mlir::tt::ttnn::ShardOrientationAttr::get(
            ShardSpecTest::getContext(),
            mlir::tt::ttnn::ShardOrientation::RowMajor)),
    mlir::tt::ttnn::ShardSpecAttr::get(
        ShardSpecTest::getContext(),
        mlir::tt::ttnn::CoreRangeSetAttr::get(
            ShardSpecTest::getContext(),
            llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                {mlir::tt::ttnn::CoreRangeAttr::get(
                     ShardSpecTest::getContext(),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         ShardSpecTest::getContext(), 0, 0),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         ShardSpecTest::getContext(), 3, 3)),
                 mlir::tt::ttnn::CoreRangeAttr::get(
                     ShardSpecTest::getContext(),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         ShardSpecTest::getContext(), 4, 0),
                     mlir::tt::ttnn::CoreCoordAttr::get(
                         ShardSpecTest::getContext(), 7, 3))})),
        mlir::tt::ttnn::ShapeAttr::get(ShardSpecTest::getContext(),
                                       {32LL, 32LL}),
        mlir::tt::ttnn::ShardOrientationAttr::get(
            ShardSpecTest::getContext(),
            mlir::tt::ttnn::ShardOrientation::ColMajor))};

INSTANTIATE_TEST_SUITE_P(ShardSpecTest, ShardSpecTest,
                         ::testing::ValuesIn(shardSpecAttrList));

//===----------------------------------------------------------------------===//
// NDShardSpec
//===----------------------------------------------------------------------===//

class NDShardSpecTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::NDShardSpecAttr,
                                     ::tt::target::ttnn::NDShardSpecT> {};

TEST_P(NDShardSpecTest, NDShardSpec) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::NDShardSpecAttr>
    ndShardSpecAttrList = {
        mlir::tt::ttnn::NDShardSpecAttr::get(
            NDShardSpecTest::getContext(),
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                NDShardSpecTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                    {mlir::tt::ttnn::CoreRangeAttr::get(
                        NDShardSpecTest::getContext(),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            NDShardSpecTest::getContext(), 0, 0),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            NDShardSpecTest::getContext(), 7, 0))})),
            mlir::tt::ttnn::ShapeAttr::get(NDShardSpecTest::getContext(),
                                           {64LL, 128LL}),
            mlir::tt::ttnn::ShardOrientationAttr::get(
                NDShardSpecTest::getContext(),
                mlir::tt::ttnn::ShardOrientation::RowMajor),
            mlir::tt::ttnn::ShardDistributionStrategyAttr::get(
                NDShardSpecTest::getContext(),
                mlir::tt::ttnn::ShardDistributionStrategy::RoundRobin1D)),
        mlir::tt::ttnn::NDShardSpecAttr::get(
            NDShardSpecTest::getContext(),
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                NDShardSpecTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                    {mlir::tt::ttnn::CoreRangeAttr::get(
                        NDShardSpecTest::getContext(),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            NDShardSpecTest::getContext(), 0, 0),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            NDShardSpecTest::getContext(), 3, 3))})),
            mlir::tt::ttnn::ShapeAttr::get(NDShardSpecTest::getContext(),
                                           {16LL, 32LL, 8LL}),
            mlir::tt::ttnn::ShardOrientationAttr::get(
                NDShardSpecTest::getContext(),
                mlir::tt::ttnn::ShardOrientation::ColMajor),
            mlir::tt::ttnn::ShardDistributionStrategyAttr::get(
                NDShardSpecTest::getContext(),
                mlir::tt::ttnn::ShardDistributionStrategy::Grid2D))};

INSTANTIATE_TEST_SUITE_P(NDShardSpecTest, NDShardSpecTest,
                         ::testing::ValuesIn(ndShardSpecAttrList));

//===----------------------------------------------------------------------===//
// MemoryConfig
//===----------------------------------------------------------------------===//

class MemoryConfigTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::MemoryConfigAttr,
                                     ::tt::target::ttnn::MemoryConfigT> {};

TEST_P(MemoryConfigTest, MemoryConfig) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::MemoryConfigAttr>
    memoryConfigAttrList = {
        mlir::tt::ttnn::MemoryConfigAttr::get(
            MemoryConfigTest::getContext(),
            mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                MemoryConfigTest::getContext(),
                mlir::tt::ttnn::TensorMemoryLayout::Interleaved),
            mlir::tt::ttnn::BufferTypeAttr::get(
                MemoryConfigTest::getContext(),
                mlir::tt::ttnn::BufferType::DRAM),
            std::nullopt, std::nullopt),
        mlir::tt::ttnn::MemoryConfigAttr::get(
            MemoryConfigTest::getContext(),
            mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                MemoryConfigTest::getContext(),
                mlir::tt::ttnn::TensorMemoryLayout::Interleaved),
            mlir::tt::ttnn::BufferTypeAttr::get(MemoryConfigTest::getContext(),
                                                mlir::tt::ttnn::BufferType::L1),
            std::nullopt, std::nullopt),
        mlir::tt::ttnn::MemoryConfigAttr::get(
            MemoryConfigTest::getContext(),
            mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                MemoryConfigTest::getContext(),
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded),
            mlir::tt::ttnn::BufferTypeAttr::get(MemoryConfigTest::getContext(),
                                                mlir::tt::ttnn::BufferType::L1),
            mlir::tt::ttnn::ShardSpecAttr::get(
                MemoryConfigTest::getContext(),
                mlir::tt::ttnn::CoreRangeSetAttr::get(
                    MemoryConfigTest::getContext(),
                    llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                        {mlir::tt::ttnn::CoreRangeAttr::get(
                            MemoryConfigTest::getContext(),
                            mlir::tt::ttnn::CoreCoordAttr::get(
                                MemoryConfigTest::getContext(), 0, 0),
                            mlir::tt::ttnn::CoreCoordAttr::get(
                                MemoryConfigTest::getContext(), 7, 0))})),
                mlir::tt::ttnn::ShapeAttr::get(MemoryConfigTest::getContext(),
                                               {128LL, 64LL}),
                mlir::tt::ttnn::ShardOrientationAttr::get(
                    MemoryConfigTest::getContext(),
                    mlir::tt::ttnn::ShardOrientation::RowMajor)),
            std::nullopt)};

INSTANTIATE_TEST_SUITE_P(MemoryConfigTest, MemoryConfigTest,
                         ::testing::ValuesIn(memoryConfigAttrList));

//===----------------------------------------------------------------------===//
// MatmulMultiCoreReuseProgramConfig
//===----------------------------------------------------------------------===//

class MatmulMultiCoreReuseProgramConfigTest
    : public ToNativeConsistencyTest<
          mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr,
          ::tt::target::ttnn::MatmulMultiCoreReuseProgramConfigT> {};

TEST_P(MatmulMultiCoreReuseProgramConfigTest,
       MatmulMultiCoreReuseProgramConfig) {
  RunTest(GetParam());
}

const std::initializer_list<
    mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr>
    matmulMultiCoreReuseAttrList = {
        mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::get(
            MatmulMultiCoreReuseProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                MatmulMultiCoreReuseProgramConfigTest::getContext(), 8, 8),
            2, 4, 4, 8, 8),
        mlir::tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::get(
            MatmulMultiCoreReuseProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                MatmulMultiCoreReuseProgramConfigTest::getContext(), 4, 4),
            1, 2, 2, 4, 4)};

INSTANTIATE_TEST_SUITE_P(MatmulMultiCoreReuseProgramConfigTest,
                         MatmulMultiCoreReuseProgramConfigTest,
                         ::testing::ValuesIn(matmulMultiCoreReuseAttrList));

//===----------------------------------------------------------------------===//
// MatmulMultiCoreReuseMultiCastProgramConfig
//===----------------------------------------------------------------------===//

class MatmulMultiCoreReuseMultiCastProgramConfigTest
    : public ToNativeConsistencyTest<
          mlir::tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr,
          ::tt::target::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigT> {};

TEST_P(MatmulMultiCoreReuseMultiCastProgramConfigTest,
       MatmulMultiCoreReuseMultiCastProgramConfig) {
  RunTest(GetParam());
}

const std::initializer_list<
    mlir::tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr>
    matmulMultiCoreReuseMultiCastAttrList = {
        mlir::tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
            MatmulMultiCoreReuseMultiCastProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                MatmulMultiCoreReuseMultiCastProgramConfigTest::getContext(), 8,
                8),
            2, 4, 4, 8, 8, 8, 8, false, mlir::tt::ttnn::UnaryWithParamAttr(),
            false),
        mlir::tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
            MatmulMultiCoreReuseMultiCastProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                MatmulMultiCoreReuseMultiCastProgramConfigTest::getContext(), 4,
                4),
            1, 2, 2, 4, 4, 4, 4, true,
            mlir::tt::ttnn::UnaryWithParamAttr::get(
                MatmulMultiCoreReuseMultiCastProgramConfigTest::getContext(),
                mlir::tt::ttnn::UnaryOpType::Relu,
                llvm::ArrayRef<mlir::FloatAttr>()),
            true)};

INSTANTIATE_TEST_SUITE_P(
    MatmulMultiCoreReuseMultiCastProgramConfigTest,
    MatmulMultiCoreReuseMultiCastProgramConfigTest,
    ::testing::ValuesIn(matmulMultiCoreReuseMultiCastAttrList));

//===----------------------------------------------------------------------===//
// MatmulMultiCoreReuseMultiCast1DProgramConfig
//===----------------------------------------------------------------------===//

class MatmulMultiCoreReuseMultiCast1DProgramConfigTest
    : public ToNativeConsistencyTest<
          mlir::tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
          ::tt::target::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigT> {};

TEST_P(MatmulMultiCoreReuseMultiCast1DProgramConfigTest,
       MatmulMultiCoreReuseMultiCast1DProgramConfig) {
  RunTest(GetParam());
}

const std::initializer_list<
    mlir::tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>
    matmulMultiCoreReuseMultiCast1DAttrList = {
        mlir::tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
            MatmulMultiCoreReuseMultiCast1DProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                MatmulMultiCoreReuseMultiCast1DProgramConfigTest::getContext(),
                8, 8),
            2, 4, 4, 8, 8, 8, 8, false, mlir::tt::ttnn::UnaryWithParamAttr(),
            true, false,
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                MatmulMultiCoreReuseMultiCast1DProgramConfigTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>()),
            0, false),
        mlir::tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
            MatmulMultiCoreReuseMultiCast1DProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                MatmulMultiCoreReuseMultiCast1DProgramConfigTest::getContext(),
                4, 4),
            1, 2, 2, 4, 4, 4, 4, true,
            mlir::tt::ttnn::UnaryWithParamAttr::get(
                MatmulMultiCoreReuseMultiCast1DProgramConfigTest::getContext(),
                mlir::tt::ttnn::UnaryOpType::Relu,
                llvm::ArrayRef<mlir::FloatAttr>()),
            false, true,
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                MatmulMultiCoreReuseMultiCast1DProgramConfigTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                    {mlir::tt::ttnn::CoreRangeAttr::get(
                        MatmulMultiCoreReuseMultiCast1DProgramConfigTest::
                            getContext(),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            MatmulMultiCoreReuseMultiCast1DProgramConfigTest::
                                getContext(),
                            0, 0),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            MatmulMultiCoreReuseMultiCast1DProgramConfigTest::
                                getContext(),
                            3, 0))})),
            2, true)};

INSTANTIATE_TEST_SUITE_P(
    MatmulMultiCoreReuseMultiCast1DProgramConfigTest,
    MatmulMultiCoreReuseMultiCast1DProgramConfigTest,
    ::testing::ValuesIn(matmulMultiCoreReuseMultiCast1DAttrList));

//===----------------------------------------------------------------------===//
// MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
//===----------------------------------------------------------------------===//

class MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigTest
    : public ToNativeConsistencyTest<
          mlir::tt::ttnn::
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr,
          ::tt::target::ttnn::
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigT> {};

TEST_P(MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigTest,
       MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig) {
  RunTest(GetParam());
}

const std::initializer_list<
    mlir::tt::ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>
    matmulDRAMShardedAttrList = {
        mlir::tt::ttnn::
            MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
                MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigTest::
                    getContext(),
                4, 8, 8, mlir::tt::ttnn::UnaryWithParamAttr()),
        mlir::tt::ttnn::
            MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
                MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigTest::
                    getContext(),
                2, 4, 4,
                mlir::tt::ttnn::UnaryWithParamAttr::get(
                    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigTest::
                        getContext(),
                    mlir::tt::ttnn::UnaryOpType::Relu,
                    llvm::ArrayRef<mlir::FloatAttr>()))};

INSTANTIATE_TEST_SUITE_P(
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigTest,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigTest,
    ::testing::ValuesIn(matmulDRAMShardedAttrList));

//===----------------------------------------------------------------------===//
// SDPAProgramConfig
//===----------------------------------------------------------------------===//

class SDPAProgramConfigTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::SDPAProgramConfigAttr,
                                     ::tt::target::ttnn::SDPAConfigT> {};

TEST_P(SDPAProgramConfigTest, SDPAProgramConfig) { RunTest(GetParam()); }

const std::initializer_list<mlir::tt::ttnn::SDPAProgramConfigAttr>
    sdpaProgramConfigAttrList = {
        mlir::tt::ttnn::SDPAProgramConfigAttr::get(
            SDPAProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                SDPAProgramConfigTest::getContext(), 8, 8),
            mlir::tt::ttnn::CoreRangeSetAttr(), 128, 128,
            mlir::BoolAttr::get(SDPAProgramConfigTest::getContext(), true),
            std::nullopt),
        mlir::tt::ttnn::SDPAProgramConfigAttr::get(
            SDPAProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                SDPAProgramConfigTest::getContext(), 4, 8),
            mlir::tt::ttnn::CoreRangeSetAttr::get(
                SDPAProgramConfigTest::getContext(),
                llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>(
                    {mlir::tt::ttnn::CoreRangeAttr::get(
                        SDPAProgramConfigTest::getContext(),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            SDPAProgramConfigTest::getContext(), 0, 0),
                        mlir::tt::ttnn::CoreCoordAttr::get(
                            SDPAProgramConfigTest::getContext(), 3, 7))})),
            64, 64,
            mlir::BoolAttr::get(SDPAProgramConfigTest::getContext(), false),
            16u),
        mlir::tt::ttnn::SDPAProgramConfigAttr::get(
            SDPAProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                SDPAProgramConfigTest::getContext(), 8, 8),
            mlir::tt::ttnn::CoreRangeSetAttr(), 32, 64, mlir::BoolAttr(),
            std::nullopt)};

INSTANTIATE_TEST_SUITE_P(SDPAProgramConfigTest, SDPAProgramConfigTest,
                         ::testing::ValuesIn(sdpaProgramConfigAttrList));

//===----------------------------------------------------------------------===//
// LayerNormShardedMultiCoreProgramConfig
//===----------------------------------------------------------------------===//

class LayerNormShardedMultiCoreProgramConfigTest
    : public ToNativeConsistencyTest<
          mlir::tt::ttnn::LayerNormShardedMultiCoreProgramConfigAttr,
          ::tt::target::ttnn::LayerNormShardedMultiCoreProgramConfigT> {};

TEST_P(LayerNormShardedMultiCoreProgramConfigTest,
       LayerNormShardedMultiCoreProgramConfig) {
  RunTest(GetParam());
}

const std::initializer_list<
    mlir::tt::ttnn::LayerNormShardedMultiCoreProgramConfigAttr>
    layerNormShardedMultiCoreProgramConfigAttrList = {
        mlir::tt::ttnn::LayerNormShardedMultiCoreProgramConfigAttr::get(
            LayerNormShardedMultiCoreProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                LayerNormShardedMultiCoreProgramConfigTest::getContext(), 8, 8),
            1, 4, 4, false),
        mlir::tt::ttnn::LayerNormShardedMultiCoreProgramConfigAttr::get(
            LayerNormShardedMultiCoreProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                LayerNormShardedMultiCoreProgramConfigTest::getContext(), 4, 4),
            1, 2, 8, true),
        mlir::tt::ttnn::LayerNormShardedMultiCoreProgramConfigAttr::get(
            LayerNormShardedMultiCoreProgramConfigTest::getContext(),
            mlir::tt::ttnn::CoreCoordAttr::get(
                LayerNormShardedMultiCoreProgramConfigTest::getContext(), 1, 1),
            1, 1, 1, false)};

INSTANTIATE_TEST_SUITE_P(
    LayerNormShardedMultiCoreProgramConfigTest,
    LayerNormShardedMultiCoreProgramConfigTest,
    ::testing::ValuesIn(layerNormShardedMultiCoreProgramConfigAttrList));
