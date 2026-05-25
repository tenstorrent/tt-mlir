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

    compareFields(native, unpacked);
  }

  virtual void compareFields(const NativeT &native,
                             const NativeT &unpacked) = 0;
};

template <typename Attr, typename NativeT>
mlir::MLIRContext ToNativeConsistencyTest<Attr, NativeT>::context;

template <typename Attr, typename NativeT>
bool ToNativeConsistencyTest<Attr, NativeT>::initialized = false;

namespace {

using ::tt::target::ttnn::CoreCoord;
using ::tt::target::ttnn::CoreRangeSetT;
using ::tt::target::ttnn::NDShardSpecT;
using ::tt::target::ttnn::ShardSpecT;
using ::tt::target::ttnn::UnaryWithParamT;

void compareCoreCoordPtr(const std::unique_ptr<CoreCoord> &native,
                         const std::unique_ptr<CoreCoord> &unpacked) {
  ASSERT_EQ(native == nullptr, unpacked == nullptr);
  if (native) {
    EXPECT_EQ(native->x(), unpacked->x());
    EXPECT_EQ(native->y(), unpacked->y());
  }
}

void compareCoreRangeSet(const CoreRangeSetT &native,
                         const CoreRangeSetT &unpacked) {
  ASSERT_EQ(native.core_ranges.size(), unpacked.core_ranges.size());
  for (size_t i = 0; i < native.core_ranges.size(); ++i) {
    EXPECT_EQ(native.core_ranges[i].start_coord().x(),
              unpacked.core_ranges[i].start_coord().x());
    EXPECT_EQ(native.core_ranges[i].start_coord().y(),
              unpacked.core_ranges[i].start_coord().y());
    EXPECT_EQ(native.core_ranges[i].end_coord().x(),
              unpacked.core_ranges[i].end_coord().x());
    EXPECT_EQ(native.core_ranges[i].end_coord().y(),
              unpacked.core_ranges[i].end_coord().y());
  }
}

void compareCoreRangeSetPtr(const std::unique_ptr<CoreRangeSetT> &native,
                            const std::unique_ptr<CoreRangeSetT> &unpacked) {
  ASSERT_EQ(native == nullptr, unpacked == nullptr);
  if (native) {
    compareCoreRangeSet(*native, *unpacked);
  }
}

void compareUnaryWithParamPtr(
    const std::unique_ptr<UnaryWithParamT> &native,
    const std::unique_ptr<UnaryWithParamT> &unpacked) {
  ASSERT_EQ(native == nullptr, unpacked == nullptr);
  if (native) {
    EXPECT_EQ(native->op_type, unpacked->op_type);
    EXPECT_EQ(native->params, unpacked->params);
  }
}

void compareShardSpecPtr(const std::unique_ptr<ShardSpecT> &native,
                         const std::unique_ptr<ShardSpecT> &unpacked) {
  ASSERT_EQ(native == nullptr, unpacked == nullptr);
  if (native) {
    compareCoreRangeSetPtr(native->core_range_set, unpacked->core_range_set);
    EXPECT_EQ(native->shape, unpacked->shape);
    EXPECT_EQ(native->orientation, unpacked->orientation);
  }
}

void compareNDShardSpecPtr(const std::unique_ptr<NDShardSpecT> &native,
                           const std::unique_ptr<NDShardSpecT> &unpacked) {
  ASSERT_EQ(native == nullptr, unpacked == nullptr);
  if (native) {
    compareCoreRangeSetPtr(native->core_range_set, unpacked->core_range_set);
    EXPECT_EQ(native->shape, unpacked->shape);
    EXPECT_EQ(native->orientation, unpacked->orientation);
    EXPECT_EQ(native->distribution_strategy, unpacked->distribution_strategy);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// UnaryWithParam
//===----------------------------------------------------------------------===//

class UnaryWithParamTest
    : public ToNativeConsistencyTest<mlir::tt::ttnn::UnaryWithParamAttr,
                                     ::tt::target::ttnn::UnaryWithParamT> {
protected:
  void
  compareFields(const ::tt::target::ttnn::UnaryWithParamT &native,
                const ::tt::target::ttnn::UnaryWithParamT &unpacked) override {
    EXPECT_EQ(native.op_type, unpacked.op_type);
    EXPECT_EQ(native.params, unpacked.params);
  }
};

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
                                     ::tt::target::ttnn::CoreRangeSetT> {
protected:
  void
  compareFields(const ::tt::target::ttnn::CoreRangeSetT &native,
                const ::tt::target::ttnn::CoreRangeSetT &unpacked) override {
    compareCoreRangeSet(native, unpacked);
  }
};

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
          ::tt::target::ttnn::DeviceComputeKernelConfigT> {
protected:
  void compareFields(
      const ::tt::target::ttnn::DeviceComputeKernelConfigT &native,
      const ::tt::target::ttnn::DeviceComputeKernelConfigT &unpacked) override {
    EXPECT_EQ(native.math_fidelity, unpacked.math_fidelity);
    EXPECT_EQ(native.math_approx_mode, unpacked.math_approx_mode);
    EXPECT_EQ(native.fp32_dest_acc_en, unpacked.fp32_dest_acc_en);
    EXPECT_EQ(native.packer_l1_acc, unpacked.packer_l1_acc);
    EXPECT_EQ(native.dst_full_sync_en, unpacked.dst_full_sync_en);
  }
};

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
                                     ::tt::target::ttnn::Conv2dConfigT> {
protected:
  void
  compareFields(const ::tt::target::ttnn::Conv2dConfigT &native,
                const ::tt::target::ttnn::Conv2dConfigT &unpacked) override {
    EXPECT_EQ(native.weights_dtype, unpacked.weights_dtype);
    EXPECT_EQ(native.deallocate_activation, unpacked.deallocate_activation);
    EXPECT_EQ(native.reallocate_halo_output, unpacked.reallocate_halo_output);
    EXPECT_EQ(native.act_block_h_override, unpacked.act_block_h_override);
    EXPECT_EQ(native.act_block_w_div, unpacked.act_block_w_div);
    EXPECT_EQ(native.reshard_if_not_optimal, unpacked.reshard_if_not_optimal);
    EXPECT_EQ(native.override_sharding_config,
              unpacked.override_sharding_config);
    EXPECT_EQ(native.shard_layout, unpacked.shard_layout);
    EXPECT_EQ(native.transpose_shards, unpacked.transpose_shards);
    EXPECT_EQ(native.output_layout, unpacked.output_layout);
    EXPECT_EQ(native.enable_act_double_buffer,
              unpacked.enable_act_double_buffer);
    EXPECT_EQ(native.enable_weights_double_buffer,
              unpacked.enable_weights_double_buffer);
    EXPECT_EQ(native.enable_kernel_stride_folding,
              unpacked.enable_kernel_stride_folding);
    EXPECT_EQ(native.config_tensors_in_dram, unpacked.config_tensors_in_dram);

    ASSERT_EQ(native.activation == nullptr, unpacked.activation == nullptr);
    if (native.activation) {
      EXPECT_EQ(native.activation->op_type, unpacked.activation->op_type);
      EXPECT_EQ(native.activation->params, unpacked.activation->params);
    }

    ASSERT_EQ(native.core_grid == nullptr, unpacked.core_grid == nullptr);
    if (native.core_grid) {
      ASSERT_EQ(native.core_grid->core_ranges.size(),
                unpacked.core_grid->core_ranges.size());
      for (size_t i = 0; i < native.core_grid->core_ranges.size(); ++i) {
        const auto &nr = native.core_grid->core_ranges[i];
        const auto &rr = unpacked.core_grid->core_ranges[i];
        EXPECT_EQ(nr.start_coord().x(), rr.start_coord().x());
        EXPECT_EQ(nr.start_coord().y(), rr.start_coord().y());
        EXPECT_EQ(nr.end_coord().x(), rr.end_coord().x());
        EXPECT_EQ(nr.end_coord().y(), rr.end_coord().y());
      }
    }
  }
};

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
                                     ::tt::target::ttnn::Conv2dSliceConfigT> {
protected:
  void compareFields(
      const ::tt::target::ttnn::Conv2dSliceConfigT &native,
      const ::tt::target::ttnn::Conv2dSliceConfigT &unpacked) override {
    EXPECT_EQ(native.slice_type, unpacked.slice_type);
    EXPECT_EQ(native.num_slices, unpacked.num_slices);
  }
};

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
                                     ::tt::target::ttnn::Conv3dConfigT> {
protected:
  void
  compareFields(const ::tt::target::ttnn::Conv3dConfigT &native,
                const ::tt::target::ttnn::Conv3dConfigT &unpacked) override {
    EXPECT_EQ(native.weights_dtype, unpacked.weights_dtype);
    EXPECT_EQ(native.t_out_block, unpacked.t_out_block);
    EXPECT_EQ(native.w_out_block, unpacked.w_out_block);
    EXPECT_EQ(native.h_out_block, unpacked.h_out_block);
    EXPECT_EQ(native.c_out_block, unpacked.c_out_block);
    EXPECT_EQ(native.c_in_block, unpacked.c_in_block);
  }
};

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
                                     ::tt::target::ttnn::ShardSpecT> {
protected:
  void compareFields(const ::tt::target::ttnn::ShardSpecT &native,
                     const ::tt::target::ttnn::ShardSpecT &unpacked) override {
    compareCoreRangeSetPtr(native.core_range_set, unpacked.core_range_set);
    EXPECT_EQ(native.shape, unpacked.shape);
    EXPECT_EQ(native.orientation, unpacked.orientation);
  }
};

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
                                     ::tt::target::ttnn::NDShardSpecT> {
protected:
  void
  compareFields(const ::tt::target::ttnn::NDShardSpecT &native,
                const ::tt::target::ttnn::NDShardSpecT &unpacked) override {
    compareCoreRangeSetPtr(native.core_range_set, unpacked.core_range_set);
    EXPECT_EQ(native.shape, unpacked.shape);
    EXPECT_EQ(native.orientation, unpacked.orientation);
    EXPECT_EQ(native.distribution_strategy, unpacked.distribution_strategy);
  }
};

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
                                     ::tt::target::ttnn::MemoryConfigT> {
protected:
  void
  compareFields(const ::tt::target::ttnn::MemoryConfigT &native,
                const ::tt::target::ttnn::MemoryConfigT &unpacked) override {
    EXPECT_EQ(native.tensor_memory_layout, unpacked.tensor_memory_layout);
    EXPECT_EQ(native.buffer_type, unpacked.buffer_type);
    compareShardSpecPtr(native.shard_spec, unpacked.shard_spec);
    compareNDShardSpecPtr(native.nd_shard_spec, unpacked.nd_shard_spec);
  }
};

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
          ::tt::target::ttnn::MatmulMultiCoreReuseProgramConfigT> {
protected:
  void compareFields(
      const ::tt::target::ttnn::MatmulMultiCoreReuseProgramConfigT &native,
      const ::tt::target::ttnn::MatmulMultiCoreReuseProgramConfigT &unpacked)
      override {
    compareCoreCoordPtr(native.compute_with_storage_grid_size,
                        unpacked.compute_with_storage_grid_size);
    EXPECT_EQ(native.in0_block_w, unpacked.in0_block_w);
    EXPECT_EQ(native.out_subblock_h, unpacked.out_subblock_h);
    EXPECT_EQ(native.out_subblock_w, unpacked.out_subblock_w);
    EXPECT_EQ(native.per_core_m, unpacked.per_core_m);
    EXPECT_EQ(native.per_core_n, unpacked.per_core_n);
  }
};

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
          ::tt::target::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigT> {
protected:
  void compareFields(
      const ::tt::target::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigT
          &native,
      const ::tt::target::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigT
          &unpacked) override {
    compareCoreCoordPtr(native.compute_with_storage_grid_size,
                        unpacked.compute_with_storage_grid_size);
    EXPECT_EQ(native.in0_block_w, unpacked.in0_block_w);
    EXPECT_EQ(native.out_subblock_h, unpacked.out_subblock_h);
    EXPECT_EQ(native.out_subblock_w, unpacked.out_subblock_w);
    EXPECT_EQ(native.out_block_h, unpacked.out_block_h);
    EXPECT_EQ(native.out_block_w, unpacked.out_block_w);
    EXPECT_EQ(native.per_core_m, unpacked.per_core_m);
    EXPECT_EQ(native.per_core_n, unpacked.per_core_n);
    EXPECT_EQ(native.transpose_mcast, unpacked.transpose_mcast);
    compareUnaryWithParamPtr(native.fused_activation,
                             unpacked.fused_activation);
    EXPECT_EQ(native.fuse_batch, unpacked.fuse_batch);
  }
};

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
          ::tt::target::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigT> {
protected:
  void compareFields(
      const ::tt::target::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigT
          &native,
      const ::tt::target::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigT
          &unpacked) override {
    compareCoreCoordPtr(native.compute_with_storage_grid_size,
                        unpacked.compute_with_storage_grid_size);
    EXPECT_EQ(native.in0_block_w, unpacked.in0_block_w);
    EXPECT_EQ(native.out_subblock_h, unpacked.out_subblock_h);
    EXPECT_EQ(native.out_subblock_w, unpacked.out_subblock_w);
    EXPECT_EQ(native.out_block_h, unpacked.out_block_h);
    EXPECT_EQ(native.out_block_w, unpacked.out_block_w);
    EXPECT_EQ(native.per_core_m, unpacked.per_core_m);
    EXPECT_EQ(native.per_core_n, unpacked.per_core_n);
    EXPECT_EQ(native.fuse_batch, unpacked.fuse_batch);
    compareUnaryWithParamPtr(native.fused_activation,
                             unpacked.fused_activation);
    EXPECT_EQ(native.mcast_in0, unpacked.mcast_in0);
    EXPECT_EQ(native.gather_in0, unpacked.gather_in0);
    compareCoreRangeSetPtr(native.hop_cores, unpacked.hop_cores);
    EXPECT_EQ(native.num_global_cb_receivers, unpacked.num_global_cb_receivers);
    EXPECT_EQ(native.untilize_out, unpacked.untilize_out);
  }
};

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
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigT> {
protected:
  void compareFields(
      const ::tt::target::ttnn::
          MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigT &native,
      const ::tt::target::ttnn::
          MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigT &unpacked)
      override {
    EXPECT_EQ(native.in0_block_w, unpacked.in0_block_w);
    EXPECT_EQ(native.per_core_m, unpacked.per_core_m);
    EXPECT_EQ(native.per_core_n, unpacked.per_core_n);
    compareUnaryWithParamPtr(native.fused_activation,
                             unpacked.fused_activation);
  }
};

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
