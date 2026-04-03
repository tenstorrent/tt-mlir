// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"
#include "ttmlir/Target/TTNN/operations/configs_generated.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>

template <typename Attr, typename RetType>
class ThreeWayConsistencyTest : public ::testing::Test,
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
    std::optional<RetType> resultA = pathA(attr);
    std::optional<RetType> resultB = pathB(attr);
    std::optional<RetType> resultC = pathC(attr);

    compareResults(resultA, resultB, resultC);
  }

  virtual std::optional<RetType> pathA(Attr attr) = 0;
  virtual std::optional<RetType> pathB(Attr attr) = 0;
  virtual std::optional<RetType> pathC(Attr attr) = 0;

  virtual void compareResults(std::optional<RetType> pathA,
                              std::optional<RetType> pathB,
                              std::optional<RetType> pathC) = 0;
};

template <typename Attr, typename RetType>
bool ThreeWayConsistencyTest<Attr, RetType>::initialized = false;

template <typename Attr, typename RetType>
mlir::MLIRContext ThreeWayConsistencyTest<Attr, RetType>::context;

class DeviceComputeKernelConfigTest
    : public ThreeWayConsistencyTest<
          mlir::tt::ttnn::DeviceComputeKernelConfigAttr,
          ::ttnn::DeviceComputeKernelConfig> {

protected:
  std::optional<::ttnn::DeviceComputeKernelConfig>
  pathA(mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfigAttr)
      override {
    return mlir::tt::ttnn::op_model::conversion::getDeviceComputeKernelConfig(
        computeConfigAttr);
  }

  std::optional<::ttnn::DeviceComputeKernelConfig>
  pathB(mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfigAttr)
      override {
    tt::target::ttnn::DeviceComputeKernelConfigT deviceComputeKernelConfigT =
        mlir::tt::toNative(computeConfigAttr);
    return unifiedOpLib::operations::utils::createDeviceComputeKernelConfig(deviceComputeKernelConfigT);
  }

  std::optional<::ttnn::DeviceComputeKernelConfig>
  pathC(mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfigAttr)
      override {
    flatbuffers::FlatBufferBuilder _fbb;
    mlir::tt::FlatbufferObjectCache cache(&_fbb);

    ::flatbuffers::Offset<::tt::target::ttnn::DeviceComputeKernelConfig>
        deviceComputeKernelConfigFB =
            mlir::tt::toFlatbuffer(cache, computeConfigAttr);
    auto *r =
        flatbuffers::GetTemporaryPointer(_fbb, deviceComputeKernelConfigFB);

    return tt::runtime::ttnn::operations::utils::
        createDeviceComputeKernelConfig(r);
  }

  void compareResults(
      std::optional<::ttnn::DeviceComputeKernelConfig> pathA,
      std::optional<::ttnn::DeviceComputeKernelConfig> pathB,
      std::optional<::ttnn::DeviceComputeKernelConfig> pathC) override {
    if (pathA == std::nullopt || pathB == std::nullopt ||
        pathC == std::nullopt) {
      if (pathA == std::nullopt && pathB == std::nullopt &&
          pathC == std::nullopt) {
        SUCCEED();
        return;
      }
      FAIL();
    }

    EXPECT_EQ(pathA->math_fidelity, pathB->math_fidelity);
    EXPECT_EQ(pathA->math_fidelity, pathC->math_fidelity);

    EXPECT_EQ(pathA->math_approx_mode, pathB->math_approx_mode);
    EXPECT_EQ(pathA->math_approx_mode, pathC->math_approx_mode);

    EXPECT_EQ(pathA->fp32_dest_acc_en, pathB->fp32_dest_acc_en);
    EXPECT_EQ(pathA->fp32_dest_acc_en, pathC->fp32_dest_acc_en);

    EXPECT_EQ(pathA->packer_l1_acc, pathB->packer_l1_acc);
    EXPECT_EQ(pathA->packer_l1_acc, pathC->packer_l1_acc);

    EXPECT_EQ(pathA->dst_full_sync_en, pathB->dst_full_sync_en);
    EXPECT_EQ(pathA->dst_full_sync_en, pathC->dst_full_sync_en);
  }
};

class Conv2dConfigTest
    : public ThreeWayConsistencyTest<mlir::tt::ttnn::Conv2dConfigAttr,
                                     ::ttnn::Conv2dConfig> {
protected:
  std::optional<::ttnn::Conv2dConfig>
  pathA(mlir::tt::ttnn::Conv2dConfigAttr conv2dConfigAttr) override {
    return mlir::tt::ttnn::op_model::conversion::getConv2dConfig(
        conv2dConfigAttr);
  }

  std::optional<::ttnn::Conv2dConfig>
  pathB(mlir::tt::ttnn::Conv2dConfigAttr conv2dConfigAttr) override {
    tt::target::ttnn::Conv2dConfigT conv2dConfigT =
        mlir::tt::toNative(conv2dConfigAttr);
    return unifiedOpLib::operations::utils::createConv2dConfig(
        conv2dConfigT);
  }

  std::optional<::ttnn::Conv2dConfig>
  pathC(mlir::tt::ttnn::Conv2dConfigAttr conv2dConfigAttr) override {
    flatbuffers::FlatBufferBuilder fbb;
    mlir::tt::FlatbufferObjectCache cache(&fbb);

    ::flatbuffers::Offset<::tt::target::ttnn::Conv2dConfig> conv2dConfigFB =
        mlir::tt::toFlatbuffer(cache, conv2dConfigAttr);
    auto *r = flatbuffers::GetTemporaryPointer(fbb, conv2dConfigFB);
    return tt::runtime::ttnn::operations::utils::createConv2dConfig(r);
  }

  void compareResults(std::optional<::ttnn::Conv2dConfig> pathA,
                      std::optional<::ttnn::Conv2dConfig> pathB,
                      std::optional<::ttnn::Conv2dConfig> pathC) override {
    if (pathA == std::nullopt || pathB == std::nullopt ||
        pathC == std::nullopt) {
      if (pathA == std::nullopt && pathB == std::nullopt &&
          pathC == std::nullopt) {
        SUCCEED();
        return;
      }
      FAIL();
    }

    EXPECT_EQ(pathA->weights_dtype, pathB->weights_dtype);
    EXPECT_EQ(pathA->weights_dtype, pathC->weights_dtype);

    auto compareUnaryWithParam =
        [](const std::optional<ttnn::operations::unary::UnaryWithParam> &a,
           const std::optional<ttnn::operations::unary::UnaryWithParam> &b) {
          if (a.has_value() != b.has_value()) {
            return false;
          }
          if (!a) {
            return true;
          }
          return a->op_type == b->op_type && a->params == b->params;
        };
    EXPECT_EQ(true,
              compareUnaryWithParam(pathA->activation, pathB->activation));
    EXPECT_EQ(true,
              compareUnaryWithParam(pathA->activation, pathC->activation));

    EXPECT_EQ(pathA->deallocate_activation, pathB->deallocate_activation);
    EXPECT_EQ(pathA->deallocate_activation, pathC->deallocate_activation);

    EXPECT_EQ(pathA->reallocate_halo_output, pathB->reallocate_halo_output);
    EXPECT_EQ(pathA->reallocate_halo_output, pathC->reallocate_halo_output);

    EXPECT_EQ(pathA->config_tensors_in_dram, pathB->config_tensors_in_dram);
    EXPECT_EQ(pathA->config_tensors_in_dram, pathC->config_tensors_in_dram);

    EXPECT_EQ(pathA->act_block_h_override, pathB->act_block_h_override);
    EXPECT_EQ(pathA->act_block_h_override, pathC->act_block_h_override);

    EXPECT_EQ(pathA->act_block_w_div, pathB->act_block_w_div);
    EXPECT_EQ(pathA->act_block_w_div, pathC->act_block_w_div);

    EXPECT_EQ(pathA->reshard_if_not_optimal, pathB->reshard_if_not_optimal);
    EXPECT_EQ(pathA->reshard_if_not_optimal, pathC->reshard_if_not_optimal);

    EXPECT_EQ(pathA->override_sharding_config, pathB->override_sharding_config);
    EXPECT_EQ(pathA->override_sharding_config, pathC->override_sharding_config);

    EXPECT_EQ(pathA->shard_layout, pathB->shard_layout);
    EXPECT_EQ(pathA->shard_layout, pathC->shard_layout);

    EXPECT_EQ(pathA->core_grid, pathB->core_grid);
    EXPECT_EQ(pathB->core_grid, pathC->core_grid);

    EXPECT_EQ(pathA->transpose_shards, pathB->transpose_shards);
    EXPECT_EQ(pathA->transpose_shards, pathC->transpose_shards);

    EXPECT_EQ(pathA->output_layout, pathB->output_layout);
    EXPECT_EQ(pathA->output_layout, pathC->output_layout);

    EXPECT_EQ(pathA->enable_act_double_buffer, pathB->enable_act_double_buffer);
    EXPECT_EQ(pathA->enable_act_double_buffer, pathC->enable_act_double_buffer);

    EXPECT_EQ(pathA->enable_weights_double_buffer,
              pathB->enable_weights_double_buffer);
    EXPECT_EQ(pathA->enable_weights_double_buffer,
              pathC->enable_weights_double_buffer);

    EXPECT_EQ(pathA->enable_kernel_stride_folding,
              pathB->enable_kernel_stride_folding);
    EXPECT_EQ(pathA->enable_kernel_stride_folding,
              pathC->enable_kernel_stride_folding);
  }
};

TEST_P(DeviceComputeKernelConfigTest, DeviceComputeKernelConfig) {
  auto params = GetParam();
  RunTest(params);
}

const std::initializer_list<mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
    deviceComputeKernelConfigAttrlist = {
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
    ::testing::ValuesIn(deviceComputeKernelConfigAttrlist));

TEST_P(Conv2dConfigTest, Conv2dConfig) {
  auto params = GetParam();
  RunTest(params);
}

const std::initializer_list<mlir::tt::ttnn::Conv2dConfigAttr>
    conv2dConfigAttrlist = {
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
            mlir::BoolAttr::get(Conv2dConfigTest::getContext(), true)),
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
                         ::testing::ValuesIn(conv2dConfigAttrlist));
