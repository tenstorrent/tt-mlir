// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"
#include "ttmlir/Target/TTNN/operations/configs_generated.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <optional>

template <typename T>
::testing::AssertionResult Equal(const char *e1, const char *e2, const char *e3,
                                 const T &v1, const T &v2, const T &v3) {
  bool ok12 = (v1 == v2);
  bool ok13 = (v1 == v3);

  if (ok12 && ok13) {
    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult failure = ::testing::AssertionFailure();
  failure << e1 << " == " << e2 << " && " << e1 << " == " << e3 << " failed;";
  return failure;
}

template <>
::testing::AssertionResult
Equal<std::optional<ttnn::operations::unary::UnaryWithParam>>(
    const char *e1, const char *e2, const char *e3,
    const std::optional<ttnn::operations::unary::UnaryWithParam> &v1,
    const std::optional<ttnn::operations::unary::UnaryWithParam> &v2,
    const std::optional<ttnn::operations::unary::UnaryWithParam> &v3) {
  bool params12 = true;
  if (v1 && v2) {
    if (v1->params.size() != v2->params.size()) {
      params12 = false;
    } else {
      for (size_t i = 0; i < v1->params.size(); i++) {
        if (v1->params[i] != v2->params[i]) {
          params12 = false;
          break;
        }
      }
    }
  }
  bool params13 = true;
  if (v1 && v3) {
    if (v1->params.size() != v3->params.size()) {
      params13 = false;
    } else {
      for (size_t i = 0; i < v1->params.size(); i++) {
        if (v1->params[i] != v3->params[i]) {
          params13 = false;
          break;
        }
      }
    }
  }
  bool ok12 =
      (!v1 && !v2) || (v1 && v2 && v1->op_type == v2->op_type && params12);
  bool ok13 =
      (!v1 && !v3) || (v1 && v3 && v1->op_type == v3->op_type && params13);

  if (ok12 && ok13) {
    return ::testing::AssertionSuccess();
  }

  ::testing::AssertionResult failure = ::testing::AssertionFailure();
  failure << e1 << " == " << e2 << " && " << e1 << " == " << e3 << " failed;";
  return failure;
}

template <typename Attr, typename RetType>
class ThreeWayConsistencyTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  void SetUp() override { context.loadDialect<mlir::tt::ttnn::TTNNDialect>(); }

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
    return tt::runtime::ttnn::operations::utils::
        createDeviceComputeKernelConfig(deviceComputeKernelConfigT);
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
      }
      FAIL();
    }

    EXPECT_PRED_FORMAT3(Equal<MathFidelity>, pathA->math_fidelity,
                        pathB->math_fidelity, pathC->math_fidelity);

    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->math_approx_mode,
                        pathB->math_approx_mode, pathC->math_approx_mode);

    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->fp32_dest_acc_en,
                        pathB->fp32_dest_acc_en, pathC->fp32_dest_acc_en);

    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->packer_l1_acc, pathB->packer_l1_acc,
                        pathC->packer_l1_acc);

    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->dst_full_sync_en,
                        pathB->dst_full_sync_en, pathC->dst_full_sync_en);
  }
};

class Conv2dConfigTest
    : public ThreeWayConsistencyTest<mlir::tt::ttnn::Conv2dConfigAttr,
                                     ::ttnn::Conv2dConfig> {
protected:
  std::optional<::ttnn::Conv2dConfig>
  pathA(mlir::tt::ttnn::Conv2dConfigAttr conv2dConfigAttr) override {
    // TODO(#2130)
    if (conv2dConfigAttr.hasCoreGrid()) {
      std::cout << "not empty core_grid\n";
      return pathC(conv2dConfigAttr);
    }
    return mlir::tt::ttnn::op_model::conversion::getConv2dConfig(
        conv2dConfigAttr);
  }

  std::optional<::ttnn::Conv2dConfig>
  pathB(mlir::tt::ttnn::Conv2dConfigAttr conv2dConfigAttr) override {
    tt::target::ttnn::Conv2dConfigT conv2dConfigT =
        mlir::tt::toNative(conv2dConfigAttr);
    return tt::runtime::ttnn::operations::utils::createConv2dConfig(
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
      }
      FAIL();
    }

    EXPECT_PRED_FORMAT3(Equal<std::optional<tt::tt_metal::DataType>>,
                        pathA->weights_dtype, pathB->weights_dtype,
                        pathC->weights_dtype);
    EXPECT_PRED_FORMAT3(
        Equal<std::optional<ttnn::operations::unary::UnaryWithParam>>,
        pathA->activation, pathB->activation, pathC->activation);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->deallocate_activation,
                        pathB->deallocate_activation,
                        pathC->deallocate_activation);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->reallocate_halo_output,
                        pathB->reallocate_halo_output,
                        pathC->reallocate_halo_output);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->act_block_h_override,
                        pathB->act_block_h_override,
                        pathC->act_block_h_override);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->act_block_w_div,
                        pathB->act_block_w_div, pathC->act_block_w_div);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->reshard_if_not_optimal,
                        pathB->reshard_if_not_optimal,
                        pathC->reshard_if_not_optimal);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->override_sharding_config,
                        pathB->override_sharding_config,
                        pathC->override_sharding_config);
    EXPECT_PRED_FORMAT3(Equal<std::optional<tt::tt_metal::TensorMemoryLayout>>,
                        pathA->shard_layout, pathB->shard_layout,
                        pathC->shard_layout);
    EXPECT_PRED_FORMAT3(Equal<std::optional<tt::tt_metal::CoreRangeSet>>,
                        pathA->core_grid, pathB->core_grid, pathC->core_grid);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->transpose_shards,
                        pathB->transpose_shards, pathC->transpose_shards);
    EXPECT_PRED_FORMAT3(Equal<tt::tt_metal::Layout>, pathA->output_layout,
                        pathB->output_layout, pathC->output_layout);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->enable_act_double_buffer,
                        pathB->enable_act_double_buffer,
                        pathC->enable_act_double_buffer);
    EXPECT_PRED_FORMAT3(Equal<bool>, pathA->enable_weights_double_buffer,
                        pathB->enable_weights_double_buffer,
                        pathC->enable_weights_double_buffer);
    EXPECT_PRED_FORMAT3(Equal<std::optional<bool>>,
                        pathA->enable_kernel_stride_folding,
                        pathB->enable_kernel_stride_folding,
                        pathC->enable_kernel_stride_folding);
    EXPECT_PRED_FORMAT3(Equal<std::optional<tt::tt_metal::TensorMemoryLayout>>,
                        pathA->shard_layout, pathB->shard_layout,
                        pathC->shard_layout);
  }
};

TEST_F(DeviceComputeKernelConfigTest, ComputeKernelConfig) {
  mlir::tt::ttnn::DeviceComputeKernelConfigAttr baseConfig =
      mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(&context);
  baseConfig = baseConfig.withMathFidelity(mlir::tt::ttnn::MathFidelity::HiFi3);
  baseConfig = baseConfig.withPackerL1Acc(true);
  RunTest(baseConfig);
}

TEST_F(Conv2dConfigTest, Conv2dConfig) {
  mlir::tt::ttnn::Conv2dConfigAttr baseConfig =
      mlir::tt::ttnn::Conv2dConfigAttr::get(&context);
  baseConfig = baseConfig.withDeallocateActivation(true);
  baseConfig = baseConfig.withShardLayout(
      mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
  baseConfig =
      baseConfig.withWeightsDtype(mlir::tt::ttcore::DataType::BFP_BFloat8);
  baseConfig = baseConfig.withConfigTensorsInDram(false);
  baseConfig = baseConfig.withActBlockWDiv(16);
  baseConfig = baseConfig.withEnableKernelStrideFolding(true);
  baseConfig = baseConfig.withActivation(mlir::tt::ttnn::UnaryOpType::Relu);

  RunTest(baseConfig);
}

TEST_F(Conv2dConfigTest, Conv2dConfigWithCoreGrid) {
  mlir::tt::ttnn::Conv2dConfigAttr baseConfig =
      mlir::tt::ttnn::Conv2dConfigAttr::get(&context);
  baseConfig = baseConfig.withDeallocateActivation(true);
  baseConfig = baseConfig.withShardLayout(
      mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
  baseConfig =
      baseConfig.withWeightsDtype(mlir::tt::ttcore::DataType::BFP_BFloat8);
  baseConfig = baseConfig.withConfigTensorsInDram(false);
  baseConfig = baseConfig.withActBlockWDiv(16);
  baseConfig = baseConfig.withEnableKernelStrideFolding(true);

  baseConfig = baseConfig.withCoreGrid(mlir::tt::ttnn::CoreRangeSetAttr::get(
  &context,
  llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>{mlir::tt::ttnn::CoreRangeAttr::get(
                  &context, mlir::tt::ttnn::CoreCoordAttr::get(&context, 0,
                  0), mlir::tt::ttnn::CoreCoordAttr::get(&context, 7, 0))}));

  baseConfig = baseConfig.withActivation(mlir::tt::ttnn::UnaryOpType::Relu);

  RunTest(baseConfig);
}
