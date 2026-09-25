// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

using namespace mlir::tt;
using namespace mlir::tt::ttnn;

// DRAM-sharded matmul eligibility and weight geometry through the matmul rule
// book. A real func.func is needed because the weight must trace to a parameter
// argument. The bias-free ttnn.linear case is only reachable here: a bias-free
// ttir.linear canonicalizes to ttir.matmul before lit could see it.
namespace {

constexpr int64_t kTile = 32;

// No metal device: eligibility and weight layout need only the DeviceAttr and
// the system descriptor.
class DSTestBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  int funcCounter = 0;

  void SetUp() override {
    context.loadDialect<ttcore::TTCoreDialect>();
    context.loadDialect<ttnn::TTNNDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    initModule(ttcore::Arch::WormholeB0);
  }

  void initModule(ttcore::Arch arch) {
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    ttcore::registerDevice(module.get(), arch);
    module->getOperation()->setAttr(utils::g_TensorL1UsageCapAttrName,
                                    builder.getF32FloatAttr(0.95f));
    // DS is off unless asked for.
    module->getOperation()->setAttr(utils::g_EnableDRAMShardedMatmulAttrName,
                                    builder.getBoolAttr(true));
  }

  TTNNLayoutAttr dramInterleaved(llvm::ArrayRef<int64_t> shape,
                                 ttcore::DataType dt) {
    auto elementType = ttcore::TileType::get(&context, {kTile, kTile}, dt);
    auto deviceAttr = ttcore::lookupDevice(module.get());
    return TTNNLayoutAttr::Builder(&context, shape, elementType)
        .setBufferType(BufferType::DRAM)
        .setMemoryLayout(TensorMemoryLayout::Interleaved)
        .setGridShape({1, 1})
        .buildWithCanonicalCorePlacement(deviceAttr);
  }

  mlir::RankedTensorType tensorOf(llvm::ArrayRef<int64_t> shape,
                                  ttcore::DataType dt) {
    return mlir::RankedTensorType::get(shape, builder.getBF16Type(),
                                       dramInterleaved(shape, dt));
  }

  // Arg 1 (the weight) is marked as a parameter.
  mlir::Block *openFunc(llvm::ArrayRef<mlir::Type> argTypes,
                        mlir::Type resultType) {
    builder.setInsertionPointToEnd(&module->getBodyRegion().front());
    auto funcType = builder.getFunctionType(argTypes, {resultType});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "ds_test_" + std::to_string(funcCounter++),
        funcType);
    func.setArgAttr(1, ttcore::ArgumentTypeAttr::name,
                    ttcore::ArgumentTypeAttr::get(
                        &context, ttcore::ArgumentType::Parameter));
    mlir::Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    return block;
  }

  MatmulOp buildMatmul(llvm::ArrayRef<int64_t> actShape,
                       llvm::ArrayRef<int64_t> weightShape,
                       llvm::ArrayRef<int64_t> outShape,
                       ttcore::DataType weightDt,
                       mlir::StringAttr activation = mlir::StringAttr()) {
    auto actType = tensorOf(actShape, ttcore::DataType::BFloat16);
    auto weightType = tensorOf(weightShape, weightDt);
    auto outType = tensorOf(outShape, ttcore::DataType::BFloat16);
    mlir::Block *block = openFunc({actType, weightType}, outType);
    auto op = builder.create<MatmulOp>(
        builder.getUnknownLoc(), outType, block->getArgument(0),
        block->getArgument(1), /*transpose_a=*/false, /*transpose_b=*/false,
        /*matmul_program_config=*/mlir::Attribute(),
        /*activation=*/activation);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         op.getResult());
    return op;
  }

  LinearOp buildLinear(llvm::ArrayRef<int64_t> actShape,
                       llvm::ArrayRef<int64_t> weightShape,
                       llvm::ArrayRef<int64_t> outShape,
                       ttcore::DataType weightDt, bool withBias,
                       mlir::StringAttr activation = mlir::StringAttr()) {
    auto actType = tensorOf(actShape, ttcore::DataType::BFloat16);
    auto weightType = tensorOf(weightShape, weightDt);
    auto outType = tensorOf(outShape, ttcore::DataType::BFloat16);

    llvm::SmallVector<int64_t> biasShape{1, weightShape.back()};
    auto biasType = tensorOf(biasShape, ttcore::DataType::BFloat16);

    llvm::SmallVector<mlir::Type> argTypes{actType, weightType};
    if (withBias) {
      argTypes.push_back(biasType);
    }
    mlir::Block *block = openFunc(argTypes, outType);

    mlir::Value bias = withBias ? block->getArgument(2) : mlir::Value();
    auto op = builder.create<LinearOp>(
        builder.getUnknownLoc(), outType, block->getArgument(0),
        block->getArgument(1), bias, /*transpose_a=*/false,
        /*transpose_b=*/false, /*activation=*/activation);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         op.getResult());
    return op;
  }

  std::vector<OpConfig> legalConfigs(llvm::ArrayRef<int64_t> outShape) {
    std::vector<OpConfig> configs;
    configs.emplace_back(dramInterleaved(outShape, ttcore::DataType::BFloat16));
    return configs;
  }

  // Whether any hint carries a DRAM-sharded matmul program config.
  static bool hasDSHint(const OutputHints &hints) {
    for (const auto &hint : hints.hints) {
      const auto *attrs = std::get_if<MatmulAttrs>(&hint.opSpecificAttrs);
      if (!attrs || !attrs->matmulProgramConfig.has_value() ||
          !attrs->matmulProgramConfig.value()) {
        continue;
      }
      if (mlir::isa<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
              attrs->matmulProgramConfig.value())) {
        return true;
      }
    }
    return false;
  }

  bool isDSEligible(mlir::Operation *op, llvm::ArrayRef<int64_t> outShape) {
    return hasDSHint(getOutputHints(op, legalConfigs(outShape)));
  }
};

class DRAMShardedEligibilityTest : public DSTestBase {};

//===----------------------------------------------------------------------===//
// Eligibility
//===----------------------------------------------------------------------===//

TEST_F(DRAMShardedEligibilityTest, MatmulEligible) {
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_TRUE(isDSEligible(op, {32, 4096}));
}

// The same computation, and how ttnn decoders write their projections.
TEST_F(DRAMShardedEligibilityTest, BiasFreeLinearEligible) {
  auto op = buildLinear({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8, /*withBias=*/false);
  EXPECT_TRUE(isDSEligible(op, {32, 4096}));
}

// The DS kernel reads a bias per DRAM bank, so it needs a DRAM width-sharded
// bias nothing produces yet; an interleaved one is read wrong silently.
TEST_F(DRAMShardedEligibilityTest, BiasedLinearDeclined) {
  auto op = buildLinear({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8, /*withBias=*/true);
  EXPECT_FALSE(isDSEligible(op, {32, 4096}));
}

// [1, 1, K, N] is the same matrix as [K, N].
TEST_F(DRAMShardedEligibilityTest, UnitBatchedWeightEligible) {
  auto op = buildMatmul({1, 1, 32, 4096}, {1, 1, 4096, 4096}, {1, 1, 32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_TRUE(isDSEligible(op, {1, 1, 32, 4096}));
}

// A non-unit batch dim needs the batched DS config, which is not emitted.
TEST_F(DRAMShardedEligibilityTest, BatchedWeightDeclined) {
  auto op = buildMatmul({1, 1, 32, 4096}, {2, 1, 4096, 4096}, {2, 1, 32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_FALSE(isDSEligible(op, {2, 1, 32, 4096}));
}

// A sub-tile batch pads up to one tile row.
TEST_F(DRAMShardedEligibilityTest, SubTileBatchEligible) {
  auto op = buildMatmul({1, 4096}, {4096, 4096}, {1, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_TRUE(isDSEligible(op, {1, 4096}));
}

// tt-metal's M == 1 assert is uncatchable, so taller is declined here.
TEST_F(DRAMShardedEligibilityTest, MultiTileMDeclined) {
  auto op = buildMatmul({64, 4096}, {4096, 4096}, {64, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_FALSE(isDSEligible(op, {64, 4096}));
}

// bfp4/bfp8 only: bf16 doubles the DRAM bytes DS streams.
TEST_F(DRAMShardedEligibilityTest, Bf16WeightDeclined) {
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFloat16);
  EXPECT_FALSE(isDSEligible(op, {32, 4096}));
}

TEST_F(DRAMShardedEligibilityTest, Bfp4WeightEligible) {
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat4);
  EXPECT_TRUE(isDSEligible(op, {32, 4096}));
}

// K in tiles must divide by the fixed 8 in0 cores; 2880 (90 tiles) is declined
// until the core count can be chosen by cost.
TEST_F(DRAMShardedEligibilityTest, KTilesNotDivisibleByIn0CoresDeclined) {
  auto op = buildMatmul({32, 2880}, {2880, 2880}, {32, 2880},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_FALSE(isDSEligible(op, {32, 2880}));
}

// The flag gates the single choke point; an absent attribute reads as off.
TEST_F(DRAMShardedEligibilityTest, DisableOptionSuppressesDS) {
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  ASSERT_TRUE(isDSEligible(op, {32, 4096}));

  module->getOperation()->setAttr(utils::g_EnableDRAMShardedMatmulAttrName,
                                  builder.getBoolAttr(false));
  EXPECT_FALSE(isDSEligible(op, {32, 4096}));

  module->getOperation()->removeAttr(utils::g_EnableDRAMShardedMatmulAttrName);
  EXPECT_FALSE(isDSEligible(op, {32, 4096}));
}

// An activation left on the matmul is declined: the op model would validate the
// DS config without it while the runtime applies it.
TEST_F(DRAMShardedEligibilityTest, FusedActivationOnMatmulDeclined) {
  auto op =
      buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                  ttcore::DataType::BFP_BFloat8, builder.getStringAttr("silu"));
  EXPECT_FALSE(isDSEligible(op, {32, 4096}));
}

// Same for a bias-free linear, which lit cannot reach.
TEST_F(DRAMShardedEligibilityTest, FusedActivationOnLinearDeclined) {
  auto op = buildLinear({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8, /*withBias=*/false,
                        builder.getStringAttr("silu"));
  EXPECT_FALSE(isDSEligible(op, {32, 4096}));
}

//===----------------------------------------------------------------------===//
// Weight geometry follows the device's DRAM bank count
//===----------------------------------------------------------------------===//

class DRAMShardedBankCountTest : public DSTestBase {
public:
  // The DS weight layout injected for operand 1.
  TTNNLayoutAttr weightReshardLayout(mlir::Operation *op) {
    std::vector<TTNNLayoutAttr> candidates =
        getRuleBook(op).getExtraInputReshardCandidates(op, /*operandIdx=*/1);
    EXPECT_EQ(candidates.size(), 1u);
    return candidates.empty() ? TTNNLayoutAttr() : candidates.front();
  }
};

// 12 banks: 128 N-tiles pad to 11 per bank.
TEST_F(DRAMShardedBankCountTest, WormholeUses12Banks) {
  initModule(ttcore::Arch::WormholeB0);
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8);

  TTNNLayoutAttr layout = weightReshardLayout(op);
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout.getGridShape(), llvm::ArrayRef<int64_t>({1, 12}));
  EXPECT_EQ(layout.getShardShape(), llvm::ArrayRef<int64_t>({128, 11}));
  EXPECT_EQ(layout.getBufferType(), BufferType::DRAM);
}

// 8 banks: 16 per bank, no padding. The count must come from the device: a
// layout over more banks than the part has is unallocatable, and nothing before
// silicon would notice.
TEST_F(DRAMShardedBankCountTest, BlackholeUses8Banks) {
  initModule(ttcore::Arch::Blackhole);
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8);

  TTNNLayoutAttr layout = weightReshardLayout(op);
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout.getGridShape(), llvm::ArrayRef<int64_t>({1, 8}));
  EXPECT_EQ(layout.getShardShape(), llvm::ArrayRef<int64_t>({128, 16}));
  EXPECT_EQ(layout.getBufferType(), BufferType::DRAM);
}

// The builders must force canonical placement: buildWithCanonicalCorePlacement
// only fills a null core range set, so a matching seed would keep its own.
TEST_F(DSTestBase, L1ShardedLayoutForcesCanonicalPlacement) {
  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(module.get());
  llvm::SmallVector<int64_t, 2> shape{kTile, 4096};
  auto elementType = ttcore::TileType::get(&context, {kTile, kTile},
                                           ttcore::DataType::BFloat16);

  // Matches the target except for a row-1 placement.
  auto offRow = CoreRangeSetAttr::get(
      &context,
      {CoreRangeAttr::get(&context, CoreCoordAttr::get(&context, 0, 1),
                          CoreCoordAttr::get(&context, 7, 1))});
  TTNNLayoutAttr seed = TTNNLayoutAttr::Builder(&context, shape, elementType)
                            .setBufferType(BufferType::L1)
                            .setMemoryLayout(TensorMemoryLayout::WidthSharded)
                            .setGridShape({1, 8})
                            .setCoreRangeSet(offRow)
                            .build();
  ASSERT_EQ(seed.getCoreRangeSet(), offRow);

  TTNNLayoutAttr out =
      buildWidthShardedLayout(&context, seed, shape, BufferType::L1,
                              /*numCores=*/8, deviceAttr);

  TTNNLayoutAttr canonical =
      TTNNLayoutAttr::Builder(&context, shape, elementType)
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayout::WidthSharded)
          .setGridShape({1, 8})
          .buildWithCanonicalCorePlacement(deviceAttr);

  EXPECT_EQ(out.getCoreRangeSet(), canonical.getCoreRangeSet());
  EXPECT_NE(out.getCoreRangeSet(), offRow);
}

} // namespace
