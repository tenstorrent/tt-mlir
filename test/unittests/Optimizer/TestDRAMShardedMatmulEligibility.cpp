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

// DRAM-sharded (DS) matmul eligibility and weight geometry, exercised through
// the matmul rule book.
//
// These build a real func.func because DS requires the weight to be const-eval
// traceable (valueTracesToConstantArgs), which means a function argument marked
// as a parameter -- the module-level ops the other strategy tests build cannot
// express that.
//
// The bias-free ttnn.linear case is only reachable here, not from lit: a
// bias-free ttir.linear canonicalizes into ttir.matmul (TTIROps.cpp), so no
// bias-free ttnn.linear survives the TTIR->TTNN pipeline. TTNN-level entry
// points (ttnn tracing, the shard advisor) are where it appears.
namespace {

constexpr int64_t kTile = 32;

// Shared scaffolding. Deliberately does not open a metal device: every path
// under test (eligibility, weight layout) needs only a DeviceAttr and the
// system descriptor.
//
// These are the fast, device-free check of the per-architecture layout the rule
// book derives; optimizer/dram_sharded_matmul_bank_count.mlir covers the same
// ground end-to-end through the pipeline.
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

  // Opens a func whose arg 1 (the weight) is marked as a parameter, and leaves
  // the builder positioned inside it.
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
                       ttcore::DataType weightDt) {
    auto actType = tensorOf(actShape, ttcore::DataType::BFloat16);
    auto weightType = tensorOf(weightShape, weightDt);
    auto outType = tensorOf(outShape, ttcore::DataType::BFloat16);
    mlir::Block *block = openFunc({actType, weightType}, outType);
    auto op = builder.create<MatmulOp>(
        builder.getUnknownLoc(), outType, block->getArgument(0),
        block->getArgument(1), /*transpose_a=*/false, /*transpose_b=*/false,
        /*matmul_program_config=*/mlir::Attribute(),
        /*activation=*/mlir::StringAttr());
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         op.getResult());
    return op;
  }

  LinearOp buildLinear(llvm::ArrayRef<int64_t> actShape,
                       llvm::ArrayRef<int64_t> weightShape,
                       llvm::ArrayRef<int64_t> outShape,
                       ttcore::DataType weightDt, bool withBias) {
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
        /*transpose_b=*/false, /*activation=*/mlir::StringAttr());
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

// Baseline: a decode-shaped bfp8 projection is DS-eligible.
TEST_F(DRAMShardedEligibilityTest, MatmulEligible) {
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_TRUE(isDSEligible(op, {32, 4096}));
}

// A bias-free ttnn.linear is the same computation the DS kernel implements, so
// it must be offered the DS config too. ttnn decoders write their projections
// as ttnn.linear, so this is the shape the path sees in practice.
TEST_F(DRAMShardedEligibilityTest, BiasFreeLinearEligible) {
  auto op = buildLinear({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8, /*withBias=*/false);
  EXPECT_TRUE(isDSEligible(op, {32, 4096}));
}

// A biased linear is declined. tt-metal's DS kernel does support a bias, but it
// reads it per DRAM bank at a bank-local offset, so the bias must be DRAM
// width-sharded across the same bank grid as the weight. Nothing produces such
// a layout for operand 2 yet, and a DRAM-interleaved bias would be read as a
// strided subset of itself -- wrong results rather than a crash, with no
// tt-metal validation to catch it.
TEST_F(DRAMShardedEligibilityTest, BiasedLinearDeclined) {
  auto op = buildLinear({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8, /*withBias=*/true);
  EXPECT_FALSE(isDSEligible(op, {32, 4096}));
}

// [1, 1, K, N] is the same matrix as [K, N]; ttnn models routinely hold
// projection weights that way.
TEST_F(DRAMShardedEligibilityTest, UnitBatchedWeightEligible) {
  auto op = buildMatmul({1, 1, 32, 4096}, {1, 1, 4096, 4096}, {1, 1, 32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_TRUE(isDSEligible(op, {1, 1, 32, 4096}));
}

// A non-unit batch dim is a real batched matmul, which tt-metal serves with a
// different program config that this path does not emit.
TEST_F(DRAMShardedEligibilityTest, BatchedWeightDeclined) {
  auto op = buildMatmul({1, 1, 32, 4096}, {2, 1, 4096, 4096}, {2, 1, 32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_FALSE(isDSEligible(op, {2, 1, 32, 4096}));
}

// A sub-tile decode batch is still one tile row, so it is eligible: tt-metal
// pads a 1..31-row activation up to one tile row and runs it.
TEST_F(DRAMShardedEligibilityTest, SubTileBatchEligible) {
  auto op = buildMatmul({1, 4096}, {4096, 4096}, {1, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_TRUE(isDSEligible(op, {1, 4096}));
}

// More than one tile row must be declined at compile time: tt-metal's
// TT_FATAL(M == 1) is an uncatchable abort, so deferring to it would crash on
// silicon rather than fall back to another config.
TEST_F(DRAMShardedEligibilityTest, MultiTileMDeclined) {
  auto op = buildMatmul({64, 4096}, {4096, 4096}, {64, 4096},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_FALSE(isDSEligible(op, {64, 4096}));
}

// bfp4/bfp8 only. bf16 is legal for tt-metal, but DS streams the weights out of
// DRAM, so bf16 moves 2x the bytes and the optimizer has no runtime estimate
// with which to rank it against 1D-mcast.
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

// Known limitation, pinned deliberately: the activation is width-sharded across
// a fixed 8 in0 cores, so K in tiles must be divisible by 8. 2880 is 90 tiles
// (the gpt-oss hidden size) and is declined. Deriving the core count from K
// would admit it -- 90 has several usable divisors -- but choosing among them
// is a cost question the optimizer cannot answer yet.
TEST_F(DRAMShardedEligibilityTest, KTilesNotDivisibleByIn0CoresDeclined) {
  auto op = buildMatmul({32, 2880}, {2880, 2880}, {32, 2880},
                        ttcore::DataType::BFP_BFloat8);
  EXPECT_FALSE(isDSEligible(op, {32, 2880}));
}

// The kill switch short-circuits the single choke point, so nothing downstream
// can reintroduce DS.
TEST_F(DRAMShardedEligibilityTest, DisableOptionSuppressesDS) {
  auto op = buildMatmul({32, 4096}, {4096, 4096}, {32, 4096},
                        ttcore::DataType::BFP_BFloat8);
  ASSERT_TRUE(isDSEligible(op, {32, 4096}));

  module->getOperation()->setAttr(utils::g_DisableDRAMShardedMatmulAttrName,
                                  builder.getBoolAttr(true));
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

// Wormhole exposes 12 DRAM banks, so N=4096 pads up to a multiple of 32*12=384
// (4224) and each bank holds 4224/12 = 352 = 11 tiles of weight width.
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

// Blackhole exposes 8. 32*8=256 already divides 4096, so there is no padding
// and each bank holds 4096/8 = 512 = 16 tiles.
//
// The count has to come from the device: a layout sharded across more banks
// than the part has is unallocatable (tensor creation aborts in
// get_dram_channel_from_logical_core), and validateTensorSpec skips the shard
// bounding-box check for DRAM buffers, so nothing before silicon would notice.
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

// The DS layout builders must land on canonical core placement whatever the
// layout they are seeded from carried. buildWithCanonicalCorePlacement only
// fills a *null* core range set, and Builder's setters each early-return when
// the value is unchanged, so a seed that already matches the target's buffer
// type, memory layout and grid would otherwise keep its own placement. No lit
// test reaches this: every DS operand starts DRAM-interleaved, so the memory
// layout always changes and invalidates the core range set on the way through.
TEST_F(DSTestBase, L1ShardedLayoutForcesCanonicalPlacement) {
  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(module.get());
  llvm::SmallVector<int64_t, 2> shape{kTile, 4096};
  auto elementType = ttcore::TileType::get(&context, {kTile, kTile},
                                           ttcore::DataType::BFloat16);

  // Same buffer type, memory layout and grid as the target, but deliberately
  // placed on row 1 rather than the canonical row 0.
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
      buildL1ShardedLayout(&context, seed, shape, /*numCores=*/8, deviceAttr);

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
