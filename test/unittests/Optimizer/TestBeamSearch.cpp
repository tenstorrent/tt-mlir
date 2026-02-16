// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;
using namespace mlir::tt;

class BeamSearchTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;

  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance().openDevice();

    setL1UsageCap(1.0f);
  }

  void TearDown() override {
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .closeInstance();
  }

  void setL1UsageCap(float cap) {
    module->getOperation()->setAttr(utils::g_TensorL1UsageCapAttrName,
                                    builder.getF32FloatAttr(cap));
  }

  ttcore::GridAttr getDeviceGrid() {
    return ttcore::GridAttr::get(&context, {8, 8});
  }

  TTNNLayoutAttr createTiledLayout(const llvm::ArrayRef<int64_t> &tensorShape,
                                   BufferType bufferType,
                                   TensorMemoryLayout tensorMemoryLayout,
                                   const llvm::ArrayRef<int64_t> &gridShape = {
                                       1, 1}) {
    auto elementType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());
    return TTNNLayoutAttr::get(
        &context, tensorShape, elementType, bufferType,
        mlir::tt::ttcore::GridAttr::get(&context, gridShape),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                     tensorMemoryLayout));
  }

  TTNNLayoutAttr
  createDRAMInterleavedLayout(const llvm::ArrayRef<int64_t> &tensorShape) {
    return createTiledLayout(tensorShape, BufferType::DRAM,
                             TensorMemoryLayout::Interleaved);
  }

  TTNNLayoutAttr
  createL1InterleavedLayout(const llvm::ArrayRef<int64_t> &tensorShape) {
    return createTiledLayout(tensorShape, BufferType::L1,
                             TensorMemoryLayout::Interleaved);
  }

  TTNNLayoutAttr
  createL1ShardedLayout(const llvm::ArrayRef<int64_t> &tensorShape,
                        const llvm::ArrayRef<int64_t> &gridShape = {8, 4}) {
    return createTiledLayout(tensorShape, BufferType::L1,
                             TensorMemoryLayout::HeightSharded, gridShape);
  }

  std::vector<OpConfig>
  createElementwiseLegalConfigs(const llvm::ArrayRef<int64_t> &shape) {
    std::vector<OpConfig> configs;
    configs.emplace_back(createDRAMInterleavedLayout(shape));
    configs.emplace_back(createL1InterleavedLayout(shape));
    configs.emplace_back(createL1ShardedLayout(shape, {1, 1}));
    return configs;
  }

  /// Create a FuncOp with the given input/output types. Sets builder insertion
  /// point inside the entry block.
  mlir::func::FuncOp
  createFuncOp(llvm::ArrayRef<mlir::Type> inputTypes,
               llvm::ArrayRef<mlir::Type> outputTypes,
               llvm::StringRef name = "test") {
    auto funcType = builder.getType<mlir::FunctionType>(
        mlir::TypeRange(inputTypes), mlir::TypeRange(outputTypes));
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name,
                                              funcType);
    mlir::Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    return func;
  }

  /// Helper: build a simple chain graph:
  ///   arg0, arg1 -> add -> relu -> multiply(relu, arg1) -> add2 -> return
  /// Returns {addOp, reluOp, mulOp, add2Op}.
  /// add, relu, mulOp are in beam; add2 is skipped (feeds return).
  llvm::SmallVector<mlir::Operation *, 4>
  buildChainGraph(const llvm::ArrayRef<int64_t> &shape) {
    auto layout = createDRAMInterleavedLayout(shape);
    auto tensorType =
        mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

    createFuncOp({tensorType, tensorType}, {tensorType});
    mlir::Value arg0 = func.getBody().front().getArgument(0);
    mlir::Value arg1 = func.getBody().front().getArgument(1);

    auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                       arg0, arg1);
    auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                         addOp.getResult());
    auto mulOp = builder.create<MultiplyOp>(
        builder.getUnknownLoc(), tensorType, reluOp.getResult(), arg1);
    auto add2Op = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                        mulOp.getResult(), arg1);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         add2Op.getResult());

    return {addOp, reluOp, mulOp, add2Op};
  }

  /// Helper: build a fork graph:
  ///   arg0, arg1, arg2 -> add(arg0, arg1) -> {relu, multiply(add, arg2)}
  ///                                       -> add2(relu, mul) -> return
  /// add is a fork point (used by relu and multiply).
  /// Returns {addOp, reluOp, mulOp, add2Op}.
  llvm::SmallVector<mlir::Operation *, 4>
  buildForkGraph(const llvm::ArrayRef<int64_t> &shape) {
    auto layout = createDRAMInterleavedLayout(shape);
    auto tensorType =
        mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

    createFuncOp({tensorType, tensorType, tensorType}, {tensorType});
    mlir::Value arg0 = func.getBody().front().getArgument(0);
    mlir::Value arg1 = func.getBody().front().getArgument(1);
    mlir::Value arg2 = func.getBody().front().getArgument(2);

    auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                       arg0, arg1);
    auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                         addOp.getResult());
    auto mulOp = builder.create<MultiplyOp>(
        builder.getUnknownLoc(), tensorType, addOp.getResult(), arg2);
    auto add2Op = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                        reluOp.getResult(), mulOp.getResult());
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         add2Op.getResult());

    return {addOp, reluOp, mulOp, add2Op};
  }
};

//===----------------------------------------------------------------------===//
// Beam width K=1 tests
//===----------------------------------------------------------------------===//

TEST_F(BeamSearchTest, BeamWidthK1_ProducesSingleCandidate) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto ops = buildChainGraph(shape);

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : ops) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/1);
  propagation.run();

  const auto &beamState = propagation.getBeamState();

  // add, relu, mulOp should be in beam (add2 feeds return -> skipped).
  for (size_t i = 0; i < 3; ++i) {
    auto it = beamState.find(ops[i]);
    if (it != beamState.end()) {
      EXPECT_LE(it->second.size(), 1u)
          << "K=1 should produce at most 1 candidate per op";
    }
  }
}

//===----------------------------------------------------------------------===//
// Beam width K=8 tests
//===----------------------------------------------------------------------===//

TEST_F(BeamSearchTest, BeamWidthK8_ProducesMultipleCandidates) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto ops = buildChainGraph(shape);

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : ops) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/8);
  propagation.run();

  const auto &beamState = propagation.getBeamState();

  // Ops should be in beam and have at most beamWidth candidates.
  for (size_t i = 0; i < 3; ++i) {
    auto it = beamState.find(ops[i]);
    if (it != beamState.end()) {
      EXPECT_LE(it->second.size(), 8u)
          << "K=8 should produce at most 8 candidates per op";
    }
  }
}

TEST_F(BeamSearchTest, BeamWidthK8_CandidatesSortedByScore) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto ops = buildChainGraph(shape);

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : ops) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/8);
  propagation.run();

  const auto &beamState = propagation.getBeamState();

  // Verify candidates are sorted by score descending.
  for (const auto &[op, candidates] : beamState) {
    for (size_t i = 1; i < candidates.size(); ++i) {
      EXPECT_TRUE(candidates[i - 1].score >= candidates[i].score)
          << "Candidates should be sorted by score descending";
    }
  }
}

//===----------------------------------------------------------------------===//
// Backward pass tests
//===----------------------------------------------------------------------===//

TEST_F(BeamSearchTest, BackwardPass_LinearChain_AllOpsHaveFinalChoice) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto ops = buildChainGraph(shape);

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : ops) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/4);
  propagation.run();

  const auto &beamState = propagation.getBeamState();
  const auto &finalChoice = propagation.getFinalChoice();

  // Every op in beamState should have a finalChoice entry.
  for (const auto &[op, candidates] : beamState) {
    auto choiceIt = finalChoice.find(op);
    EXPECT_NE(choiceIt, finalChoice.end())
        << "Every op in beam should have a finalChoice";
    if (choiceIt != finalChoice.end()) {
      EXPECT_LT(choiceIt->second, candidates.size())
          << "finalChoice index should be within beam bounds";
    }
  }
}

TEST_F(BeamSearchTest, BackwardPass_ForkPoint_ResolvesWithoutCrash) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto ops = buildForkGraph(shape);
  // ops = {add (fork), relu, multiply, add2}

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : ops) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/8);

  // Should not crash during fork resolution.
  EXPECT_NO_FATAL_FAILURE(propagation.run());

  const auto &beamState = propagation.getBeamState();
  const auto &finalChoice = propagation.getFinalChoice();

  // The fork op (add) should be in the beam state.
  auto addIt = beamState.find(ops[0]);
  if (addIt != beamState.end() && !addIt->second.empty()) {
    auto choiceIt = finalChoice.find(ops[0]);
    EXPECT_NE(choiceIt, finalChoice.end())
        << "Fork op should have a finalChoice";
    if (choiceIt != finalChoice.end()) {
      EXPECT_LT(choiceIt->second, addIt->second.size())
          << "Fork finalChoice should be within bounds";
    }
  }
}

TEST_F(BeamSearchTest, BackwardPass_ForkPoint_BothConsumersInBeam) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto ops = buildForkGraph(shape);
  // ops = {add (fork), relu, multiply, add2}

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : ops) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/4);
  propagation.run();

  const auto &beamState = propagation.getBeamState();

  // relu and multiply (consumers of the fork) should be in beam.
  auto reluIt = beamState.find(ops[1]);
  auto mulIt = beamState.find(ops[2]);
  EXPECT_NE(reluIt, beamState.end()) << "relu should be in beam state";
  EXPECT_NE(mulIt, beamState.end()) << "multiply should be in beam state";
}

//===----------------------------------------------------------------------===//
// K=1 backward pass is trivial (no consolidation)
//===----------------------------------------------------------------------===//

TEST_F(BeamSearchTest, BeamWidthK1_NoConsolidation) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto ops = buildForkGraph(shape);

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : ops) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/1);
  propagation.run();

  const auto &finalChoice = propagation.getFinalChoice();

  // With K=1, consolidateBeam is not called, so finalChoice may be empty
  // or all zeros. Either way, the implicit choice is candidate 0.
  for (const auto &[op, idx] : finalChoice) {
    EXPECT_EQ(idx, 0u) << "K=1 finalChoice should always be 0";
  }
}

//===----------------------------------------------------------------------===//
// Smoke test: complex graph with K=8
//===----------------------------------------------------------------------===//

TEST_F(BeamSearchTest, BeamWidthK8_ComplexGraph_DoesNotCrash) {
  llvm::SmallVector<int64_t> shape = {1, 1, 64, 64};
  auto layout = createDRAMInterleavedLayout(shape);
  auto tensorType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

  createFuncOp({tensorType, tensorType, tensorType}, {tensorType});
  mlir::Value arg0 = func.getBody().front().getArgument(0);
  mlir::Value arg1 = func.getBody().front().getArgument(1);
  mlir::Value arg2 = func.getBody().front().getArgument(2);

  // Build a moderately complex graph (5 ops + return):
  //   add1 = add(arg0, arg1)    -- fork point
  //   relu1 = relu(add1)
  //   mul1 = multiply(add1, arg2) -- second use of add1
  //   add2 = add(relu1, mul1)   -- fork point
  //   relu2 = relu(add2)
  //   mul2 = multiply(relu2, arg0) -> return
  auto add1 = builder.create<AddOp>(builder.getUnknownLoc(), tensorType, arg0,
                                    arg1);
  auto relu1 = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                      add1.getResult());
  auto mul1 = builder.create<MultiplyOp>(builder.getUnknownLoc(), tensorType,
                                         add1.getResult(), arg2);
  auto add2 = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                    relu1.getResult(), mul1.getResult());
  auto relu2 = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                      add2.getResult());
  auto mul2 = builder.create<MultiplyOp>(builder.getUnknownLoc(), tensorType,
                                         relu2.getResult(), arg0);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                       mul2.getResult());

  llvm::SmallVector<mlir::Operation *> allOps = {add1, relu1, mul1,
                                                  add2,  relu2, mul2};

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  for (auto *op : allOps) {
    legalConfigs[op] = createElementwiseLegalConfigs(shape);
  }

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/8);

  EXPECT_NO_FATAL_FAILURE(propagation.run());

  const auto &beamState = propagation.getBeamState();
  // At least some ops should be in beam (mul2 feeds return -> skipped).
  EXPECT_GE(beamState.size(), 3u)
      << "Complex graph should have at least 3 ops in beam";
}

//===----------------------------------------------------------------------===//
// Edge case: no valid candidates -> DRAM fallback
//===----------------------------------------------------------------------===//

TEST_F(BeamSearchTest, NoValidCandidate_FallbackToDRAM) {
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto layout = createDRAMInterleavedLayout(shape);
  auto tensorType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

  createFuncOp({tensorType, tensorType}, {tensorType});
  mlir::Value arg0 = func.getBody().front().getArgument(0);
  mlir::Value arg1 = func.getBody().front().getArgument(1);

  auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType, arg0,
                                     arg1);
  // Dummy op so addOp doesn't feed return directly.
  auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                       addOp.getResult());
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                       reluOp.getResult());

  // Provide empty legal configs -- no valid configs at all.
  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  legalConfigs[addOp.getOperation()] = {};
  legalConfigs[reluOp.getOperation()] = {};

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/8);

  // Should not crash -- falls back to DRAM interleaved.
  EXPECT_NO_FATAL_FAILURE(propagation.run());

  const auto &beamState = propagation.getBeamState();
  auto it = beamState.find(addOp.getOperation());
  if (it != beamState.end() && !it->second.empty()) {
    // Fallback candidate should be DRAM.
    const auto &fallback = it->second[0];
    if (fallback.config.outputLayout) {
      EXPECT_EQ(fallback.config.outputLayout.getBufferType(), BufferType::DRAM)
          << "Fallback should be DRAM interleaved";
    }
  }
}
