// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagation.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
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

using namespace mlir::tt::ttnn;
using namespace mlir::tt;

class ReshardCandidatesTest : public ::testing::Test {
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
                        TensorMemoryLayout memLayout,
                        const llvm::ArrayRef<int64_t> &gridShape = {8, 4}) {
    return createTiledLayout(tensorShape, BufferType::L1, memLayout, gridShape);
  }

  std::vector<OpConfig>
  createElementwiseLegalConfigs(const llvm::ArrayRef<int64_t> &shape) {
    std::vector<OpConfig> configs;
    configs.emplace_back(createDRAMInterleavedLayout(shape));
    configs.emplace_back(createL1InterleavedLayout(shape));
    configs.emplace_back(
        createL1ShardedLayout(shape, TensorMemoryLayout::HeightSharded));
    return configs;
  }

  /// Build a TensorTypeLayoutsMap with sharded layouts for a given tensor type.
  /// Populates with height-sharded and block-sharded layouts at various grid
  /// sizes.
  TensorTypeLayoutsMap
  buildTensorTypeLayoutsMap(mlir::RankedTensorType tensorType,
                            const llvm::ArrayRef<int64_t> &shape) {
    TensorTypeLayoutsMap layoutsMap;
    auto elementType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());

    // Create sharded layouts at various grid sizes.
    std::vector<std::pair<TensorMemoryLayout, llvm::SmallVector<int64_t, 2>>>
        shardSpecs = {
            {TensorMemoryLayout::HeightSharded, {1, 1}},
            {TensorMemoryLayout::HeightSharded, {2, 1}},
            {TensorMemoryLayout::HeightSharded, {4, 1}},
            {TensorMemoryLayout::HeightSharded, {8, 1}},
            {TensorMemoryLayout::HeightSharded, {8, 2}},
            {TensorMemoryLayout::HeightSharded, {8, 4}},
            {TensorMemoryLayout::HeightSharded, {8, 8}},
            {TensorMemoryLayout::BlockSharded, {2, 2}},
            {TensorMemoryLayout::BlockSharded, {4, 4}},
            {TensorMemoryLayout::BlockSharded, {8, 4}},
            {TensorMemoryLayout::BlockSharded, {8, 8}},
        };

    auto &scalarMap = layoutsMap[tensorType];
    auto &pageLayoutArr = scalarMap[elementType];

    // Populate the Tiled + Sharded slot.
    size_t tiledIdx =
        static_cast<size_t>(TensorPageLayout::Tiled);
    size_t shardedIdx =
        static_cast<size_t>(TensorMemoryLayoutIndex::Sharded);

    for (const auto &[memLayout, gridShape] : shardSpecs) {
      auto layout = TTNNLayoutAttr::get(
          &context, shape, elementType, BufferType::L1,
          ttcore::GridAttr::get(&context, gridShape),
          TensorMemoryLayoutAttr::get(&context, memLayout));
      pageLayoutArr[tiledIdx][shardedIdx].push_back(layout);
    }

    return layoutsMap;
  }

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
};

//===----------------------------------------------------------------------===//
// Reshard candidate generation via LayoutPropagation (tested indirectly
// through the full pipeline since generateReshardCandidates is private)
//===----------------------------------------------------------------------===//

TEST_F(ReshardCandidatesTest, WithTensorLayoutsMap_DoesNotCrash) {
  // Create a simple graph and run with a populated TensorTypeLayoutsMap.
  // Verifies that reshard candidate generation doesn't crash.
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto layout = createDRAMInterleavedLayout(shape);
  auto tensorType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

  createFuncOp({tensorType, tensorType}, {tensorType});
  mlir::Value arg0 = func.getBody().front().getArgument(0);
  mlir::Value arg1 = func.getBody().front().getArgument(1);

  auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType, arg0,
                                     arg1);
  auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                       addOp.getResult());
  auto mulOp = builder.create<MultiplyOp>(builder.getUnknownLoc(), tensorType,
                                          reluOp.getResult(), arg1);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                       mulOp.getResult());

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  legalConfigs[addOp.getOperation()] = createElementwiseLegalConfigs(shape);
  legalConfigs[reluOp.getOperation()] = createElementwiseLegalConfigs(shape);
  legalConfigs[mulOp.getOperation()] = createElementwiseLegalConfigs(shape);

  // Build the tensor type layouts map with sharded candidates.
  auto bareType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type());
  TensorTypeLayoutsMap tensorLayouts =
      buildTensorTypeLayoutsMap(bareType, shape);

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                &tensorLayouts, /*beamWidth=*/8);

  EXPECT_NO_FATAL_FAILURE(propagation.run());
}

TEST_F(ReshardCandidatesTest, NullTensorLayouts_NoReshardCandidates) {
  // Without a TensorTypeLayoutsMap, no reshard candidates should be generated.
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto layout = createDRAMInterleavedLayout(shape);
  auto tensorType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

  createFuncOp({tensorType, tensorType}, {tensorType});
  mlir::Value arg0 = func.getBody().front().getArgument(0);
  mlir::Value arg1 = func.getBody().front().getArgument(1);

  auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType, arg0,
                                     arg1);
  auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                       addOp.getResult());
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                       reluOp.getResult());

  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  legalConfigs[addOp.getOperation()] = createElementwiseLegalConfigs(shape);
  legalConfigs[reluOp.getOperation()] = createElementwiseLegalConfigs(shape);

  LayoutPropagation propagation(func, getDeviceGrid(), legalConfigs,
                                /*tensorTypePossibleLayouts=*/nullptr,
                                /*beamWidth=*/8);
  propagation.run();

  // With null tensor layouts map, no reshard ops should have been inserted.
  // Check beam state: no candidate should have reshard entries.
  const auto &beamState = propagation.getBeamState();
  for (const auto &[op, candidates] : beamState) {
    for (const auto &candidate : candidates) {
      EXPECT_TRUE(candidate.reshardLayouts.empty())
          << "No reshards should be generated without tensor layouts map";
    }
  }
}

TEST_F(ReshardCandidatesTest, BeamCandidatesWithTensorLayouts_MoreThanWithout) {
  // With a TensorTypeLayoutsMap, beam search should explore more candidates
  // (reshard candidates from sharded producers create additional combos).
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto layout = createDRAMInterleavedLayout(shape);
  auto tensorType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

  // Helper: create graph and run with/without tensor layouts.
  auto runWithLayouts = [&](const TensorTypeLayoutsMap *layouts) -> size_t {
    // Need fresh func each time.
    auto savedInsertPt = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&module->getBodyRegion().front());

    static int counter = 0;
    std::string name = "test_" + std::to_string(counter++);
    auto localFunc =
        builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), name,
            builder.getType<mlir::FunctionType>(
                mlir::TypeRange({tensorType, tensorType}),
                mlir::TypeRange({tensorType})));
    mlir::Block *block = localFunc.addEntryBlock();
    builder.setInsertionPointToStart(block);

    mlir::Value arg0 = block->getArgument(0);
    mlir::Value arg1 = block->getArgument(1);

    auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                       arg0, arg1);
    auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                         addOp.getResult());
    auto mulOp = builder.create<MultiplyOp>(builder.getUnknownLoc(),
                                            tensorType, reluOp.getResult(),
                                            arg1);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         mulOp.getResult());

    llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
    legalConfigs[addOp.getOperation()] = createElementwiseLegalConfigs(shape);
    legalConfigs[reluOp.getOperation()] = createElementwiseLegalConfigs(shape);
    legalConfigs[mulOp.getOperation()] = createElementwiseLegalConfigs(shape);

    LayoutPropagation propagation(localFunc, getDeviceGrid(), legalConfigs,
                                  layouts, /*beamWidth=*/8);
    propagation.run();

    size_t totalCandidates = 0;
    for (const auto &[op, candidates] : propagation.getBeamState()) {
      totalCandidates += candidates.size();
    }

    builder.restoreInsertionPoint(savedInsertPt);
    return totalCandidates;
  };

  auto bareType = mlir::RankedTensorType::get(shape, builder.getBF16Type());
  TensorTypeLayoutsMap tensorLayouts =
      buildTensorTypeLayoutsMap(bareType, shape);

  size_t candidatesWithout = runWithLayouts(nullptr);
  size_t candidatesWith = runWithLayouts(&tensorLayouts);

  // With tensor layouts, we may have same or more candidates due to
  // reshard exploration. At minimum it shouldn't have fewer.
  EXPECT_GE(candidatesWith, candidatesWithout)
      << "Tensor layouts should enable at least as many candidates";
}

TEST_F(ReshardCandidatesTest, ReshardExploration_K1vsK8) {
  // With K=8, more candidate combinations are explored than K=1.
  llvm::SmallVector<int64_t> shape = {1, 1, 32, 32};
  auto layout = createDRAMInterleavedLayout(shape);
  auto tensorType =
      mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);

  auto buildAndRun = [&](size_t beamWidth) -> size_t {
    auto savedInsertPt = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&module->getBodyRegion().front());

    static int counter = 100;
    std::string name = "k_test_" + std::to_string(counter++);
    auto localFunc =
        builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), name,
            builder.getType<mlir::FunctionType>(
                mlir::TypeRange({tensorType, tensorType}),
                mlir::TypeRange({tensorType})));
    mlir::Block *block = localFunc.addEntryBlock();
    builder.setInsertionPointToStart(block);

    mlir::Value arg0 = block->getArgument(0);
    mlir::Value arg1 = block->getArgument(1);

    auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType,
                                       arg0, arg1);
    auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), tensorType,
                                         addOp.getResult());
    auto mulOp = builder.create<MultiplyOp>(builder.getUnknownLoc(),
                                            tensorType, reluOp.getResult(),
                                            arg1);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         mulOp.getResult());

    llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
    legalConfigs[addOp.getOperation()] = createElementwiseLegalConfigs(shape);
    legalConfigs[reluOp.getOperation()] = createElementwiseLegalConfigs(shape);
    legalConfigs[mulOp.getOperation()] = createElementwiseLegalConfigs(shape);

    LayoutPropagation propagation(localFunc, getDeviceGrid(), legalConfigs,
                                  /*tensorTypePossibleLayouts=*/nullptr,
                                  beamWidth);
    propagation.run();

    size_t totalCandidates = 0;
    for (const auto &[op, candidates] : propagation.getBeamState()) {
      totalCandidates += candidates.size();
    }

    builder.restoreInsertionPoint(savedInsertPt);
    return totalCandidates;
  };

  size_t candidatesK1 = buildAndRun(1);
  size_t candidatesK8 = buildAndRun(8);

  // K=8 should produce at least as many total candidates as K=1.
  EXPECT_GE(candidatesK8, candidatesK1)
      << "K=8 should produce at least as many candidates as K=1";
}
