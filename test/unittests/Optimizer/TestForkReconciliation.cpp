// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTNN/Analysis/ForkConversionCost.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutPropagationTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseSet.h"

#include "gtest/gtest.h"

namespace mlir::tt::ttnn {
namespace {

class ForkReconciliationTest : public testing::Test {
protected:
  MLIRContext context;
  OpBuilder builder{&context};
  Block block;
  TTNNLayoutAttr dram;
  TTNNLayoutAttr l1;
  RankedTensorType tensorType;

  void SetUp() override {
    context.allowUnregisteredDialects();
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect, TTNNDialect>();
    dram = TTNNLayoutAttr::Builder(&context, {32, 32}, builder.getF32Type())
               .setBufferType(BufferType::DRAM)
               .setMemoryLayout(TensorMemoryLayout::Interleaved)
               .setGridShape({1, 1})
               .build();
    l1 = TTNNLayoutAttr::Builder(dram, {32, 32})
             .setBufferType(BufferType::L1)
             .build();
    tensorType = RankedTensorType::get({32, 32}, builder.getF32Type(), dram);
    builder.setInsertionPointToEnd(&block);
  }

  Operation *makeProducer(unsigned resultCount = 1) {
    OperationState state(builder.getUnknownLoc(), "test.producer");
    for (unsigned i = 0; i < resultCount; ++i) {
      state.addTypes(tensorType);
    }
    return builder.create(state);
  }

  Operation *makeConsumer(ValueRange operands) {
    OperationState state(builder.getUnknownLoc(), "test.consumer");
    state.addOperands(operands);
    return builder.create(state);
  }

  BeamCandidate makeCandidate(ArrayRef<size_t> indices,
                              ArrayRef<TTNNLayoutAttr> layouts) {
    BeamCandidate candidate;
    candidate.producerCandidateIndices.assign(indices.begin(), indices.end());
    candidate.inputLayouts.assign(layouts.begin(), layouts.end());
    return candidate;
  }
};

TEST_F(ForkReconciliationTest, MatchingFirstUseDoesNotHideLaterMismatch) {
  Operation *producer = makeProducer();
  Operation *consumer =
      makeConsumer({producer->getResult(0), producer->getResult(0)});
  BeamCandidate candidate = makeCandidate({0, 1}, {dram, l1});

  reconcileForkInputLayouts(producer, 0, consumer, candidate);

  ASSERT_EQ(candidate.reshardLayouts.size(), 1u);
  EXPECT_EQ(candidate.reshardLayouts.lookup(1), l1);
  EXPECT_FALSE(candidate.reshardLayouts.count(0));
}

TEST_F(ForkReconciliationTest, RepairsEveryRepeatedUse) {
  Operation *producer = makeProducer();
  Operation *consumer = makeConsumer(
      {producer->getResult(0), producer->getResult(0), producer->getResult(0)});
  BeamCandidate candidate = makeCandidate({1, 2, 1}, {dram, l1, dram});

  reconcileForkInputLayouts(producer, 0, consumer, candidate);

  ASSERT_EQ(candidate.reshardLayouts.size(), 3u);
  EXPECT_EQ(candidate.reshardLayouts.lookup(0), dram);
  EXPECT_EQ(candidate.reshardLayouts.lookup(1), l1);
  EXPECT_EQ(candidate.reshardLayouts.lookup(2), dram);
}

TEST_F(ForkReconciliationTest, DistinctResultsKeepTheirOwnTargetLayouts) {
  Operation *producer = makeProducer(2);
  Operation *consumer =
      makeConsumer({producer->getResult(0), producer->getResult(1)});
  BeamCandidate candidate = makeCandidate({1, 2}, {dram, l1});

  reconcileForkInputLayouts(producer, 0, consumer, candidate);

  ASSERT_EQ(candidate.reshardLayouts.size(), 2u);
  EXPECT_EQ(candidate.reshardLayouts.lookup(0), dram);
  EXPECT_EQ(candidate.reshardLayouts.lookup(1), l1);
}

TEST_F(ForkReconciliationTest, DoesNotPatchAnUnrelatedProducer) {
  Operation *producer = makeProducer();
  Operation *other = makeProducer();
  Operation *consumer = makeConsumer(
      {other->getResult(0), producer->getResult(0), producer->getResult(0)});
  BeamCandidate candidate = makeCandidate({1, 0, 1}, {l1, dram, l1});

  reconcileForkInputLayouts(producer, 0, consumer, candidate);

  ASSERT_EQ(candidate.reshardLayouts.size(), 1u);
  EXPECT_EQ(candidate.reshardLayouts.lookup(2), l1);
}

TEST_F(ForkReconciliationTest, SkipsNonTensorOperandsInCandidateIndices) {
  Value scalar =
      block.addArgument(builder.getI32Type(), builder.getUnknownLoc());
  Operation *producer = makeProducer();
  Operation *consumer = makeConsumer(
      {scalar, producer->getResult(0), scalar, producer->getResult(0)});
  BeamCandidate candidate = makeCandidate({0, 1}, {dram, l1});

  reconcileForkInputLayouts(producer, 0, consumer, candidate);

  ASSERT_EQ(candidate.reshardLayouts.size(), 1u);
  EXPECT_EQ(candidate.reshardLayouts.lookup(1), l1);
  EXPECT_FALSE(candidate.reshardLayouts.count(3));
}

TEST_F(ForkReconciliationTest, KeepsAnExistingReshardForAMatchingCandidate) {
  Operation *producer = makeProducer();
  Operation *consumer =
      makeConsumer({producer->getResult(0), producer->getResult(0)});
  BeamCandidate candidate = makeCandidate({0, 1}, {l1, dram});
  candidate.reshardLayouts[0] = l1;

  reconcileForkInputLayouts(producer, 0, consumer, candidate);

  ASSERT_EQ(candidate.reshardLayouts.size(), 2u);
  EXPECT_EQ(candidate.reshardLayouts.lookup(0), l1);
  EXPECT_EQ(candidate.reshardLayouts.lookup(1), dram);
}

TEST_F(ForkReconciliationTest, MatchingUsesNeedNoAdditionalReshard) {
  Operation *producer = makeProducer();
  Operation *consumer =
      makeConsumer({producer->getResult(0), producer->getResult(0)});
  BeamCandidate candidate = makeCandidate({2, 2}, {dram, dram});

  reconcileForkInputLayouts(producer, 2, consumer, candidate);

  EXPECT_TRUE(candidate.reshardLayouts.empty());
}

TEST_F(ForkReconciliationTest, CacheKeysPreserveRealResultAndTargetIdentity) {
  Operation *producer = makeProducer(2);
  using Key = std::pair<Value, Attribute>;
  ForkConversionCost<Key, llvm::SmallDenseSet<Key>> cost;
  ASSERT_TRUE(cost.add({producer->getResult(0), dram}, 8192, 8192));
  ASSERT_TRUE(cost.add({producer->getResult(0), dram}, 8192, 8192));
  ASSERT_TRUE(cost.add({producer->getResult(1), dram}, 8192, 8192));
  ASSERT_TRUE(cost.add({producer->getResult(0), l1}, 8192, 16384));
  EXPECT_EQ(cost.getConversionCount(), 3u);
  EXPECT_EQ(cost.getBytes(), 57344u);
}

} // namespace
} // namespace mlir::tt::ttnn
