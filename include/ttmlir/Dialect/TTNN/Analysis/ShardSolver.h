// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDSOLVER_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDSOLVER_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include <algorithm>
#include <bitset>
#include <llvm/ADT/StringMap.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlir::tt::ttnn {

struct OpL1MemSpec;

struct ShardSolverSolution {
  llvm::DenseMap<Operation *, tt::LayoutAttr> selectedOpLayout;
  std::unordered_set<Edge> memReconfigEdges;

  ShardSolverSolution(
      const llvm::DenseMap<Operation *, tt::LayoutAttr> &selectedOpLayout,
      const std::unordered_set<Edge> &memReconfigEdges)
      : selectedOpLayout(selectedOpLayout), memReconfigEdges(memReconfigEdges) {
  }
};

// Reconcile adjacent shard specs by using constraints on top of legal layouts.
// Generate reshard specs where needed.
// Provides a valid solution to the shard chain.
//
class ShardSolver {
private:
  static constexpr size_t kNumBitsetBits = 64;
  using Bitset = std::bitset<kNumBitsetBits>;
  static Bitset kBitsetAll;
  static constexpr Bitset kBitsetNone = Bitset{};

public:
  struct RemainingLayoutAttrs {
    class Iterator {
      std::uint64_t i = 0;
      std::vector<tt::LayoutAttr> const *p = nullptr;
      Bitset mask = 0;

    private:
      void nextValid() {
        if (mask == Bitset{}) {
          i = p->size();
          return;
        }

        while (mask.any() and not mask[i]) {
          ++i;
        }

        mask.reset(i);
      }

    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = const tt::LayoutAttr;
      using difference_type = const tt::LayoutAttr;
      using pointer = const tt::LayoutAttr *;
      using reference = const tt::LayoutAttr &;

      Iterator(std::vector<tt::LayoutAttr> const *p, const Bitset &mask,
               std::uint64_t i = 0)
          : i(i), p(p), mask(mask) {
        nextValid();
      }

      Iterator &operator++() {
        nextValid();
        return *this;
      }

      Iterator operator++(int) {
        auto r = *this;
        nextValid();
        return r;
      }

      bool operator==(Iterator other) const {
        return (p == other.p) and (i == other.i);
      }
      bool operator!=(Iterator other) const { return not(*this == other); }
      reference operator*() const { return (*p)[i]; }
    };

    RemainingLayoutAttrs(std::vector<tt::LayoutAttr> const &p,
                         const Bitset &mask)
        : p(&p), mask(mask) {}

    Iterator begin() const { return Iterator(p, mask); }
    Iterator end() const {
      return Iterator(p, 0, std::min(kNumBitsetBits, p->size()));
    }
    size_t size() const { return mask.count(); }

    std::vector<tt::LayoutAttr> const *p = nullptr;
    Bitset mask = 0;
  };

  ShardSolverSolution const finish();

private:
  static Bitset bitset(std::uint64_t bit) {
    Bitset b;
    b.set(bit);
    return b;
  }
  // is `a` a subset of `b`
  static bool isSubset(const Bitset &a, const Bitset &b) {
    return a == (a & b);
  }

  using PathSetId = int;
  using BitsetId = int;

  class OperationPathsProcessor {
  public:
    void addOp(Operation *operation) {
      if (controlSet.count(operation) == 0) {
        queue.push_back(operation);
        controlSet.insert(operation);
      }
    }

    void process(ShardSolver *shardSolver) {
      while (!queue.empty()) {
        Operation *operation = queue.back();
        queue.pop_back();
        controlSet.erase(operation);

        auto operandPathSets = shardSolver->getOperandPathSetsPts(operation);
        auto userPathSets = shardSolver->getUserPathSetsPts(operation);
        for (auto *path_set : operandPathSets) {
          path_set->updateOperationProcessor(shardSolver->bitsets, this);
        }
        for (auto *path_set : userPathSets) {
          path_set->updateOperationProcessor(shardSolver->bitsets, this);
        }
      }
    }

  private:
    llvm::SmallVector<Operation *> queue;
    llvm::DenseSet<Operation *> controlSet;
  };

  struct Path {
    std::uint16_t producerId = 0;
    std::uint16_t consumerId = 0;

    Path() = default;
    Path(std::uint16_t producerId, std::uint16_t consumerId)
        : producerId(producerId), consumerId(consumerId) {}
  };

  class PathSet {
  public:
    using Paths = llvm::SmallVector<Path, 16>;

    PathSet(BitsetId producerSetId, BitsetId consumerSetId,
            Operation *producerOperation, Operation *consumerOperation,
            Paths const &paths)
        : producerSetId(producerSetId), consumerSetId(consumerSetId),
          producerOperation(producerOperation),
          consumerOperation(consumerOperation), paths(paths) {}

    bool empty(const std::vector<Bitset> &bitsets) const {
      return paths.empty() or (bitsets[producerSetId] == 0) or
             (bitsets[consumerSetId] == 0);
    }

    bool update(std::vector<Bitset> &bitsets) {
      Bitset validProducerSet = 0;
      Bitset validConsumerSet = 0;
      Bitset producer = bitsets[producerSetId];
      Bitset consumer = bitsets[consumerSetId];

      for (size_t i = 0; i < paths.size(); i++) {
        Path const &path = paths[i];
        if (consumer[path.consumerId] and producer[path.producerId]) {
          validProducerSet.set(path.producerId);
          validConsumerSet.set(path.consumerId);
        } else {
          paths[i] = paths.back();
          paths.pop_back();
          i--;
        }
      }

      bool isProducerSub = isSubset(producer, validProducerSet);
      bool isConsumerSub = isSubset(consumer, validConsumerSet);
      bool unchanged = isProducerSub and isConsumerSub;

      if (!unchanged) {
        bitsets[producerSetId] &= validProducerSet;
        bitsets[consumerSetId] &= validConsumerSet;
      }

      return not unchanged;
    }

    void
    updateOperationProcessor(std::vector<Bitset> &bitsets,
                             OperationPathsProcessor *operation_processor) {
      Bitset validProducerSet = 0;
      Bitset validConsumerSet = 0;
      Bitset producer = bitsets[producerSetId];
      Bitset consumer = bitsets[consumerSetId];
      for (size_t i = 0; i < paths.size(); i++) {
        Path const &path = paths[i];
        if (consumer[path.consumerId] and producer[path.producerId]) {
          validProducerSet.set(path.producerId);
          validConsumerSet.set(path.consumerId);
        } else {
          paths[i] = paths.back();
          paths.pop_back();
          i--;
        }
      }

      if (!isSubset(producer, validProducerSet)) {
        operation_processor->addOp(consumerOperation);
        operation_processor->addOp(producerOperation);
        bitsets[producerSetId] &= validProducerSet;
      }

      if (!isSubset(consumer, validConsumerSet)) {
        operation_processor->addOp(producerOperation);
        operation_processor->addOp(consumerOperation);
        bitsets[consumerSetId] &= validConsumerSet;
      }
    }

    Operation *getProducerOp() const { return producerOperation; }
    Operation *getConsumerOp() const { return consumerOperation; }

  private:
  private:
    BitsetId producerSetId = -1;
    BitsetId consumerSetId = -1;
    Operation *producerOperation = nullptr;
    Operation *consumerOperation = nullptr;
    Paths paths;
  };

  const std::vector<tt::LayoutAttr> &
  getLegalLayouts(Operation *operation) const;
  void reset();

  PathSet *getPathSetPt(const Edge &edge);
  SmallVector<PathSet *> getOperandPathSetsPts(Operation *operation);
  SmallVector<PathSet *> getUserPathSetsPts(Operation *operation);

  void handleNoPathsLeftOnUpdate(bool invokedBySet);
  void updateSolver(Operation *root, bool expand_root = true,
                    bool invokedBySet = false);

  Bitset *getBitset(Operation *op);
  Bitset const *getBitset(Operation *op) const;
  Bitset *getOrInsertBitset(Operation *op, const Bitset &init);

  void resolve();
  bool resolveStep();
  void insertReshard(const Edge &edge);
  void addOperandsAndUsers(Operation *op, std::vector<Operation *> &needsUpdate,
                           Operation *ignoreOp = nullptr);

  void preprocessFirstOp();
  bool checkShardCompatible(Operation *producerOp,
                            tt::LayoutAttr const &producerLayout,
                            Operation *consumerOp,
                            tt::LayoutAttr const &consumerLayout) const;

public:
  ShardSolver(
      const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalLayouts,
      const std::vector<OpL1MemSpec> &shardSpecs,
      const llvm::DenseSet<Operation *> &shardedOps,
      const unsigned usableL1CacheSize,
      const std::unordered_set<Edge> &overrideReshardEdges);
  RemainingLayoutAttrs at(Operation *operation) const;
  void set(Operation *operation, tt::LayoutAttr const &layout);
  static bool supportsInterleavedInputShardedOutput(Operation *op);

private:
  const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> *legalLayouts;
  const std::vector<OpL1MemSpec> *shardSpecs;
  const llvm::DenseSet<Operation *> *shardedOps;
  unsigned usableL1CacheSize;
  DeviceAttr deviceAttr;

  llvm::DenseMap<Operation *, std::vector<Edge>> operandOpEdges;
  llvm::DenseMap<Operation *, std::vector<Edge>> userOpEdges;
  std::vector<PathSet> pathSets;
  std::vector<Bitset> bitsets;
  std::unordered_map<Edge, PathSetId> pathSetIds;
  std::unordered_map<Operation *, BitsetId> bitsetIds;

  llvm::DenseMap<Operation *, tt::LayoutAttr> selectedOpLayout;
  std::unordered_set<Edge> memReconfigEdges;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDSOLVER_H
