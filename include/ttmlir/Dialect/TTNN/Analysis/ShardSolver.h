// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDSOLVER_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDSOLVER_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemReconfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <bitset>
#include <unordered_map>
#include <vector>

namespace mlir::tt::ttnn {

struct OpL1MemSpec;

struct ShardSolverSolution {
  llvm::DenseMap<Operation *, OpConfig> selectedOpConfig;
  llvm::DenseMap<Edge, MemReconfigEntry> memReconfigEntryMap;

  ShardSolverSolution(
      const llvm::DenseMap<Operation *, OpConfig> &selectedOpConfig,
      const llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap)
      : selectedOpConfig(selectedOpConfig),
        memReconfigEntryMap(memReconfigEntryMap) {}
};

// Reconcile adjacent shard specs by using constraints on top of legal op
// configs. Generate reshard specs where needed. Provides a valid solution to
// the shard chain.
//
class ShardSolver {
private:
  static constexpr size_t kNumBitsetBits = 64;
  using Bitset = std::bitset<kNumBitsetBits>;
  static Bitset kBitsetAll;
  static constexpr Bitset kBitsetNone = Bitset{};

public:
  struct RemainingConfigAttrs {
    class Iterator {
      std::uint64_t i = 0;
      const std::vector<OpConfig> *p = nullptr;
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
      using value_type = const OpConfig;
      using difference_type = const OpConfig;
      using pointer = const OpConfig *;
      using reference = const OpConfig &;

      Iterator(const std::vector<OpConfig> *p, const Bitset &mask,
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
      pointer operator->() const { return get(); }
      pointer get() const { return &(*p)[i]; }
      std::uint64_t index() const { return i; }
    };

    RemainingConfigAttrs(const std::vector<OpConfig> &p, const Bitset &mask)
        : p(&p), mask(mask) {}

    Iterator begin() const { return Iterator(p, mask); }
    Iterator end() const {
      return Iterator(p, 0, std::min(kNumBitsetBits, p->size()));
    }
    size_t size() const { return mask.count(); }

    const std::vector<OpConfig> *p = nullptr;
    Bitset mask = 0;
  };

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
            const Paths &paths)
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
        const Path &path = paths[i];
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
        const Path &path = paths[i];
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
    const Paths &getPaths() const { return paths; }

  private:
    BitsetId producerSetId = -1;
    BitsetId consumerSetId = -1;
    Operation *producerOperation = nullptr;
    Operation *consumerOperation = nullptr;
    Paths paths;
  };

  const std::vector<OpConfig> &getLegalConfigs(Operation *operation) const;
  void reset();

  PathSet *getPathSetPt(const Edge &edge);
  SmallVector<PathSet *> getOperandPathSetsPts(Operation *operation);
  SmallVector<PathSet *> getUserPathSetsPts(Operation *operation);

  bool handleNoPathsLeftOnUpdate(bool invokedBySet);
  bool updateSolver(Operation *root, bool expand_root = true,
                    bool invokedBySet = false);

  Bitset *getBitset(Operation *op);
  const Bitset *getBitset(Operation *op) const;
  Bitset *getOrInsertBitset(Operation *op, const Bitset &init);

  bool resolveStep();
  bool insertReshard(const Edge &edge);
  void addOperandsAndUsers(Operation *op, std::vector<Operation *> &needsUpdate,
                           Operation *ignoreOp = nullptr);

  bool preprocessFirstOp();
  llvm::Expected<bool> checkShardCompatible(
      Value producerOperand, const TTNNLayoutAttr &producerLayout,
      Operation *consumerOp, const OpConfig &consumerConfig) const;

public:
  ShardSolver(
      const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      const std::vector<OpL1MemSpec> &shardSpecs,
      const llvm::DenseSet<Operation *> &shardedOps,
      const unsigned usableL1CacheSize,
      const llvm::DenseSet<Edge> &overrideReshardEdges,
      std::function<bool(mlir::Value, TTNNLayoutAttr, mlir::Operation *,
                         OpConfig)>
          customCheckShardCompatible = nullptr);
  RemainingConfigAttrs at(Operation *operation) const;
  void set(Operation *operation, const OpConfig &config);
  bool supportsInterleavedInputShardedOutput(Operation *op,
                                             OpConfig outputConfig);
  llvm::DenseMap<Operation *, SmallVector<float, 64>> produceMaxCoreUsage();
  ShardSolverSolution finish() const;
  bool resolve();
  bool earlyExit = false;

private:
  const TensorTypeLayoutsMap *tensorTypePossibleLayouts;
  const llvm::DenseMap<Operation *, std::vector<OpConfig>> *legalConfigs;
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

  llvm::DenseMap<Operation *, OpConfig> selectedOpConfig;

  // Map of every edge prepared for resharding. Key is the edge, value is
  // the resharding entry. The entry contains a map of consumer op config bit
  // index to vector of valid producer op configs.
  llvm::DenseMap<Edge, MemReconfigEntry> memReconfigMap;

  // Edges indicated for resharding.
  llvm::DenseSet<Edge> memReconfigEdges;

  std::function<bool(mlir::Value, TTNNLayoutAttr, mlir::Operation *, OpConfig)>
      customCheckShardCompatible;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDSOLVER_H
