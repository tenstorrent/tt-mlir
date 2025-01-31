// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Support/LLVM.h>
#include <unordered_set>
#include <utility>

namespace mlir::tt::ttnn {

ShardSolver::Bitset ShardSolver::kBitsetAll = ~kBitsetNone;

ShardSolver::ShardSolver(
    const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
        &legalLayouts,
    const std::vector<OpL1MemSpec> &shardSpecs,
    const llvm::DenseSet<Operation *> &shardedOps,
    const unsigned usableL1CacheSize,
    const std::unordered_set<Edge> &overrideReshardEdges,
    std::function<bool(Operation *, TTNNLayoutAttr const &, Operation *,
                       TTNNLayoutAttr const &)>
        customCheckShardCompatible)
    : legalLayouts(&legalLayouts), shardSpecs(&shardSpecs),
      shardedOps(&shardedOps), usableL1CacheSize(usableL1CacheSize),
      memReconfigEdges(overrideReshardEdges),
      customCheckShardCompatible(customCheckShardCompatible) {
  pathSets.reserve(shardSpecs.size());
  pathSetIds.reserve(shardSpecs.size());
  bitsets.reserve(shardedOps.size());
  bitsetIds.reserve(shardedOps.size());

  // Cache DeviceAttr.
  //
  deviceAttr = getCurrentScopeDevice(shardSpecs.front().op);

  // Populate operandOpEdges and userOpEdges.
  //
  for (const auto shardSpec : shardSpecs) {
    Operation *op = shardSpec.op;
    for (size_t operandIndex = 0; operandIndex < op->getNumOperands();
         operandIndex++) {
      Value operand = op->getOperand(operandIndex);
      Operation *operandOp = operand.getDefiningOp();
      if (operandOp && shardedOps.count(operandOp) > 0) {
        operandOpEdges[op].emplace_back(Edge(operandOp, op, operandIndex));
        userOpEdges[operandOp].emplace_back(Edge(operandOp, op, operandIndex));
      }
    }
  }

  // Resolve shard chain.
  //
  resolve();
}

void ShardSolver::reset() {
  pathSets.clear();
  pathSetIds.clear();
  bitsets.clear();
  bitsetIds.clear();
}

bool ShardSolver::resolveStep() {
  OperationPathsProcessor opProcessor;
  bitsets.reserve(shardedOps->size());
  bitsetIds.reserve(shardedOps->size());
  selectedOpLayout.reserve(shardedOps->size());
  bool reshardInserted = false;

  // We need special handling for the first op in the chain.
  //
  preprocessFirstOp();

  for (const auto shardSpec : *shardSpecs) {
    Operation *consumerOp = shardSpec.op;
    Bitset *consumerBitset = getOrInsertBitset(consumerOp, kBitsetAll);
    std::vector<TTNNLayoutAttr> const &consumerLayouts =
        getLegalLayouts(consumerOp);

    for (Edge edge : operandOpEdges[consumerOp]) {

      bool reshardOnEdge = memReconfigEdges.count(edge) > 0;

      Operation *producerOp = edge.producerOp;
      Bitset *producerBitset = getOrInsertBitset(producerOp, kBitsetAll);
      std::vector<TTNNLayoutAttr> const &producerLayouts =
          getLegalLayouts(producerOp);

      assert(not(consumerLayouts.empty() && producerLayouts.empty()));

      PathSet::Paths paths;
      std::unordered_map<std::string, int> errorCount;
      Bitset edgeProducerBitset = kBitsetNone;
      Bitset edgeConsumerBitset = kBitsetNone;
      std::uint64_t producer_count =
          std::min(kNumBitsetBits, std::max(1lu, producerLayouts.size()));
      std::uint64_t consumer_count =
          std::min(kNumBitsetBits, std::max(1lu, consumerLayouts.size()));
      for (std::uint64_t producerId = 0; producerId < producer_count;
           ++producerId) {
        // If the producer cannot accomodate this path, continue.
        // Also if this is not the TTNNLayoutAttr we selected, continue.
        //
        if (!producerBitset->test(producerId)) {
          continue;
        }

        for (std::uint64_t consumerId = 0; consumerId < consumer_count;
             ++consumerId) {

          // If the consumer cannot accomodate this path, continue.
          //
          if (!consumerBitset->test(consumerId)) {
            continue;
          }

          // TODO(nobradovic):
          // Update checkShardCompatible with op type, other input
          // spec(weight).
          //
          auto [validShardPair, error] =
              checkShardCompatible(producerOp, producerLayouts[producerId],
                                   consumerOp, consumerLayouts[consumerId]);
          if (!errorCount.count(error)) {
            errorCount.insert({error, 0});
          }
          errorCount[error]++;

          validShardPair = validShardPair || reshardOnEdge;


          if (validShardPair) {
            assert(producerId <=
                   std::numeric_limits<decltype(Path::producerId)>::max());
            assert(consumerId <=
                   std::numeric_limits<decltype(Path::consumerId)>::max());
            paths.push_back(Path(producerId, consumerId));
            edgeProducerBitset.set(producerId);
            edgeConsumerBitset.set(consumerId);
          }
        }
      }

      // IF this is totaly empty, of if just dram is left, we need to split the chain in order not to discard sharded configurations.
      // After the split, we should rerun the solver on individual chains.
      // Maybe even do this in legal layout analysis, try all inputs and outputs and discard legality of layouts based on that.
      // Then create chains based on ops having legal layouts.
      // TODO: ALSO NEED TO RUN CHECK WHEN RESHARD EDGE IS INSERTED to see which input is legal.
      if (paths.empty()) {
        llvm::outs() << "\n\nNov valid solution for edge: ";
        llvm::outs() << "Producer Op: " << producerOp->getName() << "\n";
        producerOp->dump();
        llvm::outs() << "Consumer Op: " << consumerOp->getName() << "\n";
        consumerOp->dump();
        llvm::outs() << "Error counts:\n";
        for (const auto &error : errorCount) {
          llvm::outs() << error.first.substr(80, 250) << ": " << error.second << "\n";
        }

      }

      if (paths.empty() || ((*producerBitset & edgeProducerBitset) == 0) ||
          ((*consumerBitset & edgeConsumerBitset) == 0)) {

        // No valid paths found for this edge, mark it for resharding.
        //

        insertReshard(edge);
        reshardInserted = true;
        *consumerBitset = kBitsetAll;
      }

      if (!isSubset(*producerBitset, edgeProducerBitset) && !reshardInserted) {
        opProcessor.addOp(producerOp);
      }

      *producerBitset &= edgeProducerBitset;
      *consumerBitset &= edgeConsumerBitset;
      assert(pathSetIds.find(edge) == pathSetIds.end());
      PathSetId pathSetId = static_cast<PathSetId>(pathSets.size());
      pathSets.emplace_back(bitsetIds[producerOp], bitsetIds[consumerOp],
                            producerOp, consumerOp, paths);
      pathSetIds.emplace(edge, pathSetId);
    }

    if (!reshardInserted) {
      opProcessor.process(this);
    }
  }

  if (reshardInserted) {
    return false;
  }

  for (const auto shardSpec : *shardSpecs) {
    Operation *op = shardSpec.op;

    // No need to expand root as we are calling for all ops anyway.
    //
    updateSolver(op, false /* expand_root */);
  }

  postprocessLastOp();

  return true;
}

bool ShardSolver::supportsInterleavedInputShardedOutput(Operation *op) {
  // TODO(nobradovic,mbezulj): Add check whether this op type can have sharded
  // output from interleaved inputs. For now assuming it can.
  //
  //TODO add constraint chech or false
  return false;
}

bool ShardSolver::supportsShardedInputInterleavedOutput(Operation *op) {
  // TODO(nobradovic,mbezulj): Add check whether this op type can have sharded
  // output from interleaved inputs. For now assuming it can.
  //
  //TODO add constraint chech or false
  return false;
}

// We need to check if first op requires sharded inputs and if so, insert
// reshard edge, then invalidate all sharding options which would go above L1
// size limits.
//
void ShardSolver::preprocessFirstOp() {
  // Add constraint check
  Operation *firstOp = shardSpecs->front().op;
  if (supportsInterleavedInputShardedOutput(firstOp) &&
      memReconfigEdges.count(
          Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0)) == 0) {
    return;
  }

  // Insert reshard edge for the first op to start the chain.
  //
  Edge shardChainInputEdge =
      Edge(firstOp->getOperand(0).getDefiningOp(), firstOp, 0 /*operandIndex*/);

  if (memReconfigEdges.count(shardChainInputEdge) == 0) {
    insertReshard(shardChainInputEdge);
  }

  Bitset *firstOpBitset = getOrInsertBitset(firstOp, kBitsetAll);
  std::vector<TTNNLayoutAttr> const &firstOpLayouts = getLegalLayouts(firstOp);
  Operation *operandOp = firstOp->getOperand(0).getDefiningOp();

  RankedTensorType firstOpInputTensorType =
      mlir::cast<RankedTensorType>(operandOp->getResult(0).getType());
  TTNNLayoutAttr firstOpInputLayout =
      mlir::cast<TTNNLayoutAttr>(firstOpInputTensorType.getEncoding());
  constexpr float tensorL1UsageCap = 0.8;

  for (size_t i = 0; i < firstOpLayouts.size(); ++i) {
    if (!firstOpBitset->test(i)) {
      continue;
    }

    TTNNLayoutAttr firstOpLayout = firstOpLayouts[i];
    // assert(firstOpLayout.hasShardedL1TensorMemoryLayout());

    TTNNLayoutAttr firstOpInputShardedLayout =
        firstOpInputLayout
            .withBufferType(firstOp->getContext(),
                            firstOpLayout.getBufferType())
            .withMemoryLayout(firstOp->getContext(),
                              firstOpLayout.getMemLayout())
            .withGrid(firstOp->getContext(), firstOpInputTensorType,
                      firstOpLayout.getGrid());

    uint64_t firstInputL1Usage = firstOpInputShardedLayout.getTensorSizeInBytes(
        firstOpInputTensorType.getShape(), deviceAttr);
    uint64_t firstOpL1OutputUsage = firstOpLayout.getTensorSizeInBytes(
        mlir::cast<RankedTensorType>(firstOp->getResult(0).getType())
            .getShape(),
        deviceAttr);

    if ((firstInputL1Usage + firstOpL1OutputUsage) >=
        tensorL1UsageCap * usableL1CacheSize) {
      firstOpBitset->reset(i);
    }
  }
}

// Check if we can go from a sharded layout to dram. If not add mem reconfig edge.
void ShardSolver::postprocessLastOp() {
  // Add constraint check

  // Operation *lastOp = shardSpecs->back().op;
  // if (not supportsShardedInputInterleavedOutput(lastOp) ||
  //     memReconfigEdges.count(
  //         OutputEdge(lastOp)) == 0) {
      
  //     memReconfigEdges.insert(
  //         OutputEdge(lastOp));

  // }
}

void ShardSolver::insertReshard(const Edge &edge) {
  // Same edge should not be resharded twice!
  //
  assert(memReconfigEdges.count(edge) == 0);
  memReconfigEdges.insert(edge);
}

void ShardSolver::resolve() {

  const int max_retry_step = shardedOps->size() + 1;
  int retry_step = 1;
  bool resolved = false;

  do {
    // Reset ShardSolver to default state.
    //
    reset();

    // Try to resolve shard chain. Retry if not resolved(resharding).
    //
    resolved = resolveStep();
    retry_step++;
  } while (!resolved && retry_step <= max_retry_step);

  assert(resolved);
}

ShardSolver::PathSet *ShardSolver::getPathSetPt(const Edge &edge) {
  if (pathSetIds.count(edge) > 0) {
    return &pathSets[pathSetIds.at(edge)];
  }

  return nullptr;
}

SmallVector<ShardSolver::PathSet *>
ShardSolver::getOperandPathSetsPts(Operation *op) {
  SmallVector<PathSet *> operandPathSets;
  for (auto edge : operandOpEdges[op]) {
    PathSet *el = getPathSetPt(edge);
    if (nullptr != el) {
      operandPathSets.push_back(el);
    }
  }

  return operandPathSets;
}

SmallVector<ShardSolver::PathSet *>
ShardSolver::getUserPathSetsPts(Operation *op) {
  SmallVector<PathSet *> userPathSets;
  for (auto edge : userOpEdges[op]) {
    PathSet *el = getPathSetPt(edge);
    if (nullptr != el) {
      userPathSets.push_back(el);
    }
  }

  return userPathSets;
}

void ShardSolver::addOperandsAndUsers(Operation *op,
                                      std::vector<Operation *> &needsUpdate,
                                      Operation *ignoreOp) {

  for (auto operand : op->getOperands()) {
    Operation *opOperand = operand.getDefiningOp();
    if (opOperand == nullptr ||
        !llvm::isa<mlir::DestinationStyleOpInterface>(opOperand) ||
        opOperand == ignoreOp || shardedOps->count(opOperand) == 0) {
      continue;
    }

    needsUpdate.push_back(opOperand);
  }

  for (Operation *opUser : op->getUsers()) {
    if (opUser == nullptr || opUser == ignoreOp ||
        shardedOps->count(opUser) == 0) {
      continue;
    }

    needsUpdate.push_back(opUser);
  }
}

void ShardSolver::handleNoPathsLeftOnUpdate(bool invokedBySet) {
  // We ended-up in a situation without valid solution due to circular
  // dependency.
  //
  assert(invokedBySet);

  // Invoking resolve again will use resharding to resolve the issue.
  //
  return resolve();
}

void ShardSolver::updateSolver(Operation *root, bool expand_root,
                               bool invokedBySet) {
  std::vector<Operation *> needsUpdate = {root};

  if (expand_root) {
    auto operandPathSets = getOperandPathSetsPts(root);
    auto userPathSets = getUserPathSetsPts(root);

    for (auto *path_set : operandPathSets) {
      path_set->update(bitsets);
    }

    for (auto *path_set : userPathSets) {
      path_set->update(bitsets);
    }

    // When op bitsets are updated(set of valid op layouts), we need to update
    // paths for all operands and users.
    //
    addOperandsAndUsers(root, needsUpdate);
  }

  // Iterate through the ops that need to be updated and update their operand
  // and user path sets.
  while (not needsUpdate.empty()) {
    auto *op = needsUpdate.back();

    // Get path sets for incoming edges
    auto operandPathSets = getOperandPathSetsPts(op);
    // Get path sets for outgoing edges
    auto userPathSets = getUserPathSetsPts(op);

    bool edge_changed = false;

    std::vector<bool> producersChanged(operandPathSets.size());
    for (size_t i = 0; i < operandPathSets.size(); i++) {
      auto *operandPathSet = operandPathSets[i];
      producersChanged[i] = operandPathSet->update(bitsets);

      if (operandPathSet->empty(bitsets)) {
        return handleNoPathsLeftOnUpdate(invokedBySet);
      }
    }

    std::vector<bool> consumers_changed(userPathSets.size());
    for (size_t i = 0; i < userPathSets.size(); i++) {
      auto *userPathSet = userPathSets[i];
      consumers_changed[i] = userPathSet->update(bitsets);

      if (userPathSet->empty(bitsets)) {
        return handleNoPathsLeftOnUpdate(invokedBySet);
      }
    }

    // If any of the paths between producer and this consumer changed, we need
    // to visit producer op and add its operands and users to the needsUpdate
    // list.
    for (size_t i = 0; i < producersChanged.size(); i++) {
      if (producersChanged[i]) {
        Operation *producerOp = operandPathSets[i]->getProducerOp();
        needsUpdate.push_back(producerOp);
        addOperandsAndUsers(producerOp, needsUpdate, op);

        edge_changed = true;
      }
    }

    // If any of the paths between this producer and consumer changed, we need
    // to visit consumer op and add its operands and users to the needsUpdate
    // list.
    for (size_t i = 0; i < consumers_changed.size(); i++) {
      if (consumers_changed[i]) {
        Operation *consumerOp = userPathSets[i]->getConsumerOp();
        needsUpdate.push_back(consumerOp);
        addOperandsAndUsers(consumerOp, needsUpdate, op);

        edge_changed = true;
      }
    }

    if (not edge_changed) {
      needsUpdate.pop_back();
    }
  }
}

ShardSolver::Bitset *ShardSolver::getBitset(Operation *op) {
  return &bitsets[bitsetIds.at(op)];
}

ShardSolver::Bitset const *ShardSolver::getBitset(Operation *op) const {
  return &bitsets[bitsetIds.at(op)];
}

ShardSolver::Bitset *ShardSolver::getOrInsertBitset(Operation *op,
                                                    const Bitset &init) {
  auto match = bitsetIds.find(op);
  if (match == bitsetIds.end()) {
    BitsetId bitset_id = bitsets.size();
    bitsetIds.insert({op, bitset_id});
    auto *tmp = bitsets.data();
    bitsets.push_back(init);

    // Bitsets reallocated, pointers invalid.
    //
    assert(tmp == bitsets.data());
    return &bitsets.back();
  }

  return &bitsets[match->second];
}

// Returns vector of legal LayoutAttrs for passed in op.
//
const std::vector<TTNNLayoutAttr> &
ShardSolver::getLegalLayouts(Operation *op) const {
  static std::vector<TTNNLayoutAttr> nullLayouts;

  const auto legalIt = legalLayouts->find(op);

  if (legalIt != legalLayouts->end()) {
    return legalIt->second;
  }

  return nullLayouts;
}

ShardSolver::RemainingLayoutAttrs ShardSolver::at(Operation *op) const {
  auto layouts = RemainingLayoutAttrs(getLegalLayouts(op), *getBitset(op));
  assert(layouts.begin() != layouts.end());
  return layouts;
}

void ShardSolver::set(Operation *op, TTNNLayoutAttr const &layout) {
  assert(selectedOpLayout.count(op) == 0);

  selectedOpLayout[op] = layout;

  auto const &layouts = getLegalLayouts(op);
  assert(!layouts.empty());
  size_t selection = layouts.size();
  for (size_t i = 0; i < layouts.size(); ++i) {
    if (layouts[i] == layout) {
      selection = i;
      break;
    }
  }

  Bitset *op_bitset = getBitset(op);

  assert(selection != layouts.size());
  assert((*op_bitset)[selection]);

  op_bitset->reset();
  op_bitset->set(selection);

  updateSolver(op, true /*expand_root*/, true /*invokedBySet*/);
}

std::pair<bool, std::string> ShardSolver::checkShardCompatible(
    Operation *producerOp, TTNNLayoutAttr const &producerLayout,
    Operation *consumerOp, TTNNLayoutAttr const &consumerLayout) const {

  // Custom(test) hook for shard compatibility check.
  //
  if (customCheckShardCompatible) {
    return std::make_pair(customCheckShardCompatible(producerOp, producerLayout, consumerOp,
                                      consumerLayout), "Custom check failed.");
  }
  // Figure out this const based on exec data, but will be replaced
  // with API.
  //
  constexpr float tensorL1UsageCap = 0.8;

  // HOW DOES THIS MANAGE TO CAST even tho it's not a implemented...
  if (llvm::isa<OpModel>(consumerOp)) {
    

    // CAHNGE TO USE LARGEST SHARDED GIRD INSTEAD OF l1 interlaved!!!!

    OpModel backend = dyn_cast<OpModel>(consumerOp);
    // Constraints are implemented for this op.
    //
    auto deviceAttr = mlir::tt::getCurrentScopeDevice(producerOp);
    assert(deviceAttr);
    auto workerGrid = deviceAttr.getWorkerGrid();

    // Map consumer operands to DRAM interleave or provided producerLayout
    // only one operand can be mapped to producerLayout, it's picked as first
    // operand matching producerOp output shape.

    uint32_t numOperands = consumerOp->getNumOperands();

    // Some ops have multiple operands; and some ops have output also an
    // operand. TBD if there is a more robust way to get real number of inputs.
    // TODO(odjuricic): Do we want to remove the last operand or should we use
    // DPS?
    if (llvm::isa<DestinationStyleOpInterface>(consumerOp)) {
      numOperands = numOperands - 1;
    }

    if (numOperands > 2) {
      consumerOp->emitError(
          "Ops with more than 2 operands are not supported in Optimizer");
      assert(false &&
            "Ops with more than 2 operands are not supported in Optimizer");
    }

    std::vector<TTNNLayoutAttr> inputLayouts;

    // // Assert that the first operand is the input under check.
    // llvm::outs()<<"producerOp: "<<producerOp->getName()<<"\n";
    // producerOp->dump();
    // llvm::outs()<<"consumerOp: "<<consumerOp->getName()<<"\n";
    // consumerOp->dump();

    // // Print all producerOp operands
    // for (uint32_t i = 0; i < producerOp->getNumOperands(); i++) {
    //   llvm::outs()<<"producerOp->getOperand("<<i<<"): "<<producerOp->getOperand(i).getDefiningOp()->getName()<<"\n";
    //   producerOp->getOperand(i).getDefiningOp()->dump();
    // }

    // assert(producerOp->getOperand(0).getDefiningOp() == consumerOp &&
    //        "Only first operand can be in a shard chain.");

    inputLayouts.push_back(producerLayout);

    for (uint32_t i = 1; i < numOperands; i++) {
      auto operand = consumerOp->getOperand(i);
      auto input = mlir::cast<RankedTensorType>(operand.getType());

      // this is the other input that we DRAM interleave
      // TODO just use the input as is? and assert DRAM?
      // Or fetch dram option for this op? This cannot be done now since we
      // don't pass dram configs into solver.

      auto elementType = input.getElementType();
      if (!llvm::isa<TileType>(elementType)) {
        // TODO(odjuricic): Do we need this? Jackson new change.
        elementType =
            TileType::get(consumerOp->getContext(), input.getElementType());
      }

      auto layout = TTNNLayoutAttr::get(
          consumerOp->getContext(), input.getShape(), elementType,
          BufferType::DRAM, workerGrid,
          TensorMemoryLayoutAttr::get(consumerOp->getContext(),
                                      TensorMemoryLayout::Interleaved));
      inputLayouts.push_back(layout);
    }

    auto [legal, l1Usage, errorMsg] =
        backend.getOpConstraints(inputLayouts, consumerLayout);

    constexpr bool debug = false;
    if (false == legal) {
      // early exit
      if (debug) {
        llvm::errs() << "OpModel constraints failed: ";
        llvm::errs() << producerOp->getName() << "->" << consumerOp->getName()
                     << " :: " << errorMsg.value() << "\n";
        producerLayout.dump();
        consumerLayout.dump();
      }
      return std::make_pair(false, errorMsg.value());
    }
    if (debug) {
      llvm::errs() << "OpModel constraints valid. ";
      llvm::errs() << producerOp->getName() << "->" << consumerOp->getName()
                   << "\n";
      producerLayout.dump();
      consumerLayout.dump();
    }

    // What will output tensor usage be if op is DPS?
    // For now, no op here is DPS.
    // Can we get input tensor as well? Then we don't need to calculate on our
    // side.
    auto [cBUsagePeak, tensorUsage, outputTensorUsage] = l1Usage.value();

    RankedTensorType producerTensorType =
        mlir::cast<RankedTensorType>(producerOp->getResult(0).getType());
    uint64_t producerL1OutputUsage = producerLayout.getTensorSizeInBytes(
        producerTensorType.getShape(), deviceAttr);

    // RankedTensorType consumerTensorType =
    //     mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType());
    // uint64_t consumerL1OutputUsage = consumerLayout.getTensorSizeInBytes(
    //     consumerTensorType.getShape(), deviceAttr);

    bool l1UsageValid = (producerL1OutputUsage + outputTensorUsage +
                         cBUsagePeak) < tensorL1UsageCap * usableL1CacheSize;

    if (!l1UsageValid) {
      return std::make_pair(false, "Not enough L1 memory");
    }

  } else {
    // Constraints are not implemented for this op. Use fallback.
    // Shard compat assumption. Try to keep same shard layout.
    //
    // TODO(odjurcic) Put this fallback under a flag.

    if (producerLayout.getMemLayout() != consumerLayout.getMemLayout()) {
      return std::make_pair(false, "FALLBACK: tensor memory layout mismatch");
    }

    // Calculate L1 tensor memory usage based on :
    // currentOp output tensor shard spec, nextOp exec and nextOp output
    // tensor.
    //

    uint64_t producerL1OutputUsage = 0;
    if (producerLayout.hasL1BufferType()) {
      RankedTensorType producerTensorType =
          mlir::cast<RankedTensorType>(producerOp->getResult(0).getType());
      producerL1OutputUsage = producerLayout.getTensorSizeInBytes(
          producerTensorType.getShape(), deviceAttr);
    }

    uint64_t consumerL1OutputUsage = 0;
    if (consumerLayout.hasL1BufferType()) {
      RankedTensorType consumerTensorType =
          mlir::cast<RankedTensorType>(consumerOp->getResult(0).getType());
      consumerL1OutputUsage = consumerLayout.getTensorSizeInBytes(
          consumerTensorType.getShape(), deviceAttr);
    }

    bool l1UsageValid = (producerL1OutputUsage + consumerL1OutputUsage) <
                        tensorL1UsageCap * usableL1CacheSize;

    if (!l1UsageValid) {
      return std::make_pair(false, "Not enough L1 memory");
    }
  }

  // TODO Do we run constraints enable mnist sharding test in CI?
  return std::make_pair(true, "");
}

// Preprocess ShardSolver search space to make a helper structure which links op
// layout choices to global max core usage.
// Example:
// Lets assume simple case where layouts at same index are compatible for input
// graph provided below. Tupples represent layout core
// usage (Layout0GridVolume, Layout1GridVolume, Layout2GridVolume).
//
//    Op0 ----- (4, 8, 2)
//     |
//    Op1 ----- (8, 4, 2)
//    / \
//   /   \
//  Op2  Op3 -- (4, 4, 2) (4, 4, 2)
//   \   /
//    \ /
//    Op4 ----- (2, 1, 1)
//     |
//    Op5 ----- (2, 1, 1)
//
// Here is how structure looks after preprocessing is complete:
//
//    Op0 ----- (24, 22, 10)
//     |
//    Op1 ----- (20, 14, 8)
//    / \
//   /   \
//  Op2  Op3 -- (6, 5, 3) (6, 5, 3)
//   \   /
//    \ /
//    Op4 ----- (4, 2, 2)
//     |
//    Op5 ----- (2, 1, 1)
//
// Global max of 24 core usage is achieved by selecting layout[0] for each Op.
//
// Returns map of op to vector of max core usage for each layout.
llvm::DenseMap<Operation *, SmallVector<float, 64>>
ShardSolver::produceMaxCoreUsage() {
  using Paths = llvm::SmallVector<Path, 16>;
  llvm::DenseMap<Operation *, SmallVector<float, 64>> accCoreUsage(
      shardedOps->size());

  // Start from the tail of the chain and build up the max core usage(schedule
  // in backwards).
  //
  for (auto shardSpec = shardSpecs->rbegin(); shardSpec != shardSpecs->rend();
       ++shardSpec) {
    Operation *op = shardSpec->op;
    std::vector<TTNNLayoutAttr> const &layouts = getLegalLayouts(op);
    assert(!layouts.empty());

    // Find the layout that leads to the max core usage.
    // Start with grid volume of current op.
    //
    for (size_t i = 0; i < layouts.size(); ++i) {
      TTNNLayoutAttr const &layout = layouts[i];
      uint64_t score = layout.getGrid().getGridVolume();
      if (layout.hasDRAMBufferType()) {
        score = 0;
      } else if (layout.hasInterleavedL1TensorMemoryLayout()) {
        score = 1;
      }
      accCoreUsage[op].push_back(score);
    }

    // Add core usage of current op users via live path connections.
    //
    SmallVector<ShardSolver::PathSet *> userPathSets = getUserPathSetsPts(op);
    for (size_t i = 0; i < userPathSets.size(); ++i) {
      ShardSolver::PathSet *pathSet = userPathSets[i];
      const Paths &paths = pathSet->getPaths();
      SmallVector<uint64_t, 64> maxCoreUsage(layouts.size(), 0);
      Operation *consumerOp = pathSet->getConsumerOp();
      size_t consumerInChainOperandSize =
          getOperandPathSetsPts(consumerOp).size();
      uint64_t consumerCoreUsage = 0;
      for (auto const &path : paths) {
        assert(bitsets[bitsetIds[op]].test(path.producerId));
        assert(bitsets[bitsetIds[consumerOp]].test(path.consumerId));
        consumerCoreUsage = accCoreUsage[consumerOp][path.consumerId];
        if (consumerCoreUsage > maxCoreUsage[path.producerId]) {
          maxCoreUsage[path.producerId] = consumerCoreUsage;
        }
      }

      for (size_t i = 0; i < layouts.size(); ++i) {
        // Add max core usage of consumer ops to current op layout.
        // We divide by consumerInChainOperandSize to normalize the core usage
        // based on forking factor(so that cores are not counted more than
        // once).
        //
        // Incorrect results will be produced in case chain consists of joins
        // without previous forks, ie - chain having multiple input ops. In that
        // case total sum of used cores would be a sum of maxCoreUsage generated
        // by all input ops. This is currently not needed for making a
        // decision on layout choice for maximizing core usage.
        //
        accCoreUsage[op][i] += static_cast<float>(maxCoreUsage[i]) /
                               static_cast<float>(consumerInChainOperandSize);
      }
    }
  }

  return accCoreUsage;
}

// Returns ShardSolverSolution.
//
ShardSolverSolution ShardSolver::finish() const {
  assert(selectedOpLayout.size() == shardedOps->size());
  return ShardSolverSolution(selectedOpLayout, memReconfigEdges);
}
} // namespace mlir::tt::ttnn
