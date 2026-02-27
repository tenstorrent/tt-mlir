// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DecisionTrace.h"

#include "ttmlir/Support/Logger.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/MemoryBuffer.h"

#include <string>

namespace mlir::tt::ttnn {

std::string layoutToString(TTNNLayoutAttr layout) {
  if (!layout) {
    return "null";
  }

  std::string result;
  llvm::raw_string_ostream os(result);

  // Buffer type: L1 or DRAM.
  os << layout.getBufferType();

  // Memory layout: interleaved, height_sharded, etc.
  auto memLayout = layout.getMemLayout();
  if (memLayout) {
    os << "/" << memLayout;
  }

  // Grid shape for sharded layouts.
  if (auto grid = layout.getGrid()) {
    auto shape = grid.getShape();
    if (!shape.empty()) {
      os << "/";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
          os << "x";
        }
        os << shape[i];
      }
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// JSON serialization helpers
//===----------------------------------------------------------------------===//

static llvm::json::Object scoreToJSON(bool isL1, bool isSharded,
                                      uint64_t inputDramBytes,
                                      bool requiresReshard, int64_t coreCount,
                                      uint64_t outputL1Usage) {
  llvm::json::Object obj;
  obj["isL1"] = isL1;
  obj["isSharded"] = isSharded;
  obj["inputDramBytes"] = static_cast<int64_t>(inputDramBytes);
  obj["requiresReshard"] = requiresReshard;
  obj["coreCount"] = coreCount;
  obj["outputL1Usage"] = static_cast<int64_t>(outputL1Usage);
  return obj;
}

static llvm::json::Value evalToJSON(const EvaluationRecord &e) {
  llvm::json::Object obj;
  obj["hint"] = e.hint;

  llvm::json::Array inputs;
  for (const auto &inp : e.inputs) {
    inputs.push_back(inp);
  }
  obj["inputs"] = std::move(inputs);

  obj["valid"] = e.valid;
  if (!e.valid) {
    obj["failureReason"] = e.failureReason;
  } else {
    obj["score"] = scoreToJSON(e.isL1, e.isSharded, e.inputDramBytes,
                               e.requiresReshard, e.coreCount, e.outputL1Usage);
    obj["output"] = e.output;
  }
  return llvm::json::Value(std::move(obj));
}

static llvm::json::Value
inputCandidateSetToJSON(const InputCandidateSetRecord &r) {
  llvm::json::Object obj;
  obj["operandIndex"] = static_cast<int64_t>(r.operandIndex);
  obj["fromProducerBeam"] = static_cast<int64_t>(r.fromProducerBeam);
  obj["fromReshard"] = static_cast<int64_t>(r.fromReshard);

  llvm::json::Array candidates;
  for (const auto &c : r.candidates) {
    candidates.push_back(c);
  }
  obj["candidates"] = std::move(candidates);
  return llvm::json::Value(std::move(obj));
}

static llvm::json::Value beamEntryToJSON(const BeamEntryRecord &b) {
  llvm::json::Object obj;
  obj["rank"] = static_cast<int64_t>(b.rank);
  obj["outputLayout"] = b.outputLayout;
  obj["score"] =
      scoreToJSON(b.isL1, b.isSharded, b.inputDramBytes, b.requiresReshard,
                  b.coreCount, b.outputL1Usage);
  return llvm::json::Value(std::move(obj));
}

static llvm::json::Value opDecisionToJSON(const OpDecisionRecord &op) {
  llvm::json::Object obj;
  obj["opIndex"] = static_cast<int64_t>(op.opIndex);
  obj["opName"] = op.opName;
  obj["opLocation"] = op.opLocation;

  // Input candidate sets.
  llvm::json::Array ics;
  for (const auto &ic : op.inputCandidateSets) {
    ics.push_back(inputCandidateSetToJSON(ic));
  }
  obj["inputCandidateSets"] = std::move(ics);

  // Output hints.
  llvm::json::Object hintsObj;
  hintsObj["primaryCount"] = static_cast<int64_t>(op.outputHints.primaryCount);
  hintsObj["fallbackCount"] =
      static_cast<int64_t>(op.outputHints.fallbackCount);
  hintsObj["attemptL1Sharding"] = op.outputHints.attemptL1Sharding;
  obj["outputHints"] = std::move(hintsObj);

  obj["crossProductSize"] = static_cast<int64_t>(op.crossProductSize);

  // Evaluations.
  llvm::json::Array evals;
  for (const auto &e : op.evaluations) {
    evals.push_back(evalToJSON(e));
  }
  obj["evaluations"] = std::move(evals);

  // Beam.
  llvm::json::Array beam;
  for (const auto &b : op.beam) {
    beam.push_back(beamEntryToJSON(b));
  }
  obj["beam"] = std::move(beam);

  obj["usedDramFallback"] = op.usedDramFallback;
  if (op.isInplace) {
    obj["isInplace"] = true;
  }
  return llvm::json::Value(std::move(obj));
}

static llvm::json::Value edgeToJSON(const EdgeRecord &e) {
  llvm::json::Object obj;
  obj["producerOpIndex"] = static_cast<int64_t>(e.producerOpIndex);
  obj["consumerOpIndex"] = static_cast<int64_t>(e.consumerOpIndex);
  obj["operandIndex"] = static_cast<int64_t>(e.operandIndex);
  obj["hasReshard"] = e.hasReshard;
  if (e.hasReshard) {
    obj["reshardLayout"] = e.reshardLayout;
  }
  return llvm::json::Value(std::move(obj));
}

static llvm::json::Value
forkResolutionToJSON(const ForkResolutionRecord &f) {
  llvm::json::Object obj;
  obj["opName"] = f.opName;
  obj["opLocation"] = f.opLocation;
  obj["opIndex"] = static_cast<int64_t>(f.opIndex);
  obj["chosenCandidateIndex"] = static_cast<int64_t>(f.chosenCandidateIndex);
  obj["numConsumers"] = static_cast<int64_t>(f.numConsumers);

  llvm::json::Array consumerIndices;
  for (size_t idx : f.consumerOpIndices) {
    consumerIndices.push_back(static_cast<int64_t>(idx));
  }
  obj["consumerOpIndices"] = std::move(consumerIndices);
  return llvm::json::Value(std::move(obj));
}

static llvm::json::Value finalChoiceToJSON(const FinalChoiceRecord &fc) {
  llvm::json::Object obj;
  obj["opIndex"] = static_cast<int64_t>(fc.opIndex);
  obj["opName"] = fc.opName;
  obj["chosenLayout"] = fc.chosenLayout;
  return llvm::json::Value(std::move(obj));
}

static llvm::json::Value spillEventToJSON(const SpillEventRecord &e) {
  llvm::json::Object obj;
  obj["position"] = static_cast<int64_t>(e.position);
  obj["opName"] = e.opName;
  obj["action"] = e.action;
  obj["occupiedL1Before"] = static_cast<int64_t>(e.occupiedL1Before);
  obj["occupiedL1After"] = static_cast<int64_t>(e.occupiedL1After);
  obj["opL1Usage"] = static_cast<int64_t>(e.opL1Usage);
  if (!e.victimName.empty()) {
    obj["victimName"] = e.victimName;
  }
  if (!e.details.empty()) {
    obj["details"] = e.details;
  }
  return llvm::json::Value(std::move(obj));
}

//===----------------------------------------------------------------------===//
// DecisionTrace::toJSON
//===----------------------------------------------------------------------===//

llvm::json::Value DecisionTrace::toJSON() const {
  llvm::json::Object root;
  root["version"] = 3;
  root["functionName"] = functionName;
  root["beamWidth"] = static_cast<int64_t>(beamWidth);
  root["totalOps"] = static_cast<int64_t>(totalOps);

  // Forward pass.
  llvm::json::Array forward;
  for (const auto &op : forwardPass) {
    forward.push_back(opDecisionToJSON(op));
  }
  root["forwardPass"] = std::move(forward);

  // Edges.
  llvm::json::Array edgeArray;
  for (const auto &e : edges) {
    edgeArray.push_back(edgeToJSON(e));
  }
  root["edges"] = std::move(edgeArray);

  // Backward pass.
  llvm::json::Object backward;
  llvm::json::Array forks;
  for (const auto &f : backwardPass.forkResolutions) {
    forks.push_back(forkResolutionToJSON(f));
  }
  backward["forkResolutions"] = std::move(forks);
  root["backwardPass"] = std::move(backward);

  // Final choices.
  llvm::json::Array choices;
  for (const auto &fc : finalChoices) {
    choices.push_back(finalChoiceToJSON(fc));
  }
  root["finalChoices"] = std::move(choices);

  // Spill management.
  if (!spillManagement.events.empty() || spillManagement.budget > 0) {
    root["spillManagement"] = spillManagementToJSON(spillManagement);
  }

  return llvm::json::Value(std::move(root));
}

//===----------------------------------------------------------------------===//
// DecisionTrace::writeToFile
//===----------------------------------------------------------------------===//

bool DecisionTrace::writeToFile(llvm::StringRef path) const {
  // Ensure parent directory exists.
  llvm::SmallString<256> parentDir(path);
  llvm::sys::path::remove_filename(parentDir);
  if (!parentDir.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentDir)) {
      llvm::errs() << "DecisionTrace: failed to create directory " << parentDir
                    << ": " << ec.message() << "\n";
      return false;
    }
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "DecisionTrace: failed to open " << path << ": "
                  << ec.message() << "\n";
    return false;
  }

  llvm::json::Value json = toJSON();
  os << llvm::formatv("{0:2}", json);
  return true;
}

//===----------------------------------------------------------------------===//
// DecisionTrace static helpers
//===----------------------------------------------------------------------===//

llvm::json::Object
DecisionTrace::spillManagementToJSON(const SpillManagementTrace &spill) {
  llvm::json::Object spillObj;
  spillObj["budget"] = static_cast<int64_t>(spill.budget);
  spillObj["scheduleSize"] = static_cast<int64_t>(spill.scheduleSize);
  spillObj["totalSpills"] = static_cast<int64_t>(spill.totalSpills);
  spillObj["finalOccupied"] = static_cast<int64_t>(spill.finalOccupied);
  spillObj["finalLiveTensors"] = static_cast<int64_t>(spill.finalLiveTensors);

  llvm::json::Array events;
  for (const auto &e : spill.events) {
    events.push_back(spillEventToJSON(e));
  }
  spillObj["events"] = std::move(events);
  return spillObj;
}

bool DecisionTrace::writeTraceForFunc(llvm::StringRef dir,
                                      llvm::StringRef funcName,
                                      const DecisionTrace &trace) {
  llvm::SmallString<256> path(dir);
  llvm::sys::path::append(path, funcName + "_decision_trace.json");
  return trace.writeToFile(path);
}

bool DecisionTrace::mergeSpillTrace(llvm::StringRef dir,
                                    llvm::StringRef funcName,
                                    const DecisionTrace &spillTrace) {
  llvm::SmallString<256> path(dir);
  llvm::sys::path::append(path, funcName + "_decision_trace.json");

  // Read existing trace file (written by layout propagation).
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    llvm::errs() << "DecisionTrace: could not read " << path
                 << " for spill merge: " << bufOrErr.getError().message()
                 << "\n";
    return false;
  }

  auto parsed = llvm::json::parse((*bufOrErr)->getBuffer());
  if (!parsed) {
    llvm::errs() << "DecisionTrace: failed to parse JSON in " << path << "\n";
    return false;
  }

  auto *root = parsed->getAsObject();
  if (!root) {
    llvm::errs() << "DecisionTrace: root is not a JSON object in " << path
                 << "\n";
    return false;
  }

  // Merge spill management section.
  if (!spillTrace.spillManagement.events.empty() ||
      spillTrace.spillManagement.budget > 0) {
    (*root)["spillManagement"] =
        spillManagementToJSON(spillTrace.spillManagement);
  }

  // Write merged trace back.
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "DecisionTrace: failed to write merged trace to " << path
                 << ": " << ec.message() << "\n";
    return false;
  }
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(*root)));
  return true;
}

//===----------------------------------------------------------------------===//
// DecisionTraceObserver -- LayoutPropagation callbacks
//===----------------------------------------------------------------------===//

void DecisionTraceObserver::onStart(llvm::StringRef funcName,
                                    size_t beamWidth) {
  trace.functionName = funcName.str();
  trace.beamWidth = beamWidth;
}

void DecisionTraceObserver::onOpSetup(
    Operation *op,
    const std::vector<std::vector<InputCandidate>> &inputSets,
    const OutputHints &hints, size_t crossProductSize) {
  OpDecisionRecord record;
  record.opName = op->getName().getStringRef().str();
  std::string locStr;
  llvm::raw_string_ostream locOS(locStr);
  locOS << op->getLoc();
  record.opLocation = locStr;

  // Record input candidate sets.
  for (size_t setIdx = 0; setIdx < inputSets.size(); ++setIdx) {
    InputCandidateSetRecord icr;
    icr.operandIndex = setIdx;
    size_t reshardCount = 0;
    size_t producerCount = 0;
    for (const auto &ic : inputSets[setIdx]) {
      icr.candidates.push_back(layoutToString(ic.layout));
      if (ic.isReshard) {
        ++reshardCount;
      } else {
        ++producerCount;
      }
    }
    icr.fromProducerBeam = producerCount;
    icr.fromReshard = reshardCount;
    record.inputCandidateSets.push_back(std::move(icr));
  }

  // Record output hints.
  record.outputHints.primaryCount = hints.hints.size();
  record.outputHints.fallbackCount = hints.fallbackHints.size();
  record.outputHints.attemptL1Sharding = hints.attemptL1Sharding;
  record.crossProductSize = crossProductSize;

  trace.forwardPass.push_back(std::move(record));
  currentOpRecord = &trace.forwardPass.back();
}

void DecisionTraceObserver::onEvaluation(
    Operation *op, const OpConfig &hint, size_t hintIdx,
    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts, bool valid,
    const BeamCandidate *candidate, llvm::StringRef failureReason) {
  if (!currentOpRecord) {
    return;
  }

  EvaluationRecord eval;
  eval.hint = layoutToString(hint.outputLayout);
  for (const auto &inLay : inputLayouts) {
    eval.inputs.push_back(layoutToString(inLay));
  }
  eval.valid = valid;

  if (valid && candidate) {
    eval.isL1 = candidate->score.isL1;
    eval.isSharded = candidate->score.isSharded;
    eval.inputDramBytes = candidate->score.inputDramBytes;
    eval.requiresReshard = candidate->score.requiresReshard;
    eval.coreCount = candidate->score.coreCount;
    eval.outputL1Usage = candidate->score.outputL1Usage;
    eval.output = layoutToString(candidate->config.outputLayout);
  } else {
    eval.failureReason = failureReason.str();
  }

  currentOpRecord->evaluations.push_back(std::move(eval));
}

void DecisionTraceObserver::onBeamResult(Operation *op,
                                         llvm::ArrayRef<BeamCandidate> beam,
                                         bool usedDramFallback) {
  if (!currentOpRecord) {
    return;
  }

  // Record beam survivors.
  for (size_t ci = 0; ci < beam.size(); ++ci) {
    const auto &c = beam[ci];
    BeamEntryRecord be;
    be.rank = ci;
    be.outputLayout = layoutToString(c.config.outputLayout);
    be.isL1 = c.score.isL1;
    be.isSharded = c.score.isSharded;
    be.inputDramBytes = c.score.inputDramBytes;
    be.requiresReshard = c.score.requiresReshard;
    be.coreCount = c.score.coreCount;
    be.outputL1Usage = c.score.outputL1Usage;
    currentOpRecord->beam.push_back(std::move(be));
  }
  currentOpRecord->usedDramFallback = usedDramFallback;

  // Set opIndex and register in the op->index map.
  size_t idx = trace.forwardPass.size() - 1;
  currentOpRecord->opIndex = idx;
  opToTraceIndex[op] = idx;

  // Reset current record pointer.
  currentOpRecord = nullptr;
}

void DecisionTraceObserver::onForkResolved(
    Operation *producer, size_t chosenIdx,
    llvm::ArrayRef<Operation *> consumers) {
  ForkResolutionRecord fr;
  fr.opName = producer->getName().getStringRef().str();
  std::string locStr;
  llvm::raw_string_ostream locOS(locStr);
  locOS << producer->getLoc();
  fr.opLocation = locStr;
  fr.chosenCandidateIndex = chosenIdx;
  fr.numConsumers = consumers.size();

  // Look up op index.
  auto it = opToTraceIndex.find(producer);
  if (it != opToTraceIndex.end()) {
    fr.opIndex = it->second;
  }

  // Look up consumer indices.
  for (Operation *consumer : consumers) {
    auto cit = opToTraceIndex.find(consumer);
    if (cit != opToTraceIndex.end()) {
      fr.consumerOpIndices.push_back(cit->second);
    }
  }

  trace.backwardPass.forkResolutions.push_back(std::move(fr));
}

void DecisionTraceObserver::onEdge(Operation *producer, Operation *consumer,
                                   size_t operandIdx, bool hasReshard,
                                   TTNNLayoutAttr reshardLayout) {
  EdgeRecord edge;
  auto pit = opToTraceIndex.find(producer);
  auto cit = opToTraceIndex.find(consumer);
  if (pit != opToTraceIndex.end()) {
    edge.producerOpIndex = pit->second;
  }
  if (cit != opToTraceIndex.end()) {
    edge.consumerOpIndex = cit->second;
  }
  edge.operandIndex = operandIdx;
  edge.hasReshard = hasReshard;
  if (hasReshard) {
    edge.reshardLayout = layoutToString(reshardLayout);
  }
  trace.edges.push_back(std::move(edge));
}

void DecisionTraceObserver::onFinalChoice(Operation *op, size_t opIndex,
                                          const BeamCandidate &chosen) {
  FinalChoiceRecord fc;
  fc.opIndex = opIndex;
  fc.opName = op->getName().getStringRef().str();
  fc.chosenLayout = layoutToString(chosen.config.outputLayout);
  trace.finalChoices.push_back(std::move(fc));
}

void DecisionTraceObserver::onEnd(size_t totalOps) {
  trace.totalOps = totalOps;
}

void DecisionTraceObserver::onInplaceOp(const InplaceOpInfo &info) {
  OpDecisionRecord record;
  record.isInplace = true;
  record.opName = info.op->getName().getStringRef().str();
  std::string locStr;
  llvm::raw_string_ostream locOS(locStr);
  locOS << info.op->getLoc();
  record.opLocation = locStr;

  // Record input candidate sets (one entry per tensor operand).
  for (const auto &operand : info.operands) {
    InputCandidateSetRecord icr;
    icr.operandIndex = operand.operandIdx;
    icr.fromProducerBeam = operand.producerOp ? 1 : 0;
    icr.fromReshard = 0;
    if (operand.layout) {
      icr.candidates.push_back(layoutToString(operand.layout));
    }
    record.inputCandidateSets.push_back(std::move(icr));
  }

  trace.forwardPass.push_back(std::move(record));
  size_t idx = trace.forwardPass.size() - 1;
  trace.forwardPass.back().opIndex = idx;
  opToTraceIndex[info.op] = idx;

  // Emit edges from each tracked producer to this in-place op.
  for (const auto &operand : info.operands) {
    if (!operand.producerOp) {
      continue;
    }
    auto pit = opToTraceIndex.find(operand.producerOp);
    if (pit == opToTraceIndex.end()) {
      continue;
    }
    EdgeRecord edge;
    edge.producerOpIndex = pit->second;
    edge.consumerOpIndex = idx;
    edge.operandIndex = operand.operandIdx;
    edge.hasReshard = false;
    trace.edges.push_back(std::move(edge));
  }
}

//===----------------------------------------------------------------------===//
// DecisionTraceObserver -- L1SpillObserver callbacks
//===----------------------------------------------------------------------===//

void DecisionTraceObserver::onSpillStart(llvm::StringRef funcName,
                                         uint64_t budget,
                                         size_t scheduleSize) {
  trace.spillManagement.budget = budget;
  trace.spillManagement.scheduleSize = scheduleSize;
  spillOccupiedBefore = 0;
}

void DecisionTraceObserver::onDeadRemoval(Operation *op, int64_t pos,
                                          uint64_t occupiedAfter) {
  SpillEventRecord event;
  event.position = static_cast<size_t>(pos);
  event.opName = ttmlir::opToString(op);
  event.action = "dead_removal";
  event.occupiedL1Before = spillOccupiedBefore;
  event.occupiedL1After = occupiedAfter;
  trace.spillManagement.events.push_back(std::move(event));
  spillOccupiedBefore = occupiedAfter;
}

void DecisionTraceObserver::onLiveAdded(Operation *op, int64_t pos,
                                        uint64_t opL1Usage, int64_t lastUse,
                                        uint64_t occupiedAfter) {
  SpillEventRecord event;
  event.position = static_cast<size_t>(pos);
  event.opName = ttmlir::opToString(op);
  event.action = "live_added";
  event.occupiedL1Before = spillOccupiedBefore;
  event.occupiedL1After = occupiedAfter;
  event.opL1Usage = opL1Usage;
  std::string detail;
  llvm::raw_string_ostream os(detail);
  os << "lastUse=" << lastUse;
  event.details = detail;
  trace.spillManagement.events.push_back(std::move(event));
  spillOccupiedBefore = occupiedAfter;
}

void DecisionTraceObserver::onOOM(Operation *op, int64_t pos,
                                  uint64_t occupiedL1) {
  SpillEventRecord event;
  event.position = static_cast<size_t>(pos);
  event.opName = ttmlir::opToString(op);
  event.action = "oom";
  event.occupiedL1Before = occupiedL1;
  event.occupiedL1After = occupiedL1;
  trace.spillManagement.events.push_back(std::move(event));
  spillOccupiedBefore = occupiedL1;
}

void DecisionTraceObserver::onDemotion(Operation *op, int64_t pos,
                                       bool success, uint64_t newL1Usage) {
  SpillEventRecord event;
  event.position = static_cast<size_t>(pos);
  event.opName = ttmlir::opToString(op);
  event.action = success ? "demotion_success" : "demotion_failed";
  event.occupiedL1Before = spillOccupiedBefore;
  event.opL1Usage = newL1Usage;
  // occupiedL1After will be updated by subsequent onLiveAdded.
  event.occupiedL1After = spillOccupiedBefore;
  trace.spillManagement.events.push_back(std::move(event));
}

void DecisionTraceObserver::onEviction(Operation *victim, int64_t pos,
                                       uint64_t freedBytes) {
  SpillEventRecord event;
  event.position = static_cast<size_t>(pos);
  event.opName = ttmlir::opToString(victim);
  event.action = "eviction";
  event.occupiedL1Before = spillOccupiedBefore;
  event.occupiedL1After =
      spillOccupiedBefore > freedBytes ? spillOccupiedBefore - freedBytes : 0;
  event.opL1Usage = freedBytes;
  event.victimName = ttmlir::opToString(victim);
  trace.spillManagement.events.push_back(std::move(event));
  spillOccupiedBefore = event.occupiedL1After;
}

void DecisionTraceObserver::onSelfSpill(Operation *op, int64_t pos) {
  SpillEventRecord event;
  event.position = static_cast<size_t>(pos);
  event.opName = ttmlir::opToString(op);
  event.action = "self_spill";
  event.occupiedL1Before = spillOccupiedBefore;
  event.occupiedL1After = spillOccupiedBefore;
  trace.spillManagement.events.push_back(std::move(event));
}

void DecisionTraceObserver::onRevalidationCascade(Operation *changed,
                                                  Operation *consumer,
                                                  bool outputChanged) {
  if (!outputChanged) {
    return;
  }
  SpillEventRecord event;
  event.opName = ttmlir::opToString(consumer);
  event.action = "revalidation";
  event.victimName = ttmlir::opToString(changed);
  event.details = "output layout changed after upstream spill/demotion";
  trace.spillManagement.events.push_back(std::move(event));
}

void DecisionTraceObserver::onSpillEnd(size_t totalSpills,
                                       uint64_t finalOccupied,
                                       size_t liveTensors) {
  trace.spillManagement.totalSpills = totalSpills;
  trace.spillManagement.finalOccupied = finalOccupied;
  trace.spillManagement.finalLiveTensors = liveTensors;
}

} // namespace mlir::tt::ttnn
