// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNRecoverStructureUtils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <queue>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNRECOVERSTRUCTURE
#define GEN_PASS_DEF_TTNNSIMPLIFYLOCSFORCODEGEN
#define GEN_PASS_DEF_TTNNREMOVEDEALLOCS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNRecoverStructure
    : public impl::TTNNRecoverStructureBase<TTNNRecoverStructure> {

public:
  using impl::TTNNRecoverStructureBase<
      TTNNRecoverStructure>::TTNNRecoverStructureBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // For simplicity, supporting only one for now, but could support multiple.
    // TODO (#6635): Add support for multiple candidate functions.
    //
    SmallVector<func::FuncOp> candidateFns = findCandidateFns(moduleOp);
    if (candidateFns.size() != 1) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                   "Only one candidate fn is supported now, but got {}",
                   candidateFns.size());
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Candidate fns: {}",
                   llvm::join(llvm::map_range(
                                  candidateFns,
                                  [](func::FuncOp fn) { return fn.getName(); }),
                              ", "));
      signalPassFailure();
    }
    func::FuncOp candidateFn = candidateFns.front();

    llvm::DenseMap<Operation *, PyLoc> opToLocation =
        parseOpsAndGatherLocations(candidateFn);

    logPyLocs(candidateFn, opToLocation);

    // Validate locations.
    //
    std::string errorMessage = validateLocations(opToLocation);
    if (!errorMessage.empty()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                   "RecoverStructure error: {}", errorMessage);
      signalPassFailure();
    }

    // Group operations by function.
    //
    auto funcGroups = groupOperationsByFunction(candidateFn, opToLocation);
    logFunctionGroups(funcGroups);

    // Analyze function boundaries and data flow.
    //
    auto boundaryInfos = analyzeFunctionBoundaries(funcGroups);
    logFunctionBoundaries(boundaryInfos);

    // Create new functions based on boundary information.
    //
    auto newFunctions =
        createNewFunctions(rewriter, candidateFn, boundaryInfos, funcGroups);
    logNewFunctions(newFunctions);

    // Populate function bodies with operations.
    //
    populateFunctionBodies(rewriter, boundaryInfos, newFunctions);

    // Replace operations in original function with calls to new functions.
    //
    replaceWithFunctionCalls(rewriter, candidateFn, boundaryInfos,
                             newFunctions);
  }

private:
  bool isCandidateFn(func::FuncOp funcOp) {
    return ttmlir::utils::isForwardDeviceFunc(funcOp) &&
           !funcOp.getFunctionType().getInputs().empty();
  }

  bool isCandidateOp(Operation *op) {
    // Check if ttnn op
    return isa<ttnn::TTNNDialect>(op->getDialect());
  }

  SmallVector<func::FuncOp> findCandidateFns(ModuleOp moduleOp) {
    SmallVector<func::FuncOp, 1> candidateFns;
    moduleOp->walk([&](func::FuncOp funcOp) {
      logFnInfo(funcOp);

      if (isCandidateFn(funcOp)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "Adding candidate fn: {}", funcOp.getName());
        candidateFns.push_back(funcOp);
      }
    });
    return candidateFns;
  }

  llvm::DenseMap<Operation *, PyLoc>
  parseOpsAndGatherLocations(func::FuncOp funcOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "Parsing ops and gathering locations for function: {}",
                 funcOp.getName());

    llvm::DenseMap<Operation *, PyLoc> opToLocation;

    funcOp.walk([&](Operation *op) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Parsing op: {}",
                   op->getName());
      if (!isCandidateOp(op)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\tOp is not a candidate op: {}", op->getName());
        return WalkResult::advance();
      }

      PyLoc pyLoc(op);
      if (pyLoc.isValid) {
        opToLocation.insert({op, pyLoc});
      } else {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\tPyLoc is INVALID for op: {}", op->getName());
      }
      return WalkResult::advance();
    });

    return opToLocation;
  }

  std::string
  validateLocations(const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    llvm::StringMap<PyLoc> funcPathMap;
    for (const auto &entry : opToLocation) {
      const PyLoc &pyLoc = entry.second;
      auto [it, success] = funcPathMap.try_emplace(pyLoc.funcPath, pyLoc);
      if (!success && it->second.funcName != pyLoc.funcName) {
        return "Found different func names at the same path: " +
               pyLoc.funcPath + " (first func name: " + it->second.funcName +
               ", second func name: " + pyLoc.funcName + ")";
      }
    }
    return "";
  }

  // Check if an operation is a root operation (has no internal dependencies).
  // Root operations are those whose operands are not defined within
  // opToLocation.
  //
  bool isRootOperation(Operation *op,
                       const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    for (Value operand : op->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp && opToLocation.count(definingOp)) {
        return false; // Has internal dependency
      }
    }
    return true; // No internal dependencies
  }

  // Check if all dependencies of an operation have been processed.
  // Returns true if all operands defined in opToLocation are in processedOps.
  //
  bool areAllDependenciesSatisfied(
      Operation *op, const llvm::DenseMap<Operation *, PyLoc> &opToLocation,
      const llvm::DenseSet<Operation *> &processedOps) {
    for (Value operand : op->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp && opToLocation.count(definingOp)) {
        if (!processedOps.count(definingOp)) {
          return false;
        }
      }
    }
    return true;
  }

  // Finalize and save a function group to the funcGroups map.
  // Sets the group name, generates it from module hierarchy, and stores it.
  //
  void finalizeAndSaveGroup(FuncGroup &group,
                            llvm::StringMap<FuncGroup> &funcGroups,
                            const std::string &funcPath,
                            unsigned &groupIndexCounter) {
    // Set default name, then generate one based on module hierarchy
    group.funcName = "forward_" + std::to_string(groupIndexCounter);
    group.index = groupIndexCounter;
    group.generateFuncName();

    // Create unique key for this group
    std::string uniqueKey =
        funcPath + "_group_" + std::to_string(groupIndexCounter++);
    funcGroups[uniqueKey] = group;

    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "Saved group: {}, unique key: {}", group.funcName, uniqueKey);
  }

  // Initialize the priority queue with root operations (those with no internal
  // dependencies). Root operations are the starting points for topological
  // sort.
  //
  void initializeRootOperations(
      const llvm::DenseMap<Operation *, PyLoc> &opToLocation,
      std::priority_queue<OpPyLoc, SmallVector<OpPyLoc>, CompareOpPyLoc>
          &availableOps,
      llvm::DenseSet<Operation *> &inQueue,
      llvm::DenseMap<Operation *, OpPyLoc> &opToOpPyLoc) {

    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "Adding root operations to priority queue");

    for (const auto &entry : opToLocation) {
      Operation *op = entry.first;
      const PyLoc &pyLoc = entry.second;

      // Skip deallocate operations
      if (isa<ttnn::DeallocateOp>(op)) {
        continue;
      }

      // Check if this is a root operation (no internal dependencies)
      if (isRootOperation(op, opToLocation)) {
        OpPyLoc rootOpPyLoc = {op, pyLoc, 0}; // distance = 0 for roots
        availableOps.push(rootOpPyLoc);
        inQueue.insert(op);
        opToOpPyLoc.insert({op, rootOpPyLoc});
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\tAdded root operation: {} - {} - {}", op->getName(),
                     pyLoc.opIndex, pyLoc.funcPath);
      }
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "Total added root operations: {}", inQueue.size());
  }

  // Group operations by their source function location (funcPath) using
  // topological sort with priority-based ordering.
  //
  // Algorithm Overview:
  // 1. Initialize with "root" operations (those with no internal dependencies)
  // 2. Process operations in priority order:
  //    - Operations from current group have highest priority
  //    - Non-deallocate ops prioritized over deallocates
  //    - Closer to root (lower distanceFromRoot) breaks ties
  // 3. When funcPath changes, finalize current group and start new one
  // 4. Add dependent operations to queue once all their dependencies are
  // satisfied
  //
  // Returns: Map from unique group key to FuncGroup containing operations
  llvm::StringMap<FuncGroup> groupOperationsByFunction(
      func::FuncOp &candidateFn,
      const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    llvm::StringMap<FuncGroup> funcGroups;

    // Track current group's funcPath to detect boundary changes
    std::string currentGroupFuncPath = "";
    std::string currentFuncName = "";
    FuncGroup currentGroup;
    unsigned groupIndexCounter = 0;

    // Track which operations have been processed (across all groups)
    llvm::DenseSet<Operation *> processedOps;

    // Track which operations are in the priority queue (to avoid duplicates)
    llvm::DenseSet<Operation *> opsInQueueSet;

    // Map: Op -> OpPyLoc (used for debug logging)
    llvm::DenseMap<Operation *, OpPyLoc> opToOpPyLoc;

    // Initialize comparator with reference to current group's funcPath.
    // The comparator prioritizes operations in this order:
    //   1. Operations from currentGroupFuncPath (stay in current group)
    //   2. Non-deallocate operations (deallocates processed last)
    //   3. Lower distanceFromRoot (operations closer to roots)
    CompareOpPyLoc comparator(&currentGroupFuncPath);
    std::priority_queue<OpPyLoc, SmallVector<OpPyLoc>, CompareOpPyLoc>
        readyOpsQueue(comparator);

    // Phase 1: Initialize priority queue with root operations
    initializeRootOperations(opToLocation, readyOpsQueue, opsInQueueSet,
                             opToOpPyLoc);

    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "Starting graph traversal");

    // Phase 2: Process operations in topological order, grouping by funcPath
    while (!readyOpsQueue.empty()) {
      OpPyLoc opPyLoc = readyOpsQueue.top();
      readyOpsQueue.pop();
      opsInQueueSet.erase(opPyLoc.op);
      opToOpPyLoc.erase(opPyLoc.op);

      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                   "\tProcessing op: {} - {} - {}", opPyLoc.op->getName(),
                   opPyLoc.distanceFromRoot, opPyLoc.pyLoc.funcPath);

      // Skip if already processed (can happen with duplicates in queue).
      //
      if (processedOps.count(opPyLoc.op)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\t\tOp already processed, continuing");
        continue;
      }

      const PyLoc &pyLoc = opPyLoc.pyLoc;

      // Detect group boundary: when funcPath changes, we've moved to a
      // different source function, so finalize the current group and start a
      // new one
      bool isGroupBoundary = !currentGroupFuncPath.empty() &&
                             pyLoc.funcPath != currentGroupFuncPath;
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                   "\tIs group boundary: {}", isGroupBoundary);
      if (isGroupBoundary) {
        // Log current group.
        //
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "Current group: {}", currentGroup.opPyLocs.size());
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tFunc path: {}",
                     currentGroupFuncPath);
        for ([[maybe_unused]] const OpPyLoc &opPyLoc : currentGroup.opPyLocs) {
          // Log pyloc op name and modules
          TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                       "\t- {} (modules: {})", opPyLoc.op->getName(),
                       llvm::join(llvm::map_range(opPyLoc.pyLoc.modules,
                                                  [](const auto &m) {
                                                    return m.toString();
                                                  }),
                                  ", "));
        }
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\tOps currently in queue: {}", opsInQueueSet.size());
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\t\t- {} - {} - {}", opPyLoc.op->getName(),
                     opPyLoc.distanceFromRoot, opPyLoc.pyLoc.funcPath);

        // If op is not in opToOpPyLoc, signal pass failure.
        //
        for (Operation *op : opsInQueueSet) {
          auto it = opToOpPyLoc.find(op);
          if (it == opToOpPyLoc.end()) {
            TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                         "DIDN'T FIND OP IN OPTOOPPYLOC");
            signalPassFailure();
          }

          [[maybe_unused]] const OpPyLoc &opPyLoc = it->second;
          TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                       "\t\t- {} - {} - {}", opPyLoc.op->getName(),
                       opPyLoc.distanceFromRoot, opPyLoc.pyLoc.funcPath);
        }

        // Finalize and save the current group
        finalizeAndSaveGroup(currentGroup, funcGroups, currentGroupFuncPath,
                             groupIndexCounter);
        currentGroup = FuncGroup(); // Start fresh group
      }

      // Add current op to the group.
      //
      currentGroup.opPyLocs.push_back(opPyLoc);
      processedOps.insert(opPyLoc.op);
      currentGroupFuncPath = pyLoc.funcPath;
      currentFuncName = pyLoc.funcName;

      // Add dependent operations to the priority queue.
      //
      for (Value result : opPyLoc.op->getResults()) {
        for (Operation *userOp : result.getUsers()) {
          // Skip if already processed or in queue.
          //
          if (processedOps.count(userOp) || opsInQueueSet.count(userOp)) {
            continue;
          }

          // Only add if it's in our operation set.
          //
          auto userLocIt = opToLocation.find(userOp);
          if (userLocIt != opToLocation.end()) {
            // Check if all dependencies are now satisfied.
            //
            if (areAllDependenciesSatisfied(userOp, opToLocation,
                                            processedOps)) {
              // Add to priority queue with incremented distance.
              // TODO (svuckovic): what if there's multiple paths to the same
              // op, how should we calculate the distance? Needs more thought.
              //
              readyOpsQueue.push(
                  {userOp, userLocIt->second, opPyLoc.distanceFromRoot + 1});
              opsInQueueSet.insert(userOp);
              opToOpPyLoc.insert(
                  {userOp,
                   {userOp, userLocIt->second, opPyLoc.distanceFromRoot + 1}});
            }
          }
        }
      }
    }

    // Phase 3: Save the final group if non-empty
    if (!currentGroup.opPyLocs.empty()) {
      finalizeAndSaveGroup(currentGroup, funcGroups, currentGroupFuncPath,
                           groupIndexCounter);
    }
    return funcGroups;
  }

  // Analyze data flow for each function group.
  //
  llvm::StringMap<FunctionBoundaryInfo>
  analyzeFunctionBoundaries(const llvm::StringMap<FuncGroup> &funcGroups) {
    llvm::StringMap<FunctionBoundaryInfo> boundaryInfos;

    for (const auto &[funcPath, funcGroup] : funcGroups) {
      FunctionBoundaryInfo info;
      info.funcName = funcGroup.funcName;
      info.funcPath = funcPath;
      info.opPyLocs = funcGroup.opPyLocs;

      // Build a set of operations in this group for fast lookup.
      //
      llvm::DenseSet<Operation *> opsInGroup;
      for (const OpPyLoc &opPyLoc : funcGroup.opPyLocs) {
        opsInGroup.insert(opPyLoc.op);
      }

      // Track all values defined within this group.
      //
      llvm::DenseSet<Value> valuesDefinedInGroup;
      for (const OpPyLoc &opPyLoc : funcGroup.opPyLocs) {
        for (Value result : opPyLoc.op->getResults()) {
          valuesDefinedInGroup.insert(result);
        }
      }

      // Analyze each operation's operands and results.
      //
      llvm::DenseSet<Value> inputValuesSet;
      llvm::DenseSet<Value> outputValuesSet;
      llvm::DenseSet<Value> internalValuesSet;

      for (const OpPyLoc &opPyLoc : funcGroup.opPyLocs) {
        // Check operands (inputs to the op).
        //
        for (Value operand : opPyLoc.op->getOperands()) {
          // If this value is not defined in this group, it's an input.
          //
          if (!valuesDefinedInGroup.contains(operand)) {
            inputValuesSet.insert(operand);
          }
        }

        // Check results (outputs from the op).
        //
        for (Value result : opPyLoc.op->getResults()) {
          bool usedOutside = false;
          bool usedInside = false;

          // Check if this result is used outside this group.
          //
          for (Operation *user : result.getUsers()) {
            if (opsInGroup.contains(user)) {
              usedInside = true;
            } else {
              usedOutside = true;
            }
          }

          if (usedOutside) {
            outputValuesSet.insert(result);
          } else if (usedInside) {
            internalValuesSet.insert(result);
          }
          // Note: if a value is not used at all, we don't track it.
        }
      }

      info.inputValues.assign(inputValuesSet.begin(), inputValuesSet.end());
      info.outputValues.assign(outputValuesSet.begin(), outputValuesSet.end());
      info.internalValues.assign(internalValuesSet.begin(),
                                 internalValuesSet.end());

      boundaryInfos[funcPath] = std::move(info);
    }

    return boundaryInfos;
  }

  // Create new function declarations based on boundary information.
  //
  llvm::StringMap<func::FuncOp>
  createNewFunctions(IRRewriter &rewriter, func::FuncOp candidateFn,
                     const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
                     const llvm::StringMap<FuncGroup>
                         &funcGroups) { // Needed for the execution order

    llvm::StringMap<func::FuncOp> newFunctions;

    // Track function name counts to generate unique names.
    //
    llvm::StringMap<unsigned> funcNameCounts;

    // Vector for sorting the keys by index in ascending order
    //
    SmallVector<StringRef> sortedFuncPaths;
    for (const auto &entry : boundaryInfos) {
      sortedFuncPaths.push_back(entry.getKey());
    }
    std::sort(sortedFuncPaths.begin(), sortedFuncPaths.end(),
              [&](StringRef a, StringRef b) {
                // Using .find() since the key always exists
                //
                return funcGroups.find(a)->second.index <
                       funcGroups.find(b)->second.index;
              });

    // Setting insertion point to start after _main function
    //
    rewriter.setInsertionPointAfter(candidateFn);

    // Iterate through sorted vector
    //
    for (const auto &funcPath : sortedFuncPaths) {
      const auto &info = boundaryInfos.lookup(funcPath);

      // Generate unique function name with suffix.
      //
      unsigned count = funcNameCounts[info.funcName]++;
      std::string uniqueName = info.funcName + "_" + std::to_string(count);

      // Collect input types from input values.
      //
      SmallVector<Type> inputTypes;
      for (Value input : info.inputValues) {
        inputTypes.push_back(input.getType());
      }

      // Collect output types from output values.
      //
      SmallVector<Type> outputTypes;
      for (Value output : info.outputValues) {
        outputTypes.push_back(output.getType());
      }

      // Create function type.
      //
      FunctionType funcType =
          FunctionType::get(rewriter.getContext(), inputTypes, outputTypes);

      // Create the new function.
      //
      func::FuncOp newFunc = rewriter.create<func::FuncOp>(
          candidateFn.getLoc(), uniqueName, funcType);

      // Mark as private for now.
      //
      newFunc.setPrivate();

      // Set type to ForwardDevice.
      ttmlir::utils::setFunctionType(
          newFunc, ttmlir::utils::FunctionType::ForwardDevice);

      // Store the function in the map.
      //
      newFunctions[funcPath] = newFunc;

      // Set insertion point after the new function
      //
      rewriter.setInsertionPointAfter(newFunc);
    }

    return newFunctions;
  }

  // Populate function bodies by cloning operations from the original function.
  //
  void populateFunctionBodies(
      IRRewriter &rewriter,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
      llvm::StringMap<func::FuncOp> &newFunctions) {

    // For each function, we need to:
    // 1. Create entry block with arguments
    // 2. Clone operations in order
    // 3. Map old values to new values
    // 4. Add return statement
    //
    for (const auto &entry : newFunctions) {
      llvm::StringRef funcPath = entry.getKey();
      func::FuncOp funcOp = entry.getValue();
      const FunctionBoundaryInfo &info = boundaryInfos.lookup(funcPath);

      // Create entry block.
      //
      Block *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // Create value mapping: old values -> new values.
      //
      IRMapping valueMapping;

      // Map input values to block arguments.
      //
      for (size_t i = 0; i < info.inputValues.size(); i++) {
        valueMapping.map(info.inputValues[i], entryBlock->getArgument(i));
      }

      // Clone operations in order.
      //
      for (const OpPyLoc &opPyLoc : info.opPyLocs) {
        Operation *oldOp = opPyLoc.op;
        Operation *newOp = rewriter.clone(*oldOp, valueMapping);

        // Update the value mapping with the results of the cloned operation.
        //
        for (size_t i = 0; i < oldOp->getNumResults(); i++) {
          valueMapping.map(oldOp->getResult(i), newOp->getResult(i));
        }
      }

      // Collect output values (now mapped to new function's values).
      //
      SmallVector<Value> returnValues;
      for (Value oldOutput : info.outputValues) {
        Value newOutput = valueMapping.lookup(oldOutput);
        returnValues.push_back(newOutput);
      }

      // Add return statement.
      //
      rewriter.create<func::ReturnOp>(funcOp.getLoc(), returnValues);
    }
  }

  // Replace operations in the original function with calls to new functions.
  //
  void replaceWithFunctionCalls(
      IRRewriter &rewriter, func::FuncOp candidateFn,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
      const llvm::StringMap<func::FuncOp> &newFunctions) {

    // Build a map from operations to their function info.
    //
    llvm::DenseMap<Operation *,
                   std::pair<llvm::StringRef, const FunctionBoundaryInfo *>>
        opToFuncInfo;

    for (const auto &entry : boundaryInfos) {
      llvm::StringRef funcPath = entry.getKey();
      const FunctionBoundaryInfo &info = entry.getValue();
      for (const OpPyLoc &opPyLoc : info.opPyLocs) {
        opToFuncInfo[opPyLoc.op] = {funcPath, &info};
      }
    }

    // Walk through the original function in order and collect all operations
    // that need to be replaced.
    //
    SmallVector<Operation *> opsToReplace;
    candidateFn.walk([&](Operation *op) {
      if (opToFuncInfo.count(op)) {
        opsToReplace.push_back(op);
      }
    });

    // Collect all operations by funcPath (not just consecutive ones).
    //
    llvm::StringMap<SmallVector<Operation *>> opsByFunction;

    for (Operation *op : opsToReplace) {
      auto [funcPath, info] = opToFuncInfo[op];
      opsByFunction[funcPath].push_back(op);
    }

    // Build dependency graph: funcA -> funcB means funcB depends on funcA
    // (funcB uses values produced by funcA).
    //
    llvm::StringMap<llvm::DenseSet<llvm::StringRef>> dependencies;
    llvm::StringMap<llvm::DenseSet<Operation *>> opsInFunctionSet;

    // First, build sets of operations for each function.
    //
    for (const auto &entry : opsByFunction) {
      llvm::StringRef funcPath = entry.getKey();
      for (Operation *op : entry.getValue()) {
        opsInFunctionSet[funcPath].insert(op);
      }
    }

    // Analyze dependencies: for each function, find which other functions it
    // depends on
    for (const auto &entry : opsByFunction) {
      llvm::StringRef funcPath = entry.getKey();
      llvm::DenseSet<llvm::StringRef> &deps = dependencies[funcPath];

      // Check all operations in this function.
      //
      for (Operation *op : entry.getValue()) {
        // Check all operands of this operation.
        //
        for (Value operand : op->getOperands()) {
          Operation *definingOp = operand.getDefiningOp();
          if (!definingOp) {
            continue; // Block argument, not from another function
          }

          // Find which function this operand comes from.
          //
          for (const auto &otherEntry : opsByFunction) {
            llvm::StringRef otherFuncPath = otherEntry.getKey();
            if (otherFuncPath == funcPath) {
              continue; // Skip self-dependencies
            }

            if (opsInFunctionSet[otherFuncPath].contains(definingOp)) {
              // This function depends on otherFuncPath.
              //
              deps.insert(otherFuncPath);
              break;
            }
          }
        }
      }
    }

    // Topological sort to determine function order.
    //
    SmallVector<llvm::StringRef> functionOrder;
    llvm::DenseSet<llvm::StringRef> visited;
    llvm::DenseSet<llvm::StringRef> inProgress;

    // Visit function.
    //
    std::function<bool(llvm::StringRef)> visit =
        [&](llvm::StringRef funcPath) -> bool {
      if (visited.contains(funcPath)) {
        return true;
      }
      if (inProgress.contains(funcPath)) {
        // Cycle detected! This shouldn't happen in valid IR
        llvm::errs() << "Cycle detected in function dependencies at: "
                     << funcPath << "\n";
        return false;
      }

      inProgress.insert(funcPath);

      // Visit all dependencies first
      if (dependencies.contains(funcPath)) {
        for (llvm::StringRef dep : dependencies[funcPath]) {
          if (!visit(dep)) {
            return false;
          }
        }
      }

      inProgress.erase(funcPath);
      visited.insert(funcPath);
      functionOrder.push_back(funcPath);
      return true;
    };

    // Visit all functions.
    //
    for (const auto &entry : opsByFunction) {
      if (!visit(entry.getKey())) {
        // Error: cycle in dependencies.
        //
        llvm::errs() << "Cycle detected in function dependencies at: "
                     << entry.getKey() << "\n";
        return;
      }
    }

    // Mapping from old values to new values (call results).
    //
    IRMapping valueMapping;

    // Track the last call we created for sequential insertion.
    //
    Operation *lastCallOp = nullptr;

    // Track which operations have been processed.
    //
    llvm::DenseSet<Operation *> processedOps;

    // Process each unique function in the order they first appear.
    //
    for (llvm::StringRef funcPath : functionOrder) {
      const FunctionBoundaryInfo *info =
          opToFuncInfo[opsByFunction[funcPath].front()].second;
      func::FuncOp targetFunc = newFunctions.lookup(funcPath);
      SmallVector<Operation *> &opsInFunction = opsByFunction[funcPath];

      // Prepare operands for the call (input values).
      //
      SmallVector<Value> callOperands;
      for (Value inputValue : info->inputValues) {
        // If this input was produced by a previous call, use the mapped value.
        //
        if (valueMapping.contains(inputValue)) {
          callOperands.push_back(valueMapping.lookup(inputValue));
        } else {
          callOperands.push_back(inputValue);
        }
      }

      // Insert the call after the last call we created (or before first op if
      // this is the first call).
      //
      if (lastCallOp) {
        rewriter.setInsertionPointAfter(lastCallOp);
      } else {
        Operation *firstOp = opsInFunction.front();
        rewriter.setInsertionPoint(firstOp);
      }

      // Create the call.
      //
      Operation *lastOp = opsInFunction.back();
      func::CallOp callOp = rewriter.create<func::CallOp>(
          lastOp->getLoc(), targetFunc, callOperands);

      // Track this call for the next insertion.
      //
      lastCallOp = callOp;

      // Map output values to call results.
      //
      for (size_t i = 0; i < info->outputValues.size(); i++) {
        valueMapping.map(info->outputValues[i], callOp.getResult(i));
      }

      // Replace uses of output values with call results.
      //
      FunctionBoundaryInfo &mutableInfo =
          const_cast<FunctionBoundaryInfo &>(*info);
      for (size_t i = 0; i < mutableInfo.outputValues.size(); i++) {
        mutableInfo.outputValues[i].replaceAllUsesWith(callOp.getResult(i));
      }

      // Mark operations for deletion.
      //
      for (Operation *op : opsInFunction) {
        processedOps.insert(op);
      }
    }

    // Erase replaced operations in reverse order
    for (auto it = opsToReplace.rbegin(); it != opsToReplace.rend(); ++it) {
      if (processedOps.contains(*it)) {
        rewriter.eraseOp(*it);
      }
    }
  }

  // Logging and debugging methods
  //
  void logFnInfo(func::FuncOp funcOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Fn: {}",
                 funcOp.getName());
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tInputs count: {}",
                 funcOp.getFunctionType().getInputs().size());
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tResults count: {}",
                 funcOp.getFunctionType().getResults().size());
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tIs private: {}",
                 funcOp.isPrivate());
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tIs const-eval: {}",
                 ttmlir::utils::isConstEvalFunc(funcOp));
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tIs candidate: {}",
                 isCandidateFn(funcOp));
  }

  void logPyLocs(func::FuncOp candidateFn,
                 const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "Printing PyLocs for function: {}", candidateFn.getName());
    for (const auto &entry : opToLocation) {
      const PyLoc &pyLoc = entry.second;
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tPyLoc: {}",
                   pyLoc.op->getName());
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\tLoc: {}",
                   pyLoc.op->getLoc());
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\tFunc path: {}",
                   pyLoc.funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\tFunc name: {}",
                   pyLoc.funcName);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                   "\t\tOp line num: {}", pyLoc.opLineNum);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\tOp name: {}",
                   pyLoc.opName);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\tModules: {}",
                   pyLoc.modules.size());
      for ([[maybe_unused]] const PyLoc::Module &module : pyLoc.modules) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\t\t\tModule class: {}", module.moduleClass);
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\t\t\tModule name: {}", module.moduleName);
      }
    }
  }

  void logFunctionGroups(const llvm::StringMap<FuncGroup> &funcGroups) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "=== Function Groups ===");
    for ([[maybe_unused]] const auto &[funcPath, funcGroup] : funcGroups) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Function: {}",
                   funcGroup.funcName);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tPath: {}",
                   funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tOperations: {}",
                   funcGroup.opPyLocs.size());
      for ([[maybe_unused]] const OpPyLoc &opPyLoc : funcGroup.opPyLocs) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\t\t- {} (line {})", opPyLoc.op->getName(),
                     opPyLoc.pyLoc.opLineNum);
      }
    }
  }

  void logFunctionBoundaries(
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "=== Function Boundary Analysis ===");
    for ([[maybe_unused]] const auto &[funcPath, info] : boundaryInfos) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Function: {}",
                   info.funcName);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tPath: {}",
                   info.funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tInput values: {}",
                   info.inputValues.size());
      for ([[maybe_unused]] const Value &input : info.inputValues) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\t\t- {} (type: {})", input, input.getType());
      }
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                   "\tOutput values: {}", info.outputValues.size());
      for ([[maybe_unused]] const Value &output : info.outputValues) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                     "\t\t- {} (type: {})", output, output.getType());
      }
    }
  }

  void logNewFunctions(const llvm::StringMap<func::FuncOp> &newFunctions) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "=== Created Functions ===");
    for (const auto &entry : newFunctions) {
      [[maybe_unused]] llvm::StringRef funcPath = entry.getKey();
      [[maybe_unused]] func::FuncOp funcOp = entry.getValue();

      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Function: {}",
                   funcOp.getName());
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tPath: {}",
                   funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tSignature: {}",
                   funcOp.getFunctionType());
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tInput types: {}",
                   funcOp.getFunctionType().getInputs().size());
      for ([[maybe_unused]] const Type &inputType :
           funcOp.getFunctionType().getInputs()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\t- {}",
                     inputType);
      }
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tOutput types: {}",
                   funcOp.getFunctionType().getResults().size());
      for ([[maybe_unused]] const Type &outputType :
           funcOp.getFunctionType().getResults()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\t- {}",
                     outputType);
      }
    }
  }
};

class TTNNSimplifyLocsForCodegen
    : public impl::TTNNSimplifyLocsForCodegenBase<TTNNSimplifyLocsForCodegen> {

private:
  mlir::Location simplifyNameLoc(mlir::Location loc) {
    // If not NameLoc, return pass failure
    if (!isa<NameLoc>(loc)) {
      getOperation()->emitError("Location is not a NameLoc");
      signalPassFailure();
    }

    auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc);

    // Get the name string and create a new NameLoc without nested location.
    //
    return mlir::NameLoc::get(
        // Omitting the second parameter uses UnknownLoc by default
        nameLoc.getName());
  }

public:
  using impl::TTNNSimplifyLocsForCodegenBase<
      TTNNSimplifyLocsForCodegen>::TTNNSimplifyLocsForCodegenBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Walk through all operations in the module.
    //
    moduleOp.walk([&](Operation *op) {
      // Skip UnknownLocs.
      //
      if (isa<UnknownLoc>(op->getLoc())) {
        return WalkResult::advance();
      }

      // Only process TTNN dialect ops.
      //
      if (!isa<ttnn::TTNNDialect>(op->getDialect())) {
        return WalkResult::advance();
      }

      // Simplify the location.
      //
      mlir::Location simplifiedLoc = simplifyNameLoc(op->getLoc());
      op->setLoc(simplifiedLoc);

      return WalkResult::advance();
    });
  }
};

class TTNNRemoveDeallocs
    : public impl::TTNNRemoveDeallocsBase<TTNNRemoveDeallocs> {

public:
  TTNNRemoveDeallocs() = default;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Collect all deallocate operations.
    //
    llvm::SmallVector<Operation *> deallocOps;
    moduleOp.walk([&](ttnn::DeallocateOp deallocOp) {
      deallocOps.push_back(deallocOp.getOperation());
    });

    // Erase all deallocate operations.
    //
    for (Operation *op : deallocOps) {
      op->erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
