// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
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
#define GEN_PASS_DEF_TTNNPRETTIFYFORCODEGEN
#define GEN_PASS_DEF_TTNNSIMPLIFYLOCSFORCODEGEN
#define GEN_PASS_DEF_TTNNREMOVEDEALLOCS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNPrettifyForCodegen
    : public impl::TTNNPrettifyForCodegenBase<TTNNPrettifyForCodegen> {

private:
  // This structure is used to store the location information of an operation in
  // a more user-friendly format.
  //
  struct PyLoc {
    // This structure is used to store the original module information.
    //
    struct Module {
      std::string moduleClass;
      std::string moduleName;

      std::string toString() const {
        return moduleClass + "[" + moduleName + "]";
      }
    };

    // Op location is parsed into these fields.
    //
    int opIndex;
    llvm::SmallVector<Module> modules;
    std::string funcPath;
    std::string funcName;
    int opLineNum;
    std::string opName;

    Operation *op;
    bool isValid;

    PyLoc(Operation *op) {
      this->op = op;

      // Skip unknown locs.
      if (isa<UnknownLoc>(op->getLoc())) {
        this->isValid = false;
        return;
      }

      // Non TTNN dialect ops are not target.
      if (!isa<ttnn::TTNNDialect>(op->getDialect())) {
        this->isValid = false;
        return;
      }

      // Allow NameLocs only.
      if (!isa<NameLoc>(op->getLoc())) {
        // TODO (svuckovic): should we fail the pass for this?
        this->isValid = false;
        return;
      }

      // Get location without "loc(" and ")" characters.
      std::string locStr = locationToStr(op->getLoc());

      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "Parsing loc: {}",
                   locStr);

      // Split locStr by "|" character.
      // For example, given:
      //   "7|Tail[tail]|ReLU[tail.relu]|/localdev/.../test.py:106|forward|107|aten__relu"
      // Return:
      //   ["7", "Tail[tail]", "ReLU[tail.relu]", "/localdev/.../test.py:106",
      //   "forward", "107", "aten__relu"]
      llvm::SmallVector<llvm::StringRef> locParts;
      llvm::StringRef(locStr).split(locParts, "|", -1, false);

      this->opIndex = std::stoi(locParts[0].str());

      // Validate that we have at least 4 parts (funcPath, funcName, opLineNum,
      // opName)
      size_t numParts = locParts.size();
      if (numParts < 4) {
        this->isValid = false;
        return;
      }

      // Fill in fields from back of locParts.
      this->opName = locParts[numParts - 1].str();
      this->opLineNum = std::stoi(locParts[numParts - 2].str());
      this->funcName = locParts[numParts - 3].str();
      this->funcPath = locParts[numParts - 4].str();
      this->modules = llvm::SmallVector<Module>();
      for (size_t i = 1; i < numParts - 4; i++) {
        // Split each module into class and name.
        // For example, given:
        //   "Tail[tail]"
        // Return:
        //   ["Tail", "tail"]
        llvm::SmallVector<llvm::StringRef, 2> moduleParts;
        locParts[i].split(moduleParts, "[", -1, false);
        this->modules.push_back(
            Module{/* moduleClass= */ moduleParts[0].str(),
                   // Remove trailing "]" from module name.
                   /* moduleName= */ moduleParts[1].str().substr(
                       0, moduleParts[1].str().size() - 1)});
      }
      this->isValid = true;
    }
  };

  void printFnInfo(func::FuncOp funcOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "Fn: {}",
                 funcOp.getName());
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tInputs count: {}",
                 funcOp.getFunctionType().getInputs().size());
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "\tResults count: {}",
                 funcOp.getFunctionType().getResults().size());
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tIs private: {}",
                 funcOp.isPrivate());
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "\tIs const-eval: {}", ttmlir::utils::isConstEvalFunc(funcOp));
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tIs candidate: {}",
                 isCandidateFn(funcOp));
  }

  bool isCandidateFn(func::FuncOp funcOp) {
    return !funcOp.isPrivate() && !ttmlir::utils::isConstEvalFunc(funcOp) &&
           !funcOp.getFunctionType().getInputs().empty();
  }

  bool isCandidateOp(Operation *op) {
    // Check if ttnn op
    return isa<ttnn::TTNNDialect>(op->getDialect());
  }

  static std::string locationToStr(const mlir::Location &loc) {
    std::string locStr;
    llvm::raw_string_ostream(locStr) << loc;

    // Remove the loc(" and ") characters
    if (locStr.find("loc(\"") == 0) {
      locStr = locStr.substr(5);
    }
    if (locStr.find("\")") == locStr.size() - 2) {
      locStr = locStr.substr(0, locStr.size() - 2);
    }

    return locStr;
  }

  SmallVector<func::FuncOp> findCandidateFns(ModuleOp moduleOp) {
    SmallVector<func::FuncOp, 1> candidateFns;
    moduleOp->walk([&](func::FuncOp funcOp) {
      printFnInfo(funcOp);

      if (isCandidateFn(funcOp)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "Adding candidate fn: {}", funcOp.getName());
        candidateFns.push_back(funcOp);
      }
    });
    return candidateFns;
  }

  llvm::DenseMap<Operation *, PyLoc>
  parseOpsAndGatherLocations(func::FuncOp funcOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "Parsing ops and gathering locations for function: {}",
                 funcOp.getName());

    llvm::DenseMap<Operation *, PyLoc> opToLocation;

    funcOp.walk([&](Operation *op) {
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "Parsing op: {}",
                   op->getName());
      if (!isCandidateOp(op)) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tOp is not a candidate op: {}", op->getName());
        return WalkResult::advance();
      }

      PyLoc pyLoc(op);
      if (pyLoc.isValid) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tPyLoc is valid for op: {}", op->getName());
        opToLocation.insert({op, pyLoc});
      }
      return WalkResult::advance();
    });

    return opToLocation;
  }

  void logPyLocs(func::FuncOp candidateFn,
                 const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "Printing PyLocs for function: {}", candidateFn.getName());
    for (const auto &entry : opToLocation) {
      const PyLoc &pyLoc = entry.second;
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tPyLoc: {}",
                   pyLoc.op->getName());
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t\tLoc: {}",
                   pyLoc.op->getLoc());
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "\t\tFunc path: {}", pyLoc.funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "\t\tFunc name: {}", pyLoc.funcName);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "\t\tOp line num: {}", pyLoc.opLineNum);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t\tOp name: {}",
                   pyLoc.opName);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t\tModules: {}",
                   pyLoc.modules.size());
      for ([[maybe_unused]] const PyLoc::Module &module : pyLoc.modules) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t\t\t\tModule class: {}", module.moduleClass);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t\t\t\tModule name: {}", module.moduleName);
      }
    }
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

  // This structure is used to store the operation, its PyLoc information, and
  // other metadata relevant for the "prettification" algorithm.
  //
  struct OpPyLoc {
    Operation *op;
    PyLoc pyLoc;
    int distanceFromRoot;
  };

  // This comparator is used to compare two OpPyLoc objects and determine which
  // one should be "worked on" next. Using it helps provide a topological sort
  // that is useful in handling forked paths in graph such that ops chosen first
  // are more likely to belong to the original module.
  //
  struct CompareOpPyLoc {
    std::string *currentFuncName;

    CompareOpPyLoc(std::string *currentFuncName)
        : currentFuncName(currentFuncName) {}

    bool operator()(const OpPyLoc &a, const OpPyLoc &b) const {
      bool comparatorDebug = false;

      bool isASameGroup = a.pyLoc.funcPath == *currentFuncName;
      bool isBSameGroup = b.pyLoc.funcPath == *currentFuncName;

      if (comparatorDebug) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "Comparing {} and {}", a.op->getName(), b.op->getName());
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\ta.pyLoc.funcPath: {}", a.pyLoc.funcPath);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tb.pyLoc.funcPath: {}", b.pyLoc.funcPath);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t*currentFuncName: {}", *currentFuncName);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\ta.distanceFromRoot: {}", a.distanceFromRoot);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tb.distanceFromRoot: {}", b.distanceFromRoot);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tisASameGroup: {}", isASameGroup);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tisBSameGroup: {}", isBSameGroup);
      }

      bool result;
      std::string reason;

      // return false - a wins
      // return true - b wins

      // First break by group
      //
      if (isASameGroup && !isBSameGroup) {
        result = false; // a wins
        reason = "a is same group, b is different (a has higher priority)";
      } else if (!isASameGroup && isBSameGroup) {
        result = true; // b wins
        reason = "b is same group, a is different (b has higher priority)";
      } else if (isASameGroup == isBSameGroup) {
        // Now break by deallocate ops
        //
        bool aIsDeallocate = isa<ttnn::DeallocateOp>(a.op);
        bool bIsDeallocate = isa<ttnn::DeallocateOp>(b.op);
        if (aIsDeallocate && !bIsDeallocate) {
          result = true;
          reason = "a is deallocate, b is not (b has higher priority)";
        } else if (!aIsDeallocate && bIsDeallocate) {
          result = false;
          reason = "b is deallocate, a is not (a has higher priority)";
        } else {
          // Now break by distance to root
          result = a.distanceFromRoot > b.distanceFromRoot;
          reason = result ? "a has greater distance (b wins)"
                          : "b has greater distance (a wins)";
        }
      } else {
        // This should never happen
        llvm::errs()
            << "Both ops are in the same group and are not ttnn.deallocate ops";
        exit(1);
      }

      if (comparatorDebug) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tWinner (closer to front): {} ({})",
                     (result ? b.op->getName() : a.op->getName()), reason);
      }

      return result;
    }
  };

  // This structure is used to store the operations and their PyLocs in a group
  // of operations that belong to the same module/function.
  //
  struct FuncGroup {
    // What the function will be named in the new IR.
    //
    std::string funcName;

    int index = -1;

    // The operations and their PyLocs in the group.
    //
    SmallVector<OpPyLoc> opPyLocs;

    // Generate name based on modules of the ops in the group.
    //
    void generateFuncName() {
      // clang-format off
      //
      // If we were to list module classes for each op, like this:
      //   - modules: ResNetModel ResNetEncoder ResNetStage ResNetBasicLayer Sequential ResNetConvLayer Conv2d
      //   - modules: ResNetModel ResNetEncoder ResNetStage ResNetBasicLayer Sequential ResNetConvLayer BatchNorm2d
      //   - modules: ResNetModel ResNetEncoder ResNetStage ResNetBasicLayer Sequential ResNetConvLayer ReLU
      // we want to find the LCA of them, which in this case is "ResNetConvLayer"
      //
      // clang-format on
      //
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "Func name: {}",
                   funcName);
      for ([[maybe_unused]] const OpPyLoc &opPyLoc : opPyLocs) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t- modules: {}",
                     llvm::join(llvm::map_range(
                                    opPyLoc.pyLoc.modules,
                                    [](const auto &m) { return m.toString(); }),
                                ", "));
        for ([[maybe_unused]] const PyLoc::Module &module :
             opPyLoc.pyLoc.modules) {
          TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t\t- {}",
                       module.moduleClass);
        }
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\n");
      }

      if (opPyLocs.empty()) {
        this->funcName = "forward";
        if (index >= 0) {
          this->funcName += "_" + std::to_string(index);
        }
        return;
      }

      // First find the smallest length module class list
      size_t minLength = std::numeric_limits<size_t>::max();
      for (const OpPyLoc &opPyLoc : opPyLocs) {
        if (opPyLoc.pyLoc.modules.size() < minLength) {
          minLength = opPyLoc.pyLoc.modules.size();
        }
      }

      // Then go through the module class list from behind and find the first
      // one that all the ops share
      for (size_t i = minLength - 1; i >= 0; i--) {
        bool allMatch = true;
        std::string moduleClass = opPyLocs[0].pyLoc.modules[i].moduleClass;
        for (const OpPyLoc &opPyLoc : opPyLocs) {
          if (opPyLoc.pyLoc.modules[i].moduleClass != moduleClass) {
            allMatch = false;
            break;
          }
        }

        if (allMatch) {
          this->funcName = moduleClass;
          if (index >= 0) {
            this->funcName += "_" + std::to_string(index);
          }
          return;
        }
      }
    }
  };

  // Group operations by their funcPath.
  // Returns a map: funcPath -> (funcName, FuncGroup)
  llvm::StringMap<FuncGroup> groupOperationsByFunction(
      func::FuncOp &candidateFn,
      const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    llvm::StringMap<FuncGroup> funcGroups;

    // Track previous funcPath to detect changes
    std::string prevFuncPath = "";
    std::string currentFuncName = "";
    FuncGroup currentGroup;
    unsigned groupCounter = 0;

    // Track which operations have been processed (across all groups)
    llvm::DenseSet<Operation *> processedOps;

    // Track which operations are in the priority queue (to avoid duplicates)
    llvm::DenseSet<Operation *> inQueue;

    // Map: Op -> OpPyLoc.
    llvm::DenseMap<Operation *, OpPyLoc> opToOpPyLoc;

    // Keep track of operations that are available to be added to the current
    // group. Use priority queue
    CompareOpPyLoc comparator(&prevFuncPath);
    std::priority_queue<OpPyLoc, SmallVector<OpPyLoc>, CompareOpPyLoc>
        availableOps(comparator);

    // Initialize availableOps with root operations in the function.
    // Root operations are those that have no dependencies within opToLocation.
    //
    for (const auto &entry : opToLocation) {
      Operation *op = entry.first;
      const PyLoc &pyLoc = entry.second;

      // If deallocate op, skip.
      //
      if (isa<ttnn::DeallocateOp>(op)) {
        continue;
      }

      // Check if the operation has internal dependencies (i.e. in current MLIR
      // module).
      //
      bool hasInternalDeps = false;
      for (Value operand : op->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();
        if (definingOp && opToLocation.count(definingOp)) {
          hasInternalDeps = true;
          break;
        }
      }

      if (!hasInternalDeps) {
        // Root operation - add to priority queue with distance 0.
        //
        availableOps.push({op, pyLoc, 0});
        inQueue.insert(op);
        opToOpPyLoc.insert({op, {op, pyLoc, 0}});
      }
    }

    // Print all ops in queue.
    //
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "Root ops in queue: {}", inQueue.size());
    for ([[maybe_unused]] Operation *op : inQueue) {
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t- {}",
                   op->getName());
    }

    for (Operation *op : inQueue) {
      auto it = opToOpPyLoc.find(op);
      assert(it != opToOpPyLoc.end() && "DIDN'T FIND OP IN OPTOOPPYLOC");

      [[maybe_unused]] const OpPyLoc &opPyLoc = it->second;
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t- {} - {} - {}",
                   opPyLoc.op->getName(), opPyLoc.distanceFromRoot,
                   opPyLoc.pyLoc.funcPath);
    }

    while (!availableOps.empty()) {
      OpPyLoc opPyLoc = availableOps.top();
      availableOps.pop();
      inQueue.erase(opPyLoc.op);
      opToOpPyLoc.erase(opPyLoc.op);

      // Skip if already processed (can happen with duplicates in queue).
      //
      if (processedOps.count(opPyLoc.op)) {
        continue;
      }

      const PyLoc &pyLoc = opPyLoc.pyLoc;

      // Check if funcPath changed (make a "cut")
      //
      if (!prevFuncPath.empty() && pyLoc.funcPath != prevFuncPath) {
        // Print current group.
        //
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "Current group: {}", currentGroup.opPyLocs.size());
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tFunc path: {}", prevFuncPath);
        for ([[maybe_unused]] const OpPyLoc &opPyLoc : currentGroup.opPyLocs) {
          // Print pyloc op name and modules
          TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                       "\t- {} (modules: {})", opPyLoc.op->getName(),
                       llvm::join(llvm::map_range(opPyLoc.pyLoc.modules,
                                                  [](const auto &m) {
                                                    return m.toString();
                                                  }),
                                  ", "));
        }
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tOps currently in queue: {}", inQueue.size());
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t\t- {} - {} - {}", opPyLoc.op->getName(),
                     opPyLoc.distanceFromRoot, opPyLoc.pyLoc.funcPath);

        // If op is not in opToOpPyLoc, signal pass failure.
        //
        for (Operation *op : inQueue) {
          auto it = opToOpPyLoc.find(op);
          if (it == opToOpPyLoc.end()) {
            TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                         "DIDNT FIND OP IN OPTOOPPYLOC");
            signalPassFailure();
          }

          [[maybe_unused]] const OpPyLoc &opPyLoc = it->second;
          TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                       "\t\t- {} - {} - {}", opPyLoc.op->getName(),
                       opPyLoc.distanceFromRoot, opPyLoc.pyLoc.funcPath);
        }

        // Set default group name, then try to generate one using location data.
        //
        currentGroup.funcName = "forward_" + std::to_string(groupCounter);
        currentGroup.index = groupCounter;
        currentGroup.generateFuncName();
        std::string uniqueKey =
            prevFuncPath + "_group_" + std::to_string(groupCounter++);
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\tSaving current group: {}, unique key: {}",
                     currentGroup.funcName, uniqueKey);
        funcGroups[uniqueKey] = currentGroup;
        currentGroup = FuncGroup();
      }

      // Add current op to the group.
      //
      currentGroup.opPyLocs.push_back(opPyLoc);
      processedOps.insert(opPyLoc.op);
      prevFuncPath = pyLoc.funcPath;
      currentFuncName = pyLoc.funcName;

      // Add dependent operations to the priority queue.
      //
      for (Value result : opPyLoc.op->getResults()) {
        for (Operation *userOp : result.getUsers()) {
          // Skip if already processed or in queue.
          //
          if (processedOps.count(userOp) || inQueue.count(userOp)) {
            continue;
          }

          // Only add if it's in our operation set.
          //
          auto userLocIt = opToLocation.find(userOp);
          if (userLocIt != opToLocation.end()) {
            // Check if all dependencies are now satisfied.
            //
            bool allDepsSatisfied = true;
            for (Value operand : userOp->getOperands()) {
              Operation *definingOp = operand.getDefiningOp();
              if (definingOp && opToLocation.count(definingOp)) {
                // Check if this dependency has been processed.
                //
                if (!processedOps.count(definingOp)) {
                  allDepsSatisfied = false;
                  break;
                }
              }
            }

            if (allDepsSatisfied) {
              // Add to priority queue with incremented distance.
              // TODO (svuckovic): what if there's multiple paths to the same
              // op, how should we calculate the distance? Needs more thought.
              //
              availableOps.push(
                  {userOp, userLocIt->second, opPyLoc.distanceFromRoot + 1});
              inQueue.insert(userOp);
              opToOpPyLoc.insert(
                  {userOp,
                   {userOp, userLocIt->second, opPyLoc.distanceFromRoot + 1}});
            }
          }
        }
      }
    }

    // Save the last group.
    //
    if (!currentGroup.opPyLocs.empty()) {
      currentGroup.funcName = "forward_" + std::to_string(groupCounter);
      currentGroup.index = groupCounter;
      currentGroup.generateFuncName();
      std::string uniqueKey =
          prevFuncPath + "_group_" + std::to_string(groupCounter++);
      funcGroups[uniqueKey] = currentGroup;

      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "Saving last group: {}, unique key: {}",
                   currentGroup.funcName, uniqueKey);
    }
    return funcGroups;
  }

  void printFunctionGroups(const llvm::StringMap<FuncGroup> &funcGroups) {
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "=== Function Groups ===");
    for ([[maybe_unused]] const auto &[funcPath, funcGroup] : funcGroups) {
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "Function: {}",
                   funcGroup.funcName);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tPath: {}",
                   funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tOperations: {}",
                   funcGroup.opPyLocs.size());
      for ([[maybe_unused]] const OpPyLoc &opPyLoc : funcGroup.opPyLocs) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t\t- {} (line {})", opPyLoc.op->getName(),
                     opPyLoc.pyLoc.opLineNum);
      }
    }
  }

  // Information about data flow for a function group.
  //
  struct FunctionBoundaryInfo {
    std::string funcName;
    std::string funcPath;
    SmallVector<OpPyLoc> opPyLocs;

    // Values that flow INTO this function (used but not defined here).
    //
    SmallVector<Value> inputValues;

    // Values that flow OUT of this function (defined here, used elsewhere).
    //
    SmallVector<Value> outputValues;

    // Values that are internal (defined and used only within this function).
    //
    SmallVector<Value> internalValues;
  };

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

  void printFunctionBoundaries(
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "=== Function Boundary Analysis ===");
    for ([[maybe_unused]] const auto &[funcPath, info] : boundaryInfos) {
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "Function: {}",
                   info.funcName);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tPath: {}",
                   info.funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "\tInput values: {}", info.inputValues.size());
      for ([[maybe_unused]] const Value &input : info.inputValues) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t\t- {} (type: {})", input, input.getType());
      }
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "\tOutput values: {}", info.outputValues.size());
      for ([[maybe_unused]] const Value &output : info.outputValues) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                     "\t\t- {} (type: {})", output, output.getType());
      }
    }
  }

  // Create new function declarations based on boundary information.
  //
  llvm::StringMap<func::FuncOp> createNewFunctions(
      IRRewriter &rewriter, func::FuncOp candidateFn,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    llvm::StringMap<func::FuncOp> newFunctions;

    // Track function name counts to generate unique names.
    //
    llvm::StringMap<unsigned> funcNameCounts;

    for (const auto &[funcPath, info] : boundaryInfos) {
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

      // Set insertion point to after the candidate function in the same parent
      // module.
      //
      rewriter.setInsertionPointAfter(candidateFn);

      // Create the new function.
      //
      func::FuncOp newFunc = rewriter.create<func::FuncOp>(
          candidateFn.getLoc(), uniqueName, funcType);

      // Mark as private for now.
      //
      newFunc.setPrivate();

      // Store the function in the map.
      //
      newFunctions[funcPath] = newFunc;
    }

    return newFunctions;
  }

  void printNewFunctions(const llvm::StringMap<func::FuncOp> &newFunctions) {
    TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                 "=== Created Functions ===");
    for (const auto &entry : newFunctions) {
      [[maybe_unused]] llvm::StringRef funcPath = entry.getKey();
      [[maybe_unused]] func::FuncOp funcOp = entry.getValue();

      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "Function: {}",
                   funcOp.getName());
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tPath: {}",
                   funcPath);
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\tSignature: {}",
                   funcOp.getFunctionType());
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "\tInput types: {}",
                   funcOp.getFunctionType().getInputs().size());
      for ([[maybe_unused]] const Type &inputType :
           funcOp.getFunctionType().getInputs()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t\t- {}",
                     inputType);
      }
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "\tOutput types: {}",
                   funcOp.getFunctionType().getResults().size());
      for ([[maybe_unused]] const Type &outputType :
           funcOp.getFunctionType().getResults()) {
        TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen, "\t\t- {}",
                     outputType);
      }
    }
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

public:
  using impl::TTNNPrettifyForCodegenBase<
      TTNNPrettifyForCodegen>::TTNNPrettifyForCodegenBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // For simplicity, supporting only one for now, but could support multiple.
    //
    SmallVector<func::FuncOp> candidateFns = findCandidateFns(moduleOp);
    if (candidateFns.size() != 1) {
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "Only one candidate fn is supported now, but got {}",
                   candidateFns.size());
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "Candidate fns: {}",
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
      TTMLIR_DEBUG(ttmlir::LogComponent::PrettifyForCodegen,
                   "PrettifyForCodegen error: {}", errorMessage);
      signalPassFailure();
    }

    // Group operations by function.
    //
    auto funcGroups = groupOperationsByFunction(candidateFn, opToLocation);
    printFunctionGroups(funcGroups);

    // Analyze function boundaries and data flow.
    //
    auto boundaryInfos = analyzeFunctionBoundaries(funcGroups);
    printFunctionBoundaries(boundaryInfos);

    // Create new functions based on boundary information.
    //
    auto newFunctions =
        createNewFunctions(rewriter, candidateFn, boundaryInfos);
    printNewFunctions(newFunctions);

    // Populate function bodies with operations.
    //
    populateFunctionBodies(rewriter, boundaryInfos, newFunctions);

    // Replace operations in original function with calls to new functions.
    //
    replaceWithFunctionCalls(rewriter, candidateFn, boundaryInfos,
                             newFunctions);
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
