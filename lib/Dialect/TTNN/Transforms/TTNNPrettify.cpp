// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
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
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNPrettifyForCodegen
    : public impl::TTNNPrettifyForCodegenBase<TTNNPrettifyForCodegen> {

private:
  // @dataclass
  // class LocationModuleCodegen:
  //     module_class: str
  //     module_name: str

  // @dataclass
  // class LocationCodegen:
  //     modules: list[LocationModuleCodegen]
  //     func_path: str
  //     func_name: str
  //     op_line_num: int
  //     op_name: str
  struct PyLoc {
    struct Module {
      std::string moduleClass;
      std::string moduleName;
    };

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

      // Allow NameLocs only
      if (!isa<NameLoc>(op->getLoc())) {
        // TODO (svuckovic): should we fail the pass for this?
        this->isValid = false;
        return;
      }

      // Simplify the location - remove nested locations.
      // std::cout << "Simplifying loc: " << "\n";
      // op->getLoc().print(llvm::outs());
      // std::cout << "\n";
      mlir::Location simplifiedLoc = simplifyNameLoc(op->getLoc());
      // std::cout << "Simplified loc: " << "\n";
      // simplifiedLoc.print(llvm::outs());
      // std::cout << "\n";

      // Get location without "loc(" and ")" characters.
      std::string locStr = locationToStr(simplifiedLoc);

      // op->dump();
      // std::cout << "Parsing loc: " << locStr << "\n";

      // Split locStr by "|" character.
      // For example, given:
      //   "Tail[tail]|ReLU[tail.relu]|/localdev/.../test.py:106|forward|107|aten__relu"
      // Return:
      //   ["Tail[tail]", "ReLU[tail.relu]", "/localdev/.../test.py:106",
      //   "forward", "107", "aten__relu"]
      llvm::SmallVector<llvm::StringRef, 5> locParts;
      llvm::StringRef(locStr).split(locParts, "|", -1, false);

      // Validate that we have at least 4 parts (funcPath, funcName, opLineNum,
      // opName)
      size_t n = locParts.size();
      // if (n < 4) {
      //   this->isValid = false;
      //   return;
      // }

      // Fill in fields from back of locParts.
      this->opName = locParts[n - 1].str();
      this->opLineNum = std::stoi(locParts[n - 2].str());
      this->funcName = locParts[n - 3].str();
      this->funcPath = locParts[n - 4].str();
      this->modules = llvm::SmallVector<Module>();
      for (size_t i = 0; i < n - 4; i++) {
        // Split each module into class and name.
        // For example, given:
        //   "Tail[tail]"
        // Return:
        //   ["Tail", "tail"]
        llvm::SmallVector<llvm::StringRef, 2> moduleParts;
        locParts[i].split(moduleParts, "[", -1, false);
        // std::cout << locStr << "\n";
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
    llvm::outs() << "Fn: " << funcOp.getName() << "\n";
    std::cout << "  Inputs count: "
              << funcOp.getFunctionType().getInputs().size() << "\n";
    llvm::outs() << "  Results count: "
                 << funcOp.getFunctionType().getResults().size() << "\n";
    llvm::outs() << "  Is private: " << funcOp.isPrivate() << "\n";
    llvm::outs() << "  Is const-eval: "
                 << ttmlir::utils::isConstEvalFunc(funcOp) << "\n";
    llvm::outs() << "  Is candidate: " << isCandidateFn(funcOp) << "\n";
  }

  bool isCandidateFn(func::FuncOp funcOp) {
    return !funcOp.isPrivate() && !ttmlir::utils::isConstEvalFunc(funcOp) &&
           !funcOp.getFunctionType().getInputs().empty();
  }

  bool isCandidateOp(Operation *op) {
    // Check if ttnn op
    return isa<ttnn::TTNNDialect>(op->getDialect());
  }

  static mlir::Location simplifyNameLoc(mlir::Location loc) {
    if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
      // Get the name string and create a new NameLoc without nested location
      return mlir::NameLoc::get(
          nameLoc.getName()
          // Omitting the second parameter uses UnknownLoc by default
      );
    }
    return loc;
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
      // printFnInfo(funcOp);
      if (isCandidateFn(funcOp)) {
        candidateFns.push_back(funcOp);
      }
    });
    return candidateFns;
  }

  llvm::DenseMap<Operation *, PyLoc>
  parseOpsAndGatherLocations(func::FuncOp funcOp) {
    llvm::DenseMap<Operation *, PyLoc> opToLocation;

    funcOp.walk([&](Operation *op) {
      // llvm::outs() << "Op: " << op->getName() << "\n";
      if (!isCandidateOp(op)) {
        return WalkResult::advance();
      }

      PyLoc pyLoc(op);
      if (pyLoc.isValid) {
        opToLocation.insert({op, pyLoc});
      }
      return WalkResult::advance();
    });

    return opToLocation;
  }

  void printPyLocs(func::FuncOp candidateFn,
                   const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    for (const auto &entry : opToLocation) {
      const PyLoc &pyLoc = entry.second;
      llvm::outs() << "PyLoc: " << pyLoc.op->getName() << "\n";
      llvm::outs() << "  Loc: " << pyLoc.op->getLoc() << "\n";
      llvm::outs() << "  Func path: " << pyLoc.funcPath << "\n";
      llvm::outs() << "  Func name: " << pyLoc.funcName << "\n";
      llvm::outs() << "  Op line num: " << pyLoc.opLineNum << "\n";
      llvm::outs() << "  Op name: " << pyLoc.opName << "\n";
      llvm::outs() << "  Modules: " << pyLoc.modules.size() << "\n";
      for (const PyLoc::Module &module : pyLoc.modules) {
        llvm::outs() << "    Module: " << module.moduleClass << "["
                     << module.moduleName << "]\n";
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

  struct OpPyLoc {
    Operation *op;
    PyLoc pyLoc;
    int distanceFromRoot;
  };

  struct CompareOpPyLoc {
    std::string *currentFuncName;

    CompareOpPyLoc(std::string *currentFuncName)
        : currentFuncName(currentFuncName) {}

    bool operator()(const OpPyLoc &a, const OpPyLoc &b) const {
      bool debug = false;

      bool isASameGroup = a.pyLoc.funcPath == *currentFuncName;
      bool isBSameGroup = b.pyLoc.funcPath == *currentFuncName;

      if (debug) {
        llvm::outs() << "Comparing " << a.op->getName() << " and "
                     << b.op->getName() << "\n";
        llvm::outs() << "  a.pyLoc.funcPath: " << a.pyLoc.funcPath << "\n";
        llvm::outs() << "  b.pyLoc.funcPath: " << b.pyLoc.funcPath << "\n";
        llvm::outs() << "  *currentFuncName: " << *currentFuncName << "\n";
        llvm::outs() << "  a.distanceFromRoot: " << a.distanceFromRoot << "\n";
        llvm::outs() << "  b.distanceFromRoot: " << b.distanceFromRoot << "\n";

        llvm::outs() << "  isASameGroup: " << isASameGroup << "\n";
        llvm::outs() << "  isBSameGroup: " << isBSameGroup << "\n";
      }

      bool result;
      std::string reason;

      if (isASameGroup && isBSameGroup) {
        result = a.distanceFromRoot >
                 b.distanceFromRoot; // b wins if a distance is greater
        reason = result ? "a has greater distance (lower priority)"
                        : "b has greater distance (lower priority)";
      } else if (isASameGroup && !isBSameGroup) {
        result = false; // a wins
        reason = "a is same group, b is different (a has lower priority)";
      } else if (!isASameGroup && isBSameGroup) {
        result = true; // b wins
        reason = "b is same group, a is different (b has lower priority)";
      } else {
        // This arbitrarily chooses the next op when there's a tie.
        result = a.pyLoc.funcPath < b.pyLoc.funcPath;
        reason = result ? "a.funcPath < b.funcPath (a has lower priority)"
                        : "b.funcPath < a.funcPath (b has lower priority)";
      }

      if (debug) {
        llvm::outs() << "  Winner (closer to front): "
                     << (result ? b.op->getName() : a.op->getName()) << " ("
                     << reason << ")\n";
      }

      return result;
    }
  };

  // Group operations by their funcPath.
  // Returns a map: funcPath -> (funcName, vector of PyLocs)
  llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
  groupOperationsByFunction(
      func::FuncOp &candidateFn,
      const llvm::DenseMap<Operation *, PyLoc> &opToLocation) {
    llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>> funcGroups;

    // Track previous funcPath to detect changes
    std::string prevFuncPath = "";
    std::string currentFuncName = "";
    SmallVector<PyLoc> currentGroup;
    unsigned groupCounter = 0;

    // Track which operations have been processed (across all groups)
    llvm::DenseSet<Operation *> processedOps;

    // Track which operations are in the priority queue (to avoid duplicates)
    llvm::DenseSet<Operation *> inQueue;

    // map op -> oppyloc
    llvm::DenseMap<Operation *, OpPyLoc> opToOpPyLoc;

    // Keep track of operations that are available to be added to the current
    // group. Use priority queue
    CompareOpPyLoc comparator(&prevFuncPath);
    std::priority_queue<OpPyLoc, SmallVector<OpPyLoc>, CompareOpPyLoc>
        availableOps(comparator);

    // Initialize availableOps with root operations in the function.
    // Root operations are those that have no dependencies within opToLocation
    for (const auto &entry : opToLocation) {
      Operation *op = entry.first;
      const PyLoc &pyLoc = entry.second;

      // If deallocate op, skip
      if (isa<ttnn::DeallocateOp>(op)) {
        continue;
      }

      bool hasInternalDeps = false;
      for (Value operand : op->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();
        if (definingOp && opToLocation.count(definingOp)) {
          hasInternalDeps = true;
          break;
        }
      }

      if (!hasInternalDeps) {
        // Root operation - add to priority queue with distance 0
        availableOps.push({op, pyLoc, 0});
        inQueue.insert(op);
        opToOpPyLoc.insert({op, {op, pyLoc, 0}});
      }
    }

    // Print all ops in queue
    llvm::outs() << "Root ops in queue: " << inQueue.size() << "\n";
    // for (Operation *op : inQueue) {
    //   llvm::outs() << "  - " << op->getName() << "\n";
    // }

    for (Operation *op : inQueue) {
      auto it = opToOpPyLoc.find(op);
      if (it == opToOpPyLoc.end()) {
        llvm::errs() << "DIDNT FIND OP IN OPTOOPPYLOC\n";
        exit(1);
      }

      const OpPyLoc &opPyLoc = it->second;
      llvm::outs() << "  - " << opPyLoc.op->getName() << " - "
                   << opPyLoc.distanceFromRoot << " - "
                   << opPyLoc.pyLoc.funcPath << "\n";
    }

    while (!availableOps.empty()) {
      OpPyLoc opPyLoc = availableOps.top();
      availableOps.pop();
      inQueue.erase(opPyLoc.op);
      opToOpPyLoc.erase(opPyLoc.op);

      // Skip if already processed (can happen with duplicates in queue)
      if (processedOps.count(opPyLoc.op)) {
        continue;
      }

      const PyLoc &pyLoc = opPyLoc.pyLoc;

      // Check if funcPath changed (make a "cut")
      if (!prevFuncPath.empty() && pyLoc.funcPath != prevFuncPath) {
        // Print current group
        llvm::outs() << "Current group: " << currentGroup.size() << "\n";
        llvm::outs() << "  Func path: " << prevFuncPath << "\n";
        for (PyLoc &pyLoc : currentGroup) {
          // Print pyloc op name and modules
          llvm::outs() << "  - " << pyLoc.op->getName() << " (modules: ";
          for (const PyLoc::Module &module : pyLoc.modules) {
            llvm::outs() << module.moduleClass << "[" << module.moduleName
                         << "] ";
          }
          llvm::outs() << ")\n";
        }
        llvm::outs() << "  Ops currently in queue: " << inQueue.size() << "\n";
        llvm::outs() << "    - " << opPyLoc.op->getName() << " - "
                     << opPyLoc.distanceFromRoot << " - "
                     << opPyLoc.pyLoc.funcPath << "\n";
        for (Operation *op : inQueue) {
          auto it = opToOpPyLoc.find(op);
          if (it == opToOpPyLoc.end()) {
            llvm::errs() << "DIDNT FIND OP IN OPTOOPPYLOC\n";
            exit(1);
          }
          const OpPyLoc &opPyLoc = it->second;
          llvm::outs() << "    - " << opPyLoc.op->getName() << " - "
                       << opPyLoc.distanceFromRoot << " - "
                       << opPyLoc.pyLoc.funcPath << "\n";
        }
        llvm::outs() << "\n";

        // Save the current group with a unique key
        std::string uniqueKey =
            prevFuncPath + "_group_" + std::to_string(groupCounter++);
        funcGroups[uniqueKey] = {currentFuncName, std::move(currentGroup)};
        currentGroup = SmallVector<PyLoc>();
      }

      // Add current op to the group
      currentGroup.push_back(pyLoc);
      processedOps.insert(opPyLoc.op);
      prevFuncPath = pyLoc.funcPath;
      currentFuncName = pyLoc.funcName;

      // Add dependent operations to the priority queue
      // (operations that use results of this operation)
      for (Value result : opPyLoc.op->getResults()) {
        for (Operation *userOp : result.getUsers()) {
          // Skip if already processed or in queue
          if (processedOps.count(userOp) || inQueue.count(userOp)) {
            continue;
          }

          // Only add if it's in our operation set
          auto userLocIt = opToLocation.find(userOp);
          if (userLocIt != opToLocation.end()) {
            // Check if all dependencies are now satisfied
            bool allDepsSatisfied = true;
            for (Value operand : userOp->getOperands()) {
              Operation *definingOp = operand.getDefiningOp();
              if (definingOp && opToLocation.count(definingOp)) {
                // Check if this dependency has been processed
                if (!processedOps.count(definingOp)) {
                  allDepsSatisfied = false;
                  break;
                }
              }
            }

            if (allDepsSatisfied) {
              // Add to priority queue with incremented distance
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

    // Don't forget to save the last group
    if (!currentGroup.empty()) {
      std::string uniqueKey =
          prevFuncPath + "_group_" + std::to_string(groupCounter++);
      funcGroups[uniqueKey] = {currentFuncName, std::move(currentGroup)};
    }

    return funcGroups;
  }

  void printFunctionGroups(
      const llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
          &funcGroups) {
    llvm::outs() << "\n=== Function Groups ===\n";
    for (const auto &[funcPath, funcInfo] : funcGroups) {
      const std::string &funcName = funcInfo.first;
      const SmallVector<PyLoc> &locations = funcInfo.second;

      llvm::outs() << "\nFunction: " << funcName << "\n";
      llvm::outs() << "  Path: " << funcPath << "\n";
      llvm::outs() << "  Operations: " << locations.size() << "\n";
      for (const PyLoc &pyLoc : locations) {
        llvm::outs() << "    - " << pyLoc.op->getName() << " (line "
                     << pyLoc.opLineNum << ")\n";
      }
    }
    llvm::outs() << "\n";
  }

  // Information about data flow for a function group
  struct FunctionBoundaryInfo {
    std::string funcName;
    std::string funcPath;
    SmallVector<PyLoc> locations;

    // Values that flow INTO this function (used but not defined here)
    SmallVector<Value> inputValues;

    // Values that flow OUT of this function (defined here, used elsewhere)
    SmallVector<Value> outputValues;

    // Values that are internal (defined and used only within this function)
    SmallVector<Value> internalValues;
  };

  // Analyze data flow for each function group
  llvm::StringMap<FunctionBoundaryInfo> analyzeFunctionBoundaries(
      const llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
          &funcGroups) {
    llvm::StringMap<FunctionBoundaryInfo> boundaryInfos;

    for (const auto &[funcPath, funcInfo] : funcGroups) {
      const std::string &funcName = funcInfo.first;
      const SmallVector<PyLoc> &locations = funcInfo.second;

      FunctionBoundaryInfo info;
      info.funcName = funcName;
      info.funcPath = funcPath.str();
      info.locations = locations;

      // Build a set of operations in this group for fast lookup
      llvm::DenseSet<Operation *> opsInGroup;
      for (const PyLoc &pyLoc : locations) {
        opsInGroup.insert(pyLoc.op);
      }

      // Track all values defined within this group
      llvm::DenseSet<Value> valuesDefinedInGroup;
      for (const PyLoc &pyLoc : locations) {
        for (Value result : pyLoc.op->getResults()) {
          valuesDefinedInGroup.insert(result);
        }
      }

      // Analyze each operation's operands and results
      llvm::DenseSet<Value> inputValuesSet;
      llvm::DenseSet<Value> outputValuesSet;
      llvm::DenseSet<Value> internalValuesSet;

      for (const PyLoc &pyLoc : locations) {
        // Check operands (inputs to the op)
        for (Value operand : pyLoc.op->getOperands()) {
          // If this value is not defined in this group, it's an input
          if (!valuesDefinedInGroup.contains(operand)) {
            inputValuesSet.insert(operand);
          }
        }

        // Check results (outputs from the op)
        for (Value result : pyLoc.op->getResults()) {
          bool usedOutside = false;
          bool usedInside = false;

          // Check if this result is used outside this group
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
          // Note: if a value is not used at all, we don't track it
        }
      }

      // Convert sets to vectors
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
    llvm::outs() << "\n=== Function Boundary Analysis ===\n";
    for (const auto &[funcPath, info] : boundaryInfos) {
      llvm::outs() << "\nFunction: " << info.funcName << "\n";
      llvm::outs() << "  Path: " << info.funcPath << "\n";
      llvm::outs() << "  Input values: " << info.inputValues.size() << "\n";
      for (Value input : info.inputValues) {
        llvm::outs() << "    - " << input << " (type: " << input.getType()
                     << ")\n";
      }
      llvm::outs() << "  Output values: " << info.outputValues.size() << "\n";
      for (Value output : info.outputValues) {
        llvm::outs() << "    - " << output << " (type: " << output.getType()
                     << ")\n";
      }
      llvm::outs() << "  Internal values: " << info.internalValues.size()
                   << "\n";
    }
    llvm::outs() << "\n";
  }

  // Create new function declarations based on boundary information
  llvm::StringMap<func::FuncOp> createNewFunctions(
      IRRewriter &rewriter, func::FuncOp candidateFn,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    llvm::StringMap<func::FuncOp> newFunctions;

    // Track function name counts to generate unique names
    llvm::StringMap<unsigned> funcNameCounts;

    for (const auto &[funcPath, info] : boundaryInfos) {
      // Generate unique function name with suffix
      unsigned count = funcNameCounts[info.funcName]++;
      std::string uniqueName = info.funcName + "_" + std::to_string(count);

      // Collect input types from input values
      SmallVector<Type> inputTypes;
      for (Value input : info.inputValues) {
        inputTypes.push_back(input.getType());
      }

      // Collect output types from output values
      SmallVector<Type> outputTypes;
      for (Value output : info.outputValues) {
        outputTypes.push_back(output.getType());
      }

      // Create function type
      FunctionType funcType =
          FunctionType::get(rewriter.getContext(), inputTypes, outputTypes);

      // Set insertion point to after the candidate function in the same parent
      // module
      rewriter.setInsertionPointAfter(candidateFn);

      // Create the new function
      func::FuncOp newFunc = rewriter.create<func::FuncOp>(
          candidateFn.getLoc(), uniqueName, funcType);

      // Mark as private for now (we'll make the entry point public later)
      newFunc.setPrivate();

      // Store the function
      newFunctions[funcPath] = newFunc;
    }

    return newFunctions;
  }

  void printNewFunctions(const llvm::StringMap<func::FuncOp> &newFunctions) {
    llvm::outs() << "\n=== Created Functions ===\n";
    for (const auto &entry : newFunctions) {
      llvm::StringRef funcPath = entry.getKey();
      func::FuncOp funcOp = entry.getValue();

      llvm::outs() << "\nFunction: " << funcOp.getName() << "\n";
      llvm::outs() << "  Path: " << funcPath << "\n";
      llvm::outs() << "  Signature: " << funcOp.getFunctionType() << "\n";
      llvm::outs() << "  Input types: "
                   << funcOp.getFunctionType().getInputs().size() << "\n";
      for (Type inputType : funcOp.getFunctionType().getInputs()) {
        llvm::outs() << "    - " << inputType << "\n";
      }
      llvm::outs() << "  Output types: "
                   << funcOp.getFunctionType().getResults().size() << "\n";
      for (Type outputType : funcOp.getFunctionType().getResults()) {
        llvm::outs() << "    - " << outputType << "\n";
      }
    }
    llvm::outs() << "\n";
  }

  // Populate function bodies by cloning operations from the original function
  void populateFunctionBodies(
      IRRewriter &rewriter,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
      llvm::StringMap<func::FuncOp> &newFunctions) {

    // For each function, we need to:
    // 1. Create entry block with arguments
    // 2. Clone operations in order
    // 3. Map old values to new values
    // 4. Add return statement

    for (const auto &entry : newFunctions) {
      llvm::StringRef funcPath = entry.getKey();
      func::FuncOp funcOp = entry.getValue();
      const FunctionBoundaryInfo &info = boundaryInfos.lookup(funcPath);

      // Create entry block
      Block *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // Create value mapping: old values -> new values
      IRMapping valueMapping;

      // Map input values to block arguments
      for (size_t i = 0; i < info.inputValues.size(); i++) {
        valueMapping.map(info.inputValues[i], entryBlock->getArgument(i));
      }

      // Clone operations in order
      for (const PyLoc &pyLoc : info.locations) {
        Operation *oldOp = pyLoc.op;
        Operation *newOp = rewriter.clone(*oldOp, valueMapping);

        // Update the value mapping with the results of the cloned operation
        for (size_t i = 0; i < oldOp->getNumResults(); i++) {
          valueMapping.map(oldOp->getResult(i), newOp->getResult(i));
        }
      }

      // Collect output values (now mapped to new function's values)
      SmallVector<Value> returnValues;
      for (Value oldOutput : info.outputValues) {
        Value newOutput = valueMapping.lookup(oldOutput);
        returnValues.push_back(newOutput);
      }

      // Add return statement
      rewriter.create<func::ReturnOp>(funcOp.getLoc(), returnValues);
    }
  }

  // Replace operations in the original function with calls to new functions
  void replaceWithFunctionCalls(
      IRRewriter &rewriter, func::FuncOp candidateFn,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
      const llvm::StringMap<func::FuncOp> &newFunctions) {

    // Build a map from operations to their function info
    llvm::DenseMap<Operation *,
                   std::pair<llvm::StringRef, const FunctionBoundaryInfo *>>
        opToFuncInfo;

    for (const auto &entry : boundaryInfos) {
      llvm::StringRef funcPath = entry.getKey();
      const FunctionBoundaryInfo &info = entry.getValue();
      for (const PyLoc &pyLoc : info.locations) {
        opToFuncInfo[pyLoc.op] = {funcPath, &info};
      }
    }

    // Walk through the original function in order and collect all operations
    SmallVector<Operation *> opsToReplace;
    candidateFn.walk([&](Operation *op) {
      if (opToFuncInfo.count(op)) {
        opsToReplace.push_back(op);
      }
    });

    // Collect all operations by funcPath (not just consecutive ones)
    llvm::StringMap<SmallVector<Operation *>> opsByFunction;

    for (Operation *op : opsToReplace) {
      auto [funcPath, info] = opToFuncInfo[op];
      opsByFunction[funcPath].push_back(op);
    }

    // Build dependency graph: funcA -> funcB means funcB depends on funcA
    // (funcB uses values produced by funcA)
    llvm::StringMap<llvm::DenseSet<llvm::StringRef>> dependencies;
    llvm::StringMap<llvm::DenseSet<Operation *>> opsInFunctionSet;

    // First, build sets of operations for each function for fast lookup
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

      // Check all operations in this function
      for (Operation *op : entry.getValue()) {
        // Check all operands of this operation
        for (Value operand : op->getOperands()) {
          Operation *definingOp = operand.getDefiningOp();
          if (!definingOp) {
            continue; // Block argument, not from another function
          }

          // Find which function this operand comes from
          for (const auto &otherEntry : opsByFunction) {
            llvm::StringRef otherFuncPath = otherEntry.getKey();
            if (otherFuncPath == funcPath) {
              continue; // Skip self-dependencies
            }

            if (opsInFunctionSet[otherFuncPath].contains(definingOp)) {
              // This function depends on otherFuncPath
              deps.insert(otherFuncPath);
              break;
            }
          }
        }
      }
    }

    // Topological sort to determine function order
    SmallVector<llvm::StringRef> functionOrder;
    llvm::DenseSet<llvm::StringRef> visited;
    llvm::DenseSet<llvm::StringRef> inProgress;

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

    // Visit all functions
    for (const auto &entry : opsByFunction) {
      if (!visit(entry.getKey())) {
        // Error: cycle in dependencies
        return;
      }
    }

    // Mapping from old values to new values (call results)
    IRMapping valueMapping;

    // Track the last call we created for sequential insertion
    Operation *lastCallOp = nullptr;

    // Track which operations have been processed
    llvm::DenseSet<Operation *> processedOps;

    // Process each unique function in the order they first appear
    for (llvm::StringRef funcPath : functionOrder) {
      const FunctionBoundaryInfo *info =
          opToFuncInfo[opsByFunction[funcPath].front()].second;
      func::FuncOp targetFunc = newFunctions.lookup(funcPath);
      SmallVector<Operation *> &opsInFunction = opsByFunction[funcPath];

      // Prepare operands for the call (input values)
      SmallVector<Value> callOperands;
      for (Value inputValue : info->inputValues) {
        // If this input was produced by a previous call, use the mapped value
        if (valueMapping.contains(inputValue)) {
          callOperands.push_back(valueMapping.lookup(inputValue));
        } else {
          callOperands.push_back(inputValue);
        }
      }

      // Insert the call after the last call we created (or before first op if
      // this is the first call)
      if (lastCallOp) {
        rewriter.setInsertionPointAfter(lastCallOp);
      } else {
        Operation *firstOp = opsInFunction.front();
        rewriter.setInsertionPoint(firstOp);
      }

      // Create the call
      Operation *lastOp = opsInFunction.back();
      func::CallOp callOp = rewriter.create<func::CallOp>(
          lastOp->getLoc(), targetFunc, callOperands);

      // Track this call for the next insertion
      lastCallOp = callOp;

      // Map output values to call results
      for (size_t i = 0; i < info->outputValues.size(); i++) {
        valueMapping.map(info->outputValues[i], callOp.getResult(i));
      }

      // Replace uses of output values with call results
      FunctionBoundaryInfo &mutableInfo =
          const_cast<FunctionBoundaryInfo &>(*info);
      for (size_t i = 0; i < mutableInfo.outputValues.size(); i++) {
        mutableInfo.outputValues[i].replaceAllUsesWith(callOp.getResult(i));
      }

      // Mark operations for deletion
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

    // Need a couple passes
    // 1. Find candidate fns
    // 2. Parse and gather IR locations (remember ops)
    // -----
    // 3. Build functree
    // 4. Analyze functree
    // 5. Move from original IR to functree-aligned IR

    // Open questions:
    // - What do we do with const-eval fns?
    // - What makes a fn a candidate fn?

    // For simplicity, supporting only one for now, but can support multiple.
    SmallVector<func::FuncOp> candidateFns = findCandidateFns(moduleOp);
    assert(candidateFns.size() == 1 &&
           "Only one candidate fn is supported now");

    func::FuncOp candidateFn = candidateFns.front();

    llvm::DenseMap<Operation *, PyLoc> opToLocation =
        parseOpsAndGatherLocations(candidateFn);

    // Debug prints
    // printPyLocs(candidateFn, opToLocation);

    // Validate locations
    std::string errorMessage = validateLocations(opToLocation);
    if (!errorMessage.empty()) {
      llvm::errs() << "PrettifyForCodegen error: " << errorMessage << "\n";
      signalPassFailure();
    }

    // Group operations by function
    auto funcGroups = groupOperationsByFunction(candidateFn, opToLocation);
    printFunctionGroups(funcGroups);

    // Analyze function boundaries and data flow
    auto boundaryInfos = analyzeFunctionBoundaries(funcGroups);
    printFunctionBoundaries(boundaryInfos);

    // Create new functions based on boundary information
    auto newFunctions =
        createNewFunctions(rewriter, candidateFn, boundaryInfos);
    printNewFunctions(newFunctions);

    // Populate function bodies with operations
    populateFunctionBodies(rewriter, boundaryInfos, newFunctions);

    // Replace operations in original function with calls to new functions
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

    // Get the name string and create a new NameLoc without nested location
    return mlir::NameLoc::get(
        nameLoc.getName()
        // Omitting the second parameter uses UnknownLoc by default
    );
  }

public:
  using impl::TTNNSimplifyLocsForCodegenBase<
      TTNNSimplifyLocsForCodegen>::TTNNSimplifyLocsForCodegenBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Walk through all operations in the module
    moduleOp.walk([&](Operation *op) {
      // Skip UnknownLocs
      if (isa<UnknownLoc>(op->getLoc())) {
        return WalkResult::advance();
      }

      // Only process TTNN dialect ops
      if (!isa<ttnn::TTNNDialect>(op->getDialect())) {
        return WalkResult::advance();
      }

      // Simplify the location
      mlir::Location simplifiedLoc = simplifyNameLoc(op->getLoc());
      op->setLoc(simplifiedLoc);

      return WalkResult::advance();
    });
  }
};

} // namespace

} // namespace mlir::tt::ttnn
