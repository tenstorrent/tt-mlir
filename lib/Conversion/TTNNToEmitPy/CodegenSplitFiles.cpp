// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt {
#define GEN_PASS_DEF_CODEGENSPLITFILES
#include "ttmlir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
//
// Codegen Split Files Pass
//
// This pass organizes EmitPy IR into two separate Python files:
//   - main.py: Contains the main execution logic
//   - consteval.py: Contains const-evaluated functions and caching logic
//
// Example transformation:
//
// BEFORE (single main.py):
//   import ttnn
//   import utils
//   _CONST_EVAL_CACHE = {}
//
//   def cpu_hoisted_const_eval_0(input):        # CPU-hoisted function
//       ...
//
//   def main_const_eval_0(input):               # Const-eval function
//       ...
//
//   def _main(input):                           # Main with inline caching
//       global _CONST_EVAL_CACHE
//       ...
//       utils_constEvalFuncWrapper_0 =
//       utils.constEvalFuncWrapper("main_const_eval_0", ..., _CONST_EVAL_CACHE,
//       ...) utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
//       ...
//   ...
//
// AFTER (split files):
//   main.py:
//     import ttnn
//     import utils
//     from consteval import consteval__main
//
//     def _main(input):
//         var_0 = consteval__main(input)         # Call to consteval caching
//         function
//         ...
//         result = var_0["main_const_eval_0"][0] # Lookup result in cache
//         ...
//   ...
//
//   consteval.py:
//     import ttnn
//     import utils
//     _CONST_EVAL_CACHE = {}
//
//     def cpu_hoisted_const_eval_0(input):      # Moved from main
//         ...
//
//     def main_const_eval_0(input):             # Moved from main
//         result = cpu_hoisted_const_eval_0(...)
//         ...
//
//     def consteval__main(inputs):              # Extracted consteval caching
//     logic from _main function
//         global _CONST_EVAL_CACHE
//         ...
//         utils.constEvalFuncWrapper(main_const_eval_0, ..., _CONST_EVAL_CACHE,
//         ...)
//         ...
//         return _CONST_EVAL_CACHE["_main"]
//     ...
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Data Structures and Helper Functions
//===----------------------------------------------------------------------===//

namespace {
// Constants for file names and operation names.
constexpr const char *MAIN_FILE_NAME = "main";
constexpr const char *CONSTEVAL_FILE_NAME = "consteval";
constexpr const char *CONSTEVAL_WRAPPER_PREFIX = "constEvalFuncWrapper";
constexpr const char *CACHE_CONSTEVAL_ATTR_NAME = "emitpy.cache_consteval";
constexpr const char *DEVICE_ARG_NAME = "device";

// Data structures for analysis results.
struct ModuleAnalysisResult {
  llvm::SmallVector<Operation *> importOps;
  llvm::SmallVector<Operation *> constevalOps;
  llvm::SmallVector<Operation *> uncategorizedOps;
};

struct CacheConstevalAnalysisResult {
  func::FuncOp funcOp;
  llvm::SetVector<Operation *> constevalDefUseChain;
  llvm::SmallVector<func::CallOp> crossFileCallOps;
  mlir::Value globalCacheDict = nullptr;
  mlir::Value outerDictKey = nullptr;
};

// General helper functions.
//
// Replace func::CallOp with emitpy::CallOpaqueOp in cross-file calls.
static void replaceCallWithCallOpaque(OpBuilder &builder, func::CallOp callOp) {
  builder.setInsertionPoint(callOp);
  auto callOpaqueOp = builder.create<emitpy::CallOpaqueOp>(
      callOp.getLoc(), callOp.getResultTypes(), callOp.getCallee().str(),
      callOp.getOperands());
  callOpaqueOp->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
  callOp.replaceAllUsesWith(callOpaqueOp);
  callOp->erase();
}

// Copy import operations to target file.
static void copyImportsToFile(OpBuilder &builder, emitpy::FileOp targetFile,
                              ArrayRef<Operation *> importOps) {
  builder.setInsertionPointToStart(&targetFile.getBodyRegion().front());
  for (auto *importOp : importOps) {
    builder.clone(*importOp);
  }
}

// Move operations to target file.
static void moveOperationsToFile(emitpy::FileOp file,
                                 const llvm::SmallVector<Operation *> &ops) {
  for (auto *op : ops) {
    op->moveBefore(&file.getBodyRegion().front(),
                   file.getBodyRegion().front().end());
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// Main Pass Implementation
//===----------------------------------------------------------------------===//

class CodegenSplitFiles
    : public impl::CodegenSplitFilesBase<CodegenSplitFiles> {

public:
  using impl::CodegenSplitFilesBase<CodegenSplitFiles>::CodegenSplitFilesBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    // Create main and consteval file ops.
    builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    auto mainFile =
        builder.create<emitpy::FileOp>(moduleOp->getLoc(), MAIN_FILE_NAME);
    auto constevalFile =
        builder.create<emitpy::FileOp>(moduleOp->getLoc(), CONSTEVAL_FILE_NAME);

    // Analyze module to categorize top-level module operations.
    ModuleAnalysisResult moduleAnalysis = analyzeModule(moduleOp);

    // Move consteval functions to the consteval file.
    moveOperationsToFile(constevalFile, moduleAnalysis.constevalOps);

    // Extract consteval caching logic from functions that call consteval
    // functions into separate functions in the consteval file. Collect the
    // names of the created consteval caching functions to later import them
    // into the main file.
    llvm::SmallVector<llvm::StringRef> constevalFuncNames;
    for (auto *uncategorizedOp : moduleAnalysis.uncategorizedOps) {
      if (auto funcOp = dyn_cast<func::FuncOp>(uncategorizedOp)) {
        CacheConstevalAnalysisResult analysis =
            analyzeFunction(funcOp, constevalFile, &getContext());
        // Replace func::CallOp with emitpy::CallOpaqueOp for cross-file calls.
        // This prevents symbol resolution errors when calling cpu-hoisted
        // functions that have been moved to the consteval file.
        for (auto callOp :
             llvm::make_early_inc_range(analysis.crossFileCallOps)) {
          replaceCallWithCallOpaque(builder, callOp);
        }
        // The function that does not have consteval caching logic will stay in
        // the main file as it is.
        if (!hasCacheConstevalLogic(analysis)) {
          continue;
        }
        if (failed(
                splitConstevalCacheLogic(builder, constevalFile, analysis))) {
          return;
        }
        constevalFuncNames.push_back(
            builder.getStringAttr("consteval_" + funcOp.getName()).getValue());
      }
    }

    // Create import statement for consteval functions in the main file.
    if (!constevalFuncNames.empty()) {
      builder.setInsertionPointToStart(&mainFile.getBodyRegion().front());
      auto emptyAliases = llvm::SmallVector<Attribute>(
          constevalFuncNames.size(), builder.getStringAttr(""));
      builder.create<emitpy::ImportOp>(
          moduleOp.getLoc(), builder.getStringAttr(CONSTEVAL_FILE_NAME),
          /*module_alias=*/nullptr,
          /*members_to_import=*/builder.getStrArrayAttr(constevalFuncNames),
          /*member_aliases=*/builder.getArrayAttr(emptyAliases),
          /*import_all=*/nullptr);
    }

    // Copy all import ops from the top-level module to both file ops.
    copyImportsToFile(builder, constevalFile, moduleAnalysis.importOps);
    copyImportsToFile(builder, mainFile, moduleAnalysis.importOps);

    // Erase the original import ops from the top-level module.
    for (auto *importOp : moduleAnalysis.importOps) {
      importOp->erase();
    }

    // Move remaining uncategorized operations to the main file.
    moveOperationsToFile(mainFile, moduleAnalysis.uncategorizedOps);
  }

private:
  // Helper to get commonly used types.
  struct TypeHelper {
    Type strType;
    Type tensorListType;
    Type deviceType;
    Type innerDictType;

    explicit TypeHelper(MLIRContext *ctx)
        : strType(emitpy::OpaqueType::get(ctx, "str")),
          tensorListType(emitpy::OpaqueType::get(ctx, "[ttnn.Tensor]")),
          deviceType(emitpy::OpaqueType::get(ctx, "ttnn.Device")),
          innerDictType(emitpy::DictType::get(ctx, strType, tensorListType)) {}
  };

  // Check if function has a device argument as its last parameter.
  static bool hasDeviceArg(func::FuncOp funcOp, Type deviceType) {
    if (funcOp.getNumArguments() == 0) {
      return false;
    }
    unsigned lastArgIdx = funcOp.getNumArguments() - 1;
    return funcOp.getArgument(lastArgIdx).getType() == deviceType;
  }

  // Check if function has any caching consteval logic to extract.
  static bool
  hasCacheConstevalLogic(const CacheConstevalAnalysisResult &result) {
    return !result.constevalDefUseChain.empty();
  }

  // Get topologically sorted use-def chain starting from consteval ops.
  static llvm::SetVector<Operation *> getSortedConstevalUseDefChain(
      const llvm::SmallVector<Operation *> &constevalOps) {
    llvm::SmallVector<Operation *> workload(constevalOps.begin(),
                                            constevalOps.end());
    llvm::SetVector<Operation *> useDefChain;

    while (!workload.empty()) {
      Operation *op = workload.pop_back_val();
      if (!useDefChain.insert(op)) {
        continue;
      }

      for (auto operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          workload.push_back(defOp);
        }
      }
    }

    return topologicalSort(useDefChain);
  }

  // Analyze module to categorize operations.
  static ModuleAnalysisResult analyzeModule(ModuleOp moduleOp) {
    ModuleAnalysisResult result;

    for (auto &op : moduleOp.getOps()) {
      if (isa<emitpy::FileOp>(op)) {
        continue;
      }
      // Collect the import operations separately to copy them to both files.
      if (isa<emitpy::ImportOp>(op)) {
        result.importOps.push_back(&op);
        continue;
      }
      // Collect the operation that defines the global cache dictionary as a
      // consteval operation.
      if (auto globalOp = dyn_cast<emitpy::GlobalOp>(op)) {
        result.constevalOps.push_back(&op);
        continue;
      }
      // Collect the consteval and cpu-hoisted functions.
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (ttmlir::utils::isConstEvalFunc(funcOp) ||
            ttmlir::utils::isForwardCPUFunc(funcOp)) {
          result.constevalOps.push_back(&op);
          continue;
        }
      }

      result.uncategorizedOps.push_back(&op);
    }

    return result;
  }

  // Analyze function to detect caching consteval logic.
  static CacheConstevalAnalysisResult
  analyzeFunction(func::FuncOp funcOp, emitpy::FileOp constevalFile,
                  MLIRContext *ctx) {
    CacheConstevalAnalysisResult result;
    result.funcOp = funcOp;

    llvm::SmallVector<Operation *> constevalOps;

    // Walk function to find relevant operations.
    funcOp.walk([&](Operation *op) {
      // Find global cache dictionary.
      if (isa<emitpy::GlobalStatementOp>(op)) {
        result.globalCacheDict = op->getResult(0);
      }
      // Find outer dictionary key (constant with function name) for cache
      // lookup.
      if (auto constantOp = dyn_cast<emitpy::ConstantOp>(op)) {
        auto funcNameAttr =
            emitpy::OpaqueAttr::get(ctx, "\"" + funcOp.getName().str() + "\"");
        if (constantOp.getValue() == funcNameAttr) {
          result.outerDictKey = constantOp->getResult(0);
        }
      }
      // Collect consteval caching operations (marked with cache_consteval
      // attribute).
      if (op->hasAttr(CACHE_CONSTEVAL_ATTR_NAME)) {
        constevalOps.push_back(op);
      }
      // Collect cross-file calls (to cpu-hoisted functions).
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (auto calledFunc =
                constevalFile.lookupSymbol<func::FuncOp>(callOp.getCallee())) {
          if (ttmlir::utils::isForwardCPUFunc(calledFunc)) {
            result.crossFileCallOps.push_back(callOp);
          }
        }
      }
    });

    // Build use-def chain from consteval caching operations.
    result.constevalDefUseChain = getSortedConstevalUseDefChain(constevalOps);

    return result;
  }

  // Extract dictionary key from wrapper function call operands.
  static FailureOr<emitpy::ConstantOp>
  getDictKey(emitpy::CallOpaqueOp wrapperCall, Type deviceType) {
    unsigned numOperands = wrapperCall->getNumOperands();
    if (numOperands == 0) {
      wrapperCall->emitError("wrapper call has no operands");
      return failure();
    }

    // The dict key is the last operand, unless a device argument
    // is present (target-module=true), in which case it's second-to-last.
    unsigned dictKeyIdx =
        (wrapperCall->getOperand(numOperands - 1).getType() == deviceType)
            ? numOperands - 2
            : numOperands - 1;

    auto dictKeyOp =
        wrapperCall->getOperand(dictKeyIdx).getDefiningOp<emitpy::ConstantOp>();
    if (!dictKeyOp) {
      wrapperCall->emitError("wrapper call missing dictionary key operand");
      return failure();
    }

    return dictKeyOp;
  }

  LogicalResult
  splitConstevalCacheLogic(OpBuilder &builder, emitpy::FileOp constevalFile,
                           CacheConstevalAnalysisResult &analysis) {
    TypeHelper types(&getContext());
    func::FuncOp funcOp = analysis.funcOp;

    // Create a function in the consteval file that contains all the caching
    // logic (cache lookups, wrapper calls to const-evaluated functions, cache
    // updates). This function will be called from the main file.
    llvm::StringRef constevalCachingFuncName =
        builder.getStringAttr("consteval_" + funcOp.getName()).getValue();
    builder.setInsertionPointToEnd(&constevalFile.getBodyRegion().front());
    createAndPopulateConstevalCacheFunction(builder, analysis,
                                            constevalCachingFuncName.str());

    // Insert call to the consteval caching function at the beginning of the
    // original function. This call executes the caching logic and returns a
    // cache map containing all consteval results.
    llvm::SmallVector<Value> callOperands = {funcOp.getArgument(0)};
    if (hasDeviceArg(analysis.funcOp, types.deviceType)) {
      callOperands.push_back(funcOp.getArgument(funcOp.getNumArguments() - 1));
    }
    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    auto callConstevalCacheOp = builder.create<emitpy::CallOpaqueOp>(
        funcOp.getLoc(), types.innerDictType, constevalCachingFuncName.str(),
        callOperands);

    // Patch the original function to use the results of callConstevalCacheOp
    // before erasing the consteval operations from its body.
    if (failed(patchOriginalFunctionToUseCachedResults(builder, analysis,
                                                       callConstevalCacheOp))) {
      return failure();
    }

    // Erase the consteval operations from the original function body, except
    // for operations that were not marked with the cache_consteval attribute
    // or operations that still have uses (e.g., dictionary keys for lookups).
    for (Operation *op : llvm::reverse(analysis.constevalDefUseChain)) {
      if (!op->hasAttr(CACHE_CONSTEVAL_ATTR_NAME) ||
          (isa<emitpy::ConstantOp>(op) && !op->use_empty())) {
        continue;
      }
      op->erase();
    }
    return success();
  }

  func::FuncOp createAndPopulateConstevalCacheFunction(
      OpBuilder &builder, CacheConstevalAnalysisResult &analysis,
      std::string funcName) {
    TypeHelper types(&getContext());
    func::FuncOp funcOp = analysis.funcOp;

    // Create function signature.
    llvm::SmallVector<Type> inputTypes = {types.tensorListType};
    if (hasDeviceArg(funcOp, types.deviceType)) {
      inputTypes.push_back(types.deviceType);
    }
    auto constevalCachingFunc = builder.create<func::FuncOp>(
        funcOp.getLoc(), funcName,
        builder.getFunctionType(inputTypes, {types.innerDictType}));
    constevalCachingFunc.addEntryBlock();

    // Populate function body.
    Block &entryBlock = constevalCachingFunc.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);

    // Map original funcOp arguments to the new function's arguments.
    mlir::IRMapping mapping;
    mlir::Value originalInput = funcOp.getArgument(0);
    mapping.map(originalInput, entryBlock.getArgument(0));
    if (hasDeviceArg(funcOp, types.deviceType)) {
      mlir::Value originalDeviceArg =
          funcOp.getArgument(funcOp.getNumArguments() - 1);
      mapping.map(originalDeviceArg, entryBlock.getArgument(1));
      constevalCachingFunc.setArgAttr(1, "emitpy.name",
                                      builder.getStringAttr(DEVICE_ARG_NAME));
    }

    // Clone all consteval caching-related operations into the new
    // function.
    for (Operation *op : analysis.constevalDefUseChain) {
      Operation *cloned = builder.clone(*op, mapping);
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), cloned->getResult(i));
      }
    }

    // Add return operation at the end of new function's body.
    auto dictResult = builder.create<emitpy::GetValueForDictKeyOp>(
        funcOp.getLoc(), types.innerDictType,
        mapping.lookup(analysis.globalCacheDict),
        mapping.lookup(analysis.outerDictKey));
    builder.create<func::ReturnOp>(funcOp.getLoc(), dictResult.getResult());

    return constevalCachingFunc;
  }

  LogicalResult patchOriginalFunctionToUseCachedResults(
      OpBuilder &builder, const CacheConstevalAnalysisResult &analysis,
      emitpy::CallOpaqueOp constevalCacheDict) {
    TypeHelper types(&getContext());
    llvm::DenseMap<emitpy::ConstantOp, Value> keyToLookup;
    func::FuncOp funcOp = analysis.funcOp;
    builder.setInsertionPointAfter(constevalCacheDict);

    // For each wrapper call in the caching logic (targeted for removal from the
    // original function), perform a cache lookup and replace all uses of the
    // wrapper call with the cached result. This effectively redirects
    // constEvalFuncWrapper[i] to cacheDict[consteval_function_name][i].
    for (Operation *op : analysis.constevalDefUseChain) {
      auto wrapperCall = dyn_cast<emitpy::CallOpaqueOp>(op);
      if (!wrapperCall ||
          !wrapperCall.getCallee().contains(CONSTEVAL_WRAPPER_PREFIX)) {
        continue;
      }

      // Extract the cache dictionary key from wrapper call operands.
      FailureOr<emitpy::ConstantOp> cacheKeyOp =
          getDictKey(wrapperCall, types.deviceType);
      if (failed(cacheKeyOp)) {
        return failure();
      }

      // Create dictionary lookup operation once per cache key.
      if (!keyToLookup.count(*cacheKeyOp)) {
        builder.setInsertionPointAfter(*cacheKeyOp);
        auto cachedValue = builder.create<emitpy::GetValueForDictKeyOp>(
            funcOp.getLoc(), types.tensorListType,
            constevalCacheDict.getResult(0), (*cacheKeyOp)->getResult(0));
        keyToLookup[*cacheKeyOp] = cachedValue;
      }

      // Replace all uses of the wrapper call with the cached result.
      wrapperCall->getResult(0).replaceAllUsesWith(keyToLookup[*cacheKeyOp]);
    }

    return success();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCodegenSplitFilesPass() {
  return std::make_unique<CodegenSplitFiles>();
}

} // namespace mlir::tt
