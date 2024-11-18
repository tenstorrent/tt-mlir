// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRHOISTTRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Hoist CPU ops to standalone funcs pass
//===----------------------------------------------------------------------===//
static llvm::SmallVector<int64_t, 4> getTensorRanks(mlir::Operation *op) {
  llvm::SmallVector<int64_t, 4> ranks;

  // Iterate over operands (inputs)
  for (auto operand : op->getOperands()) {
    // Check if the operand is a tensor
    if (auto tensorType = dyn_cast<mlir::RankedTensorType>(operand.getType())) {
      // Add the rank of the tensor (number of dimensions)
      ranks.push_back(tensorType.getRank());
    }
  }

  // Iterate over results (outputs)
  for (auto result : op->getResults()) {
    // Check if the result is a tensor
    if (auto tensorType = dyn_cast<mlir::RankedTensorType>(result.getType())) {
      // Add the rank of the tensor (number of dimensions)
      ranks.push_back(tensorType.getRank());
    }
  }

  return ranks;
}

static std::string generateHoistedFuncName(mlir::Operation *op) {
  std::string opName = op->getName().getStringRef().str();

  // Start building the unique function name
  std::string uniqueName = "hoisted_" + opName;

  // Iterate over operands to extract tensor shapes and types
  for (auto operand : op->getOperands()) {
    mlir::ShapedType shapedType = dyn_cast<mlir::ShapedType>(operand.getType());
    if (shapedType && shapedType.hasRank()) {
      // Append the shape (dimensions) and the element type
      std::string shapeStr = "_";
      for (auto dim : shapedType.getShape()) {
        shapeStr += std::to_string(dim) + "x";
      }

      // Append the element type (e.g., f32, i32) -- unforunately I don't think
      // there's a better way to get string from mlir::Type
      std::string elementTypeStr;
      llvm::raw_string_ostream stream(elementTypeStr);
      shapedType.getElementType().print(stream);

      uniqueName += shapeStr + elementTypeStr;
    }
  }

  // Add suffix to indicate it's a function
  uniqueName += "_func";

  return uniqueName;
}

static void hoistOperationToFunction(mlir::Operation *opToHoist,
                                     mlir::ModuleOp sourceModule,
                                     mlir::ModuleOp targetModule) {
  const auto ranks = getTensorRanks(opToHoist);

  const std::string functionName = generateHoistedFuncName(opToHoist);

  auto localFunc = llvm::dyn_cast_or_null<func::FuncOp>(
      sourceModule.lookupSymbol(functionName));

  // if we have not already emitted an equivalent function call, perform this
  // hoist
  if (localFunc == nullptr) {

    mlir::MLIRContext *context = sourceModule.getContext();

    // Gather operand and result types
    llvm::SmallVector<mlir::Type> operandTypes, resultTypes;
    for (auto operand : opToHoist->getOperands()) {
      operandTypes.push_back(operand.getType());
    }
    for (auto result : opToHoist->getResultTypes()) {
      resultTypes.push_back(result);
    }

    // Create the function signature
    mlir::FunctionType funcType =
        mlir::FunctionType::get(context, operandTypes, resultTypes);

    // Create the function in the target module
    auto hoistedFunc =
        func::FuncOp::create(opToHoist->getLoc(), functionName, funcType);
    targetModule.push_back(hoistedFunc);

    // Add a basic block to the function
    mlir::Block *block = hoistedFunc.addEntryBlock();
    mlir::OpBuilder builder(block, block->end());

    // Map operands to block arguments and clone the operation
    llvm::SmallVector<mlir::Value> newOperands;
    for (auto operand : llvm::enumerate(opToHoist->getOperands())) {
      newOperands.push_back(block->getArgument(operand.index()));
    }

    mlir::IRMapping mapping;
    for (auto operand : llvm::zip(opToHoist->getOperands(), newOperands)) {
      mapping.map(std::get<0>(operand), std::get<1>(operand));
    }

    mlir::Operation *clonedOp = builder.clone(*opToHoist, mapping);

    // Add a return operation to the function
    builder.create<mlir::func::ReturnOp>(opToHoist->getLoc(),
                                         clonedOp->getResults());

    // Declare the function prototype in the source module
    localFunc =
        func::FuncOp::create(opToHoist->getLoc(), functionName, funcType);
    localFunc.setPrivate(); // Mark as external (no body)
    sourceModule.push_back(localFunc);

    hoistedFunc->setAttr("arg_ranks", builder.getI64ArrayAttr(ranks));
  }

  // Replace the original operation with a call to the hoisted function
  mlir::OpBuilder opBuilder(opToHoist);
  auto callOp = opBuilder.create<mlir::func::CallOp>(
      opToHoist->getLoc(), localFunc, opToHoist->getOperands());

  // Replace all results of the original operation with the call results
  for (auto result : llvm::zip(opToHoist->getResults(), callOp.getResults())) {
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
  }

  // Erase the original operation
  opToHoist->erase();
}

// an analysis class which relies on manual input locations to mark ops to be
// hoisted, useful for testing
class TTIRHoistAnalyzeManual {
public:
  using HoistOpSet = std::vector<std::set<mlir::Operation *>>;

  TTIRHoistAnalyzeManual(mlir::Operation *op,
                         const std::vector<std::string> &targetLocs) {
    auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(op);
    assert(moduleOp && "somehow got non-ModuleOp in TTIRHoistAnalyzeManual!");

    moduleOp.walk([&](mlir::Operation *nestedOp) {
      if (matchesLocation(nestedOp, targetLocs)) {
        std::set<mlir::Operation *> opSet;
        opSet.insert(nestedOp);
        hoistedOps.push_back(opSet);
      }
    });
  }

  HoistOpSet getResults() { return hoistedOps; }

private:
  HoistOpSet hoistedOps;

  static bool matchesLocation(mlir::Operation *op,
                              const std::vector<std::string> &targetLocations) {
    if (const auto nameLoc = dyn_cast<NameLoc>(op->getLoc())) {
      const auto &opLocString = nameLoc.getName().str();

      for (const auto &loc : targetLocations) {
        if (opLocString == loc) {
          return true;
        }
      }
    }
    return false;
  }
};

struct TargetLocsParser : llvm::cl::parser<std::vector<std::string>> {
  TargetLocsParser(llvm::cl::Option &opt)
      : llvm::cl::parser<std::vector<std::string>>(opt) {}

  bool parse(llvm::cl::Option &opt, llvm::StringRef argName,
             llvm::StringRef argValue, std::vector<std::string> &value) {
    // Split the input string on commas
    llvm::SmallVector<llvm::StringRef, 4> names;
    argValue.split(names, ',');

    value.assign(names.begin(), names.end());

    return false; // Success
  }

  static std::string toString(const std::vector<std::string> &locList) {
    std::string result;
    for (const auto &loc : locList) {
      result += loc + ",";
    }
    return result;
  }

  static void print(llvm::raw_ostream &os,
                    const std::vector<std::string> &locList) {
    os << "hoist-locs=" << toString(locList) << "\n";
  }
};

class TTIRHoistTransform
    : public impl::TTIRHoistTransformBase<TTIRHoistTransform> {
public:
  using impl::TTIRHoistTransformBase<
      TTIRHoistTransform>::TTIRHoistTransformBase;

  // MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRHoistTransform)

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    assert(moduleOp != nullptr && "TTIRHoistTransform should run on ModuleOps, "
                                  "somehow got something else!");

    IRRewriter rewriter(&getContext());

    auto loc = moduleOp->getLoc();

    // TODO (vwells): add logic for automatic vs manual flag here when we have
    // automatic analysis pass

    // only create cpu-module etc. if we actually have ops to hoist
    if (!hoistLocs.hasValue()) {
      return;
    }
    const auto targetLocs = getLocations();
    if (targetLocs.empty()) {
      return;
    }

    TTIRHoistAnalyzeManual analysisPass(moduleOp, targetLocs);
    const auto &hoistOpSets = analysisPass.getResults();

    // Check if a "cpu_module" already exists
    mlir::ModuleOp cpuModule;
    for (auto &op : moduleOp.getBody()->getOperations()) {
      if (auto module = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        if (module->hasAttr("ttir.cpu_module")) {
          cpuModule = module;
          break;
        }
      }
    }

    // If no CPU module exists, create one
    if (!cpuModule) {
      rewriter.setInsertionPointToEnd(moduleOp.getBody());
      cpuModule = rewriter.create<mlir::ModuleOp>(loc);
      cpuModule->setAttr("ttir.cpu_module", rewriter.getUnitAttr());
      cpuModule->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                         rewriter.getStringAttr("cpu_module"));
      // try to make cpu module global
      mlir::SymbolTable::setSymbolVisibility(
          cpuModule, mlir::SymbolTable::Visibility::Public);
    }

    for (const auto &opSet : hoistOpSets) {
      assert(opSet.size() == 1 &&
             "currently don't support hoisting multiple instructions at once!");
      hoistOperationToFunction(*opSet.begin(), moduleOp, cpuModule);
    }
  }

protected:
  std::vector<std::string> getLocations() {
    std::vector<std::string> locs;
    for (const std::string &loc : hoistLocs) {
      locs.emplace_back(loc);
    }
    return locs;
  }
};

} // namespace mlir::tt::ttir
