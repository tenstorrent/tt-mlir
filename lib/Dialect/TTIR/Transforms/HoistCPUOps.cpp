// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CPUHOISTTRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Hoist CPU ops to standalone funcs pass
//===----------------------------------------------------------------------===//

namespace {
using HoistedOpsSet = llvm::SmallVector<mlir::Operation *, 4>;

// Helper function to get ranks of a subgraph's operands
// we use this to populate attrs which we need to tensor unpacking operations
// later.
static llvm::SmallVector<int64_t, 3> getSubgraphOperandTensorRanks(
    const llvm::SmallVector<mlir::Value, 4> &inputValues) {
  llvm::SmallVector<int64_t, 3> ranks;

  // Iterate over input values.
  for (auto value : inputValues) {
    // Check if the value is a tensor.
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
      // Add the rank of the tensor (number of dimensions).
      ranks.push_back(tensorType.getRank());
    }
  }

  return ranks;
}

// Generate unique name base on operation type + argument tensors dims & types.
static llvm::SmallString<16>
generateHoistedFuncName(const HoistedOpsSet &operations) {
  llvm::SmallString<16> uniqueName("hoisted_subgraph_");

  for (auto *op : operations) {
    uniqueName += op->getName().getStringRef().str();
    // TODO(dmilinkovic): tensor types
    uniqueName += "_";
  }

  uniqueName += "_func";
  std::replace(uniqueName.begin(), uniqueName.end(), '.', '_');
  return uniqueName;
}

// Helper function to hoist an arbitrary set of ops into a new function in
// targetModule, generate a matching extern prototype in the sourceModule, and
// replace the ops in the set with a callOp to the extern function.
// The last op in the set is considered the result producer.
static void hoistOperationsToFunction(HoistedOpsSet &operations,
                                      mlir::ModuleOp sourceModule,
                                      mlir::ModuleOp targetModule) {
  auto *firstOp = operations.front();

  mlir::MLIRContext *context = sourceModule.getContext();
  mlir::OpBuilder typeBuilder(firstOp);

  auto f32Type = mlir::Float32Type::get(context);
  auto i32Type =
      mlir::IntegerType::get(context, 32, mlir::IntegerType::Signless);

  // Helper lambda to unify tensor element types.
  auto convertTensorElementType = [f32Type, i32Type](Type elementType) -> Type {
    if (elementType.isInteger()) {
      return i32Type;
    }
    if (elementType.isFloat()) {
      return f32Type;
    }
    return elementType;
  };

  // Helper lambda to unify tensor types.
  auto convertTensorType =
      [&](RankedTensorType tensorType) -> RankedTensorType {
    auto elementType = tensorType.getElementType();
    auto convertedElementType = convertTensorElementType(elementType);
    if (elementType != convertedElementType) {
      return RankedTensorType::get(tensorType.getShape(), convertedElementType,
                                   tensorType.getEncoding());
    }
    return tensorType;
  };

  // Collect all input arguments to the function.
  llvm::SmallVector<mlir::Value, 4> inputArguments;
  llvm::SmallPtrSet<mlir::Value, 8> inputArgumentsSet;

  for (auto *op : operations) {
    for (auto operand : op->getOperands()) {
      // If the operand is defined outside the subgraph, it's an input.
      if (std::find(operations.begin(), operations.end(),
                    operand.getDefiningOp()) == operations.end()) {
        // Add to input values if not already present.
        if (inputArgumentsSet.insert(operand).second) {
          inputArguments.push_back(operand);
        }
      }
    }
  }

  // Currently, only single-result ops are supported for hoisting.
  auto *resultProvider = operations.back();
  assert(resultProvider->getNumResults() == 1 &&
         "Only single-result ops are supported for hoisting.");

  mlir::Value result = resultProvider->getResult(0);

  auto resultType =
      llvm::dyn_cast_or_null<mlir::RankedTensorType>(result.getType());

  assert(resultType && "Only tensor result types are supported for hoisting.");

  auto convertedResultType = convertTensorType(resultType);

  // The hoisted function needs NOT to be in DPS form.
  // For targets that require hoisted functions to be in DPS form,
  // a separate pass should be run after hoisting to adjust the functions.
  //
  // If the result producer is a DPS op, we need to check if the result
  // tensor is already part of the input arguments.
  //
  // If it is, we need to remove it from the input arguments and add a
  // placeholder empty tensor for the result instead.
  const bool isResultProducerDPSOp =
      mlir::isa<mlir::DestinationStyleOpInterface>(resultProvider);

  bool shouldAddEmptyTensorForDps = false;

  if (isResultProducerDPSOp) {
    auto dpsOp = mlir::cast<mlir::DestinationStyleOpInterface>(resultProvider);
    assert(dpsOp.getDpsInits().size() == 1 &&
           "Only single-output DPS ops are supported for hoisting.");

    auto dpsInit = dpsOp.getDpsInits().front();

    if (inputArgumentsSet.erase(dpsInit)) {
      // Remove DPS init from input arguments if present.
      inputArguments.erase(
          std::remove(inputArguments.begin(), inputArguments.end(), dpsInit),
          inputArguments.end());

      shouldAddEmptyTensorForDps = true;
    }
  }

  // Convert argument and gather types for function signature.
  llvm::SmallVector<mlir::Type> argumentTypes;
  llvm::SmallVector<mlir::Value> convertedArguments;

  for (auto argument : inputArguments) {
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(argument.getType())) {
      auto convertedTensorType = convertTensorType(tensorType);
      if (convertedTensorType != tensorType) {
        // Create converted tensor value.
        auto emptyTensor = typeBuilder.create<mlir::tt::ttir::EmptyOp>(
            firstOp->getLoc(), tensorType.getShape(),
            convertedTensorType.getElementType());
        auto converted = typeBuilder.create<mlir::tt::ttir::ToLayoutOp>(
            firstOp->getLoc(), argument, emptyTensor);
        argumentTypes.push_back(convertedTensorType);
        convertedArguments.push_back(converted->getResult(0));
      } else {
        argumentTypes.push_back(tensorType);
        convertedArguments.push_back(argument);
      }
    } else {
      argumentTypes.push_back(argument.getType());
      convertedArguments.push_back(argument);
    }
  }

  // TypeRange containing converted result type.
  mlir::TypeRange convertedResultTypes(&convertedResultType, 1);

  // Create function types.
  mlir::FunctionType localFuncType =
      mlir::FunctionType::get(context, argumentTypes, convertedResultTypes);
  mlir::FunctionType funcType =
      mlir::FunctionType::get(context, argumentTypes, convertedResultTypes);

  const llvm::SmallString<16> functionName =
      generateHoistedFuncName(operations);
  llvm::SmallString<16> localFunctionName = functionName;
  localFunctionName.append("_decl");

  auto localFunc = llvm::dyn_cast_if_present<func::FuncOp>(
      sourceModule.lookupSymbol(localFunctionName.str()));

  // Create a new hoisted function only if an equivalent one does not exist.
  if (localFunc == nullptr) {
    // Insert the function and the terminator.
    auto hoistedFunc =
        func::FuncOp::create(firstOp->getLoc(), functionName, funcType);
    targetModule.push_back(hoistedFunc);

    // Add a basic block to the function.
    mlir::Block *block = hoistedFunc.addEntryBlock();
    mlir::OpBuilder builder(block, block->end());

    // Map arguments to block arguments and clone the operation.
    llvm::SmallVector<mlir::Value> newArguments;
    for (auto operand : llvm::enumerate(inputArguments)) {
      newArguments.push_back(block->getArgument(operand.index()));
    }

    mlir::IRMapping mapping;
    for (auto operand : llvm::zip(inputArguments, newArguments)) {
      mapping.map(std::get<0>(operand), std::get<1>(operand));
    }

    // Clone each operation, but modify its type if needed.
    for (auto *opToHoist : operations) {
      auto *clonedOp = builder.clone(*opToHoist, mapping);

      // Update operand types to supported tensor types.
      for (auto operand : clonedOp->getOperands()) {
        if (auto tensorType =
                mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
          auto convertedTensorType = convertTensorType(tensorType);
          operand.setType(convertedTensorType);
        }
      }

      // Update result types to supported tensor types.
      for (auto result : clonedOp->getResults()) {
        if (auto tensorType =
                mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
          auto convertedTensorType = convertTensorType(tensorType);
          result.setType(convertedTensorType);
        }
      }

      // Check if this is the result producing op.
      if (opToHoist == resultProvider) {
        if (shouldAddEmptyTensorForDps) {
          // Prepend an empty tensor for DPS init.
          builder.setInsertionPoint(clonedOp);

          auto emptyTensor = builder.create<mlir::tt::ttir::EmptyOp>(
              firstOp->getLoc(), convertedResultType);

          auto dpsOp = mlir::cast<mlir::DestinationStyleOpInterface>(clonedOp);
          dpsOp.setDpsInitOperand(0, emptyTensor);

          builder.setInsertionPointAfter(clonedOp);
        }

        // Add a return operation to the function with the operation results.
        builder.create<mlir::func::ReturnOp>(firstOp->getLoc(),
                                             clonedOp->getResults());
      }
    }

    // Add bufferization access attributes to function arguments.
    for (auto [index, argument] : llvm::enumerate(hoistedFunc.getArguments())) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(argument.getType())) {
        hoistedFunc.setArgAttr(index, "bufferization.access",
                               builder.getStringAttr("read"));
      }
    }

    // Declare the function prototype in the source module.
    localFunc = func::FuncOp::create(firstOp->getLoc(), localFunctionName.str(),
                                     localFuncType);
    localFunc.setPrivate();

    // Add the function to the module first.
    sourceModule.push_back(localFunc);

    // Get operand ranks and set them as an attribute on the hoisted function.
    hoistedFunc->setAttr(
        "arg_ranks",
        builder.getI64ArrayAttr(getSubgraphOperandTensorRanks(inputArguments)));

    // Mark the hoisted function with the HoistedFuncAttr.
    hoistedFunc->setAttr(HoistedFuncAttr::name,
                         mlir::UnitAttr::get(firstOp->getContext()));
  }
  // Mark the function prototype with the HoistedFuncAttr.
  localFunc->setAttr(HoistedFuncAttr::name,
                     mlir::UnitAttr::get(firstOp->getContext()));

  // Create the call using already converted inputs.
  mlir::OpBuilder opBuilder(resultProvider);
  auto callOp = opBuilder.create<mlir::func::CallOp>(
      resultProvider->getLoc(), localFunc, convertedArguments);

  // Add the hoisted_call attribute.
  callOp->setAttr(HoistedCallAttr::name, UnitAttr::get(firstOp->getContext()));

  // Convert results back to original types if needed.
  mlir::Value callResult = callOp.getResult(0);
  mlir::Value finalResult = callResult;

  if (resultType != convertedResultType) {
    auto converted = opBuilder.create<mlir::tt::ttir::EmptyOp>(
        resultProvider->getLoc(), resultType.getShape(),
        resultType.getElementType());
    auto toOriginal = opBuilder.create<mlir::tt::ttir::ToLayoutOp>(
        resultProvider->getLoc(), callOp.getResult(0), converted);
    finalResult = toOriginal->getResult(0);
  }

  mlir::ValueRange finalResults(finalResult);

  // Replace output producing op with the converted results.
  resultProvider->replaceAllUsesWith(finalResults);

  // Erase the original operations in topologically-reversed order.
  for (auto *opToErase : llvm::reverse(operations)) {
    opToErase->erase();
  }
}

// Predicate type for determining whether an op should be hoisted.
using ShouldHoistPredicateType = std::function<bool(mlir::Operation *)>;

// Predicate type for determining sets of ops to hoist based on given module.
using HoistAnalyzer =
    std::function<llvm::SmallVector<HoistedOpsSet, 4>(mlir::ModuleOp)>;

// HoistAnalyzer which hoists single ops based on a predicate.
HoistAnalyzer singleOpHoistAnalyzer(ShouldHoistPredicateType predicate) {
  return [predicate](mlir::ModuleOp moduleOp) {
    llvm::SmallVector<HoistedOpsSet, 4> hoistedOpsSets;
    // Hoisting individual ops based on the predicate.
    moduleOp.walk([&](mlir::Operation *nestedOp) {
      if (predicate(nestedOp)) {
        HoistedOpsSet opSet;
        opSet.push_back(nestedOp);
        hoistedOpsSets.push_back(opSet);
      }
    });
    return hoistedOpsSets;
  };
}

// HoistAnalyzer which hoists const-eval functions as a whole.
HoistAnalyzer constEvalHoistAnalyzer() {
  return [](mlir::ModuleOp moduleOp) {
    llvm::SmallVector<HoistedOpsSet, 4> hoistedOpsSets;
    moduleOp.walk([&](func::FuncOp funcOp) {
      // Skip non-const-eval functions.
      if (!funcOp->hasAttr(ttmlir::utils::g_constEvalAttrName)) {
        return WalkResult::advance();
      }

      HoistedOpsSet opSet;
      bool interrupted = false;

      funcOp.walk([&](mlir::Operation *nestedOp) {
        // Skip funcop itself
        if (llvm::isa<func::FuncOp>(nestedOp)) {
          return WalkResult::advance();
        }
        // Skip return op
        if (llvm::isa<mlir::func::ReturnOp>(nestedOp)) {
          return WalkResult::advance();
        }
        // If there is already a hoisted call inside, skip hoisting
        // altogether to avoid nested hoisting.
        if (nestedOp->hasAttr(ttir::HoistedCallAttr::name)) {
          interrupted = true;
          return WalkResult::interrupt();
        }
        opSet.push_back(nestedOp);
        return WalkResult::advance();
      });

      if (!interrupted && !opSet.empty()) {
        hoistedOpsSets.push_back(opSet);
      }

      return WalkResult::advance();
    });

    return hoistedOpsSets;
  };
}

// Transform pass to hoist specific ops, based on the provided HoistAnalyzer
// implementation. By default, only ops manually tagged with ttir.should_hoist
// are hoisted.
class CPUHoistTransform
    : public impl::CPUHoistTransformBase<CPUHoistTransform> {
public:
  CPUHoistTransform(HoistAnalyzer analyzer) : analyzer(analyzer) {}

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();

    // If we're inside a DeviceModuleOp, go up to the root ModuleOp.
    if (rootModule->getParentOp() != nullptr) {
      rootModule = rootModule->getParentOfType<ttcore::DeviceModuleOp>()
                       ->getParentOfType<mlir::ModuleOp>();
    }

    ttcore::DeviceModuleOp deviceModule;
    for (Operation &op : rootModule.getBodyRegion().front()) {
      if (auto maybeDeviceModule = dyn_cast<ttcore::DeviceModuleOp>(op)) {
        deviceModule = maybeDeviceModule;
        break;
      }
    }
    assert(deviceModule &&
           "must run tt::WrapDeviceModulePass on IR before hoisting!");

    ModuleOp deviceInnerModule = dyn_cast_if_present<mlir::ModuleOp>(
        deviceModule.getBodyRegion().front().front());
    assert(deviceInnerModule &&
           "ttcore::DeviceModuleOp must have single ModuleOp child!");

    IRRewriter rewriter(&getContext());

    auto loc = rootModule->getLoc();

    // Gather hoistable op sets.
    auto hoistOpSets = analyzer(rootModule);

    if (hoistOpSets.empty()) {
      return;
    }

    // Check if a "cpu_module" already exists.
    ttcore::CPUModuleOp cpuModule;
    mlir::ModuleOp cpuInnerModule;
    for (auto &op : rootModule.getBody()->getOperations()) {
      if (auto module = llvm::dyn_cast<ttcore::CPUModuleOp>(op)) {
        cpuModule = module;
        cpuInnerModule = dyn_cast_if_present<mlir::ModuleOp>(
            cpuModule.getBodyRegion().front().front());
        assert(cpuInnerModule && "CPUModuleOp must contain 1 ModuleOp!");
        break;
      }
    }

    // If no CPU module exists, create one.
    if (!cpuModule) {
      rewriter.setInsertionPointToEnd(rootModule.getBody());
      cpuModule = rewriter.create<ttcore::CPUModuleOp>(loc);
      rewriter.setInsertionPointToStart(&cpuModule.getBodyRegion().front());
      cpuInnerModule = rewriter.create<mlir::ModuleOp>(loc);
    }

    // Hoist each set of ops into a new function in the CPU module.
    for (auto &opSet : hoistOpSets) {
      hoistOperationsToFunction(opSet, deviceInnerModule, cpuInnerModule);
    }
  }

private:
  HoistAnalyzer analyzer;
};
} // namespace

template <typename... Dialects>
std::unique_ptr<mlir::Pass> createCPUHoistForDialectsTransform() {
  const auto customPredicate = [](mlir::Operation *op) {
    return ((op->getDialect()->getTypeID() == TypeID::get<Dialects>()) || ...);
  };
  HoistAnalyzer analyzer = singleOpHoistAnalyzer(customPredicate);
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

template <typename... Ops>
std::unique_ptr<mlir::Pass> createCPUHoistForOpsTransform() {
  const auto customPredicate = [](mlir::Operation *op) {
    return llvm::isa<Ops...>(op);
  };
  HoistAnalyzer analyzer = singleOpHoistAnalyzer(customPredicate);
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

std::unique_ptr<mlir::Pass> createCPUHoistConstEvalTransform() {
  HoistAnalyzer analyzer = constEvalHoistAnalyzer();
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

std::unique_ptr<mlir::Pass> createCPUHoistManuallyTagedOpsTransform() {
  const auto customPredicate = [](mlir::Operation *op) {
    return op->hasAttr(ttir::ShouldHoistAttr::name);
  };
  HoistAnalyzer analyzer = singleOpHoistAnalyzer(customPredicate);
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

// Must explicitly instantiate any dialects and ops we want this pass to
// potentially fallback elsewhere due to template in .cpp file constraints.
#ifdef TTMLIR_ENABLE_STABLEHLO
template std::unique_ptr<mlir::Pass>
createCPUHoistForDialectsTransform<mlir::stablehlo::StablehloDialect>();

template std::unique_ptr<mlir::Pass>
createCPUHoistForOpsTransform<stablehlo::DynamicUpdateSliceOp,
                              stablehlo::EinsumOp>();

#endif
template std::unique_ptr<mlir::Pass>
createCPUHoistForDialectsTransform<TTIRDialect>();

} // namespace mlir::tt::ttir
