// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include <string>

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
// Helper function to get ranks of a set of input tensor values.
// We use this to populate attrs which we need to perform
// tensor unpacking operations later.
static llvm::SmallVector<int64_t, 3>
getSubgraphOperandTensorRanks(const ValuesVectorType &inputValues) {
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

// Helper function which generates a unique name based on operation type +
// argument tensors dims & types.
std::string generateHoistedFuncName(const OpsVectorType &ops) {
  llvm::SmallString<16> uniqueName("hoisted_");

  for (auto *op : ops) {
    uniqueName.append(op->getName().getStringRef());

    for (auto operand : op->getOperands()) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
        uniqueName += "_";
        llvm::raw_svector_ostream os(uniqueName);
        llvm::interleave(tensorType.getShape(), os, "x");
      }
    }
  }

  uniqueName += "_func";
  std::replace(uniqueName.begin(), uniqueName.end(), '.', '_');

  std::string result(uniqueName.begin(), uniqueName.end());
  return result;
}

// Helper function to collect input arguments to a set of operations.
//
// If an argument represents a DPS output of a result-producing op,
// it is not considered an input argument and is skipped - instead of these,
// internally created ttir.empty tensors will be used.
//
// This ugly logic related to DPS arguments should be temporary,
// as the DPS semantics should be removed from the TTIR dialect in the future.
static ValuesVectorType
collectInputArguments(const OpsVectorType &operations,
                      const OpsVectorType &resultProviders) {
  ValuesVectorType inputArguments;

  for (auto *op : operations) {
    for (auto operand : op->getOperands()) {
      // If the operand is defined inside the hoisted ops set, it is not an
      // input argument to the set of operations.
      if (llvm::is_contained(operations, operand.getDefiningOp())) {
        continue;
      }

      // Here, we know that the operand should be an input argument.

      // If the argument is a DPS output of a result-producing op, skip it.
      if (llvm::is_contained(resultProviders, op)) {
        if (auto dpsOp = mlir::dyn_cast<DestinationStyleOpInterface>(op)) {
          if (llvm::is_contained(dpsOp.getDpsInits(), operand)) {
            continue;
          }
        }
      }

      // Insert the argument if not already present.
      if (llvm::is_contained(inputArguments, operand) == false) {
        inputArguments.push_back(operand);
      }
    }
  }

  return inputArguments;
}

// Helper function to convert tensor types to CPU-compatible types.
static mlir::RankedTensorType
convertTensorType(mlir::RankedTensorType tensorType,
                  mlir::MLIRContext *context) {
  const auto f32Type = mlir::Float32Type::get(context);
  const auto i32Type =
      mlir::IntegerType::get(context, 32, mlir::IntegerType::Signless);

  const auto elementType = tensorType.getElementType();
  auto convertedElementType = tensorType.getElementType();

  if (elementType.isInteger()) {
    convertedElementType = i32Type;
  } else if (elementType.isFloat()) {
    convertedElementType = f32Type;
  }

  if (elementType != convertedElementType) {
    return mlir::RankedTensorType::get(
        tensorType.getShape(), convertedElementType, tensorType.getEncoding());
  }

  return tensorType;
}

// Helper function to convert input arguments to CPU-compatible types,
// inserting conversion ops as needed.
static void
performInputArgumentsConversion(mlir::OpBuilder &opBuilder,
                                const ValuesVectorType &inputArguments,
                                TypesVectorType &convertedArgumentTypes,
                                ValuesVectorType &convertedArguments) {
  for (auto argument : inputArguments) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(argument.getType());

    assert(tensorType && "Input argument is not a RankedTensorType!");

    auto convertedType = convertTensorType(tensorType, opBuilder.getContext());

    if (tensorType != convertedType) {
      // Create converted tensor value.
      auto emptyTensor = opBuilder.create<mlir::tt::ttir::EmptyOp>(
          argument.getLoc(), tensorType.getShape(),
          convertedType.getElementType());
      auto convertedArgument = opBuilder
                                   .create<mlir::tt::ttir::ToLayoutOp>(
                                       argument.getLoc(), argument, emptyTensor)
                                   ->getResult(0);

      convertedArguments.push_back(convertedArgument);
      convertedArgumentTypes.push_back(convertedType);
    } else {
      convertedArguments.push_back(argument);
      convertedArgumentTypes.push_back(argument.getType());
    }
  }
}

// Helper function to convert result types to CPU-compatible types.
static TypesVectorType
performResultConversions(const ValuesVectorType &outputValues) {
  TypesVectorType resultTypes;
  for (auto outputValue : outputValues) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(outputValue.getType());

    assert(tensorType && "Output value is not a RankedTensorType!");

    auto convertedType =
        convertTensorType(tensorType, outputValue.getContext());
    resultTypes.push_back(convertedType);
  }
  return resultTypes;
}

// Helper function to collect the operations producing the output values of a
// set of operations.
static OpsVectorType
collectOutputProviders(const OpsVectorType &operations,
                       const ValuesVectorType &outputValues) {
  OpsVectorType outputProivders;
  for (auto outputValue : outputValues) {
    auto *definingOp = outputValue.getDefiningOp();
    assert(definingOp && "Output value does not have a defining operation!");
    assert(llvm::is_contained(operations, definingOp) &&
           "Output value's defining operation is not in the hoisted ops set!");
    outputProivders.push_back(definingOp);
  }
  return outputProivders;
}

// Helper function to convert results of callOp back to original types,
// inserting conversion ops as needed.
static void
convertResultsBackToOriginalTypes(mlir::OpBuilder &opBuilder,
                                  mlir::ModuleOp sourceModule,
                                  ValuesVectorType &callOpOutputValues,
                                  ValuesVectorType &originalOutputValues) {
  for (auto [callOpOuput, originalOutput] :
       llvm::zip(callOpOutputValues, originalOutputValues)) {
    auto originalResultType = llvm::dyn_cast_or_null<mlir::RankedTensorType>(
        originalOutput.getType());

    auto convertedResultType =
        llvm::dyn_cast_or_null<mlir::RankedTensorType>(callOpOuput.getType());

    if (originalResultType != convertedResultType) {
      auto emptyTensor = opBuilder.create<mlir::tt::ttir::EmptyOp>(
          sourceModule->getLoc(), originalResultType.getShape(),
          originalResultType.getElementType());
      auto toOriginal = opBuilder.create<mlir::tt::ttir::ToLayoutOp>(
          sourceModule->getLoc(), callOpOuput, emptyTensor);
      // Replace all uses of the output value with the converted one.
      originalOutput.replaceAllUsesWith(toOriginal->getResult(0));
    } else {
      // Replace all uses of the output value with the call result directly.
      originalOutput.replaceAllUsesWith(callOpOuput);
    }
  }
}

// Helper function to hoist an arbitrary set of ops into a new function in
// targetModule, generate a matching extern prototype in the sourceModule, and
// replace the ops in the set with a callOp to the extern function.
static void hoistOperationsToFunction(CPUHoistedOpsDescriptor &descriptor,
                                      mlir::ModuleOp sourceModule,
                                      mlir::ModuleOp targetModule) {
  mlir::MLIRContext *context = sourceModule.getContext();

  const auto resultProviders =
      collectOutputProviders(descriptor.operations, descriptor.outputValues);

  const auto resultTypes = performResultConversions(descriptor.outputValues);

  const auto inputArguments =
      collectInputArguments(descriptor.operations, resultProviders);

  // Convert argument and gather types for function signature.
  TypesVectorType argumentTypes;
  ValuesVectorType convertedArguments;

  mlir::OpBuilder opBuilder(descriptor.operations.front());
  performInputArgumentsConversion(opBuilder, inputArguments, argumentTypes,
                                  convertedArguments);

  // Creating function types.
  mlir::FunctionType localFuncType =
      mlir::FunctionType::get(context, argumentTypes, resultTypes);
  mlir::FunctionType funcType =
      mlir::FunctionType::get(context, argumentTypes, resultTypes);

  const auto localFunctionName = descriptor.funcName + "_decl";
  auto localFunc = llvm::dyn_cast_if_present<func::FuncOp>(
      sourceModule.lookupSymbol(localFunctionName));

  // Create a new hoisted function only if an equivalent one does not exist.
  if (localFunc == nullptr) {
    // Insert the function and the terminator.
    auto hoistedFunc = func::FuncOp::create(targetModule->getLoc(),
                                            descriptor.funcName, funcType);
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
    for (auto operands : llvm::zip(inputArguments, newArguments)) {
      mapping.map(std::get<0>(operands), std::get<1>(operands));
    }

    // Clone each operation, but modify its type if needed.
    OpsVectorType clonedResultProviders;

    for (auto *opToHoist : descriptor.operations) {
      auto *clonedOp = builder.clone(*opToHoist, mapping);

      // Update operand types to supported tensor types.
      for (auto operand : clonedOp->getOperands()) {
        if (auto tensorType =
                mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
          auto convertedTensorType = convertTensorType(tensorType, context);
          operand.setType(convertedTensorType);
        }
      }

      // Update result types to supported tensor types.
      for (auto result : clonedOp->getResults()) {
        if (auto tensorType =
                mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
          auto convertedTensorType = convertTensorType(tensorType, context);
          result.setType(convertedTensorType);
        }
      }

      // Check if this is the result producing op. If it is, keep track of it
      // for later.
      if (llvm::is_contained(resultProviders, opToHoist)) {
        clonedResultProviders.push_back(clonedOp);

        // If the result producer is a DPS op, its output argument might have
        // been skipped in collectInputArguments - create an empty tensor to
        // use as the DPS init instead.
        //
        // This ugly workaround should be temporary, as in the future, the DPS
        // semantics should be removed entirely from the TTIR dialect.
        if (auto dpsOp =
                mlir::dyn_cast<DestinationStyleOpInterface>(opToHoist)) {
          auto originalDpsInit = dpsOp.getDpsInits().front();

          bool dpsInitSkipped = true;

          // DPS init doesn't exist in the input arguments.
          dpsInitSkipped &=
              !llvm::is_contained(inputArguments, originalDpsInit);

          // DPS init hasn't been produced by another op in the hoisted set.
          dpsInitSkipped &= !llvm::is_contained(
              descriptor.operations, originalDpsInit.getDefiningOp());

          if (dpsInitSkipped) {
            auto clonedDpsOp =
                llvm::dyn_cast<DestinationStyleOpInterface>(clonedOp);

            auto clonedDpsInit = clonedDpsOp.getDpsInits().front();

            builder.setInsertionPoint(clonedOp);

            auto tensorType =
                mlir::dyn_cast<mlir::RankedTensorType>(clonedDpsInit.getType());

            assert(tensorType && "DPS init is not a RankedTensorType!");

            auto emptyTensor = builder.create<mlir::tt::ttir::EmptyOp>(
                targetModule->getLoc(), tensorType.getShape(),
                tensorType.getElementType(), tensorType.getEncoding());

            clonedDpsOp.setDpsInitOperand(0, emptyTensor->getResult(0));

            builder.setInsertionPointAfter(clonedOp);
          }
        }
      }
    }

    // Add return op to the function from the cloned result providers.
    llvm::SmallVector<mlir::Value, 4> returnValues;
    for (auto *resultProvider : clonedResultProviders) {
      returnValues.push_back(resultProvider->getResult(0));
    }

    builder.create<mlir::func::ReturnOp>(targetModule->getLoc(), returnValues);

    // Add bufferization access attributes to function arguments.
    for (auto [index, argument] : llvm::enumerate(hoistedFunc.getArguments())) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(argument.getType())) {
        hoistedFunc.setArgAttr(index, "bufferization.access",
                               builder.getStringAttr("read"));
      }
    }

    // Declare the function prototype in the source module.
    localFunc = func::FuncOp::create(sourceModule->getLoc(), localFunctionName,
                                     localFuncType);
    localFunc.setPrivate();

    // Add the function to the module first.
    sourceModule.push_back(localFunc);

    // Get operand ranks and set them as an attribute on the hoisted function.
    hoistedFunc->setAttr(
        "arg_ranks",
        builder.getI64ArrayAttr(getSubgraphOperandTensorRanks(inputArguments)));

    // Mark the hoisted function with the HoistedFuncAttr.
    hoistedFunc->setAttr(HoistedFuncAttr::name, mlir::UnitAttr::get(context));
  }

  // Mark the local function with the HoistedFuncAttr.
  localFunc->setAttr(HoistedFuncAttr::name, mlir::UnitAttr::get(context));

  // Create the call using already converted inputs.
  auto callOp = opBuilder.create<mlir::func::CallOp>(
      sourceModule->getLoc(), localFunc, convertedArguments);

  // Add the hoisted_call attribute.
  callOp->setAttr(HoistedCallAttr::name, UnitAttr::get(context));

  ValuesVectorType callOpResults = callOp.getResults();

  convertResultsBackToOriginalTypes(opBuilder, sourceModule, callOpResults,
                                    descriptor.outputValues);

  // Erase the original operations in a topologically-reversed order.
  for (auto *opToErase : llvm::reverse(descriptor.operations)) {
    opToErase->erase();
  }
}

// HoistAnalyzer which hoists single ops based on a provided predicate.
CPUHoistAnalyzerType singleOpHoistAnalyzer(ShouldHoistOpType predicate) {
  return [predicate](mlir::ModuleOp moduleOp) {
    llvm::SmallVector<CPUHoistedOpsDescriptor, 4> hoistedOpsDescriptors;
    // Hoisting individual ops based on the predicate.
    moduleOp.walk([&](mlir::Operation *nestedOp) {
      if (predicate(nestedOp)) {
        OpsVectorType operations{nestedOp};
        ValuesVectorType outputValues{nestedOp->getResult(0)};

        hoistedOpsDescriptors.emplace_back(operations, outputValues,
                                           generateHoistedFuncName({nestedOp}));
      }
    });

    return hoistedOpsDescriptors;
  };
}

// HoistAnalyzer which hoists const-eval functions as a whole.
CPUHoistAnalyzerType constEvalHoistAnalyzer() {
  return [](mlir::ModuleOp moduleOp) {
    llvm::SmallVector<CPUHoistedOpsDescriptor, 4> hoistedOpsDescriptors;

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isConstEvalFunc(funcOp)) {
        return WalkResult::advance();
      }

      const auto hoistedFuncName = "hoisted_" + funcOp.getName().str();
      CPUHoistedOpsDescriptor descriptor({}, {}, hoistedFuncName);

      auto walkResult = funcOp.walk([&](mlir::Operation *nestedOp) {
        // Skip the FuncOp itself.
        if (llvm::isa<func::FuncOp>(nestedOp)) {
          return WalkResult::advance();
        }

        // Skip the ReturnOp, but collect its operands as outputs.
        if (llvm::isa<mlir::func::ReturnOp>(nestedOp)) {
          for (auto retVal : nestedOp->getOperands()) {
            descriptor.outputValues.push_back(retVal);
          }
          return WalkResult::advance();
        }

        // If there is already a CPU-hoisted call inside the const-eval
        // subgraph, skip CPU hoisting altogether to avoid nested hoisting.
        if (nestedOp->hasAttr(ttir::HoistedCallAttr::name)) {
          return WalkResult::interrupt();
        }

        descriptor.operations.push_back(nestedOp);
        return WalkResult::advance();
      });

      if (!walkResult.wasInterrupted() && !descriptor.operations.empty()) {
        hoistedOpsDescriptors.push_back(descriptor);
      }

      return WalkResult::advance();
    });

    return hoistedOpsDescriptors;
  };
}

// Transform pass to hoist specific ops, based on the provided
// HoistAnalyzerType implementation. By default, only ops manually tagged with
// ttir.should_hoist are hoisted.
class CPUHoistTransform
    : public impl::CPUHoistTransformBase<CPUHoistTransform> {
public:
  CPUHoistTransform(CPUHoistAnalyzerType analyzer) : analyzer(analyzer) {}

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();

    // We must run this transform on the root ModuleOp, since we are creating
    // new Op's within the root.
    if (rootModule->getParentOp() != nullptr) {
      return;
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
    for (auto &descriptor : analyzer(rootModule)) {
      hoistOperationsToFunction(descriptor, deviceInnerModule, cpuInnerModule);
    }
  }

private:
  CPUHoistAnalyzerType analyzer;
};
} // namespace

template <typename... Dialects>
std::unique_ptr<mlir::Pass> createCPUHoistForDialectsTransform() {
  const auto customPredicate = [](mlir::Operation *op) {
    return ((op->getDialect()->getTypeID() == TypeID::get<Dialects>()) || ...);
  };
  CPUHoistAnalyzerType analyzer = singleOpHoistAnalyzer(customPredicate);
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

template <typename... Ops>
std::unique_ptr<mlir::Pass> createCPUHoistForOpsTransform() {
  const auto customPredicate = [](mlir::Operation *op) {
    return llvm::isa<Ops...>(op);
  };
  CPUHoistAnalyzerType analyzer = singleOpHoistAnalyzer(customPredicate);
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

std::unique_ptr<mlir::Pass> createCPUHoistConstEvalTransform() {
  CPUHoistAnalyzerType analyzer = constEvalHoistAnalyzer();
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

std::unique_ptr<mlir::Pass> createCPUHoistManuallyTagedOpsTransform() {
  const auto customPredicate = [](mlir::Operation *op) {
    return op->hasAttr(ttir::ShouldHoistAttr::name);
  };
  CPUHoistAnalyzerType analyzer = singleOpHoistAnalyzer(customPredicate);
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

std::unique_ptr<mlir::Pass>
createSingleOpCPUHoistTransform(ShouldHoistOpType predicate) {
  CPUHoistAnalyzerType analyzer = singleOpHoistAnalyzer(predicate);
  auto pass = std::make_unique<CPUHoistTransform>(analyzer);
  return pass;
}

std::unique_ptr<mlir::Pass>
createCustomCPUHoistTransform(CPUHoistAnalyzerType analyzer) {
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
