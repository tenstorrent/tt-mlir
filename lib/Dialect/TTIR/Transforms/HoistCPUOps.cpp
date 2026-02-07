// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Support/IRHasher.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <optional>

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
using OpsVectorType = llvm::SmallVector<mlir::Operation *>;
using ValuesVectorType = llvm::SmallVector<mlir::Value>;
using TypesVectorType = llvm::SmallVector<mlir::Type>;

// Helper function to get ranks of tensor values.
// Used to populate attrs needed for tensor packing/unpacking operations.
static llvm::SmallVector<int64_t>
getTensorRanks(const ValuesVectorType &values) {
  llvm::SmallVector<int64_t> ranks;
  for (auto value : values) {
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
      ranks.push_back(tensorType.getRank());
    }
  }
  return ranks;
}

// Helper function to collect input arguments to a set of operations.
// If an operand is used multiple times across the ops set, it will appear
// multiple times in the returned vector - input arguments are NOT deduplicated.
static ValuesVectorType collectInputArguments(const OpsVectorType &operations) {
  ValuesVectorType inputArguments;

  for (auto *op : operations) {
    for (auto operand : op->getOperands()) {
      // If the operand is defined inside the hoisted ops set, it is not an
      // input argument to the set of operations.
      if (llvm::is_contained(operations, operand.getDefiningOp())) {
        continue;
      }

      inputArguments.push_back(operand);
    }
  }

  return inputArguments;
}

// Returns the CPU-compatible element type for the given element type.
// Both integer and float types are converted to 32-bit equivalents.
static mlir::Type getCPUCompatibleElementType(mlir::MLIRContext *context,
                                              mlir::Type elementType) {
  if (elementType.isSignedInteger()) {
    return mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed);
  }
  if (elementType.isUnsignedInteger()) {
    return mlir::IntegerType::get(context, 32, mlir::IntegerType::Unsigned);
  }
  if (elementType.isSignlessInteger()) {
    return mlir::IntegerType::get(context, 32, mlir::IntegerType::Signless);
  }
  if (elementType.isFloat()) {
    return mlir::Float32Type::get(context);
  }
  return elementType;
}

// Converts a tensor type to its CPU-compatible equivalent.
static mlir::RankedTensorType
convertTensorType(mlir::RankedTensorType tensorType) {
  auto elementType = tensorType.getElementType();
  auto convertedElementType =
      getCPUCompatibleElementType(tensorType.getContext(), elementType);

  if (elementType != convertedElementType) {
    return mlir::RankedTensorType::get(tensorType.getShape(),
                                       convertedElementType);
  }

  return tensorType;
}

// Converts DenseElementsAttr to a target tensor type.
// Returns std::nullopt if conversion is not supported.
static std::optional<mlir::DenseElementsAttr>
convertDenseElementsAttr(mlir::DenseElementsAttr denseAttr,
                         mlir::RankedTensorType targetType) {
  if (auto floatType =
          mlir::dyn_cast<mlir::FloatType>(targetType.getElementType())) {
    auto values = llvm::map_to_vector(
        denseAttr.getValues<mlir::APFloat>(), [&](mlir::APFloat value) {
          bool losesInfo;
          value.convert(floatType.getFloatSemantics(),
                        mlir::APFloat::rmNearestTiesToEven, &losesInfo);
          return value;
        });
    return mlir::DenseElementsAttr::get(targetType, values);
  }

  if (auto intType =
          mlir::dyn_cast<mlir::IntegerType>(targetType.getElementType())) {
    auto values = llvm::map_to_vector(
        denseAttr.getValues<mlir::APInt>(), [&](const mlir::APInt &value) {
          return value.sextOrTrunc(intType.getWidth());
        });
    return mlir::DenseElementsAttr::get(targetType, values);
  }

  return std::nullopt;
}

// Helper function to convert constant op value attributes to CPU-compatible
// types. This is needed because when we convert the result type of a constant
// op, we also need to convert the underlying data in the value attribute.
static void convertConstantOpValue(mlir::Operation *op) {
  auto constantOp = mlir::dyn_cast<ttir::ConstantOp>(op);
  if (!constantOp) {
    return;
  }

  auto denseAttr =
      mlir::dyn_cast<mlir::DenseElementsAttr>(constantOp.getValue());
  if (!denseAttr) {
    return;
  }

  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(denseAttr.getType());
  if (!sourceType) {
    return;
  }

  auto targetType = convertTensorType(sourceType);
  if (sourceType == targetType) {
    return;
  }

  if (auto convertedAttr = convertDenseElementsAttr(denseAttr, targetType)) {
    constantOp.setValueAttr(*convertedAttr);
  }
}

// Helper function to convert input arguments to CPU-compatible types,
// inserting conversion ops as needed.
static ValuesVectorType
performInputArgumentsConversion(mlir::OpBuilder &opBuilder,
                                const ValuesVectorType &inputArguments) {
  ValuesVectorType convertedArguments;
  for (auto argument : inputArguments) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(argument.getType());

    TT_assertv(tensorType, "Input argument is not a RankedTensorType.");

    auto convertedType = convertTensorType(tensorType);

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
    } else {
      convertedArguments.push_back(argument);
    }
  }
  return convertedArguments;
}

// Helper function to convert result types to CPU-compatible types.
static TypesVectorType
performResultConversions(const ValuesVectorType &outputValues) {
  TypesVectorType resultTypes;
  for (auto outputValue : outputValues) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(outputValue.getType());

    TT_assertv(tensorType, "Output value is not a RankedTensorType.");

    auto convertedType = convertTensorType(tensorType);
    resultTypes.push_back(convertedType);
  }
  return resultTypes;
}

// Helper function to collect the operations producing the output values of a
// set of operations.
static OpsVectorType
collectOutputProducers(const OpsVectorType &operations,
                       const ValuesVectorType &outputValues) {
  OpsVectorType outputProducers;
  for (auto outputValue : outputValues) {
    auto *definingOp = outputValue.getDefiningOp();

    TT_assertv(definingOp, "Output value does not have a defining operation.");

    TT_assertv(llvm::is_contained(operations, definingOp),
               "Output value's defining operation is not in the hoisted ops "
               "set.");

    TT_assertv(!mlir::isa<DestinationStyleOpInterface>(definingOp),
               "DPS ops as output producers are not supported.");

    TT_assertv(definingOp->getNumResults() == 1L,
               "Output producer ops with multiple results are not supported.");

    outputProducers.push_back(definingOp);
  }
  return outputProducers;
}

// Helper function to convert results of callOp back to original types,
// inserting conversion ops as needed.
static void
convertResultsBackToOriginalTypes(mlir::OpBuilder &opBuilder,
                                  mlir::ModuleOp sourceModule,
                                  ValuesVectorType &callOpOutputValues,
                                  ValuesVectorType &originalOutputValues) {
  for (auto [callOpOutput, originalOutput] :
       llvm::zip(callOpOutputValues, originalOutputValues)) {
    auto originalResultType = llvm::dyn_cast_or_null<mlir::RankedTensorType>(
        originalOutput.getType());

    auto convertedResultType =
        llvm::dyn_cast_or_null<mlir::RankedTensorType>(callOpOutput.getType());

    if (originalResultType != convertedResultType) {
      auto emptyTensor = opBuilder.create<mlir::tt::ttir::EmptyOp>(
          sourceModule->getLoc(), originalResultType.getShape(),
          originalResultType.getElementType());
      auto toOriginal = opBuilder.create<mlir::tt::ttir::ToLayoutOp>(
          sourceModule->getLoc(), callOpOutput, emptyTensor);
      // Replace all uses of the output value with the converted one.
      originalOutput.replaceAllUsesWith(toOriginal->getResult(0));
    } else {
      // Replace all uses of the output value with the call result directly.
      originalOutput.replaceAllUsesWith(callOpOutput);
    }
  }
}

// Descriptor for the set of operations which are to be CPU-hoisted.
struct CPUHoistedOpsDescriptor {
  // Vector of operations to be hoisted.
  OpsVectorType operations;
  // Values representing the outputs of the hoisted operations.
  ValuesVectorType outputValues;
  // Suffix for the hoisted function name (appears after "cpu_hoisted_",
  // and before the implementation hash).
  llvm::SmallString<64> funcNameSuffix;

  CPUHoistedOpsDescriptor(const OpsVectorType &ops,
                          const ValuesVectorType &outputs,
                          llvm::SmallString<64> suffix)
      : operations(ops), outputValues(outputs),
        funcNameSuffix(std::move(suffix)) {}
};

// Helper function to drop sign information from integer tensor types,
// used inside CPU-hoisted function definitions. The sign information is NOT
// dropped from CPU-hoisted function declarations.
//
// TODO(dmilinkovic): this workaround is needed because:
// - TOSA and Linalg ops, to which CPU-hoisted ops are getting lowered to,
//   do not support signed/unsigned integer types.
// - Rest of the graph might depend on signed/unsigned integer types,
//   so we cannot change the types in the function declaration.
// This approach should be revisited in the future - issue #6797.
static mlir::Type dropSignInformation(mlir::Type type) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) {
    return type;
  }

  auto elementType = tensorType.getElementType();

  if (!elementType.isInteger() || elementType.isSignlessInteger()) {
    return type;
  }

  auto signlessElementType = mlir::IntegerType::get(
      elementType.getContext(), elementType.getIntOrFloatBitWidth(),
      mlir::IntegerType::Signless);

  auto signlessTensorType =
      mlir::RankedTensorType::get(tensorType.getShape(), signlessElementType);

  return mlir::Type(signlessTensorType);
}

// Helper function to generate a CPU-hoisted function definition.
static func::FuncOp createCPUHoistedFunctionDefinition(
    mlir::MLIRContext *context, mlir::Location loc,
    CPUHoistedOpsDescriptor &descriptor,
    const ValuesVectorType &convertedInputArguments,
    const TypesVectorType &resultTypes, const OpsVectorType &outputProducers) {
  // Determine argument types from input arguments.
  const TypesVectorType argumentTypes =
      llvm::map_to_vector(convertedInputArguments, [](mlir::Value value) {
        return dropSignInformation(value.getType());
      });

  const TypesVectorType convertedResultTypes = llvm::map_to_vector(
      resultTypes, [](mlir::Type type) { return dropSignInformation(type); });

  // Create the function type.
  mlir::FunctionType funcType =
      mlir::FunctionType::get(context, argumentTypes, convertedResultTypes);

  // Create the function.
  auto funcDefinition =
      func::FuncOp::create(loc, "temp_cpu_hoisted_func", funcType);

  // Add a basic block to the function.
  mlir::Block *block = funcDefinition.addEntryBlock();
  mlir::OpBuilder builder(context);
  builder.setInsertionPointToStart(block);

  mlir::IRMapping mapping;
  OpsVectorType clonedOutputProducers;

  // Clone each operation, replacing input operands with block arguments.
  // We iterate in the same order as collectInputArguments, so incrementing
  // an index as we encounter input operands gives us the correct block
  // argument.
  size_t inputArgIdx = 0;

  for (auto *opToHoist : descriptor.operations) {
    auto *clonedOp = builder.clone(*opToHoist, mapping);

    // Replace input argument operands with the corresponding block arguments.
    for (unsigned opIdx = 0; opIdx < clonedOp->getNumOperands(); ++opIdx) {
      auto operand = opToHoist->getOperand(opIdx);
      // Skip operands defined within the hoisted ops set - these are handled
      // by the IRMapping above.
      if (llvm::is_contained(descriptor.operations, operand.getDefiningOp())) {
        continue;
      }
      clonedOp->setOperand(opIdx, block->getArgument(inputArgIdx++));
    }

    // Update operand types to supported tensor types.
    for (auto operand : clonedOp->getOperands()) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
        auto convertedTensorType = convertTensorType(tensorType);
        operand.setType(dropSignInformation(convertedTensorType));
      }
    }

    // Update result types to supported tensor types.
    for (auto result : clonedOp->getResults()) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(result.getType())) {
        auto convertedTensorType = convertTensorType(tensorType);
        result.setType(dropSignInformation(convertedTensorType));
      }
    }

    // Convert constant op value attributes to match the converted result type.
    convertConstantOpValue(clonedOp);

    // Check if this is the output producing op. If it is, keep track of it
    // for later.
    if (llvm::is_contained(outputProducers, opToHoist)) {
      clonedOutputProducers.push_back(clonedOp);
    }
  }

  // Add return op to the function from the cloned output producers.
  const ValuesVectorType returnValues =
      llvm::map_to_vector(clonedOutputProducers, [](mlir::Operation *op) {
        return mlir::cast<mlir::Value>(op->getResult(0));
      });

  builder.create<mlir::func::ReturnOp>(loc, returnValues);

  // Add bufferization access attributes to function arguments.
  for (auto [index, argument] :
       llvm::enumerate(funcDefinition.getArguments())) {
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(argument.getType())) {
      funcDefinition.setArgAttr(index, "bufferization.access",
                                builder.getStringAttr("read"));
    }
  }

  // Set tensor rank attributes for wrapper function generation.
  funcDefinition->setAttr("arg_ranks", builder.getI64ArrayAttr(getTensorRanks(
                                           convertedInputArguments)));

  funcDefinition->setAttr(
      "result_ranks",
      builder.getI64ArrayAttr(getTensorRanks(descriptor.outputValues)));

  // Set the type of the function.
  ttmlir::utils::setFunctionType(funcDefinition,
                                 ttmlir::utils::FunctionType::ForwardCPU);

  // Finally, hash the function implementation and set the func_hash attribute.
  auto funcHash = hashFuncOp(funcDefinition);
  funcDefinition->setAttr("func_hash", builder.getStringAttr(funcHash));

  return funcDefinition;
}

// Helper function to generate a CPU-hoisted function declaration.
static func::FuncOp createCPUHoistedFunctionDeclaration(
    mlir::MLIRContext *context, mlir::Location loc,
    CPUHoistedOpsDescriptor &descriptor, const ValuesVectorType &inputArguments,
    const TypesVectorType &resultTypes, llvm::StringRef funcHash) {
  // Determine argument types from input arguments.
  const TypesVectorType argumentTypes = llvm::map_to_vector(
      inputArguments, [](mlir::Value value) { return value.getType(); });

  // Create the function type.
  mlir::FunctionType funcType =
      mlir::FunctionType::get(context, argumentTypes, resultTypes);

  // Create the function.
  auto funcDeclaration =
      func::FuncOp::create(loc, "temp_cpu_hoisted_func_decl", funcType);

  // Set the type of the function.
  ttmlir::utils::setFunctionType(
      funcDeclaration, ttmlir::utils::FunctionType::ForwardCPUDeclaration);

  // Make the declaration private.
  funcDeclaration.setPrivate();

  // Set the func_hash attribute.
  funcDeclaration->setAttr("func_hash",
                           mlir::StringAttr::get(context, funcHash));

  return funcDeclaration;
}

static func::FuncOp lookupCPUHoistedFunction(mlir::ModuleOp module,
                                             llvm::StringRef funcHash) {
  for (auto func : module.getOps<func::FuncOp>()) {
    auto existingFuncHashAttr =
        func->getAttrOfType<mlir::StringAttr>("func_hash");
    if (existingFuncHashAttr && existingFuncHashAttr.getValue() == funcHash) {
      return func;
    }
  }
  return nullptr;
}

// Helper function to generate a unique function name for the CPU-hoisted
// function.
//
// Hoisted function names will be generated as:
// cpu_hoisted_<funcNameSuffix>_<hash>, where <hash> is a
// SHA256 hash of the function implementation.
//
static llvm::SmallString<64>
getUniqueFunctionName(const CPUHoistedOpsDescriptor &descriptor,
                      const llvm::StringRef hash) {
  llvm::SmallString<64> uniqueName;

  llvm::raw_svector_ostream os(uniqueName);
  os << "cpu_hoisted_" << descriptor.funcNameSuffix << "_" << hash;

  std::replace(uniqueName.begin(), uniqueName.end(), '.', '_');

  return uniqueName;
}

// Helper function to hoist an arbitrary set of ops into a new function in
// the CPU module, generate a matching extern prototype (declaration) in the
// Device module, and replace the ops in the set with a callOp to the extern
// function declaration.
static void hoistOperationsToFunction(CPUHoistedOpsDescriptor &descriptor,
                                      mlir::ModuleOp deviceModule,
                                      mlir::ModuleOp cpuModule) {
  mlir::MLIRContext *context = deviceModule.getContext();

  const OpsVectorType outputProducers =
      collectOutputProducers(descriptor.operations, descriptor.outputValues);

  const TypesVectorType resultTypes =
      performResultConversions(descriptor.outputValues);

  const ValuesVectorType inputArguments =
      collectInputArguments(descriptor.operations);

  mlir::OpBuilder opBuilder(descriptor.operations.front());
  const ValuesVectorType convertedInputArguments =
      performInputArgumentsConversion(opBuilder, inputArguments);

  // Create the CPU-hoisted function definition.
  func::FuncOp funcDefinition = createCPUHoistedFunctionDefinition(
      cpuModule->getContext(), cpuModule->getLoc(), descriptor,
      convertedInputArguments, resultTypes, outputProducers);

  auto funcHash =
      funcDefinition->getAttrOfType<mlir::StringAttr>("func_hash").getValue();

  // Lookup existing function declaration in the Device module by hash.
  func::FuncOp funcDeclaration =
      lookupCPUHoistedFunction(deviceModule, funcHash);

  // If the function doesn't exist, we need to insert the definition
  // into the CPU module, and create the declaration in the Device module.
  if (!funcDeclaration) {
    // Insert the function definition into the CPU module.
    cpuModule.push_back(funcDefinition);

    // Create the function declaration in the Device module.
    funcDeclaration = createCPUHoistedFunctionDeclaration(
        deviceModule->getContext(), deviceModule->getLoc(), descriptor,
        convertedInputArguments, resultTypes, funcHash);
    deviceModule.push_back(funcDeclaration);

    // Use first 8 characters of the function hash as part of the unique
    // function name.
    //
    auto hashPrefix = funcHash.substr(0, 8);

    // Rename the function definition and declaration.
    auto functionName = getUniqueFunctionName(descriptor, hashPrefix);
    funcDefinition.setName(functionName);
    funcDeclaration.setName(functionName);
  } else {
    // If the function already exists, we can discard the newly created
    // definition.
    funcDefinition.erase();
  }

  // Create the call using already converted inputs.
  opBuilder.setInsertionPointAfter(descriptor.operations.back());

  auto callOp = opBuilder.create<mlir::func::CallOp>(
      deviceModule->getLoc(), funcDeclaration, convertedInputArguments);

  // Add the hoisted_call attribute.
  callOp->setAttr(CPUHoistedCallAttr::name, UnitAttr::get(context));

  // Convert call results back to original types as needed.
  ValuesVectorType callOpResults = callOp.getResults();
  convertResultsBackToOriginalTypes(opBuilder, deviceModule, callOpResults,
                                    descriptor.outputValues);

  // Erase the original operations in a topologically-reversed order.
  for (auto *opToErase : llvm::reverse(descriptor.operations)) {
    opToErase->erase();
  }
}
} // namespace

/*
====================================================================
-------------------- CPU Hoisting Analyzers ------------------------
====================================================================
*/

namespace {
// Predicate type for determining sets of ops to hoist in the provided function.
// Returns a vector of descriptors, one for each set of ops to hoist.
//
// TODO(dmilinkovic): Currently, the implementation limits the user to hoisting
// ops inside a single function; we should consider allowing cross-function
// CPU-hoisting in the future - issue #6097.
using CPUHoistAnalyzerType =
    std::function<llvm::SmallVector<CPUHoistedOpsDescriptor>(func::FuncOp)>;

// Predicate type for determining whether an op should be hoisted in an op-by-op
// CPU hoisting analyzer.
using ShouldHoistOpType = std::function<bool(mlir::Operation *)>;

/*
====================================================================
------------------ Single op CPU-hoisting analyzer -----------------
====================================================================
*/

// HoistAnalyzer which hoists single ops based on a provided predicate.
CPUHoistAnalyzerType singleOpHoistAnalyzer(ShouldHoistOpType predicate) {
  return [predicate](func::FuncOp funcOp) {
    llvm::SmallVector<CPUHoistedOpsDescriptor> hoistedOpsDescriptors;
    // Hoisting individual ops based on the predicate.
    funcOp.walk([&](mlir::Operation *nestedOp) {
      if (predicate(nestedOp)) {
        OpsVectorType operations{nestedOp};
        ValuesVectorType outputValues{nestedOp->getResult(0)};

        // Using the op name as the CPU-hoisted function's suffix.
        hoistedOpsDescriptors.emplace_back(
            operations, outputValues,
            nestedOp->getName().getIdentifier().getValue());
      }
    });

    return hoistedOpsDescriptors;
  };
}

/*
====================================================================
----------------- Const-eval CPU-hoisting analyzer -----------------
====================================================================
*/

// Check if an op is "transparent" - it doesn't change semantic meaning,
// just format/type.
static bool isTransparentOp(mlir::Operation *op) {
  return mlir::isa<ReshapeOp, TypecastOp>(op);
}

// Walk backward from a value through transparent ops in a single traversal.
// If the chain terminates at a creation skippable op, return it.
static llvm::SmallVector<mlir::Operation *> traceCreationOpChain(Value v) {
  llvm::SmallVector<mlir::Operation *> chain;

  while (Operation *defOp = v.getDefiningOp()) {
    if (defOp->hasTrait<ttcore::Trait::TTCoreCreationOpTrait>()) {
      chain.push_back(defOp);
      return chain;
    }

    if (isTransparentOp(defOp)) {
      chain.push_back(defOp);
      v = defOp->getOperand(0);
      continue;
    }

    // Non-transparent, non-creation: chain is not skippable.
    break;
  }

  return {};
}

// CPUHoistAnalyzer which hoists operations from const-eval functions.
// Motivation for CPU-hoisting const-eval ops:
// - CPU-hoisted ops operate on 32-bit integers/floats, which should result in
//   more precise calculations compared to device execution.
// - Peak DRAM/L1 usage should be reduced, since intermediate tensors are stored
//   in host memory. This is especially beneficial for tensors which would take
//   up significantly more L1 if tilized (e.g. tensor<1024x1024x1x1).
CPUHoistAnalyzerType constEvalHoistAnalyzer() {
  return [](func::FuncOp funcOp) {
    if (!ttmlir::utils::isConstEvalFunc(funcOp)) {
      return llvm::SmallVector<CPUHoistedOpsDescriptor>{};
    }

    CPUHoistedOpsDescriptor descriptor({}, {}, llvm::StringRef("const_eval"));

    // Check if it is possible to CPU-hoist this const-eval funciton.
    auto walkResult = funcOp.walk([&](mlir::Operation *nestedOp) {
      // If there is already a CPU-hoisted call inside the const-eval
      // subgraph, skip CPU hoisting altogether to avoid nested hoisting.
      if (nestedOp->hasAttr(ttir::CPUHoistedCallAttr::name)) {
        return WalkResult::interrupt();
      }

      if (auto meshShardOp =
              mlir::dyn_cast<mlir::tt::ttir::MeshShardOp>(nestedOp)) {
        // If there is a non-identity TTIR MeshShardOp, skip CPU hoisting
        // altogether.
        // TODO(dmilinkovic) - issue #6709,
        if (meshShardOp.getShardType() != ttcore::MeshShardType::Identity) {
          return WalkResult::interrupt();
        }
      }

      // If there is any CCL op, skip CPU hoisting altogether.
      // TODO(dmilinkovic) - issue #6709
      if (mlir::isa<mlir::tt::ttir::AllGatherOp, mlir::tt::ttir::AllReduceOp,
                    mlir::tt::ttir::ReduceScatterOp,
                    mlir::tt::ttir::CollectivePermuteOp,
                    mlir::tt::ttir::AllToAllOp,
                    mlir::tt::ttir::CollectiveBroadcastOp>(nestedOp)) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      return llvm::SmallVector<CPUHoistedOpsDescriptor>{};
    }

    auto returnOp =
        llvm::cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

    llvm::SmallPtrSet<mlir::Operation *, 8> opsToSkip;

    // Skip chains of creation ops and transparent ops leading to
    // them. This is done because:
    // 1. Downstream passes might try to extract constant values from these ops,
    //    which isn't possible if these are moved to the CPU-module.
    // 2. CPU-hoisting creation ops which are results of const-eval doesn't
    //    improve PCC nor peak DRAM/L1 usage.
    for (Value retVal : returnOp.getOperands()) {
      auto chain = traceCreationOpChain(retVal);
      if (chain.empty()) {
        descriptor.outputValues.push_back(retVal);
      } else {
        opsToSkip.insert(chain.begin(), chain.end());
      }
    }

    // Skip identity MeshShard ops.
    // These ops are just semantic decorators, and are no-ops
    // from the runtime perspective.
    for (auto nestedOp : funcOp.getOps<mlir::tt::ttir::MeshShardOp>()) {
      if (nestedOp.getShardType() == ttcore::MeshShardType::Identity) {
        opsToSkip.insert(nestedOp);
      }
    }

    // Collect all ops that are not skipped.
    funcOp.walk([&](mlir::Operation *nestedOp) {
      if (llvm::isa<func::FuncOp, func::ReturnOp>(nestedOp)) {
        return;
      }

      if (opsToSkip.contains(nestedOp)) {
        return;
      }

      descriptor.operations.push_back(nestedOp);
    });

    if (descriptor.operations.empty()) {
      return llvm::SmallVector<CPUHoistedOpsDescriptor>{};
    }

    return llvm::SmallVector<CPUHoistedOpsDescriptor>{descriptor};
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

    // We must run this transform on the root ModuleOp, since we are
    // creating new Op's within the root.
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
    TT_assertv(deviceModule,
               "Must run tt::WrapDeviceModulePass on IR before hoisting.");

    ModuleOp deviceInnerModule = dyn_cast_if_present<mlir::ModuleOp>(
        deviceModule.getBodyRegion().front().front());
    TT_assertv(deviceInnerModule,
               "ttcore::DeviceModuleOp must have single ModuleOp child.");

    IRRewriter rewriter(&getContext());

    auto loc = rootModule->getLoc();

    // Collect hoisted ops descriptors by iterating over all functions.
    llvm::SmallVector<CPUHoistedOpsDescriptor> hoistedOpsDescriptors;
    deviceInnerModule.walk([&](func::FuncOp funcOp) {
      auto descriptors = analyzer(funcOp);
      for (auto &descriptor : descriptors) {
        hoistedOpsDescriptors.push_back(std::move(descriptor));
      }
    });

    // We don't want to create a CPUModuleOp etc. if we aren't hoisting any
    // ops.
    if (hoistedOpsDescriptors.empty()) {
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
        TT_assertv(cpuInnerModule, "CPUModuleOp must contain 1 ModuleOp.");
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
    for (auto &descriptor : hoistedOpsDescriptors) {
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

std::unique_ptr<mlir::Pass> createCPUHoistManuallyTaggedOpsTransform() {
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
