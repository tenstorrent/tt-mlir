// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::tt::ttnn;
using namespace mlir::tt::emitpy;

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPREPAREISOLATEDFORPYTEST
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttnn

namespace {

// Helper to extract op name from "ttnn.reshape" -> "reshape"
std::string extractOpShortName(StringRef fullOpName) {
  size_t dotPos = fullOpName.find_last_of('.');
  if (dotPos != StringRef::npos && dotPos + 1 < fullOpName.size()) {
    return fullOpName.substr(dotPos + 1).str();
  }
  return fullOpName.str();
}

// Helper to format array attribute as Python list string
std::string formatArrayAsPythonList(ArrayAttr arr) {
  std::string result = "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    Attribute attr = arr[i];
    if (auto innerArr = dyn_cast<ArrayAttr>(attr)) {
      // Nested array (e.g., shape [1, 28, 28, 1])
      result += "[";
      for (size_t j = 0; j < innerArr.size(); ++j) {
        if (auto intAttr = dyn_cast<IntegerAttr>(innerArr[j])) {
          result += std::to_string(intAttr.getInt());
        }
        if (j < innerArr.size() - 1) result += ", ";
      }
      result += "]";
    } else if (auto strAttr = dyn_cast<StringAttr>(attr)) {
      result += "'" + strAttr.getValue().str() + "'";
    } else if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      result += std::to_string(intAttr.getInt());
    }
    if (i < arr.size() - 1) result += ", ";
  }
  result += "]";
  return result;
}

// Helper to format a single opaque attribute value as Python string
std::string formatOpaqueAttrValue(mlir::tt::emitpy::OpaqueAttr attr) {
  return attr.getValue().str();
}

// Structure to hold information about varying attributes
struct VaryingAttribute {
  std::string name;                    // Parameter name (e.g., "shape", "kernel_size")
  size_t argPosition;                  // Position in call_opaque args array
  SmallVector<std::string> values;     // All values across functions
  bool isVarying;                      // Whether values actually differ
};

// Find the main call_opaque operation in a function
CallOpaqueOp findCallOpaqueOp(func::FuncOp funcOp) {
  CallOpaqueOp result;
  funcOp.walk([&](CallOpaqueOp callOp) {
    // Find the main operation (not utility ops like get_device)
    StringRef callee = callOp.getCallee();
    if (!callee.contains("get_device") && !callee.contains("DeviceGetter")) {
      result = callOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

// Extract inline opaque attributes from call_opaque args
SmallVector<std::pair<size_t, mlir::tt::emitpy::OpaqueAttr>> extractInlineOpaques(CallOpaqueOp callOp) {
  SmallVector<std::pair<size_t, mlir::tt::emitpy::OpaqueAttr>> result;

  if (!callOp) {
    return result;
  }

  auto argsOpt = callOp.getArgs();
  if (!argsOpt) {
    return result;
  }

  ArrayAttr args = *argsOpt;
  for (size_t i = 0; i < args.size(); ++i) {
    if (auto opaqueAttr = dyn_cast<mlir::tt::emitpy::OpaqueAttr>(args[i])) {
      result.push_back({i, opaqueAttr});
    }
  }

  return result;
}

// Analyze varying attributes across a group of functions
llvm::StringMap<VaryingAttribute> analyzeVaryingAttributes(
    ArrayRef<func::FuncOp> funcs,
    StringRef opName) {

  llvm::StringMap<VaryingAttribute> varyingAttrs;

  if (funcs.empty()) {
    return varyingAttrs;
  }

  // Extract inline opaques from all functions
  SmallVector<SmallVector<std::pair<size_t, mlir::tt::emitpy::OpaqueAttr>>> allInlineOpaques;
  for (func::FuncOp f : funcs) {
    CallOpaqueOp callOp = findCallOpaqueOp(f);
    if (callOp) {
      allInlineOpaques.push_back(extractInlineOpaques(callOp));
    }
  }

  if (allInlineOpaques.empty()) {
    return varyingAttrs;
  }

  // Analyze each argument position
  size_t maxArgs = 0;
  for (const auto &opaques : allInlineOpaques) {
    if (!opaques.empty()) {
      maxArgs = std::max(maxArgs, opaques.back().first + 1);
    }
  }

  for (size_t argPos = 0; argPos < maxArgs; ++argPos) {
    // Collect values at this position across all functions
    SmallVector<std::string> values;
    bool allHaveValue = true;

    for (const auto &opaques : allInlineOpaques) {
      bool foundAtPos = false;
      for (const auto &[pos, opaque] : opaques) {
        if (pos == argPos) {
          values.push_back(formatOpaqueAttrValue(opaque));
          foundAtPos = true;
          break;
        }
      }
      if (!foundAtPos) {
        allHaveValue = false;
        break;
      }
    }

    if (!allHaveValue || values.empty()) {
      continue;
    }

    // Check if values vary
    bool varies = false;
    for (size_t i = 1; i < values.size(); ++i) {
      if (values[i] != values[0]) {
        varies = true;
        break;
      }
    }

    if (varies) {
      // Determine parameter name - first try to get it from keyword_args
      std::string paramName;

      // Try to extract keyword name from the first function's call_opaque
      CallOpaqueOp firstCallOp = findCallOpaqueOp(funcs[0]);
      if (firstCallOp) {
        if (auto kwArgs = firstCallOp.getKeywordArgs()) {
          if (argPos < kwArgs->size()) {
            if (auto strAttr = dyn_cast<StringAttr>((*kwArgs)[argPos])) {
              StringRef kwName = strAttr.getValue();
              if (!kwName.empty()) {
                paramName = kwName.str();
              }
            }
          }
        }
      }

      // Fallback to heuristic naming if no keyword arg available
      if (paramName.empty()) {
        std::string shortOpName = extractOpShortName(opName);
        if (shortOpName == "reshape" && argPos == 1) {
          paramName = "shape";
        } else if (shortOpName == "permute" && argPos == 1) {
          paramName = "permutation";
        } else if (shortOpName.find("conv") != std::string::npos && argPos == 8) {
          paramName = "kernel_size";
        } else {
          paramName = "arg_" + std::to_string(argPos);
        }
      }

      VaryingAttribute attr;
      attr.name = paramName;
      attr.argPosition = argPos;
      attr.values = std::move(values);
      attr.isVarying = true;

      varyingAttrs[paramName] = std::move(attr);
    }
  }

  return varyingAttrs;
}

struct TTNNPrepareIsolatedForPytestPass
    : public ::mlir::tt::ttnn::impl::TTNNPrepareIsolatedForPytestBase<
          TTNNPrepareIsolatedForPytestPass> {
  using ::mlir::tt::ttnn::impl::TTNNPrepareIsolatedForPytestBase<
      TTNNPrepareIsolatedForPytestPass>::TTNNPrepareIsolatedForPytestBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Group isolated functions by op_name
    llvm::StringMap<SmallVector<func::FuncOp>> opGroups;

    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp->hasAttr("isolated")) {
        return;
      }

      auto opNameAttr = funcOp->getAttrOfType<StringAttr>("op_name");
      if (!opNameAttr) {
        return;
      }

      opGroups[opNameAttr.getValue()].push_back(funcOp);
    });

    // Process each operation group
    for (auto &[opName, funcs] : opGroups) {
      if (funcs.empty()) {
        continue;
      }

      // Use the first function as template
      func::FuncOp templateFunc = funcs[0];

      // Create test function name: test_reshape, test_add, etc.
      std::string testFuncName = "test_" + extractOpShortName(opName);

      // Collect varying input_shapes and input_dtypes from function attributes
      SmallVector<ArrayAttr> allInputShapes;
      SmallVector<ArrayAttr> allInputDtypes;

      for (func::FuncOp f : funcs) {
        if (auto shapes = f->getAttrOfType<ArrayAttr>("input_shapes")) {
          allInputShapes.push_back(shapes);
        }
        if (auto dtypes = f->getAttrOfType<ArrayAttr>("input_dtypes")) {
          allInputDtypes.push_back(dtypes);
        }
      }

      // Analyze varying inline attributes from call_opaque operations
      auto varyingAttrs = analyzeVaryingAttributes(funcs, opName);

      // Build new function type with additional parameters
      SmallVector<Type> newArgTypes(templateFunc.getFunctionType().getInputs());
      // Use a generic opaque type for pytest parameters (can be any Python type)
      auto opaqueType = mlir::tt::emitpy::OpaqueType::get(builder.getContext(), "typing.Any");

      // Track which parameters we're adding
      struct NewParameter {
        std::string name;
        size_t newArgIndex;
        size_t oldArgPosition; // For inline opaques, position in args array
        bool isInlineOpaque;
      };
      SmallVector<NewParameter> newParams;

      // Add parameters for varying inline attributes
      for (auto &[paramName, attr] : varyingAttrs) {
        NewParameter param;
        param.name = paramName;
        param.newArgIndex = newArgTypes.size();
        param.oldArgPosition = attr.argPosition;
        param.isInlineOpaque = true;
        newParams.push_back(param);

        newArgTypes.push_back(opaqueType);
      }

      // Add input_shape parameter if it varies
      bool hasInputShapeParam = false;
      if (!allInputShapes.empty() && allInputShapes.size() > 1) {
        bool shapesVary = false;
        for (size_t i = 1; i < allInputShapes.size(); ++i) {
          if (allInputShapes[i] != allInputShapes[0]) {
            shapesVary = true;
            break;
          }
        }
        if (shapesVary) {
          NewParameter param;
          param.name = "input_shape";
          param.newArgIndex = newArgTypes.size();
          param.isInlineOpaque = false;
          newParams.push_back(param);

          newArgTypes.push_back(opaqueType);
          hasInputShapeParam = true;
        }
      }

      // Add dtype parameter
      bool hasDtypeParam = !allInputDtypes.empty();
      if (hasDtypeParam) {
        NewParameter param;
        param.name = "dtype";
        param.newArgIndex = newArgTypes.size();
        param.isInlineOpaque = false;
        newParams.push_back(param);

        newArgTypes.push_back(opaqueType);
      }

      auto newFuncType = builder.getFunctionType(
          newArgTypes,
          templateFunc.getFunctionType().getResults());

      // Create the merged test function
      builder.setInsertionPoint(templateFunc);

      auto testFunc = builder.create<func::FuncOp>(
          templateFunc.getLoc(),
          testFuncName,
          newFuncType);

      testFunc.setPrivate();

      // Set argument names for the new parameters
      for (const auto &param : newParams) {
        testFunc.setArgAttr(param.newArgIndex, "emitpy.name",
                           builder.getStringAttr(param.name));
      }

      // Clone the body from template
      // First, create the entry block with all arguments
      Block *entryBlock = testFunc.addEntryBlock();

      IRMapping mapping;
      // Map original arguments to corresponding new function arguments
      for (auto [oldArg, newArg] : llvm::zip(templateFunc.getArguments(),
                                              testFunc.getArguments())) {
        mapping.map(oldArg, newArg);
      }

      // Clone operations from template body to new function body
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);
      for (Operation &op : templateFunc.getBody().front().getOperations()) {
        bodyBuilder.clone(op, mapping);
      }

      // Update call_opaque to use new function arguments instead of inline opaques
      testFunc.walk([&](CallOpaqueOp callOp) {
        auto argsOpt = callOp.getArgs();
        if (!argsOpt) {
          return;
        }

        // Build new args array with argument indices instead of inline opaques
        SmallVector<Attribute> newArgs;
        ArrayAttr oldArgs = *argsOpt;

        // Also track which new operands we need to add
        SmallVector<Value> newOperands(callOp.getOperands().begin(),
                                       callOp.getOperands().end());

        for (size_t i = 0; i < oldArgs.size(); ++i) {
          bool replaced = false;

          // Check if this position should be replaced with a function argument
          for (const auto &param : newParams) {
            if (param.isInlineOpaque && param.oldArgPosition == i) {
              // Replace with argument index pointing to the new operand position
              size_t newOperandIndex = newOperands.size();
              newArgs.push_back(builder.getIndexAttr(newOperandIndex));

              // Add the function argument as an operand
              newOperands.push_back(testFunc.getArgument(param.newArgIndex));
              replaced = true;
              break;
            }
          }

          if (!replaced) {
            // Keep original attribute
            newArgs.push_back(oldArgs[i]);
          }
        }

        // Update the call_opaque with new args and operands
        callOp.setArgsAttr(builder.getArrayAttr(newArgs));
        callOp.getOperandsMutable().assign(newOperands);
      });

      // Generate decorators
      SmallVector<Attribute> decorators;

      // Add decorators for varying inline attributes
      for (auto &[paramName, attr] : varyingAttrs) {
        std::string paramStr;
        paramStr += "@pytest.mark.parametrize('";
        paramStr += paramName;
        paramStr += "', [";
        for (size_t i = 0; i < attr.values.size(); ++i) {
          paramStr += attr.values[i];
          if (i < attr.values.size() - 1) {
            paramStr += ", ";
          }
        }
        paramStr += "])";
        decorators.push_back(mlir::tt::emitpy::DecoratorAttr::get(builder.getContext(), paramStr));
      }

      // Add decorator for input_shape if it varies
      if (hasInputShapeParam) {
        std::string paramStr = "@pytest.mark.parametrize('input_shape', [";
        for (size_t i = 0; i < allInputShapes.size(); ++i) {
          paramStr += formatArrayAsPythonList(allInputShapes[i]);
          if (i < allInputShapes.size() - 1) paramStr += ", ";
        }
        paramStr += "])";
        decorators.push_back(DecoratorAttr::get(builder.getContext(), paramStr));
      }

      // Add decorator for dtype
      if (hasDtypeParam) {
        std::string paramStr = "@pytest.mark.parametrize('dtype', [";
        for (size_t i = 0; i < allInputDtypes.size(); ++i) {
          paramStr += formatArrayAsPythonList(allInputDtypes[i]);
          if (i < allInputDtypes.size() - 1) paramStr += ", ";
        }
        paramStr += "])";
        decorators.push_back(DecoratorAttr::get(builder.getContext(), paramStr));
      }

      // Set decorators attribute on the function
      if (!decorators.empty()) {
        testFunc->setAttr("emitpy.decorators", builder.getArrayAttr(decorators));
      }

      // Erase all original isolated functions
      for (func::FuncOp f : funcs) {
        f.erase();
      }
    }
  }
};

} // namespace
