// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Target/Common/types_generated.h"
#include "ttmlir/Target/LLVM/LLVMToDynamicLib.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/binary_generated.h"
#include "ttmlir/Target/TTNN/operations/creation_generated.h"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttmlir/Target/TTNN/utils.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/FuncOpToProgram.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Target/Utils/Utils.h"
#include "ttmlir/Version.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include "llvm/Support/MemoryBuffer.h"
#include <regex>
#include <unordered_set>

namespace mlir::tt::ttnn {

// Helper function to parse shape from string like "Shape([1, 784])"
SmallVector<int64_t> parseShape(StringRef shapeStr) {
  SmallVector<int64_t> shape;

  // Extract the content inside brackets
  std::string str = shapeStr.str();
  std::regex shapeRegex("Shape\\(\\[(.*?)\\]\\)");
  std::smatch match;

  if (std::regex_search(str, match, shapeRegex) && match.size() > 1) {
    std::string dims = match[1].str();

    // Split by comma and convert to integers
    size_t pos = 0;
    std::string token;
    while ((pos = dims.find(',')) != std::string::npos) {
      token = dims.substr(0, pos);
      shape.push_back(std::stoi(token));
      dims.erase(0, pos + 1);
    }

    // Add the last dimension
    if (!dims.empty()) {
      shape.push_back(std::stoi(dims));
    }
  }

  return shape;
}

// Helper function to parse data type from string like "DataType::FLOAT32"
DataTypeAttr parseDataType(StringRef dataTypeStr, MLIRContext *context) {
  if (dataTypeStr.contains("FLOAT32")) {
    return DataTypeAttr::get(context, DataType::Float32);
  }
  if (dataTypeStr.contains("FLOAT16")) {
    return DataTypeAttr::get(context, DataType::Float16);
  }
  if (dataTypeStr.contains("BFLOAT16")) {
    return DataTypeAttr::get(context, DataType::BFloat16);
  }

  // Default to float32
  return DataTypeAttr::get(context, DataType::Float32);
}

// Helper function to create a shape attribute from a shape vector
ShapeAttr createShapeAttr(ArrayRef<int64_t> shape, Builder &builder) {
  // auto shapeType = builder.getIntegerType(32);
  // SmallVector<Attribute> shapeAttrs;

  // for (auto dim : shape) {
  //   shapeAttrs.push_back(builder.getIntegerAttr(shapeType, dim));
  // }

  // // return builder.getArrayAttr(shapeAttrs);
  return ShapeAttr::get(builder.getContext(), shape);
}

OwningOpRef<ModuleOp> translateTracedTTNNGraphToMLIR(llvm::SourceMgr &sourceMgr,
                                                     MLIRContext *context) {

  const llvm::MemoryBuffer *memBuffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // std::string simplifiedGraph = memBuffer->getBuffer().str();

  auto jsonOrErr = llvm::json::parse(memBuffer->getBuffer());
  if (!jsonOrErr) {
    emitError(UnknownLoc::get(context))
        << "Failed to parse JSON: " << toString(jsonOrErr.takeError());
    return nullptr;
  }

  auto json = jsonOrErr.get();

  OpBuilder builder(context);

  // Create top-level module op
  //
  ModuleOp module = builder.create<ModuleOp>(UnknownLoc::get(context));

  FunctionType funcType = builder.getFunctionType({}, {});
  // FunctionType::get(builder.getContext(), {}, {});
  auto func =
      builder.create<func::FuncOp>(module.getLoc(), "graph_main", funcType);
  ::mlir::Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Map to store node index to Value mapping
  std::unordered_map<int, Value> nodeValues;

  // First pass: create all operations
  if (auto *nodesArray = json.getAsArray()) {
    for (const auto &node : *nodesArray) {
      if (auto *nodeObj = node.getAsObject()) {
        // Get node properties
        int index = -1;
        std::string name;
        std::vector<int> children;

        if (auto indexVal = nodeObj->getInteger("index")) {
          index = *indexVal;
        }

        if (auto nameVal = nodeObj->getString("name")) {
          name = nameVal->str();
        }

        if (const auto *childrenArray = nodeObj->getArray("children")) {
          for (const auto &child : *childrenArray) {
            if (auto childIndex = child.getAsInteger()) {
              children.push_back(*childIndex);
            }
          }
        }

        // Process arguments
        SmallVector<int64_t> shape;
        DataTypeAttr dataTypeAttr =
            DataTypeAttr::get(context, DataType::Float32);

        if (const auto *argsObj = nodeObj->getObject("arguments")) {
          if (auto shapeVal = argsObj->getString("shape")) {
            shape = parseShape(*shapeVal);
          }

          if (auto dataTypeVal = argsObj->getString("dtype")) {
            dataTypeAttr = parseDataType(*dataTypeVal, context);
          }
        }

        // Create the operation based on the node name
        Location loc = builder.getUnknownLoc();
        Value result;

        if (name == "ttnn::ones") {
          // Create ones operation
          auto shapeAttr = createShapeAttr(shape, builder);
          auto onesOp = builder.create<ttnn::OnesOp>(
              // static void build(::mlir::OpBuilder &odsBuilder,
              // ::mlir::OperationState &odsState, ::mlir::Type result,
              // ::mlir::tt::ttnn::ShapeAttr shape,
              // /*optional*/::mlir::tt::DataTypeAttr dtype,
              // /*optional*/::mlir::tt::ttnn::LayoutAttr layout,
              // /*optional*/::mlir::Value device,
              // /*optional*/::mlir::tt::ttnn::MemoryConfigAttr memory_config);
              loc, RankedTensorType::get(shape, builder.getF32Type()),
              shapeAttr, dataTypeAttr, nullptr, nullptr, nullptr);
          onesOp->setAttr("shape", shapeAttr);
          result = onesOp.getResult();
        } else if (name == "ttnn::matmul") {
          // Get operands from arguments
          Value lhs, rhs;
          if (const auto *argsObj = nodeObj->getObject("arguments")) {
            if (auto lhsVal = argsObj->getInteger("lhs")) {
              auto it = nodeValues.find(*lhsVal);
              if (it != nodeValues.end()) {
                lhs = it->second;
              }
            }

            if (auto rhsVal = argsObj->getInteger("rhs")) {
              auto it = nodeValues.find(*rhsVal);
              if (it != nodeValues.end()) {
                rhs = it->second;
              }
            }
          }

          if (!lhs || !rhs) {
            module.emitError("Missing operands for matmul operation at node ")
                << index;
            return nullptr;
          }

          auto matmulOp = builder.create<ttnn::MatmulOp>(
              // static void build(::mlir::OpBuilder &odsBuilder,
              // ::mlir::OperationState &odsState, ::mlir::Type result,
              // ::mlir::Value a, ::mlir::Value b, ::mlir::BoolAttr transpose_a,
              // ::mlir::BoolAttr transpose_b, /*optional*/::mlir::Attribute
              // matmul_program_config);
              loc, RankedTensorType::get(shape, builder.getF32Type()), lhs, rhs,
              false, false, nullptr);
          // loc, RankedTensorType::get(shape, builder.getF32Type()), lhs, rhs,
          // false, false);
          result = matmulOp.getResult();
        } else if (name == "ttnn::add") {
          // Get operands from arguments
          Value lhs, rhs;
          if (const auto *argsObj = nodeObj->getObject("arguments")) {
            if (auto lhsVal = argsObj->getInteger("lhs")) {
              auto it = nodeValues.find(*lhsVal);
              if (it != nodeValues.end()) {
                lhs = it->second;
              }
            }

            if (auto rhsVal = argsObj->getInteger("rhs")) {
              auto it = nodeValues.find(*rhsVal);
              if (it != nodeValues.end()) {
                rhs = it->second;
              }
            }
          }

          if (!lhs || !rhs) {
            module.emitError("Missing operands for add operation at node ")
                << index;
            return nullptr;
          }

          auto addOp = builder.create<ttnn::AddOp>(
              loc, RankedTensorType::get(shape, builder.getF32Type()), lhs,
              rhs);
          result = addOp.getResult();
        } else if (name == "ttnn::relu") {
          // Get operand from arguments
          Value input;
          if (const auto *argsObj = nodeObj->getObject("arguments")) {
            if (auto inputVal = argsObj->getInteger("input")) {
              auto it = nodeValues.find(*inputVal);
              if (it != nodeValues.end()) {
                input = it->second;
              }
            }
          }

          if (!input) {
            module.emitError("Missing operand for relu operation at node ")
                << index;
            return nullptr;
          }

          auto reluOp = builder.create<ttnn::ReluOp>(
              loc, RankedTensorType::get(shape, builder.getF32Type()), input);
          result = reluOp.getResult();
        } else if (name == "ttnn::softmax") {
          // Get operand from arguments
          Value input;
          int64_t dim = 1; // Default dimension

          if (const auto *argsObj = nodeObj->getObject("arguments")) {
            if (auto inputVal = argsObj->getInteger("input")) {
              auto it = nodeValues.find(*inputVal);
              if (it != nodeValues.end()) {
                input = it->second;
              }
            }

            if (auto dimVal = argsObj->getInteger("dim")) {
              dim = *dimVal;
            }
          }

          if (!input) {
            module.emitError("Missing operand for softmax operation at node ")
                << index;
            return nullptr;
          }

          auto dimAttr = builder.getI64IntegerAttr(dim);
          auto softmaxOp = builder.create<ttnn::SoftmaxOp>(
              loc, RankedTensorType::get(shape, builder.getF32Type()), input);
          softmaxOp->setAttr("dim", dimAttr);
          result = softmaxOp.getResult();
        } else {
          module.emitError("Unsupported operation: ") << name;
          return nullptr;
        }

        // Store the result value with its node index
        if (result) {
          nodeValues[index] = result;
        }
      }
    }
  }

  // Find the final output node (the one that's not a child of any other node)
  std::unordered_set<int> allChildren;
  std::unordered_set<int> allNodes;

  if (auto *nodesArray = json.getAsArray()) {
    for (const auto &node : *nodesArray) {
      if (auto *nodeObj = node.getAsObject()) {
        if (auto indexVal = nodeObj->getInteger("index")) {
          allNodes.insert(*indexVal);
        }

        if (const auto *childrenArray = nodeObj->getArray("children")) {
          for (const auto &child : *childrenArray) {
            if (auto childIndex = child.getAsInteger()) {
              allChildren.insert(*childIndex);
            }
          }
        }
      }
    }
  }

  // Find nodes that are not children of any other node (outputs)
  std::vector<int> outputNodes;
  for (int node : allNodes) {
    if (allChildren.find(node) == allChildren.end()) {
      outputNodes.push_back(node);
    }
  }

  // Create return operation with output values
  SmallVector<Value> returnValues;
  for (int outputNode : outputNodes) {
    auto it = nodeValues.find(outputNode);
    if (it != nodeValues.end()) {
      returnValues.push_back(it->second);
    }
  }

  // Update function type with return values
  SmallVector<Type> returnTypes;
  for (Value val : returnValues) {
    returnTypes.push_back(val.getType());
  }

  auto newFuncType = builder.getFunctionType({}, returnTypes);
  func.setType(newFuncType);

  // Create return operation
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), returnValues);

  return module;
}

} // namespace mlir::tt::ttnn
