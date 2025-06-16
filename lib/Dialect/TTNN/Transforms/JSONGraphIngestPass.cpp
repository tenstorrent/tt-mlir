// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/JSONGraphIngestPass.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_JSONNETWORKINGEST
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

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
Type parseDataType(StringRef dataTypeStr, MLIRContext *context) {
  if (dataTypeStr.contains("FLOAT32"))
    return FloatType::getF32(context);
  else if (dataTypeStr.contains("FLOAT16") || dataTypeStr.contains("BFLOAT16"))
    return FloatType::getBF16(context);
  else if (dataTypeStr.contains("INT32"))
    return IntegerType::get(context, 32);
  else if (dataTypeStr.contains("INT16"))
    return IntegerType::get(context, 16);
  else if (dataTypeStr.contains("INT8"))
    return IntegerType::get(context, 8);
  else
    return FloatType::getF32(context); // Default to F32
}

// Helper function to create a shape attribute from a shape vector
Attribute createShapeAttr(ArrayRef<int64_t> shape, Builder &builder) {
  auto shapeType = builder.getIntegerType(32);
  SmallVector<Attribute> shapeAttrs;

  for (auto dim : shape) {
    shapeAttrs.push_back(builder.getIntegerAttr(shapeType, dim));
  }

  return builder.getArrayAttr(shapeAttrs);
}

struct JSONNetworkIngestPass
    : public impl::JSONNetworkIngestBase<JSONNetworkIngestPass> {

  JSONNetworkIngestPass() = default;
  JSONNetworkIngestPass(const JSONNetworkIngestPass &) = default;
  JSONNetworkIngestPass(StringRef jsonFilePath) {
    this->jsonFilePath = jsonFilePath.str();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    // Read JSON file
    auto fileOrErr = llvm::MemoryBuffer::getFile(jsonFilePath);
    if (std::error_code ec = fileOrErr.getError()) {
      module.emitError("Failed to open JSON file: ") << ec.message();
      return signalPassFailure();
    }

    auto jsonOrErr = llvm::json::parse(fileOrErr.get()->getBuffer());
    if (!jsonOrErr) {
      module.emitError("Failed to parse JSON: ")
          << toString(jsonOrErr.takeError());
      return signalPassFailure();
    }

    auto json = jsonOrErr.get();

    // Create a function to hold our graph
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = builder.getFunctionType({}, {});
    auto func =
        builder.create<func::FuncOp>(module.getLoc(), "graph_main", funcType);
    auto entryBlock = func.addEntryBlock();
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

          if (auto indexVal = nodeObj->getInteger("index"))
            index = *indexVal;

          if (auto nameVal = nodeObj->getString("name"))
            name = nameVal->str();

          if (auto childrenArray = nodeObj->getArray("children")) {
            for (const auto &child : *childrenArray) {
              if (auto childIndex = child.getAsInteger())
                children.push_back(*childIndex);
            }
          }

          // Process arguments
          SmallVector<int64_t> shape;
          Type dataType = FloatType::getF32(context);

          if (auto argsObj = nodeObj->getObject("arguments")) {
            if (auto shapeVal = argsObj->getString("shape"))
              shape = parseShape(*shapeVal);

            if (auto dataTypeVal = argsObj->getString("dtype"))
              dataType = parseDataType(*dataTypeVal, context);
          }

          // Create the operation based on the node name
          Location loc = builder.getUnknownLoc();
          Value result;

          if (name == "ttnn::ones") {
            // Create ones operation
            auto shapeAttr = createShapeAttr(shape, builder);
            auto onesOp = builder.create<TTNN::OnesOp>(
                loc, RankedTensorType::get(shape, dataType));
            onesOp->setAttr("shape", shapeAttr);
            result = onesOp.getResult();
          } else if (name == "ttnn::matmul") {
            // Get operands from arguments
            Value lhs, rhs;
            if (auto argsObj = nodeObj->getObject("arguments")) {
              if (auto lhsVal = argsObj->getInteger("lhs")) {
                auto it = nodeValues.find(*lhsVal);
                if (it != nodeValues.end())
                  lhs = it->second;
              }

              if (auto rhsVal = argsObj->getInteger("rhs")) {
                auto it = nodeValues.find(*rhsVal);
                if (it != nodeValues.end())
                  rhs = it->second;
              }
            }

            if (!lhs || !rhs) {
              module.emitError("Missing operands for matmul operation at node ")
                  << index;
              return signalPassFailure();
            }

            auto matmulOp = builder.create<TTNN::MatmulOp>(
                loc, RankedTensorType::get(shape, dataType), lhs, rhs);
            result = matmulOp.getResult();
          } else if (name == "ttnn::add") {
            // Get operands from arguments
            Value lhs, rhs;
            if (auto argsObj = nodeObj->getObject("arguments")) {
              if (auto lhsVal = argsObj->getInteger("lhs")) {
                auto it = nodeValues.find(*lhsVal);
                if (it != nodeValues.end())
                  lhs = it->second;
              }

              if (auto rhsVal = argsObj->getInteger("rhs")) {
                auto it = nodeValues.find(*rhsVal);
                if (it != nodeValues.end())
                  rhs = it->second;
              }
            }

            if (!lhs || !rhs) {
              module.emitError("Missing operands for add operation at node ")
                  << index;
              return signalPassFailure();
            }

            auto addOp = builder.create<TTNN::AddOp>(
                loc, RankedTensorType::get(shape, dataType), lhs, rhs);
            result = addOp.getResult();
          } else if (name == "ttnn::relu") {
            // Get operand from arguments
            Value input;
            if (auto argsObj = nodeObj->getObject("arguments")) {
              if (auto inputVal = argsObj->getInteger("input")) {
                auto it = nodeValues.find(*inputVal);
                if (it != nodeValues.end())
                  input = it->second;
              }
            }

            if (!input) {
              module.emitError("Missing operand for relu operation at node ")
                  << index;
              return signalPassFailure();
            }

            auto reluOp = builder.create<TTNN::ReluOp>(
                loc, RankedTensorType::get(shape, dataType), input);
            result = reluOp.getResult();
          } else if (name == "ttnn::softmax") {
            // Get operand from arguments
            Value input;
            int64_t dim = 1; // Default dimension

            if (auto argsObj = nodeObj->getObject("arguments")) {
              if (auto inputVal = argsObj->getInteger("input")) {
                auto it = nodeValues.find(*inputVal);
                if (it != nodeValues.end())
                  input = it->second;
              }

              if (auto dimVal = argsObj->getInteger("dim"))
                dim = *dimVal;
            }

            if (!input) {
              module.emitError("Missing operand for softmax operation at node ")
                  << index;
              return signalPassFailure();
            }

            auto dimAttr = builder.getI64IntegerAttr(dim);
            auto softmaxOp = builder.create<TTNN::SoftmaxOp>(
                loc, RankedTensorType::get(shape, dataType), input);
            softmaxOp->setAttr("dim", dimAttr);
            result = softmaxOp.getResult();
          } else {
            module.emitError("Unsupported operation: ") << name;
            return signalPassFailure();
          }

          // Store the result value with its node index
          if (result)
            nodeValues[index] = result;
        }
      }
    }

    // Find the final output node (the one that's not a child of any other node)
    std::unordered_set<int> allChildren;
    std::unordered_set<int> allNodes;

    if (auto *nodesArray = json.getAsArray()) {
      for (const auto &node : *nodesArray) {
        if (auto *nodeObj = node.getAsObject()) {
          if (auto indexVal = nodeObj->getInteger("index"))
            allNodes.insert(*indexVal);

          if (auto childrenArray = nodeObj->getArray("children")) {
            for (const auto &child : *childrenArray) {
              if (auto childIndex = child.getAsInteger())
                allChildren.insert(*childIndex);
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
  }
};

} // namespace

std::unique_ptr<Pass> createJSONNetworkIngestPass(StringRef jsonFilePath) {
  return std::make_unique<JSONNetworkIngestPass>(jsonFilePath);
}

} // namespace mlir::tt::ttnn
