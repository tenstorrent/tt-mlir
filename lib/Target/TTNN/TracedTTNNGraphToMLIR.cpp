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
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
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

// Helper function to parse layout from string like "Layout::TILE"
LayoutAttr parseLayout(StringRef layoutStr, MLIRContext *context) {
  if (layoutStr.contains("TILE")) {
    return LayoutAttr::get(context, ttnn::Layout::Tile);
  }
  if (layoutStr.contains("ROW_MAJOR")) {
    return LayoutAttr::get(context, ttnn::Layout::RowMajor);
  }
  if (layoutStr.contains("INVALID")) {
    return LayoutAttr::get(context, ttnn::Layout::Invalid);
  }

  llvm_unreachable("Unknown layout");
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

static const std::array<std::pair<int64_t, int64_t>, 1> g_defaultCollapseDims =
    {{{0, -1}}};

static const llvm::SmallDenseMap<BufferType, std::optional<TensorMemoryLayout>,
                                 4>
    g_bufferLayoutMap = {
        {BufferType::DRAM, TensorMemoryLayout::Interleaved},
        {BufferType::L1, TensorMemoryLayout::Interleaved},
        {BufferType::SystemMemory, std::nullopt},
};

static TensorMemoryLayoutAttr getMemoryLayoutAttr(MLIRContext *ctx,
                                                  BufferType bufferType) {
  std::optional<TensorMemoryLayout> layout = g_bufferLayoutMap.at(bufferType);
  if (layout) {
    return TensorMemoryLayoutAttr::get(ctx, layout.value());
  }

  return TensorMemoryLayoutAttr{};
}

static TTNNLayoutAttr createLayoutAttr(MLIRContext *ctx, GridAttr deviceGrid,
                                       RankedTensorType type,
                                       BufferType bufferType = BufferType::DRAM,
                                       bool isTiled = true) {

  std::int64_t deviceGridRank = deviceGrid.getShape().size();
  // Default to single core grid
  GridAttr tensorGrid = GridAttr::get(ctx, deviceGridRank);

  llvm::ArrayRef<std::pair<int64_t, int64_t>> collapseDimsRef(
      g_defaultCollapseDims);

  // Force TileType for tensors
  Type elementType = type.getElementType();
  // The tile type for a quantized type is the desired type.
  // Ex: for a quant p of fp32->int8, the storage type is int8.
  if (auto quantType =
          mlir::dyn_cast<mlir::quant::QuantizedType>(elementType)) {
    elementType = isTiled ? TileType::get(quantType.getStorageType())
                          : quantType.getStorageType();
  } else {
    elementType =
        isTiled ? TileType::get(type.getElementType()) : type.getElementType();
  }
  mlir::Attribute encoding = type.getEncoding();
  TensorMeshShardingAttr tensorMeshShardingAttr;
  if (auto encodingMeshSharding =
          mlir::dyn_cast_if_present<TensorMeshShardingAttr>(encoding)) {
    tensorMeshShardingAttr = encodingMeshSharding;
  } else if (auto layout =
                 mlir::dyn_cast_if_present<TTNNLayoutAttr>(encoding)) {
    tensorMeshShardingAttr = layout.getTensorMeshSharding();
  }
  TensorMemoryLayoutAttr memoryLayoutAttr =
      getMemoryLayoutAttr(ctx, bufferType);
  return TTNNLayoutAttr::get(ctx, type.getShape(), elementType, bufferType,
                             tensorGrid, memoryLayoutAttr,
                             tensorMeshShardingAttr, collapseDimsRef);
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

  builder.setInsertionPoint(module.getBody(), module.getBody()->begin());

  FunctionType funcType = builder.getFunctionType({}, {});
  // FunctionType::get(builder.getContext(), {}, {});
  auto func =
      builder.create<func::FuncOp>(module.getLoc(), "forward", funcType);
  ::mlir::Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  std::unordered_map<int, Value> indexToResultMap;

  // First pass: create all operations
  if (auto *nodesArray = json.getAsArray()) {
    for (const auto &node : *nodesArray) {
      if (const llvm::json::Object *nodeObj = node.getAsObject()) {
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
          for (const llvm::json::Value &child : *childrenArray) {
            if (auto childIndex = child.getAsInteger()) {
              children.push_back(*childIndex);
            }
          }
        }

        // Create the operation based on the node name
        Location loc = builder.getUnknownLoc();
        Value result;
        const llvm::json::Array *argsObj = nodeObj->getArray("arguments");

        if (name == "ttnn::ones") {
          // Process arguments
          SmallVector<int64_t> shapeVec;
          ShapeAttr shapeAttr;
          DataTypeAttr dataTypeAttr =
              DataTypeAttr::get(context, DataType::Float32);

          const llvm::json::Value &shapeObj = (*argsObj)[0];
          const llvm::json::Value &dtypeObj = (*argsObj)[1];
          const llvm::json::Value &layoutType = (*argsObj)[2];
          // const llvm::json::Value &layoutObj = (*arrayObj)[2];

          shapeVec = parseShape(shapeObj.getAsString()->str());

          shapeAttr = createShapeAttr(shapeVec, builder);
          dataTypeAttr = parseDataType(dtypeObj.getAsString()->str(), context);
          LayoutAttr layoutTypeAttr =
              parseLayout(layoutType.getAsString()->str(), context);

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, GridAttr::get(context),
              RankedTensorType::get(shapeVec, builder.getF32Type()));

          // Create ones operation
          auto onesOp = builder.create<ttnn::OnesOp>(
              // static void build(::mlir::OpBuilder &odsBuilder,
              // ::mlir::OperationState &odsState, ::mlir::Type result,
              // ::mlir::tt::ttnn::ShapeAttr shape,
              // /*optional*/::mlir::tt::DataTypeAttr dtype,
              // /*optional*/::mlir::tt::ttnn::LayoutAttr layout,
              // /*optional*/::mlir::Value device,
              // /*optional*/::mlir::tt::ttnn::MemoryConfigAttr memory_config);
              loc,
              RankedTensorType::get(shapeVec, builder.getF32Type(), layoutAttr),
              shapeAttr, dataTypeAttr, layoutTypeAttr, nullptr, nullptr);
          onesOp->setAttr("shape", shapeAttr);
          result = onesOp.getResult();
        } else if (name == "ttnn::matmul") {
          std::string lhsStr = (*argsObj)[0].getAsString()->str();
          std::string rhsStr = (*argsObj)[1].getAsString()->str();
          // bool transposeA = (*argsObj)[2].getAsBoolean().value();
          // bool transposeB = (*argsObj)[3].getAsBoolean().value();
          bool transposeA = false;
          bool transposeB = false;

          // lhsStr and lhsStr are of format "tensor: <index>" - need to get
          // <index>
          int lhsIndex = std::stoi(lhsStr.substr(lhsStr.find(":") + 1));
          int rhsIndex = std::stoi(rhsStr.substr(rhsStr.find(":") + 1));

          ::llvm::ArrayRef<int64_t> lhsShape =
              mlir::cast<RankedTensorType>(indexToResultMap[lhsIndex].getType())
                  .getShape();
          ::llvm::ArrayRef<int64_t> rhsShape =
              mlir::cast<RankedTensorType>(indexToResultMap[rhsIndex].getType())
                  .getShape();

          // Hack: trace doesn't provide info on which tensor is lhs and which
          // is rhs, so analyze shapes to determine
          if (lhsShape[lhsShape.size() - 1] != rhsShape[rhsShape.size() - 2]) {
            std::swap(lhsIndex, rhsIndex);
          }

          Value lhs = indexToResultMap[lhsIndex];
          Value rhs = indexToResultMap[rhsIndex];

          llvm::SmallVector<int64_t, 2> shape{
              mlir::cast<RankedTensorType>(lhs.getType()).getShape().front(),
              mlir::cast<RankedTensorType>(rhs.getType()).getShape().back()};

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, GridAttr::get(context),
              RankedTensorType::get(shape, builder.getF32Type()));

          auto matmulOp = builder.create<ttnn::MatmulOp>(
              // static void build(::mlir::OpBuilder &odsBuilder,
              // ::mlir::OperationState &odsState, ::mlir::Type result,
              // ::mlir::Value a, ::mlir::Value b, ::mlir::BoolAttr transpose_a,
              // ::mlir::BoolAttr transpose_b, /*optional*/::mlir::Attribute
              // matmul_program_config);
              loc,
              RankedTensorType::get(shape, builder.getF32Type(), layoutAttr),
              lhs, rhs, transposeA, transposeB, nullptr);
          result = matmulOp.getResult();
        } else if (name == "ttnn::add") {
          // Get operands from arguments
          // note: QueueId is first arg, skip it
          std::string lhsStr = (*argsObj)[1].getAsString()->str();
          std::string rhsStr = (*argsObj)[2].getAsString()->str();

          // lhsStr and lhsStr are of format "tensor: <index>" - need to get
          // <index>
          int lhsIndex = std::stoi(lhsStr.substr(lhsStr.find(":") + 1));
          int rhsIndex = std::stoi(rhsStr.substr(rhsStr.find(":") + 1));

          Value lhs = indexToResultMap[lhsIndex];
          Value rhs = indexToResultMap[rhsIndex];

          // std::cout << "  add lhs: " << std::endl;
          // lhs.dump();
          // std::cout << "  add rhs: " << std::endl;
          // rhs.dump();

          // Get shape from the first operand
          ::llvm::ArrayRef<int64_t> shape =
              mlir::cast<RankedTensorType>(rhs.getType()).getShape();

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, GridAttr::get(context),
              RankedTensorType::get(shape, builder.getF32Type()));

          auto addOp = builder.create<ttnn::AddOp>(
              loc,
              RankedTensorType::get(shape, builder.getF32Type(), layoutAttr),
              lhs, rhs);
          result = addOp.getResult();
        } else if (name == "ttnn::relu") {
          // Get operand from arguments
          // note: QueueId is first arg, skip it
          std::string inputStr = (*argsObj)[1].getAsString()->str();

          // inputStr is of format "tensor: <index>" - need to get <index>
          int inputIndex = std::stoi(inputStr.substr(inputStr.find(":") + 1));

          Value input = indexToResultMap[inputIndex];

          // Get shape from the input operand
          ::llvm::ArrayRef<int64_t> shape =
              mlir::cast<RankedTensorType>(input.getType()).getShape();

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, GridAttr::get(context),
              RankedTensorType::get(shape, builder.getF32Type()));

          auto reluOp = builder.create<ttnn::ReluOp>(
              loc,
              RankedTensorType::get(shape, builder.getF32Type(), layoutAttr),
              input);
          result = reluOp.getResult();
        } else if (name == "ttnn::softmax") {
          // Get operand from arguments
          std::string inputStr = (*argsObj)[0].getAsString()->str();

          int inputIndex = std::stoi(inputStr.substr(inputStr.find(":") + 1));

          Value input = indexToResultMap[inputIndex];
          // Get shape from the input operand
          ::llvm::ArrayRef<int64_t> shape =
              mlir::cast<RankedTensorType>(input.getType()).getShape();

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, GridAttr::get(context),
              RankedTensorType::get(shape, builder.getF32Type()));

          // int dim = (*argsObj)[1].getAsNumber(); // not proprely traced, hack
          // for now
          int dim = -1;
          auto dimAttr = builder.getSI32IntegerAttr(dim);

          auto softmaxOp = builder.create<ttnn::SoftmaxOp>(
              loc,
              RankedTensorType::get(shape, builder.getF32Type(), layoutAttr),
              input, dimAttr);
          result = softmaxOp.getResult();
        } else {
          module.emitError("Unsupported operation: ") << name;
          return nullptr;
        }

        // Store the result value with its node index
        if (result) {
          // result.dump();
          indexToResultMap[index] = result;
        }
      }
    }
  }

  // Find the last operation (highest index)
  int lastOpIndex = -1;
  for (const auto &pair : indexToResultMap) {
    if (pair.first > lastOpIndex) {
      lastOpIndex = pair.first;
    }
  }

  // Create return operation with only the last operation's value
  SmallVector<Value> returnValues;
  if (lastOpIndex != -1) {
    auto it = indexToResultMap.find(lastOpIndex);
    if (it != indexToResultMap.end()) {
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
