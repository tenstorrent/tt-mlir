// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
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
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
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
#include <optional>
#include <regex>
#include <string>
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
ttcore::DataTypeAttr parseDataType(StringRef dataTypeStr,
                                   MLIRContext *context) {
  if (dataTypeStr == "DataType::FLOAT32") {
    return ttcore::DataTypeAttr::get(context, ttcore::DataType::Float32);
  }
  if (dataTypeStr == "DataType::BFLOAT16") {
    return ttcore::DataTypeAttr::get(context, ttcore::DataType::BFloat16);
  }
  if (dataTypeStr == "DataType::FLOAT16") {
    return ttcore::DataTypeAttr::get(context, ttcore::DataType::Float16);
  }

  // Default to float32
  return ttcore::DataTypeAttr::get(context, ttcore::DataType::Float32);
}

// Helper function to parse layout from string like "Layout::TILE"
std::optional<LayoutAttr> parseLayout(StringRef layoutStr,
                                      MLIRContext *context) {
  if (layoutStr.contains("TILE")) {
    return LayoutAttr::get(context, ttnn::Layout::Tile);
  }
  if (layoutStr.contains("ROW_MAJOR")) {
    return LayoutAttr::get(context, ttnn::Layout::RowMajor);
  }
  if (layoutStr.contains("INVALID")) {
    return LayoutAttr::get(context, ttnn::Layout::Invalid);
  }
  if (layoutStr.contains("nullopt")) {
    return std::nullopt;
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

static TTNNLayoutAttr createLayoutAttr(MLIRContext *ctx,
                                       ttcore::GridAttr deviceGrid,
                                       RankedTensorType type,
                                       BufferType bufferType = BufferType::DRAM,
                                       bool isTiled = true) {

  std::int64_t deviceGridRank = deviceGrid.getShape().size();
  // Default to single core grid
  ttcore::GridAttr tensorGrid = ttcore::GridAttr::get(ctx, deviceGridRank);

  llvm::ArrayRef<std::pair<int64_t, int64_t>> collapseDimsRef(
      g_defaultCollapseDims);

  // Force TileType for tensors
  Type elementType = type.getElementType();
  // The tile type for a quantized type is the desired type.
  // Ex: for a quant p of fp32->int8, the storage type is int8.
  if (auto quantType =
          mlir::dyn_cast<mlir::quant::QuantizedType>(elementType)) {
    elementType = isTiled ? ttcore::TileType::get(quantType.getStorageType())
                          : quantType.getStorageType();
  } else {
    elementType = isTiled ? ttcore::TileType::get(type.getElementType())
                          : type.getElementType();
  }
  mlir::Attribute encoding = type.getEncoding();
  ttcore::TensorMeshShardingAttr tensorMeshShardingAttr;
  if (auto encodingMeshSharding =
          mlir::dyn_cast_if_present<ttcore::TensorMeshShardingAttr>(encoding)) {
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

llvm::SmallVector<int32_t> parseArray(StringRef arrayStr) {
  std::string str = arrayStr.str();

  // Parse each value from string
  llvm::SmallVector<int32_t, 2> result;
  size_t pos = 0;
  std::string token;
  while ((pos = str.find(',')) != std::string::npos) {
    token = str.substr(0, pos);
    result.push_back(std::stoi(token));
    str.erase(0, pos + 1);
  }
  if (!str.empty()) {
    result.push_back(std::stoi(str));
  }
  return result;
}

inline std::pair<int, int>
calculateOutputDims(int inputY, int inputX,
                    std::pair<int, int> kernelSize,    // (kh, kw)
                    std::pair<int, int> stride,        // (sh, sw)
                    const std::array<int, 4> &padding, // [l, r, t, b]
                    int dilation = 1, bool ceilMode = false) {
  for (int p : padding) {
    assert(p >= 0);
  }

  auto computeDim = [&](int in, int padBefore, int padAfter, int k,
                        int s) -> int {
    const int effectiveFilter = dilation * (k - 1) + 1;
    const int numerator = in + padBefore + padAfter - effectiveFilter;
    if (ceilMode) {
      return static_cast<int>(
                 std::ceil(static_cast<double>(numerator + 1) / s)) +
             1;
    }
    return numerator / s + 1; // integer floor division
  };

  int outY = computeDim(inputY, padding[2], padding[3], kernelSize.first,
                        stride.first);
  int outX = computeDim(inputX, padding[0], padding[1], kernelSize.second,
                        stride.second);

  return {outY, outX};
}

/* Convenience overload: one integer for kernel & stride (e.g. k=3, s=2). */
inline std::pair<int, int> calculateOutputDims(int inputY, int inputX,
                                               int kernelSize, int stride,
                                               int padding, int dilation = 1,
                                               bool ceilMode = false) {
  return calculateOutputDims(
      inputY, inputX, {kernelSize, kernelSize}, {stride, stride},
      {padding, padding, padding, padding}, dilation, ceilMode);
}

ModuleOp hoistInputTensorOps(ModuleOp module, OpBuilder &builder) {
  // Find all functions in the module
  module.walk([&](func::FuncOp funcOp) {
    // Use a vector to maintain the order of tensor creation ops as they're
    // encountered
    llvm::SmallVector<Operation *> tensorCreationOps;
    llvm::SmallVector<Value> newArguments;
    llvm::SmallVector<Type> newArgumentTypes;
    llvm::SmallVector<ttcore::ArgumentTypeAttr> newArgumentTypeAttrs;

    // Set to track which ops we've already seen to avoid duplicates
    llvm::DenseSet<Operation *> seenOps;

    // Find all tensor creation operations in the function in traversal order
    funcOp.walk([&](Operation *op) {
      if (isa<ttnn::OnesOp, ttnn::ZerosOp, ttnn::EmptyOp, ttnn::FullOp>(op)) {
        // Only add each op once
        if (seenOps.insert(op).second) {
          // Store the tensor creation op in order of traversal
          tensorCreationOps.push_back(op);
        }
      }
    });

    // If no tensor creation ops found, return early
    if (tensorCreationOps.empty()) {
      return;
    }

    // Create a new function type with additional arguments for tensor creation
    // ops
    FunctionType oldFuncType = funcOp.getFunctionType();
    llvm::SmallVector<Type> inputTypes(oldFuncType.getInputs().begin(),
                                       oldFuncType.getInputs().end());
    llvm::SmallVector<Type> resultTypes(oldFuncType.getResults().begin(),
                                        oldFuncType.getResults().end());

    // Collect types for new arguments in traversal order
    for (Operation *op : tensorCreationOps) {
      Value result = op->getResult(0);
      newArguments.push_back(result);
      newArgumentTypes.push_back(result.getType());
      // Create tt ArgumentType attribute for the new argument
      newArgumentTypeAttrs.push_back(ttcore::ArgumentTypeAttr::get(
          builder.getContext(), ttcore::ArgumentType::Input));
    }

    // Add new argument types to the function type
    inputTypes.append(newArgumentTypes.begin(), newArgumentTypes.end());
    FunctionType newFuncType = builder.getFunctionType(inputTypes, resultTypes);

    // Update function type
    funcOp.setType(newFuncType);

    // Add argument attributes for the new arguments
    for (unsigned i = 0; i < newArgumentTypeAttrs.size(); ++i) {
      unsigned argIndex = oldFuncType.getNumInputs() + i;
      funcOp.setArgAttr(argIndex, newArgumentTypeAttrs[i].name,
                        newArgumentTypeAttrs[i]);
    }

    // Create new block arguments for the tensor creation ops
    Block &entryBlock = funcOp.getBlocks().front();
    for (Type type : newArgumentTypes) {
      entryBlock.addArgument(type, builder.getUnknownLoc());
    }

    // Replace uses of tensor creation op results with the new block arguments
    for (unsigned i = 0; i < newArguments.size(); ++i) {
      Value oldValue = newArguments[i];
      Value newValue = entryBlock.getArgument(oldFuncType.getNumInputs() + i);
      oldValue.replaceAllUsesWith(newValue);

      // Erase the tensor creation op
      Operation *op = tensorCreationOps[i];
      op->erase();
    }
  });

  return module;
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

  // Register device
  ttcore::registerDevice(module);

  mlir::IRRewriter rewriter(builder);
  rewriter.setInsertionPointToStart(entryBlock);
  // Value device = ttnn::utils::getOrInsertDevice(rewriter, func).getResult();
  Value device = ttnn::utils::getOrInsertDeviceInsideBlock(rewriter, entryBlock)
                     .getResult();

  std::unordered_map<int, Value> indexToResultMap;

  // First pass: create all operations
  if (auto *nodesArray = json.getAsArray()) {
    for (const auto &node : *nodesArray) {
      static int counter = 0;
      counter++;
      if (counter > 99999) {
        // if (counter > 10) {
        break;
      }
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
          ttcore::DataTypeAttr dataTypeAttr =
              ttcore::DataTypeAttr::get(context, ttcore::DataType::Float32);

          const llvm::json::Value &shapeObj = (*argsObj)[0];
          const llvm::json::Value &dtypeObj = (*argsObj)[1];
          const llvm::json::Value &layoutType = (*argsObj)[2];
          // const llvm::json::Value &layoutObj = (*arrayObj)[2];

          shapeVec = parseShape(shapeObj.getAsString()->str());

          shapeAttr = createShapeAttr(shapeVec, builder);
          dataTypeAttr = parseDataType(dtypeObj.getAsString()->str(), context);
          std::optional<LayoutAttr> layoutTypeAttrOpt =
              parseLayout(layoutType.getAsString()->str(), context);

          LayoutAttr layoutTypeAttr =
              layoutTypeAttrOpt ? layoutTypeAttrOpt.value() : nullptr;

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(shapeVec, builder.getBF16Type()));

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
              RankedTensorType::get(shapeVec, builder.getBF16Type(),
                                    layoutAttr),
              shapeAttr, dataTypeAttr, layoutTypeAttr, nullptr, nullptr);
          onesOp->setAttr("shape", shapeAttr);
          result = onesOp.getResult();
        } else if (name == "ttnn::conv2d") {
          // static ResultWithOptions invoke(
          //   QueueId queue_id,
          //   const ttnn::Tensor& input_tensor,
          //   const ttnn::Tensor& weight_tensor,
          //   MeshDevice* device,
          //   uint32_t in_channels,
          //   uint32_t out_channels,
          //   uint32_t batch_size,
          //   uint32_t input_height,
          //   uint32_t input_width,
          //   std::array<uint32_t, 2> kernel_size,
          //   std::array<uint32_t, 2> stride = std::array<uint32_t, 2>{1, 1},
          //   std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>
          //   padding = std::array<uint32_t, 2>{0, 0}, std::array<uint32_t, 2>
          //   dilation = std::array<uint32_t, 2>{1, 1}, uint32_t groups = 1,
          //   const std::optional<const ttnn::Tensor>& bias_tensor =
          //   std::nullopt, const std::optional<const Conv2dConfig>&
          //   conv_config_ = std::nullopt, const std::optional<const
          //   DeviceComputeKernelConfig>& compute_config_ = std::nullopt, const
          //   std::optional<const MemoryConfig>& memory_config_ = std::nullopt,
          //   const std::optional<const Conv2dSliceConfig>& dram_slice_config_
          //   = std::nullopt, bool return_output_dim = false, bool
          //   return_weights_and_bias = false);

          // note: QueueId is first arg, skip it
          std::string inputStr = (*argsObj)[1].getAsString()->str();
          std::string weightsStr = (*argsObj)[2].getAsString()->str();
          // device = (*argsObj)[3].getAsString()->str();
          uint32_t in_channels = std::stoul((*argsObj)[4].getAsString()->str());
          uint32_t out_channels =
              std::stoul((*argsObj)[5].getAsString()->str());
          uint32_t batch_size = std::stoul((*argsObj)[6].getAsString()->str());
          uint32_t input_height =
              std::stoul((*argsObj)[7].getAsString()->str());
          uint32_t input_width = std::stoul((*argsObj)[8].getAsString()->str());
          llvm::SmallVector<int32_t, 2> kernel_size =
              parseArray((*argsObj)[9].getAsString()->str());
          llvm::SmallVector<int32_t, 2> stride =
              parseArray((*argsObj)[10].getAsString()->str());
          llvm::SmallVector<int32_t, 2> padding =
              parseArray((*argsObj)[11].getAsString()->str());
          llvm::SmallVector<int32_t, 2> dilation =
              parseArray((*argsObj)[12].getAsString()->str());
          uint32_t groups = std::stoul((*argsObj)[13].getAsString()->str());
          // bias
          // conv_config
          // compute_config
          // memory_config
          // dram_slice_config
          // return_output_dim
          // return_weights_and_bias

          int inputIndex = std::stoi(inputStr.substr(inputStr.find(":") + 1));
          int weightsIndex =
              std::stoi(weightsStr.substr(weightsStr.find(":") + 1));

          Value input = indexToResultMap[inputIndex];
          Value weights = indexToResultMap[weightsIndex];

          // Calculate output shape
          std::pair<int, int> outputDims = calculateOutputDims(
              /*inputY=*/input_height, /*inputX=*/input_width,
              /*kernelSize=*/kernel_size[0], /*stride=*/stride[0],
              /*padding=*/padding[0], /*dilation=*/dilation[0],
              /*ceilMode=*/false);

          llvm::SmallVector<int64_t, 4> outputShape = {
              1, 1, batch_size * outputDims.first * outputDims.second,
              out_channels};

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(outputShape, builder.getBF16Type()));

          //   static void build(::mlir::OpBuilder &odsBuilder,
          //   ::mlir::OperationState &odsState, ::mlir::Type result,
          //   ::mlir::Value input, ::mlir::Value weight,
          //   /*optional*/::mlir::Value bias, ::mlir::Value device, uint32_t
          //   in_channels, uint32_t out_channels, uint32_t batch_size, uint32_t
          //   input_height, uint32_t input_width, ::llvm::ArrayRef<int32_t>
          //   kernel_size, ::llvm::ArrayRef<int32_t> stride,
          //   ::llvm::ArrayRef<int32_t> padding, ::llvm::ArrayRef<int32_t>
          //   dilation, uint32_t groups,
          //   /*optional*/::mlir::tt::ttnn::Conv2dConfigAttr conv2d_config,
          //   /*optional*/::mlir::tt::ttnn::DeviceComputeKernelConfigAttr
          //   compute_config);

          auto conv2dOp = builder.create<ttnn::Conv2dOp>(
              loc,
              RankedTensorType::get(outputShape, builder.getBF16Type(),
                                    layoutAttr),
              input, weights, /*bias=*/nullptr, device, in_channels,
              out_channels, batch_size, input_height, input_width, kernel_size,
              stride, padding, dilation, groups, nullptr, nullptr);
          result = conv2dOp.getResult();
        } else if (name == "ttnn::max_pool2d") {
          // "\u0000",
          // "tensor: 3",
          // "8",
          // "112",
          // "112",
          // "64",
          // "[ unsupported type , std::reference_wrapper<std::array<unsigned
          // int, 2ul> >]",
          // "[ unsupported type , std::reference_wrapper<std::array<unsigned
          // int, 2ul> >]",
          // "[ unsupported type , std::reference_wrapper<std::array<unsigned
          // int, 2ul> >]",
          // "[ unsupported type , std::reference_wrapper<std::array<unsigned
          // int, 2ul> >]", "0",
          // "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)",
          // "[ unsupported type ,
          // std::reference_wrapper<std::optional<tt::tt_metal::TensorMemoryLayout
          // const> const>]", "0"

          // static Tensor invoke(
          //   QueueId queue_id,
          //   const Tensor& input_tensor,
          //   uint32_t batch_size,
          //   uint32_t input_h,
          //   uint32_t input_w,
          //   uint32_t channels,
          //   std::array<uint32_t, 2> kernel_size,
          //   std::array<uint32_t, 2> stride,
          //   std::array<uint32_t, 2> padding,
          //   std::array<uint32_t, 2> dilation,
          //   bool ceil_mode = false,
          //   const std::optional<const MemoryConfig>& memory_config =
          //   std::nullopt, std::optional<const TensorMemoryLayout>
          //   applied_shard_scheme = std::nullopt, bool in_place_halo = false);

          // note: QueueId is first arg, skip it
          Value input = indexToResultMap[std::stoi(
              (*argsObj)[1].getAsString()->str().substr(
                  (*argsObj)[1].getAsString()->str().find(":") + 1))];
          uint32_t batch_size = std::stoul((*argsObj)[2].getAsString()->str());
          uint32_t input_h = std::stoul((*argsObj)[3].getAsString()->str());
          uint32_t input_w = std::stoul((*argsObj)[4].getAsString()->str());
          uint32_t channels = std::stoul((*argsObj)[5].getAsString()->str());
          llvm::SmallVector<int32_t, 2> kernel_size =
              mlir::tt::ttnn::parseArray((*argsObj)[6].getAsString()->str());
          llvm::SmallVector<int32_t, 2> stride =
              mlir::tt::ttnn::parseArray((*argsObj)[7].getAsString()->str());
          llvm::SmallVector<int32_t, 2> padding =
              mlir::tt::ttnn::parseArray((*argsObj)[8].getAsString()->str());
          llvm::SmallVector<int32_t, 2> dilation =
              mlir::tt::ttnn::parseArray((*argsObj)[9].getAsString()->str());
          bool ceil_mode = false;

          // Calculate output shape
          std::pair<int, int> outputDims = calculateOutputDims(
              /*inputY=*/input_h, /*inputX=*/input_w,
              /*kernelSize=*/kernel_size[0], /*stride=*/stride[0],
              /*padding=*/padding[0], /*dilation=*/dilation[0],
              /*ceilMode=*/false);

          llvm::SmallVector<int64_t, 4> outputShape = {
              1, 1, batch_size * outputDims.first * outputDims.second,
              channels};

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(outputShape, builder.getBF16Type()));

          //   static void build(::mlir::OpBuilder &odsBuilder,
          //   ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes,
          //   ::mlir::Value input, int32_t batch_size, int32_t input_height,
          //   int32_t input_width, int32_t channels, ::llvm::ArrayRef<int32_t>
          //   kernel_size, ::llvm::ArrayRef<int32_t> stride,
          //   ::llvm::ArrayRef<int32_t> padding, ::llvm::ArrayRef<int32_t>
          //   dilation, /*optional*/::mlir::tt::ttnn::MemoryConfigAttr
          //   memory_config,
          //   /*optional*/::mlir::tt::ttnn::TensorMemoryLayoutAttr
          //   applied_shard_scheme, bool ceil_mode, bool in_place_halo);
          auto maxpool2dOp = builder.create<ttnn::MaxPool2dOp>(
              loc,
              RankedTensorType::get(outputShape, builder.getBF16Type(),
                                    layoutAttr),
              input, batch_size, input_h, input_w, channels, kernel_size,
              stride, padding, dilation, /*memory_config=*/nullptr,
              /*applied_shard_scheme=*/nullptr, ceil_mode,
              /*in_place_halo=*/false);
          result = maxpool2dOp.getResult();
        } else if (name == "ttnn::avg_pool2d") {
          // static Tensor invoke(
          //   QueueId queue_id,
          //   const Tensor& input_tensor,
          //   uint32_t batch_size,
          //   uint32_t input_h,
          //   uint32_t input_w,
          //   uint32_t channels,
          //   std::array<uint32_t, 2> kernel_size,
          //   std::array<uint32_t, 2> stride,
          //   std::array<uint32_t, 2> padding,
          //   bool ceil_mode = false,
          //   bool count_include_pad = true,
          //   std::optional<int32_t> divisor_override = std::nullopt,
          //   const std::optional<const MemoryConfig>& memory_config =
          //   std::nullopt, std::optional<const TensorMemoryLayout>
          //   applied_shard_scheme = std::nullopt, bool in_place_halo = false);

          // "\u0000",
          // "tensor: 172",
          // "8",
          // "7",
          // "7",
          // "2048",
          // "[ unsupported type , std::reference_wrapper<std::array<unsigned
          // int, 2ul> >]",
          // "[ unsupported type , std::reference_wrapper<std::array<unsigned
          // int, 2ul> >]",
          // "[ unsupported type , std::reference_wrapper<std::array<unsigned
          // int, 2ul> >]", "0", "1", "nullopt",
          // "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)",
          // "[ unsupported type ,
          // std::reference_wrapper<std::optional<tt::tt_metal::TensorMemoryLayout
          // const> const>]", "0"

          // note: QueueId is first arg, skip it
          Value input = indexToResultMap[std::stoi(
              (*argsObj)[1].getAsString()->str().substr(
                  (*argsObj)[1].getAsString()->str().find(":") + 1))];
          uint32_t batch_size = std::stoul((*argsObj)[2].getAsString()->str());
          uint32_t input_h = std::stoul((*argsObj)[3].getAsString()->str());
          uint32_t input_w = std::stoul((*argsObj)[4].getAsString()->str());
          uint32_t channels = std::stoul((*argsObj)[5].getAsString()->str());
          llvm::SmallVector<int32_t, 2> kernel_size =
              mlir::tt::ttnn::parseArray((*argsObj)[6].getAsString()->str());
          llvm::SmallVector<int32_t, 2> stride =
              mlir::tt::ttnn::parseArray((*argsObj)[7].getAsString()->str());
          llvm::SmallVector<int32_t, 2> padding =
              mlir::tt::ttnn::parseArray((*argsObj)[8].getAsString()->str());
          bool ceil_mode = false;

          // Calculate output shape
          std::pair<int, int> outputDims = calculateOutputDims(
              /*inputY=*/input_h, /*inputX=*/input_w,
              /*kernelSize=*/kernel_size[0], /*stride=*/stride[0],
              /*padding=*/padding[0], /*dilation=*/1,
              /*ceilMode=*/false);

          llvm::SmallVector<int64_t, 4> outputShape = {
              1, 1, batch_size * outputDims.first * outputDims.second,
              channels};

          TTNNLayoutAttr layoutAttr = createLayoutAttr(
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(outputShape, builder.getBF16Type()));

          //   static void build(::mlir::OpBuilder &odsBuilder,
          //   ::mlir::OperationState &odsState, ::mlir::Type result,
          //   ::mlir::Value input, int32_t batch_size, int32_t input_height,
          //   int32_t input_width, int32_t channels, ::llvm::ArrayRef<int32_t>
          //   kernel_size, ::llvm::ArrayRef<int32_t> stride,
          //   ::llvm::ArrayRef<int32_t> padding, ::llvm::ArrayRef<int32_t>
          //   dilation, /*optional*/::mlir::tt::ttnn::MemoryConfigAttr
          //   memory_config,
          //   /*optional*/::mlir::tt::ttnn::TensorMemoryLayoutAttr
          //   applied_shard_scheme, bool ceil_mode, bool in_place_halo);
          auto avgpool2dOp = builder.create<ttnn::AvgPool2dOp>(
              loc,
              RankedTensorType::get(outputShape, builder.getBF16Type(),
                                    layoutAttr),
              input, batch_size, input_h, input_w, channels, kernel_size,
              stride, padding, /*dilation=*/llvm::SmallVector<int32_t, 2>{1, 1},
              /*memory_config=*/nullptr,
              /*applied_shard_scheme=*/nullptr, ceil_mode,
              /*in_place_halo=*/false);
          result = avgpool2dOp.getResult();
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
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(shape, builder.getBF16Type()));

          auto matmulOp = builder.create<ttnn::MatmulOp>(
              // static void build(::mlir::OpBuilder &odsBuilder,
              // ::mlir::OperationState &odsState, ::mlir::Type result,
              // ::mlir::Value a, ::mlir::Value b, ::mlir::BoolAttr transpose_a,
              // ::mlir::BoolAttr transpose_b, /*optional*/::mlir::Attribute
              // matmul_program_config);
              loc,
              RankedTensorType::get(shape, builder.getBF16Type(), layoutAttr),
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
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(shape, builder.getBF16Type()));

          auto addOp = builder.create<ttnn::AddOp>(
              loc,
              RankedTensorType::get(shape, builder.getBF16Type(), layoutAttr),
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
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(shape, builder.getBF16Type()));

          auto reluOp = builder.create<ttnn::ReluOp>(
              loc,
              RankedTensorType::get(shape, builder.getBF16Type(), layoutAttr),
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
              context, ttcore::GridAttr::get(context),
              RankedTensorType::get(shape, builder.getBF16Type()));

          // int dim = (*argsObj)[1].getAsNumber(); // not proprely traced, hack
          // for now
          int dim = -1;
          auto dimAttr = builder.getSI32IntegerAttr(dim);

          auto softmaxOp = builder.create<ttnn::SoftmaxOp>(
              loc,
              RankedTensorType::get(shape, builder.getBF16Type(), layoutAttr),
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

  // module->dump();

  hoistInputTensorOps(module, builder);

  return module;
}

} // namespace mlir::tt::ttnn
