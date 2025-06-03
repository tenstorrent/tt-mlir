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
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn {

constexpr uint64_t kHostAllocatedSize = 0;

#define GEN_PASS_DEF_TTNNSERIALIZETOBINARY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

static std::vector<::tt::target::Dim2dRange>
getTensorValueCoreRangeSet(FlatbufferObjectCache &cache, Value value) {
  DeviceAttr deviceAttr = lookupDevice(value.getParentBlock()->getParentOp());
  assert(deviceAttr);
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(value.getType());
  ttnn::TTNNLayoutAttr layoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
  std::vector<::tt::target::Dim2dRange> coreRangeSet =
      toFlatbuffer(cache, layoutAttr.getGrid(), deviceAttr.getWorkerGrid());
  return coreRangeSet;
}

static ttnn::MemoryConfigAttr
getMemoryConfigAttr(::mlir::tt::ttnn::TTNNLayoutAttr layoutAttr,
                    GridAttr deviceGrid) {
  MLIRContext *ctx = layoutAttr.getContext();
  ttnn::BufferTypeAttr bufferTypeAttr =
      ttnn::BufferTypeAttr::get(ctx, layoutAttr.getBufferType());

  ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
      ctx, layoutAttr.getMemLayout(), bufferTypeAttr,
      utils::createShardSpecIfNeeded(layoutAttr, deviceGrid));
  return memoryConfigAttr;
}

static ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig>
getMemoryConfigFromTensorTypeIfNeeded(FlatbufferObjectCache &cache,
                                      Value tensor) {
  auto tensorType = mlir::cast<RankedTensorType>(tensor.getType());
  auto layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
  ttnn::BufferType bufferType = layoutAttr.getBufferType();

  ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig> memoryConfig = 0;
  if (isDeviceBufferType(bufferType)) {
    DeviceAttr deviceAttr =
        lookupDevice(tensor.getParentBlock()->getParentOp());
    auto memoryConfigAttr =
        getMemoryConfigAttr(layoutAttr, deviceAttr.getWorkerGrid());
    memoryConfig = toFlatbuffer(cache, memoryConfigAttr);
  }

  return memoryConfig;
}

template <typename OpType>
static ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig>
getMemoryConfigIfNeeded(FlatbufferObjectCache &cache, OpType op) {
  // TODO (#620): Once we add a full support for shard spec, we can
  // remove obtaining tileShape and coreRangeSet from output tensor.
  // TODO (#2415): Once we have this pass, we can remove ternary if
  // and just get memory config attr from the op.
  auto result = op.getResult();
  return op.getMemoryConfig()
             ? toFlatbuffer(cache, *op.getMemoryConfig())
             : getMemoryConfigFromTensorTypeIfNeeded(cache, result);
}

::flatbuffers::Offset<::tt::target::DeviceRef>
createDeviceRef(FlatbufferObjectCache &cache, Value device) {
  auto desc = lookupDevice(device.getParentBlock()->getParentOp());
  auto chipIds = desc.getChipIds();
  return ::tt::target::CreateDeviceRef(*cache.fbb, chipIds[0]);
}

flatbuffers::Offset<::tt::target::ttnn::TensorDesc>
tensorTypeToFlatbuffer(FlatbufferObjectCache &cache, Type type,
                       DeviceAttr deviceAttr) {
  auto tensorType = mlir::cast<RankedTensorType>(type);
  auto shapeInt64 = tensorType.getShape();
  std::vector<int32_t> shape;
  shape.reserve(shapeInt64.size());
  std::transform(
      shapeInt64.begin(), shapeInt64.end(), std::back_inserter(shape),
      [](int64_t val) -> int32_t { return static_cast<int32_t>(val); });
  auto layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
  // Set meshShape to {1, 1} for single device tensor.
  std::vector<int32_t> meshShape = {1, 1};
  if (layoutAttr.getTensorMeshSharding()) {
    meshShape.clear();
    // Set meshShape to {x, y} for multi device tensor.
    auto meshShapeInt64 = deviceAttr.getMeshShape();
    meshShape =
        std::vector<int32_t>(meshShapeInt64.begin(), meshShapeInt64.end());
  }
  return ::tt::target::ttnn::CreateTensorDescDirect(
      *cache.fbb, &shape, &meshShape,
      cache.getOrCreate(layoutAttr, ttnnLayoutAttrToFlatbuffer, deviceAttr));
}

flatbuffers::Offset<::tt::target::ttnn::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t size) {
  auto deviceAttr = lookupDevice(value.getParentBlock()->getParentOp());
  assert(deviceAttr);
  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  mlir::Type elementType = tensorType.getElementType();
  // If the element type is quantized, use the desired type.
  // Ex: for a quant op of fp32->int8, the storage type is int8.
  if (auto quantType =
          mlir::dyn_cast<mlir::quant::QuantizedType>(elementType)) {
    elementType = quantType.getStorageType();
    tensorType = mlir::RankedTensorType::get(tensorType.getShape(), elementType,
                                             tensorType.getEncoding());
  }
  auto tensorDesc =
      cache.getOrCreate(tensorType, tensorTypeToFlatbuffer, deviceAttr);
  return ::tt::target::ttnn::CreateTensorRef(*cache.fbb, cache.global_id++,
                                             size, tensorDesc);
}

template <typename OpT>
::flatbuffers::Offset<::tt::target::ttnn::Operation>
createOperation(FlatbufferObjectCache &cache, ::flatbuffers::Offset<OpT> op,
                const std::string &debugString, const std::string &locInfo) {
  return CreateOperationDirect(
      *cache.fbb, ::tt::target::ttnn::OpTypeTraits<OpT>::enum_value, op.Union(),
      debugString.c_str(), locInfo.c_str());
}

::flatbuffers::Offset<::tt::target::ttnn::GetDeviceOp>
createOp(FlatbufferObjectCache &cache, GetDeviceOp op) {
  auto result = op.getResult();
  auto desc = lookupDevice(op);
  auto meshShape = desc.getMeshShape();
  auto meshVolume = ttmlir::utils::volume(meshShape);
  ::tt::target::Dim2d mesh;
  if (meshVolume > 1) {
    mesh = ::tt::target::Dim2d(meshShape[0], meshShape[1]);
  } else {
    mesh = ::tt::target::Dim2d(1, 1);
  }

  ::tt::target::Dim2d offset(0, 0);
  if (auto offsetAttr = op.getMeshOffset()) {
    offset = ::tt::target::Dim2d(offsetAttr->getY(), offsetAttr->getX());
  }

  auto chipIds = toFlatbuffer(cache, desc.getChipIds());
  auto out = cache.getOrCreate(result, createDeviceRef);
  return ::tt::target::ttnn::CreateGetDeviceOp(*cache.fbb, &mesh, &offset,
                                               chipIds, out);
}

::flatbuffers::Offset<::tt::target::ttnn::ToMemoryConfigOp>
createOp(FlatbufferObjectCache &cache, ToMemoryConfigOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));

  // TODO (jnie): Disabled `cache.getOrCreate` because identical MLIR memory
  // configs may produce different flatbuffer memory configs. One-to-one mapping
  // needed.
  auto memoryConfig = toFlatbuffer(cache, op.getMemoryConfig());

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  return ::tt::target::ttnn::CreateToMemoryConfigOp(*cache.fbb, input,
                                                    memoryConfig, output);
}

::flatbuffers::Offset<::tt::target::ttnn::ToLayoutOp>
createOp(FlatbufferObjectCache &cache, ToLayoutOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  ::tt::target::TensorLayout layout =
      ::mlir::tt::ttnn::utils::toTargetTensorLayout(op.getLayout());
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  ::flatbuffers::Optional<::tt::target::DataType> dtype =
      toFlatbuffer(cache, op.getDtype());
  std::optional<::mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();
  ::mlir::Value device = op.getDevice();
  if (device) {
    device = getOperandThroughDPSOps(device);
  }
  return ::tt::target::ttnn::CreateToLayoutOp(
      *cache.fbb, input, layout, dtype,
      memoryConfig ? toFlatbuffer(cache, *memoryConfig) : 0,
      device ? cache.at<::tt::target::DeviceRef>(device) : 0, output);
}

::flatbuffers::Offset<::tt::target::ttnn::ToDTypeOp>
createOp(FlatbufferObjectCache &cache, ToDTypeOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  ::tt::target::DataType dtype =
      ::mlir::tt::ttnn::utils::toTargetDataType(op.getDtype());
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  return ::tt::target::ttnn::CreateToDTypeOp(*cache.fbb, input, dtype, output);
}

::flatbuffers::Offset<::tt::target::ttnn::TypecastOp>
createOp(FlatbufferObjectCache &cache, TypecastOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  ::tt::target::DataType dtype =
      ::mlir::tt::ttnn::utils::toTargetDataType(op.getDtype());
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  return ::tt::target::ttnn::CreateTypecastOp(*cache.fbb, input, dtype, output);
}

::flatbuffers::Offset<::tt::target::ttnn::ToDeviceOp>
createOp(FlatbufferObjectCache &cache, ToDeviceOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto device = getOperandThroughDPSOps(op.getDevice());

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  if (!op.getMemoryConfig()) {
    return ::tt::target::ttnn::CreateToDeviceOp(
        *cache.fbb, input, cache.at<::tt::target::DeviceRef>(device),
        /* memoryConfig */ 0, output);
  }
  auto memoryConfig = toFlatbuffer(cache, op.getMemoryConfig().value());

  return ::tt::target::ttnn::CreateToDeviceOp(
      *cache.fbb, input, cache.at<::tt::target::DeviceRef>(device),
      memoryConfig, output);
}

::flatbuffers::Offset<::tt::target::ttnn::FromDeviceOp>
createOp(FlatbufferObjectCache &cache, FromDeviceOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  return ::tt::target::ttnn::CreateFromDeviceOp(*cache.fbb, input, output);
}

::flatbuffers::Offset<::tt::target::ttnn::CpuOp>
createCpuOp(FlatbufferObjectCache &cache, func::CallOp op, uint32_t dylib_id) {
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> ins;
  for (auto input : op.getOperands()) {
    ins.push_back(cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(input)));
  }

  // For now, assume we will get exactly 1 result tensor from our call -- this
  // is hardcoded assumption for all ops AFAICT.
  auto output = cache.getOrCreate(*op.getResults().begin(),
                                  tensorValueToFlatbuffer, kHostAllocatedSize);

  llvm::SmallString<24> funcName =
      tt::utils::convertDylibFuncName(op.getCallee());

  return ::tt::target::ttnn::CreateCpuOp(
      *cache.fbb, cache.fbb->CreateVector(ins), output,
      cache.fbb->CreateString(funcName.c_str()), dylib_id);
}

::flatbuffers::Offset<::tt::target::ttnn::DistributionStrategy>
createDistributionStrategy(FlatbufferObjectCache &cache,
                           const Value &deviceValue,
                           const RankedTensorType &type, uint32_t &numShards) {
  auto noneDistributionStrategy = [&cache]() {
    ::flatbuffers::Offset<void> distribution = 0;
    return ::tt::target::ttnn::CreateDistributionStrategy(
        *cache.fbb, ::tt::target::ttnn::DistributedTensorConfig::NONE,
        distribution);
  };

  // Skip single device tensors if it includes TensorMeshShardingAttr.
  auto layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(type.getEncoding());
  if (!deviceValue || !layoutAttr.isMeshDeviceTensor()) {
    return noneDistributionStrategy();
  }

  auto deviceOp = mlir::cast<GetDeviceOp>(
      getOperandThroughDPSOps(deviceValue).getDefiningOp());
  auto desc = lookupDevice(deviceOp);
  ::llvm::ArrayRef<int64_t> meshShape = desc.getMeshShape();
  numShards = ttmlir::utils::volume(meshShape);

  if (numShards == 1) {
    return noneDistributionStrategy();
  }

  assert(meshShape.size() <= 2 && "expected 2D mesh shape");

  // One-dimensional tensor sharding strategy. Tensor is sliced by the number of
  // devices at a certain dimension. For EmptyOp and FullOp, we assume that the
  // tensor is sliced at the fastest dimension.
  if (meshShape[0] == 1 || meshShape[1] == 1) {
    assert(type.getShape().size() > 0 && "expected non-zero tensor shape");
    uint32_t target_dim = type.getShape().size() - 1;
    auto strategy =
        ::tt::target::ttnn::CreateShardTensor(*cache.fbb, target_dim);
    return ::tt::target::ttnn::CreateDistributionStrategy(
        *cache.fbb, ::tt::target::ttnn::DistributedTensorConfig::ShardTensor,
        strategy.Union());
  }

  const ::tt::target::Dim2d shard_mesh(meshShape[0], meshShape[1]);
  auto strategy =
      ::tt::target::ttnn::CreateShardTensor2D(*cache.fbb, &shard_mesh);
  return ::tt::target::ttnn::CreateDistributionStrategy(
      *cache.fbb, ::tt::target::ttnn::DistributedTensorConfig::ShardTensor2D,
      strategy.Union());
}

::flatbuffers::Offset<::tt::target::ttnn::EmptyOp>
createOp(FlatbufferObjectCache &cache, EmptyOp op) {
  ::llvm::ArrayRef<int64_t> shape = op.getShape().getShape();
  ::tt::target::DataType dtype =
      ::mlir::tt::ttnn::utils::toTargetDataType(op.getDtype());
  ::tt::target::TensorLayout layout =
      ::mlir::tt::ttnn::utils::toTargetTensorLayout(op.getLayout());

  auto output = getOperandThroughDPSOps(op.getResult());
  auto device = getOperandThroughDPSOps(op.getDevice());
  auto memoryConfig = toFlatbuffer(cache, op.getMemoryConfig());

  return ::tt::target::ttnn::CreateEmptyOp(
      *cache.fbb, cache.fbb->CreateVector<int64_t>(shape), dtype, layout,
      cache.at<::tt::target::DeviceRef>(device), memoryConfig,
      cache.getOrCreate(output, tensorValueToFlatbuffer, kHostAllocatedSize));
}

::flatbuffers::Offset<::tt::target::ttnn::FullOp>
createOp(FlatbufferObjectCache &cache, FullOp op) {
  auto shape = op.getShape().getShape().vec();
  auto device = cache.at<::tt::target::DeviceRef>(
      getOperandThroughDPSOps(op.getDevice()));
  ::tt::target::ttnn::FillValueType fillValueType;
  ::flatbuffers::Offset<void> fillValue;
  if (auto fillValueAttr = mlir::dyn_cast<mlir::FloatAttr>(op.getFillValue())) {
    fillValueType = ::tt::target::ttnn::FillValueType::FP;
    fillValue = ::tt::target::ttnn::CreateFloatingPointType(
                    *cache.fbb, fillValueAttr.getValue().convertToFloat())
                    .Union();
  } else if (auto fillValueAttr =
                 mlir::dyn_cast<mlir::IntegerAttr>(op.getFillValue())) {
    fillValueType = ::tt::target::ttnn::FillValueType::I32;
    fillValue = ::tt::target::ttnn::CreateIntegralType(
                    *cache.fbb, fillValueAttr.getValue().getSExtValue())
                    .Union();
  } else {
    llvm_unreachable("fill value must be float or integer");
  }
  auto dtype = toFlatbuffer(cache, op.getDtype());
  auto layout = toFlatbuffer(cache, op.getLayout());
  auto memoryConfig = toFlatbuffer(cache, op.getMemoryConfig()).value_or(0);
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  return ::tt::target::ttnn::CreateFullOpDirect(
      *cache.fbb, &shape, fillValueType, fillValue, dtype, layout, device,
      memoryConfig, output);
}

::flatbuffers::Offset<::tt::target::ttnn::ArangeOp>
createOp(FlatbufferObjectCache &cache, ArangeOp op) {
  flatbuffers::Optional<::tt::target::DataType> dtype =
      toFlatbuffer(cache, op.getDtype());
  auto device =
      op.getDevice() ? cache.at<::tt::target::DeviceRef>(op.getDevice()) : 0;

  auto memoryConfig = op.getMemoryConfig().has_value()
                          ? toFlatbuffer(cache, op.getMemoryConfig().value())
                          : 0;

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  return ::tt::target::ttnn::CreateArangeOp(
      *cache.fbb, static_cast<float>(op.getStart()),
      static_cast<float>(op.getEnd()), static_cast<float>(op.getStep()),
      dtype /* optional */, device /* optional */, memoryConfig /* optional */,
      output);
}

template <typename OpTy>
::flatbuffers::Offset<::tt::target::ttnn::NamedFullOp>
createNamedFullOp(FlatbufferObjectCache &cache, OpTy op) {
  ::tt::target::ttnn::NamedFullOpType type;
  if constexpr (std::is_same_v<OpTy, ttnn::ZerosOp>) {
    type = ::tt::target::ttnn::NamedFullOpType::Zeros;
  } else if constexpr (std::is_same_v<OpTy, ttnn::OnesOp>) {
    type = ::tt::target::ttnn::NamedFullOpType::Ones;
  } else {
    static_assert(ttmlir::utils::always_false<OpTy>(),
                  "Unsupported NamedFullOp type");
  }

  ::flatbuffers::Offset<::flatbuffers::Vector<int64_t>> shape =
      cache.fbb->CreateVector<int64_t>(op.getShape().getShape());

  ::flatbuffers::Optional<::tt::target::DataType> dtype =
      toFlatbuffer(cache, op.getDtype());

  ::flatbuffers::Optional<::tt::target::TensorLayout> layout =
      toFlatbuffer(cache, op.getLayout());

  flatbuffers::Offset<::tt::target::DeviceRef> device =
      op.getDevice() ? cache.at<::tt::target::DeviceRef>(op.getDevice()) : 0;

  auto memoryConfig = op.getMemoryConfig().has_value()
                          ? toFlatbuffer(cache, op.getMemoryConfig().value())
                          : 0;

  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  return ::tt::target::ttnn::CreateNamedFullOp(
      *cache.fbb, type, shape, dtype, layout, device, memoryConfig, output);
}

::flatbuffers::Offset<::tt::target::ttnn::LinearOp>
createOp(FlatbufferObjectCache &cache, LinearOp op) {
  auto a = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getA()));
  auto b = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getB()));
  auto bias = op.getBias()
                  ? cache.at<::tt::target::ttnn::TensorRef>(
                        getOperandThroughDPSOps(op.getBias()))
                  : flatbuffers::Offset<::tt::target::ttnn::TensorRef>();
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  return ::tt::target::ttnn::CreateLinearOp(
      *cache.fbb, a, b, bias, output, op.getTransposeA(), op.getTransposeB());
}

// ANCHOR: adding_an_op_matmul_serialize_to_binary
::flatbuffers::Offset<::tt::target::ttnn::MatmulOp>
createOp(FlatbufferObjectCache &cache, MatmulOp op) {
  auto a = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getA()));
  auto b = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getB()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  using MatmulConfigType = ::tt::target::ttnn::MatmulProgramConfig;
  MatmulConfigType matmulProgramConfigType = MatmulConfigType::NONE;
  ::flatbuffers::Offset<void> matmulProgramConfigDesc;
  if (auto matmulProgramConfig = op.getMatmulProgramConfigAttr()) {
    if (auto config =
            mlir::dyn_cast<ttnn::MatmulMultiCoreReuseProgramConfigAttr>(
                matmulProgramConfig)) {
      matmulProgramConfigType =
          MatmulConfigType::MatmulMultiCoreReuseProgramConfig;
      matmulProgramConfigDesc = toFlatbuffer(cache, config).Union();
    } else if (auto config = mlir::dyn_cast<
                   ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr>(
                   matmulProgramConfig)) {
      matmulProgramConfigType =
          MatmulConfigType::MatmulMultiCoreReuseMultiCastProgramConfig;
      matmulProgramConfigDesc = toFlatbuffer(cache, config).Union();
    } else if (auto config = mlir::dyn_cast<
                   ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(
                   matmulProgramConfig)) {
      matmulProgramConfigType =
          MatmulConfigType::MatmulMultiCoreReuseMultiCast1DProgramConfig;
      matmulProgramConfigDesc = toFlatbuffer(cache, config).Union();
    } else if (
        auto config = mlir::dyn_cast<
            ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
            matmulProgramConfig)) {
      matmulProgramConfigType = MatmulConfigType::
          MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig;
      matmulProgramConfigDesc = toFlatbuffer(cache, config).Union();
    }
  }

  return ::tt::target::ttnn::CreateMatmulOp(
      *cache.fbb, a, b, output, op.getTransposeA(), op.getTransposeB(),
      matmulProgramConfigType, matmulProgramConfigDesc);
}
// ANCHOR_END: adding_an_op_matmul_serialize_to_binary

::flatbuffers::Offset<::tt::target::ttnn::MorehCumSumOp>
createOp(FlatbufferObjectCache &cache, MorehCumSumOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto outputType = op.getResult();
  auto output = cache.getOrCreate(outputType, tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  auto coreRangeSet = getTensorValueCoreRangeSet(cache, outputType);
  auto memoryConfig = op.getMemoryConfig()
                          ? toFlatbuffer(cache, op.getMemoryConfig().value())
                          : 0;

  return ::tt::target::ttnn::CreateMorehCumSumOp(*cache.fbb, in, output,
                                                 op.getDim(), memoryConfig);
}

::flatbuffers::Offset<::tt::target::ttnn::PrepareConv2dWeightsOp>
createOp(FlatbufferObjectCache &cache, PrepareConv2dWeightsOp op) {
  auto weightTensor = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getWeightTensor()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig> memoryConfig =
      toFlatbuffer(cache, op.getInputMemoryConfig());

  ::tt::target::TensorLayout inputTensorLayout =
      toFlatbuffer(cache, op.getInputTensorLayout());
  ::flatbuffers::Offset<::flatbuffers::String> weightsFormat =
      toFlatbuffer(cache, op.getWeightsFormat());

  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> kernelSize =
      toFlatbuffer(cache, op.getKernelSize());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> stride =
      toFlatbuffer(cache, op.getStride());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> padding =
      toFlatbuffer(cache, op.getPadding());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> dilation =
      toFlatbuffer(cache, op.getDilation());
  auto device = getOperandThroughDPSOps(op.getDevice());

  std::optional<::flatbuffers::Offset<::tt::target::ttnn::Conv2dConfig>>
      conv2dConfig = toFlatbuffer(cache, op.getConv2dConfig());

  return ::tt::target::ttnn::CreatePrepareConv2dWeightsOp(
      *cache.fbb, weightTensor, output, memoryConfig, inputTensorLayout,
      weightsFormat, op.getInChannels(), op.getOutChannels(), op.getBatchSize(),
      op.getInputHeight(), op.getInputWidth(), kernelSize, stride, padding,
      dilation, op.getHasBias(), op.getGroups(),
      cache.at<::tt::target::DeviceRef>(device), conv2dConfig.value_or(0));
}

::flatbuffers::Offset<::tt::target::ttnn::Conv2dOp>
createOp(FlatbufferObjectCache &cache, Conv2dOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto weight = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getWeight()));
  auto bias = op.getODSOperands(2).empty()
                  ? flatbuffers::Offset<::tt::target::ttnn::TensorRef>()
                  : cache.at<::tt::target::ttnn::TensorRef>(
                        getOperandThroughDPSOps(op.getBias()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  auto device = getOperandThroughDPSOps(op.getDevice());

  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> kernelSize =
      toFlatbuffer(cache, op.getKernelSize());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> stride =
      toFlatbuffer(cache, op.getStride());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> padding =
      toFlatbuffer(cache, op.getPadding());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> dilation =
      toFlatbuffer(cache, op.getDilation());

  std::optional<::flatbuffers::Offset<::tt::target::ttnn::Conv2dConfig>>
      conv2dConfig = toFlatbuffer(cache, op.getConv2dConfig());

  std::optional<
      ::flatbuffers::Offset<::tt::target::ttnn::DeviceComputeKernelConfig>>
      computeConfig = toFlatbuffer(cache, op.getComputeConfig());

  return ::tt::target::ttnn::CreateConv2dOp(
      *cache.fbb, input, weight, bias, output,
      cache.at<::tt::target::DeviceRef>(device), op.getInChannels(),
      op.getOutChannels(), op.getBatchSize(), op.getInputHeight(),
      op.getInputWidth(), kernelSize, stride, padding, dilation, op.getGroups(),
      conv2dConfig.value_or(0), computeConfig.value_or(0));
}

::flatbuffers::Offset<::tt::target::ttnn::ConvTranspose2dOp>
createOp(FlatbufferObjectCache &cache, ConvTranspose2dOp op) {
  auto in0 = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto in1 = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getWeight()));
  auto in2 = op.getODSOperands(2).empty()
                 ? flatbuffers::Offset<::tt::target::ttnn::TensorRef>()
                 : cache.at<::tt::target::ttnn::TensorRef>(
                       getOperandThroughDPSOps(op.getBias()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  auto device = getOperandThroughDPSOps(op.getDevice());

  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> kernelSize =
      toFlatbuffer(cache, op.getKernelSize());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> stride =
      toFlatbuffer(cache, op.getStride());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> padding =
      toFlatbuffer(cache, op.getPadding());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> outputPadding =
      toFlatbuffer(cache, op.getOutputPadding());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> dilation =
      toFlatbuffer(cache, op.getDilation());

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  return ::tt::target::ttnn::CreateConvTranspose2dOp(
      *cache.fbb, in0, in1, in2, output,
      cache.at<::tt::target::DeviceRef>(device), op.getInChannels(),
      op.getOutChannels(), op.getBatchSize(), op.getInputHeight(),
      op.getInputWidth(), kernelSize, stride, padding, outputPadding, dilation,
      op.getGroups(), memoryConfig);
}

::flatbuffers::Offset<::tt::target::ttnn::AllGatherOp>
createOp(FlatbufferObjectCache &cache, AllGatherOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  auto device = getOperandThroughDPSOps(op.getDevice());
  return ::tt::target::ttnn::CreateAllGatherOp(
      *cache.fbb, input, output, cache.at<::tt::target::DeviceRef>(device),
      op.getAllGatherDim(), op.getClusterAxis(), op.getNumLinks());
}

::flatbuffers::Offset<::tt::target::ttnn::ReduceScatterOp>
createOp(FlatbufferObjectCache &cache, ReduceScatterOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  auto device = getOperandThroughDPSOps(op.getDevice());
  return ::tt::target::ttnn::CreateReduceScatterOp(
      *cache.fbb, input, output, cache.at<::tt::target::DeviceRef>(device),
      op.getScatterDim(), static_cast<uint32_t>(op.getReduceType()),
      op.getClusterAxis(), op.getNumLinks());
}

::flatbuffers::Offset<::tt::target::ttnn::CollectivePermuteOp>
createOp(FlatbufferObjectCache &cache, CollectivePermuteOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  auto device = getOperandThroughDPSOps(op.getDevice());
  auto sourceTargetPairs = op.getSourceTargetPairs().getValues<int64_t>();
  std::vector<int64_t> sourceTargetPairsVec(sourceTargetPairs.begin(),
                                            sourceTargetPairs.end());
  return ::tt::target::ttnn::CreateCollectivePermuteOp(
      *cache.fbb, input, output, cache.at<::tt::target::DeviceRef>(device),
      cache.fbb->CreateVector<int64_t>(sourceTargetPairsVec));
}

::flatbuffers::Offset<::tt::target::ttnn::MeshShardOp>
createOp(FlatbufferObjectCache &cache, MeshShardOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  auto device = getOperandThroughDPSOps(op.getDevice());
  const mlir::tt::MeshShardDirection shardDirection = op.getShardDirection();
  const mlir::tt::MeshShardType shardType = op.getShardType();
  llvm::ArrayRef<int64_t> shardShape = op.getShardShape();
  llvm::ArrayRef<int64_t> shardDims = op.getShardDims();

  ::tt::target::ttnn::MeshShardDirection meshShardDirection;
  if (shardDirection == mlir::tt::MeshShardDirection::FullToShard) {
    meshShardDirection =
        ::tt::target::ttnn::MeshShardDirection::FullToShardShape;
  } else if (shardDirection == mlir::tt::MeshShardDirection::ShardToFull) {
    meshShardDirection =
        ::tt::target::ttnn::MeshShardDirection::ShardToFullShape;
  } else {
    llvm_unreachable("unhandled mesh_shard direction");
  }

  ::tt::target::ttnn::MeshShardType meshShardType;
  if (shardType == mlir::tt::MeshShardType::Replicate) {
    meshShardType = ::tt::target::ttnn::MeshShardType::Replicate;
  } else if (shardType == mlir::tt::MeshShardType::Devices) {
    meshShardType = ::tt::target::ttnn::MeshShardType::Devices;
  } else if (shardType == mlir::tt::MeshShardType::Identity) {
    meshShardType = ::tt::target::ttnn::MeshShardType::Identity;
  } else {
    llvm_unreachable("unhandled mesh_shard type");
  }

  return ::tt::target::ttnn::CreateMeshShardOp(
      *cache.fbb, input, output, cache.at<::tt::target::DeviceRef>(device),
      meshShardDirection, meshShardType,
      cache.fbb->CreateVector<int64_t>(shardShape),
      cache.fbb->CreateVector<int64_t>(shardDims));
}

::flatbuffers::Offset<::tt::target::ttnn::PermuteOp>
createOp(FlatbufferObjectCache &cache, PermuteOp op) {
  flatbuffers::Offset<::tt::target::ttnn::TensorRef> input =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getInput()));
  flatbuffers::Offset<flatbuffers::Vector<int64_t>> permutation =
      toFlatbuffer(cache, op.getPermutation());
  std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();
  float padValue = op.getPadValue().convertToFloat();
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  auto coreRangeSet = getTensorValueCoreRangeSet(cache, op.getResult());
  return ::tt::target::ttnn::CreatePermuteOp(
      *cache.fbb, input, permutation,
      memoryConfig ? toFlatbuffer(cache, memoryConfig.value()) : 0, padValue,
      output);
}

::flatbuffers::Offset<::tt::target::ttnn::BatchNormOp>
createOp(FlatbufferObjectCache &cache, BatchNormOp op) {
  flatbuffers::Offset<::tt::target::ttnn::TensorRef> input =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getInput()));
  ::flatbuffers::Offset<::tt::target::ttnn::TensorRef> runningMean =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getRunningMean()));
  ::flatbuffers::Offset<::tt::target::ttnn::TensorRef> runningVar =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getRunningVar()));
  ::flatbuffers::Offset<::tt::target::ttnn::TensorRef> weight =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getWeight()));
  ::flatbuffers::Offset<::tt::target::ttnn::TensorRef> bias =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getBias()));

  ::flatbuffers::Offset<::tt::target::ttnn::TensorRef> output =
      cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                        kHostAllocatedSize);

  ::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig> memoryConfig =
      op.getMemoryConfig() ? toFlatbuffer(cache, *op.getMemoryConfig()) : 0;

  return ::tt::target::ttnn::CreateBatchNormOp(
      *cache.fbb, input, runningMean, runningVar, op.getTraining(),
      op.getEpsilon().convertToFloat(), op.getMomentum().convertToFloat(),
      weight, bias, memoryConfig, output);
}

::flatbuffers::Offset<::tt::target::ttnn::UpsampleOp>
createOp(FlatbufferObjectCache &cache, UpsampleOp op) {
  flatbuffers::Offset<::tt::target::ttnn::TensorRef> input =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getInput()));
  flatbuffers::Offset<flatbuffers::String> mode =
      toFlatbuffer(cache, op.getMode());

  auto coreRangeSet = getTensorValueCoreRangeSet(cache, op.getResult());
  flatbuffers::Offset<::tt::target::ttnn::MemoryConfig> memoryConfig =
      op.getMemoryConfig() ? toFlatbuffer(cache, op.getMemoryConfig().value())
                           : 0;
  flatbuffers::Offset<::tt::target::ttnn::TensorRef> output = cache.getOrCreate(
      op.getResult(), tensorValueToFlatbuffer, kHostAllocatedSize);

  ::tt::target::ttnn::Scale2D scaleType;
  ::flatbuffers::Offset<void> scaleFactor;
  if (auto uniformScaleFactor =
          mlir::dyn_cast<IntegerAttr>(op.getScaleFactor())) {
    scaleType = ::tt::target::ttnn::Scale2D::UniformScale2D;
    scaleFactor = ::tt::target::ttnn::CreateUniformScale2D(
                      *cache.fbb, uniformScaleFactor.getSInt())
                      .Union();
  } else if (auto nonUniformScaleFactor =
                 mlir::dyn_cast<DenseI32ArrayAttr>(op.getScaleFactor())) {
    scaleType = ::tt::target::ttnn::Scale2D::NonUniformScale2D;
    scaleFactor =
        ::tt::target::ttnn::CreateNonUniformScale2D(
            *cache.fbb, toFlatbuffer(cache, nonUniformScaleFactor.asArrayRef()))
            .Union();
  } else {
    assert(false && "Unhandled scale factor type");
  }

  return ::tt::target::ttnn::CreateUpsampleOp(
      *cache.fbb, input, scaleType, scaleFactor, mode, memoryConfig, output);
}

::flatbuffers::Offset<::tt::target::ttnn::UpdateCacheOp>
createOp(FlatbufferObjectCache &cache, UpdateCacheOp op) {
  auto cacheOperand = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getCache()));
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto updateIndex = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getUpdateIndex()));

  return ::tt::target::ttnn::CreateUpdateCacheOp(
      *cache.fbb, cacheOperand, input, updateIndex, op.getBatchOffset());
}

::flatbuffers::Offset<::tt::target::ttnn::FillCacheOp>
createOp(FlatbufferObjectCache &cache, FillCacheOp op) {
  auto cacheOperand = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getCache()));
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));

  return ::tt::target::ttnn::CreateFillCacheOp(*cache.fbb, cacheOperand, input,
                                               op.getBatchOffset());
}

::flatbuffers::Offset<::tt::target::ttnn::ConstantOp>
createOp(FlatbufferObjectCache &cache, ttnn::ConstantOp op) {
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  std::vector<uint8_t> rawVector;
  if (auto data =
          mlir::dyn_cast<mlir::DenseResourceElementsAttr>(op.getValue())) {
    ArrayRef<char> rawData = data.getData();
    rawVector = std::vector<uint8_t>(rawData.begin(), rawData.end());
  } else if (auto data =
                 mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue())) {
    ArrayRef<char> rawData = data.getRawData();
    rawVector = std::vector<uint8_t>(rawData.begin(), rawData.end());
  } else {
    llvm_unreachable("Unknown constant value attribute type");
  }

  return ::tt::target::ttnn::CreateConstantOpDirect(*cache.fbb, output,
                                                    &rawVector);
}

template <typename EltwiseBinaryOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryOp>
createEltwiseBinaryOp(FlatbufferObjectCache &cache, EltwiseBinaryOp op) {

  ::tt::target::ttnn::EltwiseBinaryOpType type;
  if constexpr (std::is_same_v<EltwiseBinaryOp, AddOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::Add;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, MultiplyOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::Multiply;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, SubtractOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::Subtract;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, EqualOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::Equal;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, NotEqualOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::NotEqual;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, GreaterEqualOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::GreaterEqual;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, GreaterThanOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::GreaterThan;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, LessEqualOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::LessEqual;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, LessThanOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::LessThan;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, DivideOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::Divide;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, LogicalAndOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::LogicalAnd;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, LogicalOrOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::LogicalOr;
  } else if constexpr (std::is_same_v<EltwiseBinaryOp, LogicalXorOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryOpType::LogicalXor;
  } else {
    llvm_unreachable("unhandled EltwiseBinaryOp");
  }
  auto lhs = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getLhs()));

  auto rhs = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getRhs()));

  auto result = op.getResult();

  auto out =
      cache.getOrCreate(result, tensorValueToFlatbuffer, kHostAllocatedSize);

  // TODO (#2856): we should be getting output data type and memory config from
  // the binary op directly instead of deriving from the output tensor.
  // Requires compiler support.
  auto outputType = mlir::cast<RankedTensorType>(result.getType());
  DataType outputDtype = elementTypeToDataType(outputType.getElementType());
  ::tt::target::DataType targetDtype =
      ::mlir::tt::ttnn::utils::toTargetDataType(outputDtype);

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  return ::tt::target::ttnn::CreateEltwiseBinaryOp(
      *cache.fbb, type, lhs, rhs, targetDtype, memoryConfig, out);
}

template <typename EltwiseBinaryCompositeOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseBinaryCompositeOp>
createEltwiseBinaryCompositeOp(FlatbufferObjectCache &cache,
                               EltwiseBinaryCompositeOp op) {

  ::tt::target::ttnn::EltwiseBinaryCompositeOpType type;
  if (std::is_same_v<EltwiseBinaryCompositeOp, MaximumOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Maximum;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, MinimumOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Minimum;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, RemainderOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Remainder;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, ScatterOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Scatter;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, PowOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Pow;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, Atan2Op>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Atan2;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, BitwiseAndOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseAnd;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, BitwiseOrOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseOr;
  } else if (std::is_same_v<EltwiseBinaryCompositeOp, BitwiseXorOp>) {
    type = ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseXor;
  } else {
    llvm_unreachable("unhandled EltwiseBinaryCompositeOp");
  }
  auto lhs = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getLhs()));

  auto rhs = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getRhs()));

  auto result = op.getResult();

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  auto out =
      cache.getOrCreate(result, tensorValueToFlatbuffer, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateEltwiseBinaryCompositeOp(
      *cache.fbb, type, lhs, rhs, memoryConfig, out);
}

::flatbuffers::Offset<::tt::target::ttnn::EltwiseTernaryWhereOp>
createEltwiseTernaryWhereOp(FlatbufferObjectCache &cache, WhereOp op) {

  auto first = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getFirst()));
  auto second = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getSecond()));
  auto third = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getThird()));

  auto result = op.getResult();

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  auto out =
      cache.getOrCreate(result, tensorValueToFlatbuffer, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateEltwiseTernaryWhereOp(
      *cache.fbb, first, second, third, memoryConfig, out);
}

// Helper to walk typecast (optional) -> to_device -> to_layout -> from_device
// -> full and extract attribute to get the original value for the constant
// quantization scale or zero point.
template <typename AttrType>
static AttrType getAttrFromConstantChain(mlir::Value tensorVal,
                                         const char *expectedTypeMsg) {
  mlir::Value firstInput = tensorVal;
  if constexpr (std::is_same_v<AttrType, int32_t>) {
    // typecast first op for per-tensor zp
    if (auto typeCastOp = firstInput.getDefiningOp<ttnn::TypecastOp>()) {
      firstInput = typeCastOp.getInput();
    } else {
      llvm_unreachable(
          "Expected ttnn.typecast as defining op for per-tensor zp.");
    }
  }
  ttnn::ToDeviceOp toDeviceOp =
      mlir::dyn_cast<ttnn::ToDeviceOp>(firstInput.getDefiningOp());
  if (!toDeviceOp) {
    llvm_unreachable(
        "Expected ttnn.to_device as defining op for per-tensor scale/zp.");
  }
  ttnn::ToLayoutOp toLayoutOp =
      mlir::dyn_cast<ttnn::ToLayoutOp>(toDeviceOp.getInput().getDefiningOp());
  if (!toLayoutOp) {
    llvm_unreachable(
        "Expected ttnn.to_layout as defining op for per-tensor scale/zp.");
  }
  ttnn::FromDeviceOp fromDeviceOp =
      mlir::dyn_cast<ttnn::FromDeviceOp>(toLayoutOp.getInput().getDefiningOp());
  if (!fromDeviceOp) {
    llvm_unreachable(
        "Expected ttnn.from_device as defining op for per-tensor scale/zp.");
  }
  ttnn::FullOp fullOp =
      mlir::dyn_cast<ttnn::FullOp>(fromDeviceOp.getInput().getDefiningOp());
  if (!fullOp) {
    llvm_unreachable(
        "Expected ttnn.full as defining op for per-tensor scale/zp.");
  }
  if constexpr (std::is_same_v<AttrType, float>) {
    if (auto fillValueAttr =
            mlir::dyn_cast<mlir::FloatAttr>(fullOp.getFillValue())) {
      return fillValueAttr.getValue().convertToDouble();
    }
  } else if constexpr (std::is_same_v<AttrType, int32_t>) {
    if (auto fillValueAttr =
            mlir::dyn_cast<mlir::IntegerAttr>(fullOp.getFillValue())) {
      return fillValueAttr.getValue().getSExtValue();
    }
  }
  llvm_unreachable(expectedTypeMsg);
}

// Process scale tensor (float values)
static std::pair<::tt::target::ttnn::QuantizationScale,
                 flatbuffers::Offset<void>>
processScaleTensor(FlatbufferObjectCache &cache, mlir::Value scale) {
  ::tt::target::ttnn::QuantizationScale scaleType;
  flatbuffers::Offset<void> scaleUnion;

  mlir::Value scaleTensor = getOperandThroughDPSOps(scale);
  mlir::RankedTensorType scaleTensorType =
      mlir::cast<mlir::RankedTensorType>(scaleTensor.getType());

  if (scaleTensorType.getNumElements() == 1) {
    // In the per-tensor case, the scale is a float scalar.
    scaleType = ::tt::target::ttnn::QuantizationScale::PerTensorScale;
    float scaleValue = getAttrFromConstantChain<float>(
        scaleTensor, "Scale tensor constant must be a float attribute for "
                     "per-tensor quantization.");
    scaleUnion =
        ::tt::target::ttnn::CreatePerTensorScale(*cache.fbb, scaleValue)
            .Union();
  } else {
    // In the per-axis case, the scale is a tensor.
    scaleType = ::tt::target::ttnn::QuantizationScale::PerAxisScale;
    flatbuffers::Offset<::tt::target::ttnn::TensorRef> scaleTensorRef =
        cache.at<::tt::target::ttnn::TensorRef>(scaleTensor);
    scaleUnion =
        ::tt::target::ttnn::CreatePerAxisScale(*cache.fbb, scaleTensorRef)
            .Union();
  }

  return {scaleType, scaleUnion};
}

// Process zero point tensor (int32 values)
static std::pair<::tt::target::ttnn::QuantizationZeroPoint,
                 flatbuffers::Offset<void>>
processZeroPointTensor(FlatbufferObjectCache &cache, mlir::Value zeroPoint) {
  ::tt::target::ttnn::QuantizationZeroPoint zeroPointType;
  flatbuffers::Offset<void> zeroPointUnion;

  mlir::Value zeroPointTensor = getOperandThroughDPSOps(zeroPoint);
  mlir::RankedTensorType zeroPointTensorType =
      mlir::cast<mlir::RankedTensorType>(zeroPointTensor.getType());

  // In the per-tensor case, the zero point is an int32 scalar.
  if (zeroPointTensorType.getNumElements() == 1) {
    zeroPointType =
        ::tt::target::ttnn::QuantizationZeroPoint::PerTensorZeroPoint;
    int32_t zeroPointValue = getAttrFromConstantChain<int32_t>(
        zeroPointTensor, "Zero point tensor constant must be an integer "
                         "attribute for per-tensor quantization.");
    zeroPointUnion =
        ::tt::target::ttnn::CreatePerTensorZeroPoint(*cache.fbb, zeroPointValue)
            .Union();
  } else {
    // In the per-axis case, the zero point is a tensor.
    zeroPointType = ::tt::target::ttnn::QuantizationZeroPoint::PerAxisZeroPoint;
    flatbuffers::Offset<::tt::target::ttnn::TensorRef> zeroPointTensorRef =
        cache.at<::tt::target::ttnn::TensorRef>(zeroPointTensor);
    zeroPointUnion = ::tt::target::ttnn::CreatePerAxisZeroPoint(
                         *cache.fbb, zeroPointTensorRef)
                         .Union();
  }

  return {zeroPointType, zeroPointUnion};
}

template <typename EltwiseQuantizationOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseQuantizationOp>
createEltwiseQuantizationOp(FlatbufferObjectCache &cache,
                            EltwiseQuantizationOp op) {

  auto createQuantDequantParams =
      [&cache](std::variant<QuantizeOp, DequantizeOp> opVariant) {
        return std::visit(
            [&cache](auto &&op) {
              // Process scale.
              auto [scaleType, scaleValue] =
                  processScaleTensor(cache, op.getScale());

              // Process zero point.
              auto [zeroPointType, zeroPointValue] =
                  processZeroPointTensor(cache, op.getZeroPoint());

              return ::tt::target::ttnn::CreateQuantizeDequantizeOpParams(
                  *cache.fbb, scaleType, scaleValue, zeroPointType,
                  zeroPointValue);
            },
            opVariant);
      };

  auto createRequantOpParams = [&cache](RequantizeOp op) {
    // Process in_scale.
    auto [inScaleType, inScaleValue] =
        processScaleTensor(cache, op.getInScale());

    // Process in_zero_point.
    auto [inZeroPointType, inZeroPointValue] =
        processZeroPointTensor(cache, op.getInZeroPoint());

    // Process out_scale.
    auto [outScaleType, outScaleValue] =
        processScaleTensor(cache, op.getOutScale());

    // Process out_zero_point.
    auto [outZeroPointType, outZeroPointValue] =
        processZeroPointTensor(cache, op.getOutZeroPoint());

    return ::tt::target::ttnn::CreateRequantizeOpParams(
        *cache.fbb, inScaleType, inScaleValue, inZeroPointType,
        inZeroPointValue, outScaleType, outScaleValue, outZeroPointType,
        outZeroPointValue);
  };

  ::tt::target::ttnn::EltwiseQuantizationOpType type;
  ::tt::target::ttnn::EltwiseQuantizationOpParams paramsType =
      ::tt::target::ttnn::EltwiseQuantizationOpParams::NONE;
  ::flatbuffers::Offset<void> params = 0;

  if constexpr (std::is_same_v<EltwiseQuantizationOp, QuantizeOp>) {
    type = ::tt::target::ttnn::EltwiseQuantizationOpType::Quantize;
    paramsType = ::tt::target::ttnn::EltwiseQuantizationOpParams::
        QuantizeDequantizeOpParams;
    params = createQuantDequantParams(op).Union();
  } else if constexpr (std::is_same_v<EltwiseQuantizationOp, DequantizeOp>) {
    type = ::tt::target::ttnn::EltwiseQuantizationOpType::Dequantize;
    paramsType = ::tt::target::ttnn::EltwiseQuantizationOpParams::
        QuantizeDequantizeOpParams;
    params = createQuantDequantParams(op).Union();
  } else if constexpr (std::is_same_v<EltwiseQuantizationOp, RequantizeOp>) {
    type = ::tt::target::ttnn::EltwiseQuantizationOpType::Requantize;
    paramsType =
        ::tt::target::ttnn::EltwiseQuantizationOpParams::RequantizeOpParams;
    params = createRequantOpParams(op).Union();
  } else {
    llvm_unreachable("unhandled EltwiseQuantizationOp");
  }

  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));

  ::flatbuffers::Optional<int32_t> axis = toFlatbuffer(cache, op.getAxis());
  ::flatbuffers::Optional<::tt::target::DataType> outputDType =
      toFlatbuffer(cache, op.getOutputDtype());

  auto result = op.getResult();

  // TODO (#2858): we should be getting memory config from the quantization op
  // directly instead of deriving from the output tensor.
  // Although the mlir op has a memory config attribute, it's not being set.
  auto memoryConfig = getMemoryConfigFromTensorTypeIfNeeded(cache, result);

  auto out =
      cache.getOrCreate(result, tensorValueToFlatbuffer, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateEltwiseQuantizationOp(
      *cache.fbb, type, in, axis, outputDType, memoryConfig, out, paramsType,
      params);
}

template <typename EltwiseUnaryOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseUnaryOp>
createEltwiseUnaryOp(FlatbufferObjectCache &cache, EltwiseUnaryOp op) {

  ::tt::target::ttnn::EltwiseUnaryOpType type;
  ::tt::target::ttnn::EltwiseUnaryOpParams paramsType =
      ::tt::target::ttnn::EltwiseUnaryOpParams::NONE;
  ::flatbuffers::Offset<void> params = 0;

  if constexpr (std::is_same_v<EltwiseUnaryOp, AbsOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Abs;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, CeilOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Ceil;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, CosOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Cos;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, FloorOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Floor;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, GeluOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Gelu;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, IsFiniteOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::IsFinite;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, LogicalNotOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::LogicalNot;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, NegOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Neg;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, ReluOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Relu;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, SqrtOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Sqrt;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, RsqrtOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Rsqrt;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, SigmoidOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, SinOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Sin;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, ReciprocalOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Reciprocal;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, SignOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Sign;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, TanOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Tan;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, TanhOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Tanh;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, AtanOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Atan;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, ExpOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Exp;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, ErfOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Erf;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, ErfcOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Erfc;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, LogOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Log;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, Expm1Op>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::Expm1;
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, LeakyReluOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu;
    paramsType =
        ::tt::target::ttnn::EltwiseUnaryOpParams::EltwiseOpWithFloatParams;
    auto parameter = op.getParameter().convertToFloat();
    params = ::tt::target::ttnn::CreateEltwiseOpWithFloatParams(*cache.fbb,
                                                                parameter)
                 .Union();
  } else if constexpr (std::is_same_v<EltwiseUnaryOp, BitwiseNotOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryOpType::BitwiseNot;
  } else {
    llvm_unreachable("unhandled EltwiseUnaryOp");
  }

  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));

  auto result = op.getResult();

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  auto out =
      cache.getOrCreate(result, tensorValueToFlatbuffer, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateEltwiseUnaryOp(
      *cache.fbb, type, in, memoryConfig, out, paramsType, params);
}

template <typename EltwiseUnaryCompositeOp>
::flatbuffers::Offset<::tt::target::ttnn::EltwiseUnaryCompositeOp>
createEltwiseUnaryCompositeOp(FlatbufferObjectCache &cache,
                              EltwiseUnaryCompositeOp op) {

  ::tt::target::ttnn::EltwiseUnaryCompositeOpType type;
  ::tt::target::ttnn::EltwiseUnaryCompositeOpParams paramsType =
      ::tt::target::ttnn::EltwiseUnaryCompositeOpParams::NONE;
  ::flatbuffers::Offset<void> params = 0;

  if constexpr (std::is_same_v<EltwiseUnaryCompositeOp, CbrtOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Cbrt;
  } else if constexpr (std::is_same_v<EltwiseUnaryCompositeOp, ClampScalarOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampScalar;
    paramsType =
        ::tt::target::ttnn::EltwiseUnaryCompositeOpParams::ClampScalarOpParams;
    auto min = op.getMin().convertToFloat();
    auto max = op.getMax().convertToFloat();
    params = ::tt::target::ttnn::CreateClampScalarOpParams(*cache.fbb, min, max)
                 .Union();
  } else if constexpr (std::is_same_v<EltwiseUnaryCompositeOp, ClampTensorOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor;
    paramsType =
        ::tt::target::ttnn::EltwiseUnaryCompositeOpParams::ClampTensorOpParams;
    auto min = cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(op.getMin()));
    auto max = cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(op.getMax()));
    params = ::tt::target::ttnn::CreateClampTensorOpParams(*cache.fbb, min, max)
                 .Union();
  } else if constexpr (std::is_same_v<EltwiseUnaryCompositeOp, Log1pOp>) {
    type = ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Log1p;
  } else {
    llvm_unreachable("unhandled EltwiseUnaryCompositeOp");
  }

  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));

  auto result = op.getResult();

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  auto out =
      cache.getOrCreate(result, tensorValueToFlatbuffer, kHostAllocatedSize);

  return ::tt::target::ttnn::CreateEltwiseUnaryCompositeOp(
      *cache.fbb, type, in, memoryConfig, out, paramsType, params);
}

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionOp>
createReductionOp(FlatbufferObjectCache &cache, ReductionOp op) {
  ::tt::target::ttnn::ReductionOpType type;
  if constexpr (std::is_same_v<ReductionOp, SumOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Sum;
  } else if constexpr (std::is_same_v<ReductionOp, MeanOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Mean;
  } else if constexpr (std::is_same_v<ReductionOp, MaxOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Max;
  } else if constexpr (std::is_same_v<ReductionOp, MinOp>) {
    type = ::tt::target::ttnn::ReductionOpType::Min;
  } else {
    llvm_unreachable("unhandled ReductionOp");
  }

  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);
  auto dimArg =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int>(cache, op.getDimArg());

  return ::tt::target::ttnn::CreateReductionOp(*cache.fbb, type, in, output,
                                               dimArg, op.getKeepDim());
}

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionArgMaxOp>
createReductionArgMaxOp(FlatbufferObjectCache &cache, ReductionOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  ::flatbuffers::Optional<int32_t> dim = toFlatbuffer(cache, op.getDim());

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  return ::tt::target::ttnn::CreateReductionArgMaxOp(
      *cache.fbb, in, output, dim, op.getKeepDim(), op.getUseMulticore(),
      memoryConfig);
}

template <typename ReductionOp>
::flatbuffers::Offset<::tt::target::ttnn::ReductionProdOp>
createReductionProdOp(FlatbufferObjectCache &cache, ReductionOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto output = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                                  kHostAllocatedSize);

  ::flatbuffers::Optional<int64_t> dimArg = toFlatbuffer(cache, op.getDimArg());

  auto memoryConfig = op.getMemoryConfig()
                          ? toFlatbuffer(cache, op.getMemoryConfig().value())
                          : 0;

  return ::tt::target::ttnn::CreateReductionProdOp(
      *cache.fbb, in, output, dimArg, op.getKeepDim(), memoryConfig);
}

::flatbuffers::Offset<::tt::target::ttnn::TransposeOp>
createTransposeOp(FlatbufferObjectCache &cache, TransposeOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);
  int32_t dim0 = op.getDim0();
  int32_t dim1 = op.getDim1();

  return ::tt::target::ttnn::CreateTransposeOp(*cache.fbb, in, out, dim0, dim1);
}

::flatbuffers::Offset<::tt::target::ttnn::ConcatOp>
createConcatOp(FlatbufferObjectCache &cache, ConcatOp op) {
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> ins;
  for (auto input : op.getInputs()) {
    ins.push_back(cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(input)));
  }

  auto outputType = op.getResult();
  auto out = cache.getOrCreate(outputType, tensorValueToFlatbuffer,
                               kHostAllocatedSize);
  int32_t dim = op.getDim();

  std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();

  return ::tt::target::ttnn::CreateConcatOpDirect(
      *cache.fbb, &ins, out, dim,
      memoryConfig ? toFlatbuffer(cache, memoryConfig.value()) : 0);
}

::flatbuffers::Offset<::tt::target::ttnn::EmbeddingOp>
createEmbeddingOp(FlatbufferObjectCache &cache, EmbeddingOp op) {
  auto in0 = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto in1 = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getWeight()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);
  return ::tt::target::ttnn::CreateEmbeddingOp(*cache.fbb, in0, in1, out);
}

template <typename EmbeddingBackwardOp>
::flatbuffers::Offset<::tt::target::ttnn::EmbeddingBackwardOp>
createEmbeddingBackwardOp(FlatbufferObjectCache &cache,
                          EmbeddingBackwardOp op) {
  auto in0 = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto in1 = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getWeight()));
  auto in2 = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInGradient()));
  ::flatbuffers::Optional<::tt::target::DataType> dtype =
      toFlatbuffer(cache, op.getDtype());
  std::optional<::mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();

  auto outputType = op.getResult();
  auto out = cache.getOrCreate(outputType, tensorValueToFlatbuffer,
                               kHostAllocatedSize);

  return ::tt::target::ttnn::CreateEmbeddingBackwardOp(
      *cache.fbb, in0, in1, in2, dtype,
      memoryConfig ? toFlatbuffer(cache, memoryConfig.value()) : 0, out);
}

::flatbuffers::Offset<::tt::target::ttnn::ReshapeOp>
createReshapeOp(FlatbufferObjectCache &cache, ReshapeOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto shape =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int32_t>(cache, op.getShape());
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);

  std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();

  return ::tt::target::ttnn::CreateReshapeOp(
      *cache.fbb, in, out, shape,
      memoryConfig ? toFlatbuffer(cache, memoryConfig.value()) : 0);
}

template <typename RepeatOp>
::flatbuffers::Offset<::tt::target::ttnn::RepeatOp>
createRepeatOp(FlatbufferObjectCache &cache, RepeatOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  ::llvm::ArrayRef<int64_t> repeatDims = op.getRepeatDims().getShape();
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);

  return ::tt::target::ttnn::CreateRepeatOp(
      *cache.fbb, in, out, cache.fbb->CreateVector<int64_t>(repeatDims));
}

::flatbuffers::Offset<::tt::target::ttnn::PadOp>
createPadOp(FlatbufferObjectCache &cache, PadOp op) {
  flatbuffers::Offset<::tt::target::ttnn::TensorRef> in =
      cache.at<::tt::target::ttnn::TensorRef>(
          getOperandThroughDPSOps(op.getInput()));
  std::vector<uint32_t> padding(op.getPadding().begin(), op.getPadding().end());
  float value = op.getValue().convertToFloat();
  flatbuffers::Offset<::tt::target::ttnn::TensorRef> out = cache.getOrCreate(
      op.getResult(), tensorValueToFlatbuffer, kHostAllocatedSize);

  auto memoryConfig = op.getMemoryConfig().has_value()
                          ? toFlatbuffer(cache, op.getMemoryConfig().value())
                          : 0;
  return ::tt::target::ttnn::CreatePadOp(
      *cache.fbb, in, out, cache.fbb->CreateVector<uint32_t>(padding), value,
      op.getUseMulticore(), memoryConfig);
}

::flatbuffers::Offset<::tt::target::ttnn::SliceOp>
createSliceOp(FlatbufferObjectCache &cache, SliceOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);
  auto begins =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int64_t>(cache, op.getBegins());
  auto ends =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int64_t>(cache, op.getEnds());
  auto step =
      arrayAttrToFlatbuffer<mlir::IntegerAttr, int64_t>(cache, op.getStep());

  return ::tt::target::ttnn::CreateSliceOp(*cache.fbb, in, out, begins, ends,
                                           step);
}

template <typename Pool2dOp>
::flatbuffers::Offset<::tt::target::ttnn::Pool2dOp>
createPool2dOp(FlatbufferObjectCache &cache, Pool2dOp op) {
  ::tt::target::ttnn::Pool2dOpType type;
  if constexpr (std::is_same_v<Pool2dOp, AvgPool2dOp>) {
    type = ::tt::target::ttnn::Pool2dOpType::AvgPool2d;
  } else if constexpr (std::is_same_v<Pool2dOp, MaxPool2dOp>) {
    type = ::tt::target::ttnn::Pool2dOpType::MaxPool2d;
  } else {
    llvm_unreachable("unhandled EltwiseUnaryOp");
  }
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);

  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> kernelSize =
      toFlatbuffer(cache, op.getKernelSize());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> stride =
      toFlatbuffer(cache, op.getStride());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> padding =
      toFlatbuffer(cache, op.getPadding());
  ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> dilation =
      toFlatbuffer(cache, op.getDilation());

  auto memoryConfig = getMemoryConfigIfNeeded(cache, op);

  return ::tt::target::ttnn::CreatePool2dOp(
      *cache.fbb, type, in, out, op.getBatchSize(), op.getInputHeight(),
      op.getInputWidth(), op.getChannels(), kernelSize, stride, padding,
      dilation, memoryConfig, toFlatbuffer(cache, op.getAppliedShardScheme()),
      op.getCeilMode(), op.getInPlaceHalo());
}

::flatbuffers::Offset<::tt::target::ttnn::RepeatInterleaveOp>
createRepeatInterleaveOp(FlatbufferObjectCache &cache, RepeatInterleaveOp op) {
  auto input = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);
  std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig =
      op.getMemoryConfig();
  uint32_t repeats = op.getRepeats();
  int32_t dim = op.getDim();

  return ::tt::target::ttnn::CreateRepeatInterleaveOp(
      *cache.fbb, input, out, repeats, dim,
      memoryConfig ? toFlatbuffer(cache, memoryConfig.value()) : 0);
}

::flatbuffers::Offset<::tt::target::ttnn::SoftmaxOp>
createSoftmaxOp(FlatbufferObjectCache &cache, SoftmaxOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto out = cache.getOrCreate(op.getResult(), tensorValueToFlatbuffer,
                               kHostAllocatedSize);
  int32_t dimension = op.getDimension();

  return ::tt::target::ttnn::CreateSoftmaxOp(*cache.fbb, in, out, dimension);
}

::flatbuffers::Offset<::tt::target::ttnn::DeallocateOp>
createDeallocateOp(FlatbufferObjectCache &cache, DeallocateOp op) {
  auto in = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(op.getInput()));
  auto force = op.getForceAttr().getValue();
  return ::tt::target::ttnn::CreateDeallocateOp(*cache.fbb, in, force);
}

::flatbuffers::Offset<::tt::target::ttnn::LoadCachedOp>
createOp(FlatbufferObjectCache &cache, tt::LoadCachedOp op,
         const llvm::StringMap<uint32_t> &programIndexMap) {
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> ins;
  for (auto input : op.getInputs()) {
    ins.push_back(cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(input)));
  }

  // Collect output tensors
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> outputs;
  for (auto result : op.getResults()) {
    outputs.push_back(
        cache.getOrCreate(result, tensorValueToFlatbuffer, kHostAllocatedSize));
  }

  auto it = programIndexMap.find(op.getCallee().str());
  assert(it != programIndexMap.end() &&
         "Program name not found in program index map!");
  const uint32_t programIdx = it->second;

  // Create the LoadCachedOp with indices instead of inputs
  return ::tt::target::ttnn::CreateLoadCachedOpDirect(
      *cache.fbb, &ins, op.getCallee().str().c_str(), programIdx, &outputs);
}
::flatbuffers::Offset<::tt::target::ttnn::Operation>
emitTTNNOperation(FlatbufferObjectCache &cache, Operation *op,
                  const llvm::StringMap<uint32_t> &programIndexMap,
                  const std::string &debugString, const std::string &locInfo) {
  if (auto getDeviceOp = dyn_cast<GetDeviceOp>(op); getDeviceOp) {
    return createOperation(cache, createOp(cache, getDeviceOp), debugString,
                           locInfo);
  }
  if (auto toMemoryConfigOp = dyn_cast<ToMemoryConfigOp>(op);
      toMemoryConfigOp) {
    return createOperation(cache, createOp(cache, toMemoryConfigOp),
                           debugString, locInfo);
  }
  if (auto toLayoutOp = dyn_cast<ToLayoutOp>(op); toLayoutOp) {
    return createOperation(cache, createOp(cache, toLayoutOp), debugString,
                           locInfo);
  }
  if (auto toDTypeOp = dyn_cast<ToDTypeOp>(op); toDTypeOp) {
    return createOperation(cache, createOp(cache, toDTypeOp), debugString,
                           locInfo);
  }
  if (auto typecastOp = dyn_cast<TypecastOp>(op); typecastOp) {
    return createOperation(cache, createOp(cache, typecastOp), debugString,
                           locInfo);
  }
  if (auto toDeviceOp = dyn_cast<ToDeviceOp>(op); toDeviceOp) {
    return createOperation(cache, createOp(cache, toDeviceOp), debugString,
                           locInfo);
  }
  if (auto fromDeviceOp = dyn_cast<FromDeviceOp>(op); fromDeviceOp) {
    return createOperation(cache, createOp(cache, fromDeviceOp), debugString,
                           locInfo);
  }
  if (auto emptyOp = dyn_cast<EmptyOp>(op); emptyOp) {
    return createOperation(cache, createOp(cache, emptyOp), debugString,
                           locInfo);
  }
  if (auto fullOp = dyn_cast<FullOp>(op); fullOp) {
    return createOperation(cache, createOp(cache, fullOp), debugString,
                           locInfo);
  }
  if (auto arangeOp = dyn_cast<ArangeOp>(op); arangeOp) {
    return createOperation(cache, createOp(cache, arangeOp), debugString,
                           locInfo);
  }
  if (auto zerosOp = dyn_cast<ZerosOp>(op); zerosOp) {
    return createOperation(cache, createNamedFullOp(cache, zerosOp),
                           debugString, locInfo);
  }
  if (auto onesOp = dyn_cast<OnesOp>(op); onesOp) {
    return createOperation(cache, createNamedFullOp(cache, onesOp), debugString,
                           locInfo);
  }
  if (auto addOp = dyn_cast<AddOp>(op); addOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, addOp),
                           debugString, locInfo);
  }
  if (auto multiplyOp = dyn_cast<MultiplyOp>(op); multiplyOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, multiplyOp),
                           debugString, locInfo);
  }
  if (auto subtractOp = dyn_cast<SubtractOp>(op); subtractOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, subtractOp),
                           debugString, locInfo);
  }
  if (auto divOp = dyn_cast<DivideOp>(op); divOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, divOp),
                           debugString, locInfo);
  }
  if (auto eqOp = dyn_cast<EqualOp>(op); eqOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, eqOp),
                           debugString, locInfo);
  }
  if (auto neOp = dyn_cast<NotEqualOp>(op); neOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, neOp),
                           debugString, locInfo);
  }
  if (auto geOp = dyn_cast<GreaterEqualOp>(op); geOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, geOp),
                           debugString, locInfo);
  }
  if (auto gtOp = dyn_cast<GreaterThanOp>(op); gtOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, gtOp),
                           debugString, locInfo);
  }
  if (auto leOp = dyn_cast<LessEqualOp>(op); leOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, leOp),
                           debugString, locInfo);
  }
  if (auto ltOp = dyn_cast<LessThanOp>(op); ltOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, ltOp),
                           debugString, locInfo);
  }
  if (auto logicalAndOp = dyn_cast<LogicalAndOp>(op); logicalAndOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, logicalAndOp),
                           debugString, locInfo);
  }
  if (auto logicalOrOp = dyn_cast<LogicalOrOp>(op); logicalOrOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, logicalOrOp),
                           debugString, locInfo);
  }
  if (auto logicalXorOp = dyn_cast<LogicalXorOp>(op); logicalXorOp) {
    return createOperation(cache, createEltwiseBinaryOp(cache, logicalXorOp),
                           debugString, locInfo);
  }
  if (auto bitwiseAndOp = dyn_cast<BitwiseAndOp>(op); bitwiseAndOp) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, bitwiseAndOp),
                           debugString, locInfo);
  }
  if (auto bitwiseOrOp = dyn_cast<BitwiseOrOp>(op); bitwiseOrOp) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, bitwiseOrOp),
                           debugString, locInfo);
  }
  if (auto bitwiseXorOp = dyn_cast<BitwiseXorOp>(op); bitwiseXorOp) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, bitwiseXorOp),
                           debugString, locInfo);
  }
  if (auto maximumOp = dyn_cast<MaximumOp>(op); maximumOp) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, maximumOp),
                           debugString, locInfo);
  }
  if (auto minimumOp = dyn_cast<MinimumOp>(op); minimumOp) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, minimumOp),
                           debugString, locInfo);
  }
  if (auto powOp = dyn_cast<PowOp>(op); powOp) {
    return createOperation(cache, createEltwiseBinaryCompositeOp(cache, powOp),
                           debugString, locInfo);
  }
  if (auto remainderOp = dyn_cast<RemainderOp>(op); remainderOp) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, remainderOp),
                           debugString, locInfo);
  }
  if (auto scatterOp = dyn_cast<ScatterOp>(op); scatterOp) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, scatterOp),
                           debugString, locInfo);
  }
  if (auto atan2Op = dyn_cast<Atan2Op>(op); atan2Op) {
    return createOperation(cache,
                           createEltwiseBinaryCompositeOp(cache, atan2Op),
                           debugString, locInfo);
  }
  if (auto whereOp = dyn_cast<WhereOp>(op); whereOp) {
    return createOperation(cache, createEltwiseTernaryWhereOp(cache, whereOp),
                           debugString, locInfo);
  }
  if (auto quantizeOp = dyn_cast<QuantizeOp>(op); quantizeOp) {
    return createOperation(cache,
                           createEltwiseQuantizationOp(cache, quantizeOp),
                           debugString, locInfo);
  }
  if (auto dequantizeOp = dyn_cast<DequantizeOp>(op); dequantizeOp) {
    return createOperation(cache,
                           createEltwiseQuantizationOp(cache, dequantizeOp),
                           debugString, locInfo);
  }
  if (auto requantizeOp = dyn_cast<RequantizeOp>(op); requantizeOp) {
    return createOperation(cache,
                           createEltwiseQuantizationOp(cache, requantizeOp),
                           debugString, locInfo);
  }
  if (auto absOp = dyn_cast<AbsOp>(op); absOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, absOp),
                           debugString, locInfo);
  }
  if (auto floorOp = dyn_cast<FloorOp>(op); floorOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, floorOp),
                           debugString, locInfo);
  }
  if (auto isFiniteOp = dyn_cast<IsFiniteOp>(op); isFiniteOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, isFiniteOp),
                           debugString, locInfo);
  }
  if (auto logicalNotOp = dyn_cast<LogicalNotOp>(op); logicalNotOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, logicalNotOp),
                           debugString, locInfo);
  }
  if (auto bitwiseNotOp = dyn_cast<BitwiseNotOp>(op); bitwiseNotOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, bitwiseNotOp),
                           debugString, locInfo);
  }
  if (auto negOp = dyn_cast<NegOp>(op); negOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, negOp),
                           debugString, locInfo);
  }
  if (auto reluOp = dyn_cast<ReluOp>(op); reluOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, reluOp),
                           debugString, locInfo);
  }
  if (auto sqrtOp = dyn_cast<SqrtOp>(op); sqrtOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, sqrtOp),
                           debugString, locInfo);
  }
  if (auto rsqrtOp = dyn_cast<RsqrtOp>(op); rsqrtOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, rsqrtOp),
                           debugString, locInfo);
  }
  if (auto signOp = dyn_cast<SignOp>(op); signOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, signOp),
                           debugString, locInfo);
  }
  if (auto expOp = dyn_cast<ExpOp>(op); expOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, expOp),
                           debugString, locInfo);
  }
  if (auto erfOp = dyn_cast<ErfOp>(op); erfOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, erfOp),
                           debugString, locInfo);
  }
  if (auto erfcOp = dyn_cast<ErfcOp>(op); erfcOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, erfcOp),
                           debugString, locInfo);
  }
  if (auto logOp = dyn_cast<LogOp>(op); logOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, logOp),
                           debugString, locInfo);
  }
  if (auto expm1Op = dyn_cast<Expm1Op>(op); expm1Op) {
    return createOperation(cache, createEltwiseUnaryOp(cache, expm1Op),
                           debugString, locInfo);
  }
  if (auto sigmoidOp = dyn_cast<SigmoidOp>(op); sigmoidOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, sigmoidOp),
                           debugString, locInfo);
  }
  if (auto reciprocalOp = dyn_cast<ReciprocalOp>(op); reciprocalOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, reciprocalOp),
                           debugString, locInfo);
  }
  if (auto leakyReluOp = dyn_cast<LeakyReluOp>(op); leakyReluOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, leakyReluOp),
                           debugString, locInfo);
  }
  if (auto ceilOp = dyn_cast<CeilOp>(op); ceilOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, ceilOp),
                           debugString, locInfo);
  }
  if (auto cosOp = dyn_cast<CosOp>(op); cosOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, cosOp),
                           debugString, locInfo);
  }
  if (auto sinOp = dyn_cast<SinOp>(op); sinOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, sinOp),
                           debugString, locInfo);
  }
  if (auto geluOp = dyn_cast<GeluOp>(op); geluOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, geluOp),
                           debugString, locInfo);
  }
  if (auto tanOp = dyn_cast<TanOp>(op); tanOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, tanOp),
                           debugString, locInfo);
  }
  if (auto tanhOp = dyn_cast<TanhOp>(op); tanhOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, tanhOp),
                           debugString, locInfo);
  }
  if (auto atanOp = dyn_cast<AtanOp>(op); atanOp) {
    return createOperation(cache, createEltwiseUnaryOp(cache, atanOp),
                           debugString, locInfo);
  }
  if (auto cbrtOp = dyn_cast<CbrtOp>(op); cbrtOp) {
    return createOperation(cache, createEltwiseUnaryCompositeOp(cache, cbrtOp),
                           debugString, locInfo);
  }
  if (auto log1pOp = dyn_cast<Log1pOp>(op); log1pOp) {
    return createOperation(cache, createEltwiseUnaryCompositeOp(cache, log1pOp),
                           debugString, locInfo);
  }
  if (auto clampScalarOp = dyn_cast<ClampScalarOp>(op); clampScalarOp) {
    return createOperation(cache,
                           createEltwiseUnaryCompositeOp(cache, clampScalarOp),
                           debugString, locInfo);
  }
  if (auto clampTensorOp = dyn_cast<ClampTensorOp>(op); clampTensorOp) {
    return createOperation(cache,
                           createEltwiseUnaryCompositeOp(cache, clampTensorOp),
                           debugString, locInfo);
  }
  if (auto linearOp = dyn_cast<LinearOp>(op); linearOp) {
    return createOperation(cache, createOp(cache, linearOp), debugString,
                           locInfo);
  }
  if (auto matmulOp = dyn_cast<MatmulOp>(op); matmulOp) {
    return createOperation(cache, createOp(cache, matmulOp), debugString,
                           locInfo);
  }
  if (auto morehCumSumOp = dyn_cast<MorehCumSumOp>(op); morehCumSumOp) {
    return createOperation(cache, createOp(cache, morehCumSumOp), debugString,
                           locInfo);
  }
  if (auto sumOp = dyn_cast<SumOp>(op); sumOp) {
    return createOperation(cache, createReductionOp(cache, sumOp), debugString,
                           locInfo);
  }
  if (auto meanOp = dyn_cast<MeanOp>(op); meanOp) {
    return createOperation(cache, createReductionOp(cache, meanOp), debugString,
                           locInfo);
  }
  if (auto maxOp = dyn_cast<MaxOp>(op); maxOp) {
    return createOperation(cache, createReductionOp(cache, maxOp), debugString,
                           locInfo);
  }
  if (auto minOp = dyn_cast<MinOp>(op); minOp) {
    return createOperation(cache, createReductionOp(cache, minOp), debugString,
                           locInfo);
  }
  if (auto argMaxOp = dyn_cast<ArgMaxOp>(op); argMaxOp) {
    return createOperation(cache, createReductionArgMaxOp(cache, argMaxOp),
                           debugString, locInfo);
  }
  if (auto prodOp = dyn_cast<ProdOp>(op); prodOp) {
    return createOperation(cache, createReductionProdOp(cache, prodOp),
                           debugString, locInfo);
  }
  if (auto embeddingOp = dyn_cast<EmbeddingOp>(op); embeddingOp) {
    return createOperation(cache, createEmbeddingOp(cache, embeddingOp),
                           debugString, locInfo);
  }
  if (auto embeddingBackwardOp = dyn_cast<EmbeddingBackwardOp>(op);
      embeddingBackwardOp) {
    return createOperation(
        cache, createEmbeddingBackwardOp(cache, embeddingBackwardOp),
        debugString, locInfo);
  }
  if (auto repeatInterleaveOp = dyn_cast<RepeatInterleaveOp>(op);
      repeatInterleaveOp) {
    return createOperation(cache,
                           createRepeatInterleaveOp(cache, repeatInterleaveOp),
                           debugString, locInfo);
  }
  if (auto softmaxOp = dyn_cast<SoftmaxOp>(op); softmaxOp) {
    return createOperation(cache, createSoftmaxOp(cache, softmaxOp),
                           debugString, locInfo);
  }
  if (auto transposeOp = dyn_cast<TransposeOp>(op); transposeOp) {
    return createOperation(cache, createTransposeOp(cache, transposeOp),
                           debugString, locInfo);
  }
  if (auto prepareConv2dWeightsOp = dyn_cast<PrepareConv2dWeightsOp>(op);
      prepareConv2dWeightsOp) {
    return createOperation(cache, createOp(cache, prepareConv2dWeightsOp),
                           debugString, locInfo);
  }
  if (auto conv2dOp = dyn_cast<Conv2dOp>(op); conv2dOp) {
    return createOperation(cache, createOp(cache, conv2dOp), debugString,
                           locInfo);
  }
  if (auto conv_transpose2dOp = dyn_cast<ConvTranspose2dOp>(op);
      conv_transpose2dOp) {
    return createOperation(cache, createOp(cache, conv_transpose2dOp),
                           debugString, locInfo);
  }
  if (auto allGatherOp = dyn_cast<AllGatherOp>(op); allGatherOp) {
    return createOperation(cache, createOp(cache, allGatherOp), debugString,
                           locInfo);
  }
  if (auto reduceScatterOp = dyn_cast<ReduceScatterOp>(op); reduceScatterOp) {
    return createOperation(cache, createOp(cache, reduceScatterOp), debugString,
                           locInfo);
  }
  if (auto collectivePermuteOp = dyn_cast<CollectivePermuteOp>(op);
      collectivePermuteOp) {
    return createOperation(cache, createOp(cache, collectivePermuteOp),
                           debugString, locInfo);
  }
  if (auto meshShardOp = dyn_cast<MeshShardOp>(op); meshShardOp) {
    return createOperation(cache, createOp(cache, meshShardOp), debugString,
                           locInfo);
  }
  if (auto concatOp = dyn_cast<ConcatOp>(op); concatOp) {
    return createOperation(cache, createConcatOp(cache, concatOp), debugString,
                           locInfo);
  }
  if (auto reshapeOp = dyn_cast<ReshapeOp>(op); reshapeOp) {
    return createOperation(cache, createReshapeOp(cache, reshapeOp),
                           debugString, locInfo);
  }
  if (auto repeatOp = dyn_cast<RepeatOp>(op); repeatOp) {
    return createOperation(cache, createRepeatOp(cache, repeatOp), debugString,
                           locInfo);
  }
  if (auto padOp = dyn_cast<PadOp>(op); padOp) {
    return createOperation(cache, createPadOp(cache, padOp), debugString,
                           locInfo);
  }
  if (auto sliceOp = dyn_cast<SliceOp>(op); sliceOp) {
    return createOperation(cache, createSliceOp(cache, sliceOp), debugString,
                           locInfo);
  }
  if (auto avg_pool2dOp = dyn_cast<AvgPool2dOp>(op); avg_pool2dOp) {
    return createOperation(cache, createPool2dOp(cache, avg_pool2dOp),
                           debugString, locInfo);
  }
  if (auto max_pool2dOp = dyn_cast<MaxPool2dOp>(op); max_pool2dOp) {
    return createOperation(cache, createPool2dOp(cache, max_pool2dOp),
                           debugString, locInfo);
  }
  if (auto deallocateOp = dyn_cast<DeallocateOp>(op); deallocateOp) {
    return createOperation(cache, createDeallocateOp(cache, deallocateOp),
                           debugString, locInfo);
  }
  if (auto updateCacheOp = dyn_cast<UpdateCacheOp>(op); updateCacheOp) {
    return createOperation(cache, createOp(cache, updateCacheOp), debugString,
                           locInfo);
  }
  if (auto fillCacheOp = dyn_cast<FillCacheOp>(op); fillCacheOp) {
    return createOperation(cache, createOp(cache, fillCacheOp), debugString,
                           locInfo);
  }
  if (auto permuteOp = dyn_cast<PermuteOp>(op); permuteOp) {
    return createOperation(cache, createOp(cache, permuteOp), debugString,
                           locInfo);
  }
  if (auto upsampleOp = dyn_cast<UpsampleOp>(op); upsampleOp) {
    return createOperation(cache, createOp(cache, upsampleOp), debugString,
                           locInfo);
  }
  if (auto batchNormOp = dyn_cast<BatchNormOp>(op); batchNormOp) {
    return createOperation(cache, createOp(cache, batchNormOp), debugString,
                           locInfo);
  }
  if (auto constantOp = dyn_cast<ConstantOp>(op); constantOp) {
    return createOperation(cache, createOp(cache, constantOp), debugString,
                           locInfo);
  }
  if (auto callOp = dyn_cast<func::CallOp>(op); callOp) {
    // TODO (#2355): Here dylib_id is hardcoded to 0.  In the long run, we want
    // to support multiple dylibs per flatbuffer, but the exact schema is not so
    // clear.
    return createOperation(cache, createCpuOp(cache, callOp, 0), debugString,
                           locInfo);
  }
  if (auto loadCachedOp = dyn_cast<tt::LoadCachedOp>(op); loadCachedOp) {
    return createOperation(cache,
                           createOp(cache, loadCachedOp, programIndexMap),
                           debugString, locInfo);
  }

  llvm_unreachable("unhandled op in emitTTNNOperation");
}

std::shared_ptr<void> ttnnToFlatbuffer(
    Operation *op,
    const std::unordered_map<std::string, GoldenTensor> &goldenMap,
    const std::vector<std::pair<std::string, std::string>> &moduleCache) {
  ModuleOp rootModule = dyn_cast<ModuleOp>(op);
  assert(rootModule && "Expected ModuleOp as top level operation");

  // If we have a nested module structure, we want to use nested module inside
  // DeviceModule for most conversions.
  ModuleOp module = rootModule;
  if (auto deviceModule =
          mlir::tt::utils::findOpAtTopLevel<tt::DeviceModuleOp>(module)) {
    module = dyn_cast_if_present<mlir::ModuleOp>(
        deviceModule.getBodyRegion().front().front());
    assert(module && "Found tt::DeviceModuleOp but it didn't contain a single "
                     "mlir::ModuleOp!");
  }

  ::flatbuffers::FlatBufferBuilder fbb;
  FlatbufferObjectCache cache(&fbb);

  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                      ttmlirVersion.patch);

  auto systemDesc =
      toFlatbuffer(cache, mlir::cast<tt::SystemDescAttr>(
                              module->getAttr(tt::SystemDescAttr::name)));

  std::string cpp;
  llvm::raw_string_ostream os(cpp);
  auto result = mlir::tt::ttnn::emitTTNNAsCpp(module, os);
  (void)result;

  flatbuffers::Offset<::tt::target::DebugInfo> debugInfo =
      debugInfoToFlatbuffer(fbb, "ttnn", rootModule, goldenMap, moduleCache,
                            cpp.c_str());

  // Handle dylib creation and packaging, if needed.
  // Currently, we only have 1 CPUModuleOp and 1 top-level ModuleOp; we use a
  // vector here in case in the future we support more complex arrangements.
  std::vector<::flatbuffers::Offset<::tt::target::DynamicLib>> dylibs;
  if (auto cpuModule =
          mlir::tt::utils::findOpAtTopLevel<tt::CPUModuleOp>(rootModule);
      cpuModule != nullptr) {
    mlir::ModuleOp cpuNestedModule = dyn_cast_if_present<mlir::ModuleOp>(
        cpuModule.getBodyRegion().front().front());
    llvm::SmallVector<char, 2048> binaryBuffer;
    llvm::raw_svector_ostream dylibStream(binaryBuffer);
    auto result = mlir::tt::llvm_to_cpu::translateLLVMToDyLib(cpuNestedModule,
                                                              dylibStream);
    if (llvm::succeeded(result)) {
      auto rawFileVector = fbb.CreateVector(
          reinterpret_cast<const uint8_t *>(binaryBuffer.data()),
          binaryBuffer.size());
      dylibs.emplace_back(
          ::tt::target::CreateDynamicLib(fbb, 0, rawFileVector));
    }
  }

  size_t programIdx = 0;
  llvm::StringMap<uint32_t> programIdxMap;
  // Preserve original ordering by skipping const-eval in the first pass.
  module->walk([&](func::FuncOp func) {
    if (ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }
    programIdxMap[func.getSymName().str()] = programIdx++;
  });

  // Add const-eval funcs after normal funcs.
  module->walk([&](func::FuncOp func) {
    if (!ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }
    programIdxMap[func.getSymName().str()] = programIdx++;
  });

  std::vector<::flatbuffers::Offset<::tt::target::ttnn::Program>> programs;
  // Again, process original funcs in order first to perserve input order.
  module->walk([&](func::FuncOp func) {
    if (ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }
    Program<::tt::target::ttnn::Operation> program =
        funcOpToProgram<::tt::target::ttnn::Operation>(
            cache, func, emitTTNNOperation, tensorValueToFlatbuffer,
            programIdxMap);
    programs.push_back(::tt::target::ttnn::CreateProgramDirect(
        fbb, program.name, &program.inputs, &program.outputs, &program.ops,
        &dylibs, debugInfo, /*private=*/false));
  });
  // Then process const-eval funcs in 2nd pass.
  module->walk([&](func::FuncOp func) {
    if (!ttmlir::utils::isConstEvalFunc(func)) {
      return;
    }
    Program<::tt::target::ttnn::Operation> program =
        funcOpToProgram<::tt::target::ttnn::Operation>(
            cache, func, emitTTNNOperation, tensorValueToFlatbuffer,
            programIdxMap);
    programs.push_back(::tt::target::ttnn::CreateProgramDirect(
        fbb, program.name, &program.inputs, &program.outputs, &program.ops,
        &dylibs, debugInfo, /*private=*/true));
  });

  auto binary = ::tt::target::ttnn::CreateTTNNBinaryDirect(
      fbb, &binaryVersion, ::ttmlir::getGitHash(), systemDesc, &programs);

  ::tt::target::ttnn::FinishSizePrefixedTTNNBinaryBuffer(fbb, binary);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  ::tt::target::ttnn::VerifySizePrefixedTTNNBinaryBuffer(verifier);

  uint8_t *buf = fbb.GetBufferPointer();
  std::size_t size = fbb.GetSize();

  std::shared_ptr<void> bufferPtr =
      std::shared_ptr<void>(std::malloc(size), std::free);
  std::memcpy(bufferPtr.get(), buf, size);
  return bufferPtr;
}

LogicalResult translateTTNNToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    const std::unordered_map<std::string, GoldenTensor> &goldenMap,
    const std::vector<std::pair<std::string, std::string>> &moduleCache) {
  std::shared_ptr<void> data = ttnnToFlatbuffer(op, goldenMap, moduleCache);
  std::size_t size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(data.get()));
  os.write(reinterpret_cast<const char *>(data.get()), size);
  return success();
}
} // namespace mlir::tt::ttnn
