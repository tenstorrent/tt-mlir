// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"
#include "ttmlir/Dialect/TTMetal/Transforms/KernelsToCpp.h"
#include "ttmlir/Target/TTTarget.h"
#include "ttmlir/Version.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_TTMETALSERIALIZETOBINARY
#include "ttmlir/Dialect/TTMetal/Passes.h.inc"

struct FlatbufferObjectCache {
  ::flatbuffers::FlatBufferBuilder *fbb;
  DenseMap<void const *, ::flatbuffers::uoffset_t> objectMap;
  uint32_t global_id = 0;

  FlatbufferObjectCache(::flatbuffers::FlatBufferBuilder *fbb) : fbb(fbb) {}

  template <typename T> struct offset_extract_t;
  template <template <typename> typename OffsetT, typename T>
  struct offset_extract_t<OffsetT<T>> {
    using type = T;
  };

  template <typename MLIRTypeOrAttr> bool exists(MLIRTypeOrAttr obj) const {
    return objectMap.contains(obj.getAsOpaquePointer());
  }

  template <typename MLIRTypeOrAttr, typename SchemaType>
  flatbuffers::Offset<SchemaType>
  insert(MLIRTypeOrAttr obj, flatbuffers::Offset<SchemaType> offset) {
    assert(!exists(obj) && "object already exists");
    objectMap.insert(std::make_pair(obj.getAsOpaquePointer(), offset.o));
    return offset;
  }

  template <typename SchemaType, typename MLIRTypeOrAttr>
  flatbuffers::Offset<SchemaType> at(MLIRTypeOrAttr obj) const {
    assert(exists(obj) && "object does not exist");
    return flatbuffers::Offset<SchemaType>(
        objectMap.at(obj.getAsOpaquePointer()));
  }

  template <typename MLIRTypeOrAttr, typename CreateFn, typename... Args>
  std::invoke_result_t<CreateFn, FlatbufferObjectCache &, MLIRTypeOrAttr,
                       Args...>
  getOrCreate(MLIRTypeOrAttr obj, CreateFn createFn, Args... args) {
    using SchemaType = typename offset_extract_t<std::invoke_result_t<
        CreateFn, FlatbufferObjectCache &, MLIRTypeOrAttr, Args...>>::type;

    if (exists(obj))
      return at<SchemaType, MLIRTypeOrAttr>(obj);
    return insert(obj, createFn(*this, obj, args...));
  }
};

struct CQBuilder {
  ::flatbuffers::FlatBufferBuilder *fbb;
  const char *name;
  std::vector<::flatbuffers::Offset<::tt::TensorRef>> inputs;
  std::vector<::flatbuffers::Offset<::tt::TensorRef>> outputs;
  std::vector<::flatbuffers::Offset<::tt::Command>> commands;
  OpPrintingFlags printFlags;

  CQBuilder(::flatbuffers::FlatBufferBuilder *fbb) : fbb(fbb) {
    printFlags = printFlags.elideLargeElementsAttrs()
                     .elideLargeResourceString()
                     .skipRegions()
                     .enableDebugInfo();
  }

  std::string getDebugString(mlir::Operation *op) {
    std::string str;
    llvm::raw_string_ostream os(str);
    op->print(os, printFlags);
    return str;
  };

  template <typename CommandT>
  ::flatbuffers::Offset<::tt::Command>
  appendCommand(::flatbuffers::Offset<CommandT> commandT, mlir::Operation *op) {
    auto debugString = getDebugString(op);
    commands.push_back(::tt::CreateCommandDirect(
        *fbb, ::tt::CommandTypeTraits<CommandT>::enum_value, commandT.Union(),
        debugString.c_str()));
    return commands.back();
  }
};

::tt::OOBVal toFlatbuffer(OOBVal oobVal) {
  switch (oobVal) {
  case OOBVal::Undef:
    return ::tt::OOBVal::Undef;
  case OOBVal::Zero:
    return ::tt::OOBVal::Zero;
  case OOBVal::One:
    return ::tt::OOBVal::One;
  case OOBVal::Inf:
    return ::tt::OOBVal::Inf;
  case OOBVal::NegInf:
    return ::tt::OOBVal::NegInf;
  }
}

::tt::DataType toFlatbuffer(DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return ::tt::DataType::Float32;
  case DataType::Float16:
    return ::tt::DataType::Float16;
  case DataType::BFloat16:
    return ::tt::DataType::BFloat16;
  case DataType::BC_Float8:
    return ::tt::DataType::BC_Float8;
  case DataType::BC_BFloat8:
    return ::tt::DataType::BC_BFloat8;
  case DataType::BC_Float4:
    return ::tt::DataType::BC_Float4;
  case DataType::BC_BFloat4:
    return ::tt::DataType::BC_BFloat4;
  case DataType::BC_Float2:
    return ::tt::DataType::BC_Float2;
  case DataType::BC_BFloat2:
    return ::tt::DataType::BC_BFloat2;
  case DataType::UInt32:
    return ::tt::DataType::UInt32;
  case DataType::UInt16:
    return ::tt::DataType::UInt16;
  case DataType::UInt8:
    return ::tt::DataType::UInt8;
  }
}

::tt::MemorySpace toFlatbuffer(MemorySpace memspace) {
  switch (memspace) {
  case MemorySpace::System:
    return ::tt::MemorySpace::System;
  case MemorySpace::SystemMMIO:
    return ::tt::MemorySpace::SystemMMIO;
  case MemorySpace::DeviceDRAM:
    return ::tt::MemorySpace::DeviceDRAM;
  case MemorySpace::DeviceL1:
    return ::tt::MemorySpace::DeviceL1;
  }
}

::tt::SourceType toFlatbuffer(ttkernel::ThreadType threadType) {
  switch (threadType) {
  case ttkernel::ThreadType::Noc0:
    return ::tt::SourceType::Noc0;
  case ttkernel::ThreadType::Noc1:
    return ::tt::SourceType::Noc1;
  case ttkernel::ThreadType::Tensix:
    return ::tt::SourceType::Tensix;
  case ttkernel::ThreadType::Ethernet:
    return ::tt::SourceType::Ethernet;
  }
}

::tt::Dim2dRange toFlatbuffer(CoreRangeAttr coreRange) {
  auto offset = coreRange.getOffset();
  auto size = coreRange.getSize();
  return ::tt::Dim2dRange(::tt::Dim2d(offset[0], offset[1]),
                          ::tt::Dim2d(size[0], size[1]));
}

inline DataType elementTypeToDataType(Type elementType) {
  DataType dtype = DataType::Float32;
  if (isa<FloatType>(elementType)) {
    auto floatType = elementType.cast<FloatType>();
    if (floatType.getWidth() == 32) {
      dtype = DataType::Float32;
    } else if (floatType.getWidth() == 16) {
      dtype = DataType::Float16;
    } else {
      assert(false && "unsupported float type");
    }
  } else if (isa<IntegerType>(elementType)) {
    auto intType = elementType.cast<IntegerType>();
    if (intType.getWidth() == 32) {
      dtype = DataType::UInt32;
    } else if (intType.getWidth() == 16) {
      dtype = DataType::UInt16;
    } else if (intType.getWidth() == 8) {
      dtype = DataType::UInt8;
    } else {
      assert(false && "unsupported integer type");
    }
  }
  return dtype;
}

inline flatbuffers::Offset<::tt::MemoryDesc>
memrefAttrToFlatbuffer(FlatbufferObjectCache &cache, MemRefType memref) {
  auto shapeInt64 = memref.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  DataType dtype = DataType::Float32;
  ::tt::Dim2d tileShape(0, 0);
  Type elementType = memref.getElementType();
  if (isa<TileType>(elementType)) {
    auto tileType = elementType.cast<TileType>();
    dtype = tileType.getDataType();
    tileShape = ::tt::Dim2d(tileType.getHeight(), tileType.getWidth());
  } else {
    dtype = elementTypeToDataType(elementType);
  }

  return ::tt::CreateMemoryDescDirect(
      *cache.fbb, &shape, &tileShape, toFlatbuffer(dtype),
      toFlatbuffer(memref.getMemorySpace().cast<MemorySpaceAttr>().getValue()));
}

inline flatbuffers::Offset<::tt::LayoutDesc>
layoutAttrToFlatbuffer(FlatbufferObjectCache &cache, Attribute attr) {
  assert(attr.isa<LayoutAttr>() && "expected a tensor type");
  auto layoutAttr = attr.cast<LayoutAttr>();
  auto stridesInt64 = layoutAttr.getStrides();
  std::vector<int32_t> strides(stridesInt64.begin(), stridesInt64.end());
  auto gridAttr = layoutAttr.getGrid();
  auto gridShape = gridAttr.getShape();
  assert(gridShape.size() == 2 && "expected a 2D grid");
  ::tt::Dim2dRange grid(::tt::Dim2d(0, 0),
                        ::tt::Dim2d(gridShape[0], gridShape[1]));
  return ::tt::CreateLayoutDescDirect(
      *cache.fbb, &strides, toFlatbuffer(layoutAttr.getOobVal()), &grid,
      cache.getOrCreate(layoutAttr.getMemref(), memrefAttrToFlatbuffer));
}

inline flatbuffers::Offset<::tt::TensorDesc>
tensorTypeToFlatbuffer(FlatbufferObjectCache &cache, Type type) {
  auto tensorType = type.cast<RankedTensorType>();
  auto shapeInt64 = tensorType.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  return ::tt::CreateTensorDescDirect(
      *cache.fbb, &shape,
      cache.getOrCreate(tensorType.getEncoding(), layoutAttrToFlatbuffer));
}

inline flatbuffers::Offset<::tt::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        uint64_t address, uint64_t size) {
  auto tensorType = value.getType().cast<RankedTensorType>();
  auto tensorDesc = cache.getOrCreate(tensorType, tensorTypeToFlatbuffer);
  return ::tt::CreateTensorRef(*cache.fbb, cache.global_id++, address, size,
                               tensorDesc);
}

class TTMetalSerializeToBinary
    : public impl::TTMetalSerializeToBinaryBase<TTMetalSerializeToBinary> {
public:
  using impl::TTMetalSerializeToBinaryBase<
      TTMetalSerializeToBinary>::TTMetalSerializeToBinaryBase;

  Value getOperandThroughDPSOps(Value value) {
    auto *op = value.getDefiningOp();
    if (!op)
      return value;
    while (isa<DestinationStyleOpInterface>(op)) {
      assert(op->getResults().size() == 1);
      auto dps = cast<DestinationStyleOpInterface>(op);
      assert(dps.getNumDpsInits() == 1);
      auto opOperand = dps.getDpsInitOperand(0);
      value = opOperand->get();
      op = value.getDefiningOp();
    }
    return value;
  }

  void runOnOperation() final {
    constexpr uint64_t kHostAllocatedAddress = 0;
    constexpr uint64_t kHostAllocatedSize = 0;

    ::flatbuffers::FlatBufferBuilder fbb;
    FlatbufferObjectCache cache(&fbb);
    CQBuilder cqBuilder(&fbb);

    ModuleOp module = getOperation();
    func::FuncOp entry = dyn_cast<func::FuncOp>(*module.getRegion().op_begin());
    assert(entry && "expected an entry function");
    cqBuilder.name = entry.getSymName().data();

    for (auto &input : entry.getBody().getArguments()) {
      cqBuilder.inputs.push_back(
          cache.getOrCreate(input, tensorValueToFlatbuffer,
                            kHostAllocatedAddress, kHostAllocatedSize));
    }

    module->walk([&](mlir::Operation *op) {
      if (auto dispatchOp = dyn_cast_or_null<tt::ttmetal::DispatchOp>(op);
          dispatchOp) {
        std::vector<::flatbuffers::Offset<::tt::TensorRef>> operands;
        for (auto operand : dispatchOp.getOperands()) {
          operands.push_back(
              cache.at<::tt::TensorRef>(getOperandThroughDPSOps(operand)));
        }

        std::vector<::flatbuffers::Offset<::tt::KernelDesc>> kernels;
        for (auto &region : dispatchOp.getRegions()) {
          std::string source;
          llvm::raw_string_ostream os(source);
          auto result = emitDispatchOpRegionAsCpp(dispatchOp,
                                                  region.getRegionNumber(), os);
          assert(succeeded(result) &&
                 "failed to emit dispatch op region as cpp");
          auto threadType =
              dispatchOp.getThreadTypes()[region.getRegionNumber()]
                  .cast<ttkernel::ThreadTypeAttr>()
                  .getValue();
          ::tt::Dim2dRange core_range =
              toFlatbuffer(dispatchOp.getCoreRanges()[region.getRegionNumber()]
                               .cast<CoreRangeAttr>());
          ::tt::Dim2dRange(::tt::Dim2d(0, 0), ::tt::Dim2d(0, 0));
          std::vector<::flatbuffers::Offset<::tt::CBRef>> cbs;
          kernels.push_back(::tt::CreateKernelDescDirect(
              fbb, ::tt::Kernel::KernelSource,
              ::tt::CreateKernelSourceDirect(fbb, toFlatbuffer(threadType),
                                             source.c_str())
                  .Union(),
              &core_range, &cbs, nullptr /*TODO debug info*/));
        }
        std::vector<::flatbuffers::Offset<::tt::DispatchProgram>> programs = {
            ::tt::CreateDispatchProgramDirect(fbb, &kernels),
        };

        cqBuilder.appendCommand(
            ::tt::CreateDispatchCommandDirect(fbb, &operands, &programs), op);
      } else if (auto allocOp = dyn_cast_or_null<tt::ttmetal::AllocOp>(op);
                 allocOp) {
        cqBuilder.appendCommand(
            ::tt::CreateHostAllocCommand(
                fbb,
                cache.getOrCreate(allocOp.getResult(), tensorValueToFlatbuffer,
                                  allocOp.getAddress(), allocOp.getSize())),
            op);
      } else if (auto deallocOp = dyn_cast_or_null<tt::ttmetal::DeallocOp>(op);
                 deallocOp) {
        cqBuilder.appendCommand(
            ::tt::CreateHostDeallocCommand(
                fbb, cache.at<::tt::TensorRef>(
                         getOperandThroughDPSOps(deallocOp.getInput()))),
            op);
      } else if (auto hostReadOp =
                     dyn_cast_or_null<tt::ttmetal::HostReadOp>(op);
                 hostReadOp) {
        cqBuilder.appendCommand(
            ::tt::CreateHostReadCommand(
                fbb,
                cache.at<::tt::TensorRef>(
                    getOperandThroughDPSOps(hostReadOp.getInput())),
                cache.at<::tt::TensorRef>(
                    getOperandThroughDPSOps(hostReadOp.getOutput()))),
            op);
      } else if (auto hostWriteOp =
                     dyn_cast_or_null<tt::ttmetal::HostWriteOp>(op);
                 hostWriteOp) {
        cqBuilder.appendCommand(
            ::tt::CreateHostReadCommand(
                fbb,
                cache.at<::tt::TensorRef>(
                    getOperandThroughDPSOps(hostWriteOp.getInput())),
                cache.at<::tt::TensorRef>(
                    getOperandThroughDPSOps(hostWriteOp.getOutput()))),
            op);
      } else if (auto returnOp = dyn_cast_or_null<func::ReturnOp>(op);
                 returnOp) {
        for (auto output : returnOp.getOperands()) {
          cqBuilder.outputs.push_back(
              cache.at<::tt::TensorRef>(getOperandThroughDPSOps(output)));
        }
      }
    });

    ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
    ::tt::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.release);

    ::tt::Dim2d deviceGrid(8, 8);
    auto deviceDesc =
        ::tt::CreateDeviceDesc(fbb, ::tt::DeviceArch::Wormhole_b0, &deviceGrid);

    std::vector<::flatbuffers::Offset<::tt::CommandQueue>> commandQueues = {
        ::tt::CreateCommandQueueDirect(fbb, cqBuilder.name, &cqBuilder.inputs,
                                       &cqBuilder.outputs, &cqBuilder.commands),
    };
    auto binary =
        ::tt::CreateBinaryDirect(fbb, &binaryVersion, ::ttmlir::getGitHash(),
                                 deviceDesc, &commandQueues);

    FinishSizePrefixedBinaryBuffer(fbb, binary);
    ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
    ::tt::VerifySizePrefixedBinaryBuffer(verifier);

    uint8_t *buf = fbb.GetBufferPointer();
    auto size = fbb.GetSize();

#if 1
    std::ofstream ttb("out.ttb", std::ios::out | std::ios::binary);
    ttb.write((char const *)buf, size);
    ttb.close();
#endif
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
  }
};

} // namespace mlir::tt::ttmetal
