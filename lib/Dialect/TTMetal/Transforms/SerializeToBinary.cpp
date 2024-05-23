// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/Passes.h"

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
  flatbuffers::Offset<SchemaType> insert(MLIRTypeOrAttr obj,
                                        flatbuffers::Offset<SchemaType> offset) {
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
  std::invoke_result_t<CreateFn, FlatbufferObjectCache &, MLIRTypeOrAttr, Args...>
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

inline flatbuffers::Offset<::tt::MemoryDesc>
memrefAttrToFlatbuffer(FlatbufferObjectCache &cache, MemRefType memref) {
  auto shapeInt64 = memref.getShape();
  std::vector<int32_t> shape(shapeInt64.begin(), shapeInt64.end());
  ::tt::Dim2d tileShape(0, 0);
  // asdf
  return ::tt::CreateMemoryDescDirect(*cache.fbb, &shape, &tileShape,
                                      ::tt::DataType::Float32,
                                      ::tt::MemorySpace::System);
}

inline flatbuffers::Offset<::tt::LayoutDesc>
layoutAttrToFlatbuffer(FlatbufferObjectCache &cache, Attribute attr) {
  assert(attr.isa<LayoutAttr>() && "expected a tensor type");
  auto layoutAttr = attr.cast<LayoutAttr>();
  auto stridesInt64 = layoutAttr.getStrides();
  std::vector<int32_t> strides(stridesInt64.begin(), stridesInt64.end());
  ::tt::Dim2dRange grid(::tt::Dim2d(0, 0), ::tt::Dim2d(1, 1));
  return ::tt::CreateLayoutDescDirect(
      *cache.fbb, &strides, ::tt::OOBVal::Undef, // asdf
      &grid,
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
    : public impl::TTMetalSerializeToBinaryBase<
          TTMetalSerializeToBinary> {
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

#if 0
    for (auto &output : entry.getResults()) {
      cqBuilder.outputs.push_back(
          cache.getOrCreate(output, tensorValueToFlatbuffer,
                            kHostAllocatedAddress, kHostAllocatedSize));
    }
#endif

    module->walk([&](mlir::Operation *op) {
      if (auto dispatchOp = dyn_cast_or_null<tt::ttmetal::DispatchOp>(op);
          dispatchOp) {
#if 0
      for (auto &region : dispatchOp.getRegions()) {
        for (auto &op : region.getOps()) {
          if (isa<ModuleOp>(op)) {
            auto res = emitc::translateToCpp(&op, llvm::outs());
            (void)res;
          }
        }
      }
#endif
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
      }
    });


    ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
    ::tt::Version binaryVersion(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.release);

    ::tt::Dim2d deviceGrid(8, 8);
    auto deviceDesc = ::tt::CreateDeviceDesc(
        fbb, ::tt::DeviceArch::Wormhole_b0, &deviceGrid);

    std::vector<::flatbuffers::Offset<::tt::CommandQueue>> commandQueues = {
        ::tt::CreateCommandQueueDirect(fbb, cqBuilder.name,
                                       &cqBuilder.inputs, &cqBuilder.outputs,
                                       &cqBuilder.commands),
    };
    auto binary = ::tt::CreateBinaryDirect(fbb, &binaryVersion,
                                           ::ttmlir::getGitHash(), deviceDesc,
                                           &commandQueues);

    FinishSizePrefixedBinaryBuffer(fbb, binary);
    ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(),
                                     fbb.GetSize());
    ::tt::VerifySizePrefixedBinaryBuffer(verifier);

    uint8_t *buf = fbb.GetBufferPointer();
    auto size = fbb.GetSize();

#if 1
    std::ofstream ttb("out.ttb",
                      std::ios::out | std::ios::binary | std::ios::app);
    ttb.write((char const *)buf, size);
    ttb.close();
#endif
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

} // namespace mlir::tt::ttmetal
