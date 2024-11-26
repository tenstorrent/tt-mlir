
#pragma clang diagnostic ignored "-Wunused-variable"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Dialect/TT/Utils/PhysicalCoreCoord.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt {
namespace ttmetal {

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

class TTIRToTTMetalMatmulRewriter : public OpRewritePattern<ttir::MatmulOp> {
public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  // Start copy/paste from TTIRToTTMetal.cpp ~~ Just Testing These ~~

  Value i32(std::int32_t value, OpBuilder &builder) const {
    return builder
        .create<arith::ConstantOp>(builder.getUnknownLoc(),
                                   builder.getIntegerType(32),
                                   builder.getI32IntegerAttr(value))
        .getResult();
  }

  struct NocTx {
    enum class Type { Read, Write };

    Type type;
    PhysicalCoreCoord coreCoord;
    std::int64_t srcOffset = 0;
    std::int64_t dstOffset = 0;
    std::int64_t size = 0;
    std::int32_t numElements = 0;

    NocTx(Type type, PhysicalCoreCoord coreCoord, std::int64_t srcOffset,
          std::int64_t dstOffset, std::int64_t size, std::int32_t numElements)
        : type(type), coreCoord(coreCoord), srcOffset(srcOffset),
          dstOffset(dstOffset), size(size), numElements(numElements) {}

    bool isContiguous(PhysicalCoreCoord nextCoord, std::int64_t nextSrcOffset,
                      std::int64_t nextDstOffset) const {
      return (nextCoord == coreCoord) && (nextSrcOffset == srcOffset + size) &&
             (nextDstOffset == dstOffset + size);
    }
  };

  // This routine calculates the data movement for a tensor layout change by
  // tracing the walk order of the src and dst affine maps.  The sample routine
  // is just a helper function that iterates over the tensor shape and calls the
  // lambda with the current index.  It walks the shape in innermost-major
  // order. It also coalesces the noc transactions.
  //
  // The return value is a map of physical cores where each core has
  // an associated list of noc reads/writes to be performed.
  static llvm::MapVector<PhysicalCoreCoord, mlir::SmallVector<NocTx>>
  calculateDataMovementFromAffine(ArrayRef<int64_t> tensorShape,
                                  std::int64_t elemSize, AffineMap src,
                                  AffineMap dst, NocTx::Type type,
                                  std::int64_t dstCapacity) {
    bool read = type == NocTx::Type::Read;
    llvm::MapVector<PhysicalCoreCoord, mlir::SmallVector<NocTx>> txMap;
    assert(src.getNumResults() == MemoryMapResultIdx::NumIndices);
    assert(dst.getNumResults() == MemoryMapResultIdx::NumIndices);

    ::ttmlir::utils::sample(
        tensorShape, [&txMap, src, dst, elemSize, read, type,
                      dstCapacity](ArrayRef<std::int64_t> index) {
          SmallVector<int64_t> srcResults = src.compose(index);
          SmallVector<int64_t> dstResults = dst.compose(index);
          assert(srcResults.size() == src.getNumResults());
          assert(dstResults.size() == dst.getNumResults());
          PhysicalCoreCoord srcCoord(srcResults);
          PhysicalCoreCoord dstCoord(dstResults);
          std::int64_t srcOffset = srcResults.back() * elemSize;
          std::int64_t dstOffset = dstResults.back() * elemSize;

          SmallVector<NocTx> &txs = txMap[read ? dstCoord : srcCoord];
          if (not txs.empty() &&
              txs.back().isContiguous(read ? srcCoord : dstCoord, srcOffset,
                                      dstOffset) &&
              txs.back().size + elemSize <= dstCapacity) {
            txs.back().size += elemSize;
            txs.back().numElements++;
          } else {
            txs.push_back(NocTx(type, read ? srcCoord : dstCoord, srcOffset,
                                dstOffset, elemSize, 1));
          }
        });

    return txMap;
  }

  static void
  buildNocAsyncTx(mlir::Location loc, std::int64_t inputBaseAddress,
                  std::int64_t outputBaseAddress, std::int64_t addressAlignment,
                  NocTx nocTx,
                  PhysicalCoreCoordMapping const &physicalCoordMapping,
                  mlir::OpBuilder &nocBuilder) {
    assert(nocTx.srcOffset % addressAlignment == 0);
    assert(nocTx.dstOffset % addressAlignment == 0);
    assert(nocTx.size % addressAlignment == 0);
    auto [yPhys, xPhys] = physicalCoordMapping[nocTx.coreCoord];
    auto y = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(yPhys));
    auto x = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(xPhys));
    auto srcLocalL1Addr = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(inputBaseAddress + nocTx.srcOffset));
    auto dstLocalL1Addr = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(outputBaseAddress + nocTx.dstOffset));
    auto size = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(nocTx.size));
    if (nocTx.type == NocTx::Type::Read) {
      auto srcRemoteNocAddr =
          nocBuilder.create<ttkernel::GetNocAddrOp>(loc, x, y, srcLocalL1Addr);
      nocBuilder.create<ttkernel::NocAsyncReadOp>(loc, srcRemoteNocAddr,
                                                  dstLocalL1Addr, size);
    } else {
      auto dstRemoteNocAddr =
          nocBuilder.create<ttkernel::GetNocAddrOp>(loc, x, y, dstLocalL1Addr);
      nocBuilder.create<ttkernel::NocAsyncWriteOp>(loc, srcLocalL1Addr,
                                                   dstRemoteNocAddr, size);
    }
  }

  static void createDataMovementThread(
      Location loc, OpBuilder nocBuilder, int64_t inputBaseAddress,
      int64_t outputBaseAddress, ArrayRef<NocTx> transactions,
      const PhysicalCoreCoordMapping &physicalCoordMapping,
      std::int64_t addressAlignment, Value *inputCB = nullptr) {

    assert(inputBaseAddress);
    assert(outputBaseAddress);
    assert(inputBaseAddress % addressAlignment == 0);
    assert(outputBaseAddress % addressAlignment == 0);
    NocTx::Type type = transactions.front().type;
    for (auto tx : transactions) {
      assert(tx.type == type);
      if (inputCB) {
        auto numElementsConst = nocBuilder.create<arith::ConstantOp>(
            loc, nocBuilder.getI32Type(),
            nocBuilder.getI32IntegerAttr(tx.numElements));
        nocBuilder.create<ttkernel::CBReserveBackOp>(loc, *inputCB,
                                                     numElementsConst);
      }
      buildNocAsyncTx(loc, inputBaseAddress, outputBaseAddress,
                      addressAlignment, tx, physicalCoordMapping, nocBuilder);
      if (inputCB) {
        auto numElementsConst = nocBuilder.create<arith::ConstantOp>(
            loc, nocBuilder.getI32Type(),
            nocBuilder.getI32IntegerAttr(tx.numElements));
        nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(loc);
        nocBuilder.create<ttkernel::CBPushBackOp>(loc, *inputCB,
                                                  numElementsConst);
      }
    }
    if (!inputCB) {
      if (type == NocTx::Type::Read) {
        nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(loc);
      } else {
        nocBuilder.create<ttkernel::NocAsyncWriteBarrierOp>(loc);
      }
    }
  }

  llvm::MapVector<PhysicalCoreCoord, SmallVector<NocTx>>
  calculateDataMovement(const RankedTensorType &src,
                        const RankedTensorType &dst, DeviceAttr device) const {
    auto srcLayout = mlir::cast<tt::LayoutAttr>(src.getEncoding());
    assert(srcLayout.isTiled());

    auto dstLayout = mlir::cast<tt::LayoutAttr>(dst.getEncoding());
    assert(dstLayout.isTiled());

    auto srcMap = srcLayout.getIdentityTileLinearMap();

    auto srcShape = srcMap.compose(srcLayout.getTiledShape(src.getShape()));
    auto srcProjection = srcLayout.projectOnto(
        srcMap, device.getMapForMemorySpace(srcLayout.getMemorySpace()));

    auto dstMap = dstLayout.getIdentityTileLinearMap();
    auto dstShape = dstLayout.getTiledShape(dst.getShape());
    auto dstProjection = dstLayout.projectOnto(
        dstMap, device.getMapForMemorySpace(dstLayout.getMemorySpace()));

    // dstProjection is composed with srcMap to cover the case where srcMap is
    // transposed. Then its shape is transposed too, therefore dstProjection
    // must work with transposed shape.
    auto dm = calculateDataMovementFromAffine(
        srcShape, srcLayout.getElementSizeBytes(), srcProjection,
        dstProjection.compose(srcMap), NocTx::Type::Read,
        dstLayout.getMemrefSizeBytes());

    return dm;
  }

  // End copy/paste from TTIRToTTMetal.cpp

  std::pair<SmallVector<Attribute, 5>, SmallVector<Attribute, 5>>
  generate2DMMAttributes(ArrayRef<int64_t> &gridShape,
                         PatternRewriter &rewriter) const {
    SmallVector<Attribute, 5> coreRanges;
    SmallVector<Attribute, 5> kernelConfigs;

    // Compute (whole worker grid)
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{gridShape[0], gridShape[1]}));
    kernelConfigs.push_back(rewriter.getAttr<ttkernel::TensixConfigAttr>(
        ttkernel::MathFidelity::HiFi4, false, false));

    // in0 senders
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{1, gridShape[1]}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc0));

    // in1 senders/writers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{gridShape[0], 1}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc1));

    // in0 receivers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{1, 0},
        llvm::ArrayRef<int64_t>{gridShape[0] - 1, gridShape[1]}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc0));

    // in1 receivers/writers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 1},
        llvm::ArrayRef<int64_t>{gridShape[0], gridShape[1] - 1}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc1));

    return std::pair<SmallVector<Attribute, 5>, SmallVector<Attribute, 5>>{
        coreRanges, kernelConfigs};
  }

  void generateComputeBlock(ttmetal::DispatchOp &metalDispatch,
                            PatternRewriter &rewriter,
                            SmallVector<ttkernel::CBType, 3> &cbs) const {
    Block *computeBlock = rewriter.createBlock(&metalDispatch.getRegion(0));
    OpBuilder computeBuilder(computeBlock, computeBlock->begin());

    computeBlock->addArgument(cbs[0], metalDispatch.getLoc());
    computeBlock->addArgument(cbs[1], metalDispatch.getLoc());
    computeBlock->addArgument(cbs[2], metalDispatch.getLoc());

    // kernel here

    computeBuilder.create<ttkernel::ReturnOp>(metalDispatch.getLoc());
  }

  SmallVector<ttkernel::SemaphoreType, 4>
  generate2DMMSemaphores(PatternRewriter &rewriter) const {
    auto in0_sender_sem = rewriter.getType<ttkernel::SemaphoreType>(0);
    auto in0_receiver_sem = rewriter.getType<ttkernel::SemaphoreType>(0);
    auto in1_sender_sem = rewriter.getType<ttkernel::SemaphoreType>(0);
    auto in1_receiver_sem = rewriter.getType<ttkernel::SemaphoreType>(0);

    return SmallVector<ttkernel::SemaphoreType, 4>{
        in0_sender_sem, in0_receiver_sem, in1_sender_sem, in1_receiver_sem};
  }

  SmallVector<Value> addSemaphores(ttmetal::DispatchOp &metalDispatch,
                                   OpBuilder &builder,
                                   SmallVector<Value> &rt_args) const {
    auto sender_sempahore_id = builder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size(), builder));
    auto sender_sem_addr = builder.create<ttkernel::GetSemaphoreOp>(
        metalDispatch.getLoc(), sender_sempahore_id);
    rt_args.push_back(sender_sem_addr);

    auto receiver_semaphore_id = builder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size(), builder));
    auto receiver_sem_addr = builder.create<ttkernel::GetSemaphoreOp>(
        metalDispatch.getLoc(), receiver_semaphore_id);
    rt_args.push_back(receiver_sem_addr);

    return SmallVector<Value>{sender_sem_addr, receiver_sem_addr};
  }

  void gatherAndMcastTensor(ttmetal::DispatchOp &metalDispatch,
                            OpBuilder &readerBuilder, PatternRewriter &rewriter,
                            SmallVector<ttkernel::CBType, 3> &cbs, Value in0,
                            Value in1, Value out0, DeviceAttr &device,
                            SystemDescAttr &sysDesc,
                            SmallVector<Value> &rt_args, bool isIn1) const {
    RankedTensorType in0Type = mlir::cast<RankedTensorType>(in0.getType());
    LayoutAttr in0Encoding = mlir::cast<LayoutAttr>(in0Type.getEncoding());

    RankedTensorType in1Type = mlir::cast<RankedTensorType>(in1.getType());
    LayoutAttr in1Encoding = mlir::cast<LayoutAttr>(in1Type.getEncoding());

    RankedTensorType out0Type = mlir::cast<RankedTensorType>(out0.getType());
    LayoutAttr out0Encoding = mlir::cast<LayoutAttr>(out0Type.getEncoding());

    // in0 CB Initialization
    auto in0Cb = readerBuilder.getBlock()->getArgument(0);
    auto in0CbType = mlir::cast<ttkernel::CBType>(in0Cb.getType());

    // in1 CB Initialization
    auto in1Cb = readerBuilder.getBlock()->getArgument(1);
    auto in1CbType = mlir::cast<ttkernel::CBType>(in1Cb.getType());

    // Working CB Initialization
    auto workingCb = isIn1 ? in1Cb : in0Cb;
    auto workingCbType = isIn1 ? in1CbType : in0CbType;

    auto sender_sem_l1_ptr = readerBuilder.create<ttkernel::CastToL1PtrOp>(
        metalDispatch.getLoc(), rt_args[0]);
    auto receiver_sem_l1_ptr = readerBuilder.create<ttkernel::CastToL1PtrOp>(
        metalDispatch.getLoc(), rt_args[1]);

    // Which block to start reading from per core
    auto start_block_id = readerBuilder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size(), readerBuilder));

    // Block dimensions
    uint64_t block_k_dim;
    uint64_t block_v_dim;
    if (isIn1) {
      // in1
      block_k_dim = in1Type.getShape().front() / TILE_HEIGHT;
      block_v_dim = in1Type.getShape().back() / TILE_WIDTH /
                    out0Encoding.getGrid().getShape().back();
    } else {
      // in0
      block_k_dim = in0Type.getShape().back() / TILE_WIDTH;
      block_v_dim = in0Type.getShape().front() / TILE_HEIGHT /
                    out0Encoding.getGrid().getShape().front();
    }

    auto block_k = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(
            block_k_dim)); // this calc will only work for 2D MM
    auto block_v = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(
            block_v_dim)); // this calc will only work for 2D MM

    // Size of one tile in bytes - this is wrong atm idk why... need tile size *
    // datatype size
    auto tile_size_bytes = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(TILE_HEIGHT * TILE_WIDTH *
                                        in0Encoding.getElementSizeBytes()));
    auto block_size = readerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_k, block_v);
    // Size of block in bytes
    auto block_size_bytes = readerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_size, tile_size_bytes);

    // Remote address for in0
    auto start_in0_addr = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(lookupAddress(in0)));
    // Use the start_block_id for in0 to get stride & address
    auto start_block_stride = readerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), start_block_id, block_size_bytes);
    auto in0_addr = readerBuilder.create<arith::AddIOp>(
        metalDispatch.getLoc(), start_in0_addr, start_block_stride);

    // Reserve space in in0 CB for the block
    auto in0_block_size_tiles = readerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_k, block_v);
    auto cbReserve = readerBuilder.create<ttkernel::CBReserveBackOp>(
        metalDispatch.getLoc(), workingCb, in0_block_size_tiles);

    // Initial values for NoC and CB addresses for read loop
    auto start_core_v = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(0));
    auto start_core_k = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(0));
    auto start_cb_addr = readerBuilder.create<ttkernel::GetWritePtrOp>(
        metalDispatch.getLoc(),
        i32(static_cast<int32_t>(workingCbType.getPort()), readerBuilder));
    llvm::SmallVector<mlir::Value, 2> outer_iter_args = {start_core_v};
    llvm::SmallVector<mlir::Value, 2> inner_iter_args = {start_core_k,
                                                         start_cb_addr};

    // Outer Read Loop
    scf::ForOp outerCoreReadLoop = readerBuilder.create<scf::ForOp>(
        metalDispatch.getLoc(), i32(0, readerBuilder), block_v,
        i32(1, readerBuilder), outer_iter_args);

    readerBuilder.setInsertionPointToStart(outerCoreReadLoop.getBody());
    // Inner Read Loop
    scf::ForOp innerCoreReadLoop = readerBuilder.create<scf::ForOp>(
        metalDispatch.getLoc(), i32(0, readerBuilder), block_k,
        i32(1, readerBuilder), inner_iter_args);
    readerBuilder.setInsertionPointToStart(innerCoreReadLoop.getBody());

    // Begin Inner Read Loop
    ttkernel::GetNocAddrOp noc_addr;
    if (isIn1) {
      // in1
      noc_addr = readerBuilder.create<ttkernel::GetNocAddrOp>(
          metalDispatch.getLoc(), outerCoreReadLoop.getRegionIterArg(0),
          innerCoreReadLoop.getRegionIterArg(0), in0_addr);
    } else {
      // in0
      noc_addr = readerBuilder.create<ttkernel::GetNocAddrOp>(
          metalDispatch.getLoc(), innerCoreReadLoop.getRegionIterArg(0),
          outerCoreReadLoop.getRegionIterArg(0), in0_addr);
    }
    auto noc_read = readerBuilder.create<ttkernel::NocAsyncReadOp>(
        metalDispatch->getLoc(), noc_addr,
        innerCoreReadLoop.getRegionIterArg(1),
        tile_size_bytes); // this is not always tile_size_bytes CHANGE ME
    auto inc_core_k = readerBuilder.create<arith::AddIOp>(
        metalDispatch.getLoc(), innerCoreReadLoop.getRegionIterArg(0),
        i32(1, readerBuilder)); // increment core_k by 1 core
    auto inc_l1_addr = readerBuilder.create<arith::AddIOp>(
        metalDispatch.getLoc(), innerCoreReadLoop.getRegionIterArg(1),
        tile_size_bytes); // inc local l1 addr by tile_size ?  (probably
                          // subblock_h * tile_size more generally)
    auto yieldInner = readerBuilder.create<scf::YieldOp>(
        metalDispatch.getLoc(),
        llvm::SmallVector<mlir::Value>{inc_core_k.getResult(),
                                       inc_l1_addr.getResult()});
    // End Inner Read Loop
    readerBuilder.setInsertionPointAfter(innerCoreReadLoop);
    auto inc_core_v = readerBuilder.create<arith::AddIOp>(
        metalDispatch.getLoc(), outerCoreReadLoop.getRegionIterArg(0),
        i32(1, readerBuilder)); // increment core_v by 1 core
    auto yieldOuter = readerBuilder.create<scf::YieldOp>(
        metalDispatch.getLoc(),
        llvm::SmallVector<mlir::Value>{inc_core_v.getResult()});

    // End Outer Read Loop
    readerBuilder.setInsertionPointAfter(outerCoreReadLoop);

    auto readBarrier = readerBuilder.create<ttkernel::NocAsyncReadBarrierOp>(
        metalDispatch.getLoc());
    auto cbPushBack = readerBuilder.create<ttkernel::CBPushBackOp>(
        metalDispatch.getLoc(), workingCb, block_size);

    // Start Mcast -- TODO: Make dynamic for in0/in1

    // Set Local L1 Sender Semaphore to VALID, to be mcasted after data mcast
    auto valid = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(1));
    auto setLocalValid = readerBuilder.create<ttkernel::NocSemaphoreSetOp>(
        metalDispatch.getLoc(), receiver_sem_l1_ptr, valid);

    // Wait for all receivers to be ready
    auto num_receivers = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(
            out0Encoding.getGrid().getShape().back() - 1));
    auto receiverWait = readerBuilder.create<ttkernel::NocSemaphoreWaitOp>(
        metalDispatch.getLoc(), sender_sem_l1_ptr, num_receivers);
    auto resetReceiverReady = readerBuilder.create<ttkernel::NocSemaphoreSetOp>(
        metalDispatch.getLoc(), sender_sem_l1_ptr, i32(0, readerBuilder));

    // Data Transfer
    auto nocDataMcastAddr =
        readerBuilder.create<ttkernel::GetNocMulticastAddrOp>(
            metalDispatch.getLoc(), i32(1, readerBuilder),
            start_block_id.getArgVal(), num_receivers,
            start_block_id.getArgVal(), start_cb_addr.getWritePtr());
    auto nocMcastWrite =
        readerBuilder.create<ttkernel::NocAsyncWriteMulticastOp>(
            metalDispatch.getLoc(), start_cb_addr, nocDataMcastAddr,
            block_size_bytes, num_receivers, readerBuilder.getBoolAttr(true),
            readerBuilder.getBoolAttr(true));
    // Write Barrier
    auto writerBarrier = readerBuilder.create<ttkernel::NocAsyncWriteBarrierOp>(
        metalDispatch.getLoc());

    // Signal receivers that data is ready
    auto nocReceiverSemaphoreMcastAddr =
        readerBuilder.create<ttkernel::GetNocMulticastAddrOp>(
            metalDispatch.getLoc(), i32(1, readerBuilder),
            start_block_id.getArgVal(), num_receivers,
            start_block_id.getArgVal(), rt_args[1]);
    auto mcastValid =
        readerBuilder.create<ttkernel::NocSemaphoreSetMulticastOp>(
            metalDispatch.getLoc(), rt_args[1], nocReceiverSemaphoreMcastAddr,
            num_receivers, readerBuilder.getBoolAttr(true),
            readerBuilder.getBoolAttr(true));

    // End Mcast

    return;
  }

  void generateReaderBlocks(ttmetal::DispatchOp &metalDispatch,
                            PatternRewriter &rewriter,
                            SmallVector<ttkernel::CBType, 3> &cbs,
                            DeviceAttr &device, SystemDescAttr sysDesc) const {
    // generate 4 reader blocks, block 0 is the compute block, blocks 1-4 are
    // the reader/writer blocks

    SmallVector<ttkernel::SemaphoreType, 4> semaphores =
        generate2DMMSemaphores(rewriter);

    for (int i = 1; i < 5; i++) {
      Block *readerBlock = rewriter.createBlock(&metalDispatch.getRegion(i));
      OpBuilder readerBuilder(readerBlock, readerBlock->begin());

      readerBlock->addArgument(cbs[0], metalDispatch.getLoc());
      readerBlock->addArgument(cbs[1], metalDispatch.getLoc());

      if (i == 1 || i == 3) {
        readerBlock->addArgument(semaphores[0], metalDispatch.getLoc());
        readerBlock->addArgument(semaphores[1], metalDispatch.getLoc());
      } else {
        readerBlock->addArgument(semaphores[2], metalDispatch.getLoc());
        readerBlock->addArgument(semaphores[3], metalDispatch.getLoc());
      }

      SmallVector<Value> rt_args;
      auto semaphores = addSemaphores(metalDispatch, readerBuilder, rt_args);

      // kernels for each block here (use createDataMovementThread /
      // buildNocAsyncTx, etc. (TTIRToTTMetal.cpp))
      if (i == 1) {
        gatherAndMcastTensor(
            metalDispatch, readerBuilder, rewriter, cbs,
            metalDispatch.getInputs()[0], metalDispatch.getInputs()[1],
            metalDispatch.getOutputs()[0], device, sysDesc, rt_args, false);
      } else if (i == 2) {
        gatherAndMcastTensor(
            metalDispatch, readerBuilder, rewriter, cbs,
            metalDispatch.getInputs()[0], metalDispatch.getInputs()[1],
            metalDispatch.getOutputs()[0], device, sysDesc, rt_args, true);
      }

      readerBuilder.create<ttkernel::ReturnOp>(metalDispatch.getLoc());
    }
  }

  LogicalResult matchAndRewrite(ttir::MatmulOp op,
                                PatternRewriter &rewriter) const final {
    RankedTensorType tensorA = op.getA().getType();
    RankedTensorType tensorB = op.getB().getType();
    RankedTensorType outputTensor = op.getOutput().getType();
    DeviceAttr device = op.getDevice();
    // ArrayAttr constraints = op.getOperandConstraints();

    // // Operands must be DRAM OR L1 AND Tile Layout
    // if ((std::find(constraints.begin(), constraints.end(),
    // OperandConstraint::DRAM) == constraints.end() &&
    //     std::find(constraints.begin(), constraints.end(),
    //     OperandConstraint::L1) == constraints.end()) ||
    //     std::find(constraints.begin(), constraints.end(),
    //     OperandConstraint::Tile) == constraints.end()) {
    //       return failure();
    // }

    uint32_t tensorARank = tensorA.getRank();
    uint32_t tensorBRank = tensorB.getRank();
    uint32_t outputTensorRank = outputTensor.getRank();

    // Input A must be tile aligned
    if ((tensorA.getShape()[tensorARank - 1] % TILE_WIDTH != 0 ||
         tensorA.getShape()[tensorARank - 2] % TILE_HEIGHT != 0)) {
      return failure();
    }

    // Input B must be tile aligned
    if ((tensorB.getShape()[tensorBRank - 1] % TILE_WIDTH != 0 ||
         tensorB.getShape()[tensorBRank - 2] % TILE_HEIGHT != 0)) {
      return failure();
    }

    // Output must be tile aligned
    if ((outputTensor.getShape()[outputTensorRank - 1] % TILE_WIDTH != 0 ||
         outputTensor.getShape()[outputTensorRank - 2] % TILE_HEIGHT != 0)) {
      return failure();
    }

    llvm::errs() << "MatmulOp matched\n";

    auto in0TensorLayout = mlir::cast<LayoutAttr>(tensorA.getEncoding());
    auto in1TensorLayout = mlir::cast<LayoutAttr>(tensorB.getEncoding());
    auto out0TensorLayout = mlir::cast<LayoutAttr>(outputTensor.getEncoding());
    auto outputTensorGrid = out0TensorLayout.getGrid().getShape();

    auto [coreRanges, kernelConfigs] =
        generate2DMMAttributes(outputTensorGrid, rewriter);

    SmallVector<Value> operands = {op.getA(), op.getB()};
    SmallVector<Value> outputs = {op.getOutput()};
    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op->getResults().getTypes(), operands, outputs,
        rewriter.getArrayAttr(coreRanges), rewriter.getArrayAttr(kernelConfigs),
        coreRanges.size());

    std::int64_t in0BaseAddress = lookupAddress(op.getA());
    std::int64_t in1BaseAddress = lookupAddress(op.getB());
    std::int64_t out0BaseAddress = lookupAddress(op.getOutput());

    ttkernel::CBType in0CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::In0, in0BaseAddress, in0TensorLayout.getMemref(),
        in0TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 1);
    ttkernel::CBType in1CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::In1, in1BaseAddress, in1TensorLayout.getMemref(),
        in1TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 1);
    ttkernel::CBType out0CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::Out0, out0BaseAddress, out0TensorLayout.getMemref(),
        out0TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 1);

    SmallVector<ttkernel::CBType, 3> cbTypes = {in0CBTy, in1CBTy, out0CBTy};

    generateComputeBlock(metalDispatch, rewriter, cbTypes);
    generateReaderBlocks(metalDispatch, rewriter, cbTypes, device,
                         op.getSystemDesc());
    rewriter.replaceOp(op, metalDispatch);
    return success();
  }
};

} // namespace ttmetal

} // namespace mlir::tt
