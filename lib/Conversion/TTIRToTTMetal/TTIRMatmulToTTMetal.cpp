
#pragma clang diagnostic ignored "-Wunused-variable"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Dialect/TT/Utils/PhysicalCoreCoord.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

namespace mlir::tt {
namespace ttmetal {

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

class TTIRToTTMetalMatmulRewriter : public OpRewritePattern<ttir::MatmulOp> {
public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  Value i32(std::int32_t value, OpBuilder &builder) const {
    return builder
        .create<arith::ConstantOp>(builder.getUnknownLoc(),
                                   builder.getIntegerType(32),
                                   builder.getI32IntegerAttr(value))
        .getResult();
  }

  Value i64(std::int32_t value, OpBuilder &builder) const {
    return builder
        .create<arith::ConstantOp>(builder.getUnknownLoc(),
                                   builder.getIntegerType(64),
                                   builder.getI64IntegerAttr(value))
        .getResult();
  }

  std::pair<Value, Value> logicalToPhysicalRT(Location loc, OpBuilder &builder,
                                              Value trans_x, Value trans_y,
                                              Value logical_x,
                                              Value logical_y) const {
    auto x3 = builder.create<arith::MulIOp>(loc, logical_x, i32(3, builder));
    auto y3 = builder.create<arith::MulIOp>(loc, logical_y, i32(3, builder));

    auto shift_x = builder.create<arith::ShRUIOp>(loc, trans_x, x3);
    auto shift_y = builder.create<arith::ShRUIOp>(loc, trans_y, y3);

    auto bm3 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                                 builder.getI32IntegerAttr(7));

    auto masked_x = builder.create<arith::AndIOp>(loc, shift_x, bm3);
    auto masked_y = builder.create<arith::AndIOp>(loc, shift_y, bm3);

    auto x_final = builder.create<arith::AddIOp>(loc, masked_x, logical_x);
    auto y_final = builder.create<arith::AddIOp>(loc, masked_y, logical_y);

    return std::make_pair(x_final, y_final);
  }

  std::pair<SmallVector<Attribute, 5>, SmallVector<Attribute, 5>>
  generate2DMMAttributes(ArrayRef<int64_t> &gridShape,
                         PatternRewriter &rewriter) const {
    SmallVector<Attribute, 5> coreRanges;
    SmallVector<Attribute, 5> kernelConfigs;

    // Compute (whole worker grid)
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{gridShape[0], gridShape[1]}));
    kernelConfigs.push_back(rewriter.getAttr<ttkernel::TensixConfigAttr>());

    // in0 senders
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{gridShape[1], 1}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc0));

    // in1 senders/writers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{1, gridShape[0]}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc1));

    // in0 receivers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 1},
        llvm::ArrayRef<int64_t>{gridShape[1], gridShape[0] - 1}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc0));

    // in1 receivers/writers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{1, 0},
        llvm::ArrayRef<int64_t>{gridShape[1] - 1, gridShape[0]}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc1));

    return std::pair<SmallVector<Attribute, 5>, SmallVector<Attribute, 5>>{
        coreRanges, kernelConfigs};
  }

  void generateComputeBlock(ttmetal::DispatchOp &metalDispatch,
                            PatternRewriter &rewriter,
                            SmallVector<ttkernel::CBType, 3> &cbs, Value in0,
                            Value in1, Value out0) const {
    Block *computeBlock = rewriter.createBlock(&metalDispatch.getRegion(0));
    OpBuilder computeBuilder(computeBlock, computeBlock->begin());

    computeBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"Start compute\" << ENDL();");

    computeBlock->addArgument(cbs[0], metalDispatch.getLoc());
    computeBlock->addArgument(cbs[1], metalDispatch.getLoc());
    computeBlock->addArgument(cbs[2], metalDispatch.getLoc());

    // kernel here
    RankedTensorType in0Type = mlir::cast<RankedTensorType>(in0.getType());
    MetalLayoutAttr in0Encoding = mlir::cast<MetalLayoutAttr>(in0Type.getEncoding());

    RankedTensorType in1Type = mlir::cast<RankedTensorType>(in1.getType());
    MetalLayoutAttr in1Encoding = mlir::cast<MetalLayoutAttr>(in1Type.getEncoding());

    RankedTensorType out0Type = mlir::cast<RankedTensorType>(out0.getType());
    MetalLayoutAttr out0Encoding = mlir::cast<MetalLayoutAttr>(out0Type.getEncoding());

    auto in0_block_k = computeBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), computeBuilder.getIntegerType(32),
        computeBuilder.getI32IntegerAttr(in0Type.getShape().back() /
                                         TILE_WIDTH));
    auto in0_block_v = computeBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), computeBuilder.getIntegerType(32),
        computeBuilder.getI32IntegerAttr(
            in0Type.getShape().front() / TILE_HEIGHT /
            out0Encoding.getGrid().getShape().front()));
    auto in1_block_v = computeBuilder.create<arith::ConstantOp>(
        metalDispatch->getLoc(), computeBuilder.getIntegerType(32),
        computeBuilder.getI32IntegerAttr(
            in1Type.getShape().back() / TILE_WIDTH /
            out0Encoding.getGrid().getShape().back()));
    auto in0_cb_id = computeBuilder.create<arith::ConstantOp>(
        metalDispatch->getLoc(), computeBuilder.getIntegerType(32),
        computeBuilder.getI32IntegerAttr(
            static_cast<int32_t>(cbs[0].getPort())));
    auto in1_cb_id = computeBuilder.create<arith::ConstantOp>(
        metalDispatch->getLoc(), computeBuilder.getIntegerType(32),
        computeBuilder.getI32IntegerAttr(
            static_cast<int32_t>(cbs[1].getPort())));
    auto out_cb_id = computeBuilder.create<arith::ConstantOp>(
        metalDispatch->getLoc(), computeBuilder.getIntegerType(32),
        computeBuilder.getI32IntegerAttr(
            static_cast<int32_t>(cbs[2].getPort())));

    computeBuilder.create<emitc::CallOpaqueOp>(
        metalDispatch.getLoc(), TypeRange{}, "ckernel::matmul_compute_main",
        SmallVector<Value>{in0_block_k, in0_block_v, in1_block_v, in0_cb_id,
                           in1_cb_id, out_cb_id});

    computeBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"End compute\" << ENDL();");

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
                            OpBuilder &readerBuilder, Value in, Value out0,
                            SmallVector<Value> &rt_args, bool isIn1,
                            SystemDescAttr sysDesc) const {

    readerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v6 << ENDL();");
    readerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v7 << ENDL();");
    readerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v9 << ENDL();");
    readerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v10 << ENDL();");

    readerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << \"Start sender: \";");
    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), isIn1 ? "DPRINT << \"in1\" << ENDL();"
                                      : "DPRINT << \"in0\" << ENDL();");

    RankedTensorType inType = mlir::cast<RankedTensorType>(in.getType());
    MetalLayoutAttr inEncoding = mlir::cast<MetalLayoutAttr>(inType.getEncoding());

    RankedTensorType out0Type = mlir::cast<RankedTensorType>(out0.getType());
    MetalLayoutAttr out0Encoding = mlir::cast<MetalLayoutAttr>(out0Type.getEncoding());

    // Working CB Initialization
    auto workingCb = readerBuilder.getBlock()->getArgument(isIn1);
    auto workingCbType = mlir::cast<ttkernel::CBType>(workingCb.getType());

    auto sender_sem_l1_ptr = readerBuilder.create<ttkernel::CastToL1PtrOp>(
        metalDispatch.getLoc(), rt_args[0]);
    auto receiver_sem_l1_ptr = readerBuilder.create<ttkernel::CastToL1PtrOp>(
        metalDispatch.getLoc(), rt_args[1]);

    // Which row/col is this core for in0/in1
    auto core_id = readerBuilder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size(), readerBuilder));

    auto trans_x = readerBuilder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size() + 1, readerBuilder));
    auto trans_y = readerBuilder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size() + 2, readerBuilder));

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"pre block dims\" << ENDL();");

    // Block dimensions
    uint64_t block_k_dim;
    uint64_t block_v_dim;
    if (isIn1) {
      // in1
      block_k_dim = inType.getShape().front() / TILE_HEIGHT;
      block_v_dim = inType.getShape().back() / TILE_WIDTH /
                    out0Encoding.getGrid().getShape().back();
    } else {
      // in0
      block_k_dim = inType.getShape().back() / TILE_WIDTH;
      block_v_dim = inType.getShape().front() / TILE_HEIGHT /
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
        readerBuilder.getI32IntegerAttr(inEncoding.getElementSizeBytes()));
    auto block_size = readerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_k, block_v);
    // Size of block in bytes
    auto block_size_bytes = readerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_size, tile_size_bytes);

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post block calcs\" << ENDL();");

    // Remote address for in0
    auto in0_addr = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(lookupAddress(in)));
    // Use the start_block_id for in0 to get stride & address
    // auto start_block_stride = readerBuilder.create<arith::MulIOp>(
    //     metalDispatch.getLoc(), core_id, block_size_bytes);
    // auto in0_addr = readerBuilder.create<arith::AddIOp>(
    //     metalDispatch.getLoc(), start_in0_addr, start_block_stride);

    // Reserve space in in0 CB for the block
    auto in0_block_size_tiles = readerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_k, block_v);
    auto cbReserve = readerBuilder.create<ttkernel::CBReserveBackOp>(
        metalDispatch.getLoc(), workingCb, in0_block_size_tiles);

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(),
        "DPRINT << \"post addresses and cb reserve\" << ENDL();");

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
    // llvm::SmallVector<mlir::Value, 2> outer_iter_args = {start_core_v};
    // llvm::SmallVector<mlir::Value, 2> inner_iter_args = {start_cb_addr};

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"pre read loop\" << ENDL();");

    auto cb_addr = workingCbType.getAddress();
    for (uint32_t inner = 0; inner < block_k_dim; inner++) {
      for (uint32_t outer = 0; outer < block_v_dim; outer++) {

        auto core_outer = readerBuilder.create<arith::ConstantOp>(
            metalDispatch->getLoc(), readerBuilder.getI32Type(),
            readerBuilder.getI32IntegerAttr(outer));
        auto core_inner = readerBuilder.create<arith::ConstantOp>(
            metalDispatch->getLoc(), readerBuilder.getI32Type(),
            readerBuilder.getI32IntegerAttr(inner));

        auto core_outer_offset = readerBuilder.create<arith::MulIOp>(
            metalDispatch->getLoc(), core_id, block_v);

        auto core_outer_logical = readerBuilder.create<arith::AddIOp>(
            metalDispatch->getLoc(), core_outer_offset, core_outer);
        auto core_inner_logical = readerBuilder.create<arith::AddIOp>(
            metalDispatch->getLoc(), i32(0, readerBuilder), core_inner);

        auto logical_x = isIn1 ? core_outer_logical.getResult()
                               : core_inner_logical.getResult();
        auto logical_y = isIn1 ? core_inner_logical.getResult()
                               : core_outer_logical.getResult();

        auto [x_final, y_final] = logicalToPhysicalRT(
            metalDispatch.getLoc(), readerBuilder, trans_x.getResult(),
            trans_y.getResult(), logical_x, logical_y);

        auto noc_addr = readerBuilder.create<ttkernel::GetNocAddrOp>(
            metalDispatch.getLoc(), x_final, y_final, in0_addr);
        auto noc_read = readerBuilder.create<ttkernel::NocAsyncReadOp>(
            metalDispatch->getLoc(), noc_addr, i32(cb_addr, readerBuilder),
            tile_size_bytes); // this is not always tile_size_bytes CHANGE ME -
                              // TODO
        cb_addr += inEncoding.getElementSizeBytes();
      }
    }

    /* Memref Single Loop Code - not working
    auto memRefType =
        MemRefType::get(block_k_dim * block_v_dim, readerBuilder.getI32Type());
    auto memRefAllocX = readerBuilder.create<memref::AllocaOp>(
        metalDispatch.getLoc(), memRefType);
    auto memRefAllocY = readerBuilder.create<memref::AllocaOp>(
        metalDispatch.getLoc(), memRefType);

    for (uint32_t i = 0; i < block_k_dim*block_v_dim; i++) {
      auto index = readerBuilder.create<arith::ConstantOp>(
          metalDispatch.getLoc(), readerBuilder.getIndexType(),
          readerBuilder.getIndexAttr(i));
      auto memRefStoreX = readerBuilder.create<memref::StoreOp>(
          metalDispatch.getLoc(), i32(physXCoords[i], readerBuilder),
    memRefAllocX, ArrayRef<Value>{index}); auto memRefStoreY =
    readerBuilder.create<memref::StoreOp>( metalDispatch.getLoc(),
    i32(physYCoords[i], readerBuilder), memRefAllocY, ArrayRef<Value>{index});
    }


    scf::ForOp coreReadLoop = readerBuilder.create<scf::ForOp>(
        metalDispatch.getLoc(), i32(1, readerBuilder),
        i32(block_k_dim * block_v_dim, readerBuilder), i32(1, readerBuilder),
        inner_iter_args);

    readerBuilder.setInsertionPointToStart(coreReadLoop.getBody());
    auto iv_i64 = readerBuilder.create<arith::IndexCastOp>(
        metalDispatch.getLoc(), readerBuilder.getIndexType(),
        coreReadLoop.getInductionVar());
    auto core_x = readerBuilder.create<memref::LoadOp>(
        metalDispatch->getLoc(), memRefAllocX, ArrayRef<Value>{iv_i64});
    auto core_y = readerBuilder.create<memref::LoadOp>(
        metalDispatch->getLoc(), memRefAllocY, ArrayRef<Value>{iv_i64});
    auto noc_addr = readerBuilder.create<ttkernel::GetNocAddrOp>(
        metalDispatch.getLoc(), core_x.getResult(), core_y.getResult(),
        in0_addr);
    auto noc_read = readerBuilder.create<ttkernel::NocAsyncReadOp>(
        metalDispatch->getLoc(), noc_addr,
        coreReadLoop.getRegionIterArg(0), tile_size_bytes); // this is not
    always tile_size_bytes CHANGE ME auto inc_l1_addr =
    readerBuilder.create<arith::AddIOp>( metalDispatch.getLoc(),
    coreReadLoop.getRegionIterArg(0), tile_size_bytes); // inc local l1 addr by
    tile_size ?  (probably
                          // subblock_h * tile_size more generally)
    auto yieldInner = readerBuilder.create<scf::YieldOp>(
        metalDispatch.getLoc(),
        llvm::SmallVector<mlir::Value>{inc_l1_addr.getResult()});
    readerBuilder.setInsertionPointAfter(coreReadLoop);
    */

    /* Double-looped code
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
      readerBuilder.create<emitc::VerbatimOp>(
          metalDispatch.getLoc(), "DPRINT << v41 << \", \" << v47 << ENDL();");
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
    */

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post read loop\" << ENDL();");

    auto readBarrier = readerBuilder.create<ttkernel::NocAsyncReadBarrierOp>(
        metalDispatch.getLoc());
    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post read barrier\" << ENDL();");
    auto cbPushBack = readerBuilder.create<ttkernel::CBPushBackOp>(
        metalDispatch.getLoc(), workingCb, block_size);

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post cb push\" << ENDL();");

    // Start Mcast -- TODO: Make dynamic for in0/in1

    // Set Local L1 Sender Semaphore to VALID, to be mcasted after data mcast
    auto valid = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(1));
    auto setLocalValid = readerBuilder.create<ttkernel::NocSemaphoreSetOp>(
        metalDispatch.getLoc(), receiver_sem_l1_ptr, valid);

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post semaphore init\" << ENDL();");

    // Wait for all receivers to be ready
    uint64_t num_receivers_c;
    if (isIn1) {
      // in1
      num_receivers_c = out0Encoding.getGrid().getShape().front() - 1;
    } else {
      // in0
      num_receivers_c = out0Encoding.getGrid().getShape().back() - 1;
    }

    auto num_receivers = readerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), readerBuilder.getIntegerType(32),
        readerBuilder.getI32IntegerAttr(num_receivers_c));
    auto receiverWait = readerBuilder.create<ttkernel::NocSemaphoreWaitOp>(
        metalDispatch.getLoc(), sender_sem_l1_ptr, num_receivers);
    auto resetReceiverReady = readerBuilder.create<ttkernel::NocSemaphoreSetOp>(
        metalDispatch.getLoc(), sender_sem_l1_ptr, i32(0, readerBuilder));

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post sem wait\" << ENDL();");

    // Data Transfer
    auto start_x = isIn1 ? core_id.getArgVal() : i32(1, readerBuilder);
    auto start_y = isIn1 ? i32(1, readerBuilder) : core_id.getArgVal();
    auto end_x = isIn1 ? core_id.getArgVal() : num_receivers.getResult();
    auto end_y = isIn1 ? num_receivers.getResult() : core_id.getArgVal();
    auto [start_x_final, start_y_final] =
        logicalToPhysicalRT(metalDispatch->getLoc(), readerBuilder, trans_x,
                            trans_y, start_x, start_y);
    auto [end_x_final, end_y_final] = logicalToPhysicalRT(
        metalDispatch->getLoc(), readerBuilder, trans_x, trans_y, end_x, end_y);
    auto nocDataMcastAddr =
        readerBuilder.create<ttkernel::GetNocMulticastAddrOp>(
            metalDispatch.getLoc(), start_x_final, start_y_final, end_x_final,
            end_y_final, start_cb_addr.getWritePtr());
    auto nocMcastWrite =
        readerBuilder.create<ttkernel::NocAsyncWriteMulticastOp>(
            metalDispatch.getLoc(), start_cb_addr, nocDataMcastAddr,
            block_size_bytes, num_receivers, readerBuilder.getBoolAttr(true),
            readerBuilder.getBoolAttr(true));
    // Write Barrier
    auto writerBarrier = readerBuilder.create<ttkernel::NocAsyncWriteBarrierOp>(
        metalDispatch.getLoc());

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post data mcast\" << ENDL();");

    // Signal receivers that data is ready
    auto nocReceiverSemaphoreMcastAddr =
        readerBuilder.create<ttkernel::GetNocMulticastAddrOp>(
            metalDispatch.getLoc(), start_x_final, start_y_final, end_x_final,
            end_y_final, rt_args[1]);
    auto mcastValid =
        readerBuilder.create<ttkernel::NocSemaphoreSetMulticastOp>(
            metalDispatch.getLoc(), rt_args[1], nocReceiverSemaphoreMcastAddr,
            num_receivers, readerBuilder.getBoolAttr(true),
            readerBuilder.getBoolAttr(true));

    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"post datavalid\" << ENDL();");

    // End Mcast

    readerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << \"End sender: \";");
    readerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), isIn1 ? "DPRINT << \"in1\" << ENDL();"
                                      : "DPRINT << \"in0\" << ENDL();");
  }

  void receiveAndWriteTensor(ttmetal::DispatchOp &metalDispatch,
                             OpBuilder &writerBuilder, Value in, Value out0,
                             SmallVector<Value> &rt_args, bool isIn1) const {

    writerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << \"Start receiver: \";");
    writerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), isIn1 ? "DPRINT << \"in1\" << ENDL();"
                                      : "DPRINT << \"in0\" << ENDL();");

    writerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v8 << ENDL();");
    writerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v9 << ENDL();");
    writerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v11 << ENDL();");
    writerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << v12 << ENDL();");

    auto core_id = writerBuilder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size(), writerBuilder));

    RankedTensorType inType = mlir::cast<RankedTensorType>(in.getType());
    MetalLayoutAttr inEncoding = mlir::cast<MetalLayoutAttr>(inType.getEncoding());

    RankedTensorType out0Type = mlir::cast<RankedTensorType>(out0.getType());
    MetalLayoutAttr out0Encoding = mlir::cast<MetalLayoutAttr>(out0Type.getEncoding());

    // Working In CB Initialization
    auto workingInCb = writerBuilder.getBlock()->getArgument(isIn1);
    auto workingInCbType = mlir::cast<ttkernel::CBType>(workingInCb.getType());

    // Working Out CB Initialization
    auto workingOutCb = writerBuilder.getBlock()->getArgument(2);
    auto workingOutCbType =
        mlir::cast<ttkernel::CBType>(workingOutCb.getType());

    auto sender_sem_l1_ptr = writerBuilder.create<ttkernel::CastToL1PtrOp>(
        metalDispatch.getLoc(), rt_args[0]);
    auto receiver_sem_l1_ptr = writerBuilder.create<ttkernel::CastToL1PtrOp>(
        metalDispatch.getLoc(), rt_args[1]);

    auto trans_x = writerBuilder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size() + 1, writerBuilder));
    auto trans_y = writerBuilder.create<ttkernel::GetArgValOp>(
        metalDispatch.getLoc(), i32(rt_args.size() + 2, writerBuilder));

    writerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"Post SemPtrCasts\" << ENDL();");

    // Block dimensions
    uint64_t block_k_dim;
    uint64_t block_v_dim;
    if (isIn1) {
      // in1
      block_k_dim = inType.getShape().front() / TILE_HEIGHT;
      block_v_dim = inType.getShape().back() / TILE_WIDTH /
                    out0Encoding.getGrid().getShape().back();
    } else {
      // in0
      block_k_dim = inType.getShape().back() / TILE_WIDTH;
      block_v_dim = inType.getShape().front() / TILE_HEIGHT /
                    out0Encoding.getGrid().getShape().front();
    }

    auto block_k = writerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), writerBuilder.getIntegerType(32),
        writerBuilder.getI32IntegerAttr(
            block_k_dim)); // this calc will only work for 2D MM
    auto block_v = writerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), writerBuilder.getIntegerType(32),
        writerBuilder.getI32IntegerAttr(
            block_v_dim)); // this calc will only work for 2D MM

    auto tile_size_bytes = writerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), writerBuilder.getIntegerType(32),
        writerBuilder.getI32IntegerAttr(inEncoding.getElementSizeBytes()));
    auto block_size = writerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_k, block_v);
    // Size of block in bytes
    auto block_size_bytes = writerBuilder.create<arith::MulIOp>(
        metalDispatch.getLoc(), block_size, tile_size_bytes);

    writerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"Pre-CB Reserve\" << ENDL();");
    writerBuilder.create<ttkernel::CBReserveBackOp>(metalDispatch.getLoc(),
                                                    workingInCb, block_size);
    writerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"CB Reserved\" << ENDL();");

    auto valid = writerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), writerBuilder.getIntegerType(32),
        writerBuilder.getI32IntegerAttr(1));
    auto invalid = writerBuilder.create<arith::ConstantOp>(
        metalDispatch.getLoc(), writerBuilder.getIntegerType(32),
        writerBuilder.getI32IntegerAttr(0));
    writerBuilder.create<ttkernel::NocSemaphoreSetOp>(
        metalDispatch->getLoc(), receiver_sem_l1_ptr, invalid);

    writerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(),
        "DPRINT << \"Local receiver semaphore set 0\" << ENDL();");

    auto x = isIn1 ? core_id.getArgVal() : i32(0, writerBuilder);
    auto y = isIn1 ? i32(0, writerBuilder) : core_id.getArgVal();

    auto [x_final, y_final] =
        logicalToPhysicalRT(metalDispatch.getLoc(), writerBuilder,
                            trans_x.getResult(), trans_y.getResult(), x, y);
    auto sender_sem_noc_addr = writerBuilder.create<ttkernel::GetNocAddrOp>(
        metalDispatch->getLoc(), x_final, y_final, rt_args[0]);

    writerBuilder.create<ttkernel::NocSemaphoreIncOp>(
        metalDispatch->getLoc(), sender_sem_noc_addr, i32(1, writerBuilder),
        isIn1 ? i32(1, writerBuilder) : i32(0, writerBuilder)); // CHANGE NOC INDEX TO MATCH
                                // NCRISC/BRISC CONFIG
                                // TODO

    writerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), "DPRINT << \"Sender sem inc\" << ENDL();");

    writerBuilder.create<ttkernel::NocSemaphoreWaitOp>(
        metalDispatch->getLoc(), receiver_sem_l1_ptr, valid);
    writerBuilder.create<ttkernel::CBPushBackOp>(metalDispatch->getLoc(),
                                                 workingInCb, block_size);

    auto out_r = out0Type.getShape().front() / TILE_HEIGHT / out0Encoding.getGrid().getShape().front();
    auto out_k = out0Type.getShape().back() / TILE_WIDTH / out0Encoding.getGrid().getShape().back();

    writerBuilder.create<ttkernel::CBWaitFrontOp>(metalDispatch->getLoc(),
                                                  workingOutCb, i32(out_r * out_k, writerBuilder));

    writerBuilder.create<emitc::VerbatimOp>(metalDispatch.getLoc(),
                                            "DPRINT << \"End receiver: \";");
    writerBuilder.create<emitc::VerbatimOp>(
        metalDispatch.getLoc(), isIn1 ? "DPRINT << \"in1\" << ENDL();"
                                      : "DPRINT << \"in0\" << ENDL();");
  }

  void generateMMArgs(ttmetal::DispatchOp &metalDispatch,
                      PatternRewriter &rewriter, Block &block, bool isIn1,
                      SystemDescAttr sysDesc) const {
    RankedTensorType outputTensor =
        mlir::cast<RankedTensorType>(metalDispatch.getOutputs()[0].getType());
    MetalLayoutAttr out0Encoding =
        mlir::cast<MetalLayoutAttr>(outputTensor.getEncoding());
    auto shape_max = isIn1 ? out0Encoding.getGrid().getShape().back()
                           : out0Encoding.getGrid().getShape().front();
    SmallVector<uint32_t> core_ids;
    for (int i = 0; i < shape_max; i++) {
      core_ids.push_back(i);
    }
    auto rt_arg = rewriter.getType<ttkernel::MMArgsType>(
        ArrayRef<uint32_t>(core_ids),
        PhysicalCoreCoordMapping::getXMapping(sysDesc.getChipDescs()[0]),
        PhysicalCoreCoordMapping::getYMapping(sysDesc.getChipDescs()[0]));
    block.addArgument(rt_arg, metalDispatch.getLoc());
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

      if (i == 3 || i == 4) {
        readerBlock->addArgument(cbs[2], metalDispatch.getLoc());
      }

      if (i == 1 || i == 3) {
        readerBlock->addArgument(semaphores[0], metalDispatch.getLoc());
        readerBlock->addArgument(semaphores[1], metalDispatch.getLoc());
        generateMMArgs(metalDispatch, rewriter, *readerBlock, false, sysDesc);
      } else {
        readerBlock->addArgument(semaphores[2], metalDispatch.getLoc());
        readerBlock->addArgument(semaphores[3], metalDispatch.getLoc());
        generateMMArgs(metalDispatch, rewriter, *readerBlock, true, sysDesc);
      }

      SmallVector<Value> rt_args;
      auto semaphores = addSemaphores(metalDispatch, readerBuilder, rt_args);
      // kernels for each block here (use createDataMovementThread /
      // buildNocAsyncTx, etc. (TTIRToTTMetal.cpp))

      if (i == 1) {
        // in0 sender
        gatherAndMcastTensor(
            metalDispatch, readerBuilder, metalDispatch.getInputs()[0],
            metalDispatch.getOutputs()[0], rt_args, false, sysDesc);
      } else if (i == 2) {
        // in1 sender
        gatherAndMcastTensor(
            metalDispatch, readerBuilder, metalDispatch.getInputs()[1],
            metalDispatch.getOutputs()[0], rt_args, true, sysDesc);
      } else if (i == 3) {
        receiveAndWriteTensor(metalDispatch, readerBuilder,
                              metalDispatch.getInputs()[0],
                              metalDispatch.getOutputs()[0], rt_args, false);
      } else if (i == 4) {
        receiveAndWriteTensor(metalDispatch, readerBuilder,
                              metalDispatch.getInputs()[1],
                              metalDispatch.getOutputs()[0], rt_args, true);
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

    auto in0TensorLayout = mlir::cast<MetalLayoutAttr>(tensorA.getEncoding());
    auto in1TensorLayout = mlir::cast<MetalLayoutAttr>(tensorB.getEncoding());
    auto out0TensorLayout = mlir::cast<MetalLayoutAttr>(outputTensor.getEncoding());
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
        ttkernel::CBPort::In0, 111104, in0TensorLayout.getMemref(),
        in0TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 8); // TODO HELP ME
    ttkernel::CBType in1CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::In1, 127488, in1TensorLayout.getMemref(),
        in1TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 16); // TODO HELP ME
    ttkernel::CBType out0CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::Out0, out0BaseAddress, out0TensorLayout.getMemref(),
        out0TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 2); // TODO HELP ME 

    SmallVector<ttkernel::CBType, 3> cbTypes = {in0CBTy, in1CBTy, out0CBTy};

    generateComputeBlock(
        metalDispatch, rewriter, cbTypes, metalDispatch.getInputs()[0],
        metalDispatch.getInputs()[1], metalDispatch.getOutputs()[0]);
    generateReaderBlocks(metalDispatch, rewriter, cbTypes, device,
                         op.getSystemDesc());
    rewriter.replaceOp(op, metalDispatch);
    return success();
  }
};

} // namespace ttmetal

} // namespace mlir::tt