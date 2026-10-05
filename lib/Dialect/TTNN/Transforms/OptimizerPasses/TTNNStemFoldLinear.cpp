// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// clang-format off
// TTNNStemFoldLinear
// ==================
//
// Places the stem that TTIRStemFold produced. P = N*H/R*W/R pixels, Cp padded
// input channels, Cout output channels, G = number of worker cores. After
// lowering and the optimizer the stem reaches this pass as
//
//   %pu   = ttnn.pixel_unshuffle(%x) {downscale_factor = R, channel_major, channels_last, padded_channels = Cp}
//           : N x Cin x H x W (rm, dram) -> N x H/R x W/R x Cp   dram interleaved     (no op-model: left in DRAM)
//   %flat = ttnn.reshape(%pu) [1,1,P,Cp]                          -> 1 x 1 x P x Cp   dram interleaved
//   %y    = ttnn.conv2d(%flat, %W Cout x Cp x 1 x 1 (host), %b 1 x 1 x 1 x Cout (host))
//           {kernel [1,1], stride [1,1], padding 0, groups 1, conv2d_config<activation, shard_layout, ...>}
//           -> 1 x 1 x P x Cout   layout chosen by the optimizer (height-sharded in the measured graphs)
//
// and leaves as
//
//   %pu   = ttnn.pixel_unshuffle(%x) {... same attrs ...,
//             memory_config = <#l1, height_sharded, shard_spec<full grid, [P/G, Cp], row_major>>}
//           -> N x H/R x W/R x Cp   rm, l1, height_sharded G x 1, shard [P/G, Cp]
//   %flat = ttnn.reshape(%pu) [1,1,P,Cp]                          -> same shards (a view)
//   %t    = ttnn.to_layout(%flat)                                 -> tile, l1, height_sharded, [P/G/32, Cp/32] tiles
//           (emitted as ttnn.to_tensor_spec; TTNNDecomposeLayouts makes it the single in-place tilize)
//   %y    = ttnn.linear(%t, %Wt Cp x Cout (tile, dram, conv2d weight dtype), %b 1 x 1 x 1 x Cout (tile, dram))
//           {matmul_program_config = 1d<grid, in0_block_w Cp/32, out_subblock 1 x min(2, Cout/32),
//                                       out_block_h 2 (1 if P/G/32 is odd), out_block_w Cout/32,
//                                       per_core_m P/G/32, per_core_n Cout/32, fuse_batch,
//                                       fused_activation = the conv's activation, mcast_in0 false>,
//            compute_config = the conv's}
//           -> 1 x 1 x P x Cout   tile, l1, height_sharded G x 1, [P/G/32, Cout/32] tiles   (what the next op reads)
//
//   %Wt = typecast(transpose(reshape(to_device(%W), [Cout, Cp]), 0, 1), dtype)   [K = Cp, N = Cout]; const-eval hoisted
//
// Steps, in order:
//   1. drop the to_tensor_spec (tilize) the layout pass put on the input: the kernel reads row-major;
//   2. give the pixel_unshuffle result ROW_MAJOR / L1 / HEIGHT_SHARDED over the full worker grid
//      (shard [P/G, Cp]) and the matching memory_config attribute;
//   3. recreate the flatten reshape as a view with the same shards;
//   4. tilize in place (to_tensor_spec -> TILE, same shards);
//   5. move the conv weight and bias to DRAM (conv2d keeps them on host), reshape the weight
//      [Cout, Cp, 1, 1] -> [Cout, Cp], transpose to [Cp, Cout], cast to the conv2d weight dtype;
//   6. replace the conv2d by ttnn.linear with the conv's compute config and fused activation inside an
//      explicit 1-D height-sharded program config (out_block_h = 2), output TILE / L1 / HEIGHT_SHARDED
//      [P/G, Cout].
//
// Why linear and not conv2d: conv2d's automatic matmul config uses
// out_block_h = per_core_M; when the per-core shard is large (hundreds of KB)
// its static circular buffers grow with it, clash with the input and output
// shards and the program cannot be built.
// out_block_h = 2 runs the same kernel with small buffers.
//
// Requirements: P divisible by G, P/G and Cp and Cout multiples of 32;
// otherwise the stem is left as the optimizer placed it (with a warning).
//
// Runs inside the optimizer device-pass group after OperationValidationAndFallback
// and before conv2d weight preparation (and, without the optimizer, in the layout
// decomposition stage), so the raw conv weight is still the operand and the
// transposed weight it builds is const-eval hoisted afterwards.

// clang-format on

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/BFPDtypeParser.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSTEMFOLDLINEAR
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static bool allEqual(DenseI32ArrayAttr a, int32_t x) {
  return a && !a.empty() &&
         llvm::all_of(a.asArrayRef(), [x](int32_t e) { return e == x; });
}

// The single non-deallocate user of a value, or null.
static Operation *singleUser(Value v) {
  Operation *found = nullptr;
  for (Operation *u : v.getUsers()) {
    if (mlir::isa<DeallocateOp>(u)) {
      continue;
    }
    if (found) {
      return nullptr;
    }
    found = u;
  }
  return found;
}

static int64_t largestDivisorUpTo(int64_t value, int64_t maxDivisor) {
  for (int64_t d = std::min(value, maxDivisor); d >= 1; --d) {
    if (value % d == 0) {
      return d;
    }
  }
  return 1;
}

// The folded stem as it arrives here.
struct FoldedStem {
  PixelUnshuffleOp pu;
  SmallVector<Operation *, 4>
      viewChain; // reshape / layout ops between pu and conv
  Conv2dOp conv;
  int64_t pixels = 0, Cp = 0, Cout = 0;
};

// pixel_unshuffle(channels_last)
//   -> [reshape | to_tensor_spec | to_memory_config]* -> 1x1 conv2d.
static std::optional<FoldedStem> matchFoldedStem(PixelUnshuffleOp pu) {
  FoldedStem s;
  s.pu = pu;
  auto puTy = mlir::cast<RankedTensorType>(pu.getType());
  if (puTy.getRank() != 4) {
    return std::nullopt;
  }
  s.pixels = puTy.getDimSize(0) * puTy.getDimSize(1) * puTy.getDimSize(2);
  s.Cp = puTy.getDimSize(3);

  Value cur = pu.getResult();
  while (Operation *u = singleUser(cur)) {
    if (auto c = mlir::dyn_cast<Conv2dOp>(u)) {
      s.conv = c;
      break;
    }
    if (!mlir::isa<ReshapeOp, ToTensorSpecOp, ToMemoryConfigOp>(u)) {
      break;
    }
    s.viewChain.push_back(u);
    cur = u->getResult(0);
  }
  if (!s.conv) {
    pu.emitWarning(
        "stem fold: pixel_unshuffle(channels_last) is not consumed by a "
        "1x1 conv2d; leaving the stem unplaced");
    return std::nullopt;
  }
  Conv2dOp conv = s.conv;
  auto inTy = mlir::cast<RankedTensorType>(conv.getInput().getType());
  auto outTy = mlir::cast<RankedTensorType>(conv.getType());
  if (!allEqual(conv.getKernelSizeAttr(), 1) ||
      !allEqual(conv.getStrideAttr(), 1) ||
      !allEqual(conv.getPaddingAttr(), 0) || conv.getGroups() != 1 ||
      conv.getInChannels() != s.Cp || inTy.getRank() != 4 ||
      inTy.getDimSize(2) != s.pixels || inTy.getDimSize(3) != s.Cp ||
      outTy.getRank() != 4 || outTy.getDimSize(2) != s.pixels) {
    pu.emitWarning(
        "stem fold: consumer is not the expected flattened 1x1 conv2d; "
        "leaving the stem unplaced");
    return std::nullopt;
  }
  s.Cout = outTy.getDimSize(3);
  return s;
}

class TTNNStemFoldLinearPass
    : public impl::TTNNStemFoldLinearBase<TTNNStemFoldLinearPass> {
public:
  using impl::TTNNStemFoldLinearBase<
      TTNNStemFoldLinearPass>::TTNNStemFoldLinearBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    ttcore::DeviceAttr device = ttcore::lookupDevice(moduleOp);
    if (!device || device.getWorkerGrid().getShape().size() != 2) {
      return;
    }
    ArrayRef<int64_t> grid = device.getWorkerGrid().getShape();

    SmallVector<PixelUnshuffleOp, 8> ops;
    moduleOp.walk([&](PixelUnshuffleOp op) {
      if (op.getChannelsLast()) {
        ops.push_back(op);
      }
    });
    for (PixelUnshuffleOp pu : ops) {
      if (std::optional<FoldedStem> stem = matchFoldedStem(pu)) {
        place(*stem, grid);
      }
    }
  }

private:
  void place(FoldedStem &s, ArrayRef<int64_t> grid) {
    MLIRContext *ctx = &getContext();
    const int64_t numCores = grid[0] * grid[1];
    const int64_t rowsPerCore = s.pixels / numCores;
    if (s.pixels % numCores != 0 || rowsPerCore % 32 != 0 || s.Cp % 32 != 0 ||
        s.Cout % 32 != 0) {
      s.pu.emitWarning("stem fold: ")
          << s.pixels << " pixels x " << s.Cp << " -> " << s.Cout
          << " channels do not tile over " << numCores
          << " cores; leaving the stem unplaced";
      return;
    }

    // 1. The kernel reads row-major: bypass the tilize of the model input.
    if (auto spec = s.pu.getInput().getDefiningOp<ToTensorSpecOp>()) {
      auto inLayout = mlir::dyn_cast<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(spec.getInput().getType())
              .getEncoding());
      auto outLayout = mlir::dyn_cast<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(spec.getType()).getEncoding());
      if (inLayout && outLayout && !inLayout.isTiled() &&
          inLayout.isDeviceBufferType() &&
          inLayout.getElementType() == outLayout.getScalarElementType() &&
          spec.getResult().hasOneUse()) {
        s.pu.getInputMutable().assign(spec.getInput());
        spec.erase();
      }
    }

    // Full worker grid as one core range, height-sharded 64x1.
    auto crs = CoreRangeSetAttr::get(
        ctx,
        CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 0, 0),
                           CoreCoordAttr::get(ctx, grid[1] - 1, grid[0] - 1)));
    auto sharded = [&](RankedTensorType ty, Layout layout) {
      return utils::RankedTensorTypeFactory::create(
          ty, TTNNLayoutAttr::Builder(ty)
                  .setBufferType(BufferType::L1)
                  .setLayout(layout)
                  .setMemoryLayout(TensorMemoryLayout::HeightSharded)
                  .setGridShape({numCores, 1})
                  .setCoreRangeSet(crs)
                  .build());
    };

    // 2. pixel_unshuffle result: ROW_MAJOR, L1, HEIGHT_SHARDED,
    //    shard [rowsPerCore, Cp].
    auto puTy =
        sharded(mlir::cast<RankedTensorType>(s.pu.getType()), Layout::RowMajor);
    s.pu.getResult().setType(puTy);
    s.pu.setMemoryConfigAttr(
        MemoryConfigAttr::get(mlir::cast<TTNNLayoutAttr>(puTy.getEncoding())));

    OpBuilder b(s.conv);
    Location loc = s.conv.getLoc();

    // 3. Flatten view [1, 1, pixels, Cp], same shards.
    auto flatRmTy =
        utils::RankedTensorTypeFactory::create(puTy, {1, 1, s.pixels, s.Cp});
    Value flat = b.create<ReshapeOp>(
                      loc, flatRmTy, s.pu.getResult(),
                      b.getI32ArrayAttr({1, 1, static_cast<int32_t>(s.pixels),
                                         static_cast<int32_t>(s.Cp)}))
                     .getResult();

    // 4. In-place sharded tilize (decomposed later into one ttnn.to_layout).
    auto flatTileTy =
        utils::RankedTensorTypeFactory::create(flatRmTy, Layout::Tile);
    Value tiled = b.create<ToTensorSpecOp>(loc, flatTileTy, flat).getResult();

    // 5. Weight [Cout, Cp, 1, 1] (host) -> DRAM -> [Cout, Cp] -> [Cp, Cout]
    //    -> conv weight dtype.
    auto toDevice = [&](Value v) -> Value {
      auto ty = mlir::cast<RankedTensorType>(v.getType());
      auto l = mlir::dyn_cast<TTNNLayoutAttr>(ty.getEncoding());
      if (!l || l.isDeviceBufferType()) {
        return v;
      }
      auto devLayout = TTNNLayoutAttr::Builder(ty)
                           .setBufferType(BufferType::DRAM)
                           .setMemoryLayout(TensorMemoryLayout::Interleaved)
                           .setLayout(Layout::Tile)
                           .build();
      return b
          .create<ToTensorSpecOp>(
              loc, utils::RankedTensorTypeFactory::create(ty, devLayout), v)
          .getResult();
    };
    Value w = toDevice(s.conv.getWeight());
    auto w2Ty = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(w.getType()), {s.Cout, s.Cp});
    Value w2 =
        b.create<ReshapeOp>(loc, w2Ty, w,
                            b.getI32ArrayAttr({static_cast<int32_t>(s.Cout),
                                               static_cast<int32_t>(s.Cp)}))
            .getResult();
    auto wtTy = utils::RankedTensorTypeFactory::create(w2Ty, {s.Cp, s.Cout});
    Value wt = b.create<TransposeOp>(loc, wtTy, w2, b.getSI32IntegerAttr(0),
                                     b.getSI32IntegerAttr(1))
                   .getResult();
    if (targetDtype != BFPDtype::None) {
      wt = b.create<TypecastOp>(loc,
                                utils::RankedTensorTypeFactory::create(
                                    wtTy, bfpDtypeToDataType(targetDtype)),
                                wt)
               .getResult();
    }
    Value bias = s.conv.getBias() ? toDevice(s.conv.getBias()) : Value();

    // 6. linear: output TILE / L1 / HEIGHT_SHARDED [rowsPerCore, Cout],
    //    1-D program config with small output blocks and the conv's fused
    //    activation.
    auto outTy =
        sharded(mlir::cast<RankedTensorType>(s.conv.getType()), Layout::Tile);
    const int64_t perCoreM = rowsPerCore / 32, perCoreN = s.Cout / 32,
                  Kt = s.Cp / 32;
    UnaryWithParamAttr fusedActivation =
        s.conv.getConv2dConfigAttr()
            ? s.conv.getConv2dConfigAttr().getActivation()
            : UnaryWithParamAttr{};
    auto [gridX, gridY] = utils::getPhysicalGridDimensions(
        mlir::cast<TTNNLayoutAttr>(outTy.getEncoding()));
    auto programConfig = MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
        ctx, CoreCoordAttr::get(ctx, gridX, gridY),
        /*in0_block_w=*/static_cast<uint64_t>(Kt),
        /*out_subblock_h=*/1,
        /*out_subblock_w=*/
        static_cast<uint64_t>(largestDivisorUpTo(perCoreN, 2)),
        /*out_block_h=*/static_cast<uint64_t>(perCoreM % 2 == 0 ? 2 : 1),
        /*out_block_w=*/static_cast<uint64_t>(perCoreN),
        /*per_core_m=*/static_cast<uint64_t>(perCoreM),
        /*per_core_n=*/static_cast<uint64_t>(perCoreN),
        /*fuse_batch=*/true, fusedActivation, /*mcast_in0=*/false,
        /*gather_in0=*/false, CoreRangeSetAttr::get(ctx, {}),
        /*num_global_cb_receivers=*/0, /*untilize_out=*/false);
    auto linear = b.create<LinearOp>(
        loc, outTy, tiled, wt, bias, /*transpose_a=*/false,
        /*transpose_b=*/false, programConfig, /*activation=*/StringAttr{},
        s.conv.getComputeConfigAttr());

    s.conv.getResult().replaceAllUsesWith(linear.getResult());
    s.conv.erase();
    for (Operation *op : llvm::reverse(s.viewChain)) {
      if (op->use_empty()) {
        op->erase();
      }
    }
    TTMLIR_DEBUG(ttmlir::LogComponent::Optimizer,
                 "stem fold: pixel_unshuffle -> to_layout -> linear placed "
                 "height-sharded "
                 "over {0} cores ({1} pixels x {2} -> {3})",
                 numCores, s.pixels, s.Cp, s.Cout);
  }
};

} // namespace
} // namespace mlir::tt::ttnn
