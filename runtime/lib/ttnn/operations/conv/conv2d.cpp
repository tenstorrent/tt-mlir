// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

#include <cmath>

#include "ttnn/operations/experimental/quasar/conv2d/conv2d.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/quasar/pad/pad.hpp"
#include "ttnn/operations/experimental/quasar/slice/slice.hpp"
#include "ttnn/operations/experimental/quasar/to_device/to_device.hpp"
#include "ttnn/operations/experimental/quasar/tilize/tilize.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/operations/experimental/quasar/matmul/matmul.hpp"
#include "ttnn/operations/experimental/quasar/binary/binary.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose.hpp"

#include <cstdlib>

#include <vector>

#include <string>

namespace tt::runtime::ttnn::operations::conv {
using ::ttnn::Conv2dResultWithOptions;
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  LOG_ASSERT(op->kernel_size()->size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(op->stride()->size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(op->padding()->size() == 2 || op->padding()->size() == 4,
             "Padding expected to have 2 or 4 elements");
  LOG_ASSERT(op->dilation()->size() == 2,
             "Dilation expected to have 2 elements");

  std::array<uint32_t, 2> kernelSize, stride, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(op->padding()->begin(), 2, symPadding.begin());
    padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(op->padding()->begin(), 4, asymPadding.begin());
    padding = asymPadding;
  }

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  ::ttnn::Conv2dConfig conv2dConfig;
  if (op->conv2d_config()) {
    conv2dConfig = utils::createConv2dConfig(op->conv2d_config());
  }

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();
  // conv2d can carry a FUSED ACTIVATION in its Conv2dConfig
  // (utils.cpp:353 reads it from the flatbuffer). Both Quasar paths below
  // reimplement the convolution, so they have to apply it themselves -- dropping
  // it is silent and looks like a plausible-but-wrong result.
  //
  // This is what made the full ResNet-50 wrong while every op passed on its own:
  // the frontend fuses each relu that directly follows a convolution into it, so
  // the stem's graph has one conv2d and no ttnn.relu at all, and layer1 has 11
  // convolutions but only the 3 post-residual relus. Measured with the
  // activation dropped: stem pcc 0.954 (positives right, negatives never
  // clamped, magnitudes unchanged), layer1 0.339, whole model 0.071.
  //
  // Applied as add(x, 0) with the activation fused on the LHS, the same idiom
  // the runtime's own relu uses on Quasar (eltwise/unary/unary.cpp): a
  // tensor-scalar add with LHS activation fusion is inside the validated slice,
  // and adding 0.0f is exact in bf16.
  auto applyFusedActivation =
      [&](const ::ttnn::Tensor &t,
          const std::optional<::ttnn::MemoryConfig> &memoryConfig) {
        if (!conv2dConfig.activation.has_value()) {
          return t;
        }
        // UnaryWithParam (float params) converts to EltwiseUnaryWithParam (a
        // variant over float/int32/uint32) through its converting constructor,
        // so any fused activation carries over, not just parameterless ones.
        const std::array<::ttnn::operations::unary::EltwiseUnaryWithParam, 1>
            lhsActivations{
                ::ttnn::operations::unary::EltwiseUnaryWithParam(
                    *conv2dConfig.activation)};
        return ::ttnn::operations::experimental::quasar::binary::add(
            t, 0.0f, /*dtype=*/std::nullopt, memoryConfig,
            /*output=*/std::nullopt, /*post_activations=*/{},
            /*lhs_activations=*/lhsActivations);
      };


  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
  if (op->conv2d_slice_config()) {
    sliceConfig = utils::createConv2dSliceConfig(op->conv2d_slice_config());
  }

  // Quasar reimplements conv2d; the mainline op's program spec rejects the
  // Quasar compute config (TT_FATAL on holds_alternative<ComputeGen2Config> in
  // tt_metal/impl/metal2_host_api/program_spec.cpp). The Quasar entry point
  // takes the same arguments -- Conv2dConfig and Conv2dSliceConfig are the same
  // underlying types -- but returns its own, structurally identical, result
  // variant, so each branch unwraps its own.
  // ------------------------------------------------------------------
  // Quasar 1x1 fast path: a 1x1, stride-1, dilation-1, unit-group convolution
  // with no padding is exactly a matmul over the channel dimension --
  //     out[m][n] = sum_k in[m][k] * weight[n][k]
  // -- so decompose it into reshape + matmul/linear rather than going through
  // Quasar's conv2d factory, whose output writer does not publish its result
  // (tt-metal #48552). Every op used here is numerically verified on Quasar.
  // Not a heuristic: the decomposition is exact for these parameters.
  const bool paddingAllZero = std::visit(
      [](const auto &p) {
        return std::all_of(p.begin(), p.end(), [](uint32_t v) { return v == 0; });
      },
      padding);
  // Stride > 1 is allowed: a 1x1 kernel with stride s is a stride-s subsample of
  // the activation followed by the same matmul, and the subsample is a strided
  // quasar::slice. That matters because ResNet-50's three downsample convs are
  // 1x1 stride 2, and Quasar's own conv2d hangs on them -- it writes its DFB
  // configs and then never leaves the core barrier.
  const bool quasarMatmulEquivalent =
      utils::isQuasar() && kernelSize[0] == 1 && kernelSize[1] == 1 &&
      stride[0] >= 1 && stride[1] >= 1 && dilation[0] == 1 &&
      dilation[1] == 1 && op->groups() == 1 && paddingAllZero;

  if (quasarMatmulEquivalent) {
    const uint32_t M = op->batch_size() * op->input_height() * op->input_width();
    const uint32_t K = op->in_channels();
    const uint32_t N = op->out_channels();

    // Activations arrive channel-last as [1, 1, N*H*W, C_in] (the frontend
    // permutes NCHW->NHWC and flattens ahead of the conv), and the weight is
    // [C_out, C_in, 1, 1]. Fold the activation to [1, 1, M, K] -- for a 1x1 conv
    // that is already its shape, so the reshape costs nothing.
    const std::vector<int32_t> actShape = {1, 1, static_cast<int32_t>(M),
                                           static_cast<int32_t>(K)};
    // The weight (and bias) reach conv2d as host, row-major tensors -- ttnn's
    // conv2d normally prepares them on device itself. Do that here before the
    // matmul, using the Quasar to_device/to_layout paths.
    auto onDevice = [&](const ::ttnn::Tensor &t) {
      return ::ttnn::is_device_tensor(t)
                 ? t
                 : ::ttnn::operations::experimental::quasar::to_device(
                       t, &targetDevice, std::nullopt);
    };
    // Tilize through logical data, not through a device layout op.
    //
    // Quasar's device ROW_MAJOR -> TILE conversion corrupts some shapes:
    // measured, a ROW_MAJOR DRAM tensor of 64x64 comes back differing from its
    // input by 5.65625 (full scale, confined to the first tile-row) while 64x256
    // in the same graph is exact. quasar::tilize and quasar::to_layout give
    // byte-identical wrong results -- they share the device op -- and a device
    // barrier makes no difference, so it is the conversion itself. That single
    // call made an identity-skip ResNet-50 block's third convolution 83% wrong
    // while the two convolutions beside it were correct.
    //
    // Tensor::from_vector tilizes on the host and copies down, the same route
    // the weight already uses. It costs a host round trip per convolution, which
    // is slow but correct; the device path can come back once the conversion is
    // fixed in tt-metal.
    auto tilize = [&](const ::ttnn::Tensor &t) -> ::ttnn::Tensor {
      if (t.layout() == ::ttnn::Layout::TILE) {
        return t;
      }
      // Narrowly gated. The device conversion is measured wrong for a 64x64
      // tensor and exact for 64x256, [1,64] and [1,256], so the host route is
      // taken only for a square-ish tile block: both trailing extents at least
      // one tile, and the width no wider than two tiles. Everything else keeps
      // the device path, which matters because the extra host round trips change
      // the DRAM allocation pattern and that alone was enough to hang a later
      // convolution (a 3x3 stride-2 512->512 at 2x2, which completes without
      // them).
      //
      // The rule is fitted to one measured shape; the general condition under
      // which Quasar's tilize corrupts is not known. It is a workaround for a
      // tt-metal defect, not a characterisation of it.
      const auto &tl0 = t.logical_shape();
      const bool deviceTilizeSuspect =
          tl0.rank() >= 2 &&
          tl0[tl0.rank() - 2] >= ::tt::constants::TILE_HEIGHT &&
          tl0[tl0.rank() - 1] >= ::tt::constants::TILE_WIDTH &&
          tl0[tl0.rank() - 1] <= 2 * ::tt::constants::TILE_WIDTH;
      if (!deviceTilizeSuspect) {
        return ::ttnn::operations::experimental::quasar::to_layout(
            t, ::ttnn::Layout::TILE, std::nullopt, std::nullopt);
      }
      auto viaHost = [&](auto probe) -> ::ttnn::Tensor {
        using T = decltype(probe);
        std::vector<T> data = t.to_vector<T>();
        const ::tt::tt_metal::TensorSpec spec(
            t.logical_shape(),
            ::tt::tt_metal::TensorLayout(t.dtype(), ::ttnn::Layout::TILE,
                                         t.memory_config()));
        return ::ttnn::Tensor::from_vector(std::move(data), spec,
                                          &targetDevice);
      };
      if (t.dtype() == ::ttnn::DataType::FLOAT32) {
        return viaHost(float{});
      }
      if (t.dtype() == ::ttnn::DataType::BFLOAT16) {
        return viaHost(bfloat16());
      }
      return ::ttnn::operations::experimental::quasar::to_layout(
          t, ::ttnn::Layout::TILE, std::nullopt, std::nullopt);
    };

    auto onDeviceTiled = [&](const ::ttnn::Tensor &t) {
      return tilize(onDevice(t));
    };

    ::ttnn::Tensor act = onDevice(input);
    if (stride[0] > 1 || stride[1] > 1) {
      // The activation arrives flattened as [1, 1, N*H*W, C]. Unflatten it so H
      // and W are separate axes, take every stride-th row and column, and
      // reflatten. With a 1x1 kernel and no padding that is exactly what the
      // strided convolution selects.
      const uint32_t H = op->input_height();
      const uint32_t W = op->input_width();
      const uint32_t batch = op->batch_size();
      const std::vector<int32_t> spatialShape = {
          static_cast<int32_t>(batch), static_cast<int32_t>(H),
          static_cast<int32_t>(W), static_cast<int32_t>(K)};
      // ROW_MAJOR for the same reason as the tap path: a strided slice of a TILE
      // tensor is wrong once the second-last dim crosses a tile boundary, and
      // ResNet-50's 1x1 stride-2 convolutions run at 56, 28 and 14 spatial.
      if (act.layout() != ::ttnn::Layout::ROW_MAJOR) {
        act = ::ttnn::operations::experimental::quasar::to_layout(
            act, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
      }
      ::ttnn::Tensor spatial =
          ::ttnn::operations::experimental::quasar::reshape(act, spatialShape,
                                                            std::nullopt);
      // Call the Span overload directly: the SmallVector one forwards to it in a
      // way that is ambiguous at instantiation (slice.hpp:32).
      const std::vector<int32_t> begins = {0, 0, 0, 0};
      const std::vector<int32_t> ends = {
          static_cast<int32_t>(batch), static_cast<int32_t>(H),
          static_cast<int32_t>(W), static_cast<int32_t>(K)};
      const std::vector<int32_t> steps = {
          1, static_cast<int32_t>(stride[0]), static_cast<int32_t>(stride[1]),
          1};
      ::ttnn::Tensor strided =
          ::ttnn::operations::experimental::quasar::slice<int32_t>(
              spatial, ::ttsl::Span<const int32_t>(begins),
              ::ttsl::Span<const int32_t>(ends),
              ::ttsl::Span<const int32_t>(steps), std::nullopt, std::nullopt,
              std::nullopt, std::nullopt);
      // Ceiling division: with stride s, rows 0, s, 2s, ... are kept.
      const uint32_t outH = (H + stride[0] - 1) / stride[0];
      const uint32_t outW = (W + stride[1] - 1) / stride[1];
      const std::vector<int32_t> stridedFlat = {
          1, 1, static_cast<int32_t>(batch * outH * outW),
          static_cast<int32_t>(K)};
      act = ::ttnn::operations::experimental::quasar::reshape(
          strided, stridedFlat, std::nullopt);
      act = tilize(act);
    } else {
      act = ::ttnn::operations::experimental::quasar::reshape(tilize(act),
                                                              actShape,
                                                              std::nullopt);
    }
    // Prepare the weight through logical data instead of through layout ops.
    // Tensor::to_vector() returns elements in logical row-major order whatever
    // the physical layout is, and Tensor::from_vector() tilizes on the host and
    // copies to the device without launching a kernel. That sidesteps every
    // layout contract that has bitten this path: device ROW_MAJOR padding of a
    // weight whose last dim is 1, and quasar::transpose of an already-tiled
    // matrix (measured: values preserved, positions scrambled -- pcc 0.116,
    // sorted-multiset 0.978). The [C_out, C_in] -> [C_in, C_out] transpose
    // becomes a plain CPU loop, and matmul receives a [K, N] operand so it needs
    // no transpose_b (which lowers to mainline ttnn::prim::transpose and is
    // refused on Quasar).
    const ::ttnn::MemoryConfig weightMemoryConfig =
        outputMemoryConfig.value_or(::ttnn::DRAM_MEMORY_CONFIG);
    auto transposeWeightToDevice = [&](auto probe) -> ::ttnn::Tensor {
      using T = decltype(probe);
      const std::vector<T> src = weight.to_vector<T>();
      LOG_ASSERT(src.size() == static_cast<size_t>(N) * static_cast<size_t>(K),
                 "Weight element count does not match C_out * C_in");
      std::vector<T> dst(src.size());
      for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k = 0; k < K; k++) {
          dst[static_cast<size_t>(k) * N + n] =
              src[static_cast<size_t>(n) * K + k];
        }
      }
      const ::tt::tt_metal::TensorSpec spec(
          ::ttnn::Shape({K, N}),
          ::tt::tt_metal::TensorLayout(weight.dtype(), ::ttnn::Layout::TILE,
                                       weightMemoryConfig));
      return ::ttnn::Tensor::from_vector(std::move(dst), spec, &targetDevice);
    };
    ::ttnn::Tensor wt = weight.dtype() == ::ttnn::DataType::FLOAT32
                            ? transposeWeightToDevice(float{})
                            : transposeWeightToDevice(bfloat16());
    if (bias.has_value()) {
      bias = onDeviceTiled(*bias);
    }

    // Same factory choice as runtime/lib/ttnn/operations/matmul/matmul.cpp:
    // auto-selection lands on the mcast-1d factory whose writer never consumes
    // the packed output.
    const std::optional<
        ::ttnn::operations::experimental::quasar::matmul::MatmulProgramConfig>
        mmConfig = ::ttnn::operations::experimental::quasar::matmul::
            MatmulMultiCoreProgramConfig{};

    ::ttnn::Tensor flat =
        bias.has_value()
            ? ::ttnn::operations::experimental::quasar::matmul::linear(
                  act, wt, bias, /*transpose_a=*/false, /*transpose_b=*/false,
                  outputMemoryConfig, outputDtype, mmConfig,
                  /*activation=*/std::nullopt,
                  /*compute_kernel_config=*/computeConfig)
            : ::ttnn::operations::experimental::quasar::matmul::matmul(
                  act, wt, /*transpose_a=*/false, /*transpose_b=*/false,
                  outputMemoryConfig, outputDtype, mmConfig,
                  /*activation=*/std::nullopt,
                  /*compute_kernel_config=*/computeConfig);

    // Does the PREPARED WEIGHT still hold the original weight's data? The matmul
    // check reads `wt` back off the device, so a bad transpose here is invisible
    // there: the reference and the device would share the same wrong operand and
    // agree. Same blind spot as checking act against the convolution's input.
    if (std::getenv("TTMLIR_CONV_CHECK") &&
        weight.dtype() == ::ttnn::DataType::BFLOAT16) {
      const std::vector<bfloat16> w0 = weight.to_vector<bfloat16>();
      const std::vector<bfloat16> wdev = wt.to_vector<bfloat16>();
      double maxAbs = 0.0;
      size_t bad = 0;
      ssize_t firstBad = -1;
      if (w0.size() == static_cast<size_t>(N) * K &&
          wdev.size() == static_cast<size_t>(K) * N) {
        for (uint32_t n = 0; n < N; n++) {
          for (uint32_t k = 0; k < K; k++) {
            const double d =
                std::abs(static_cast<float>(w0[static_cast<size_t>(n) * K + k]) -
                         static_cast<float>(wdev[static_cast<size_t>(k) * N + n]));
            if (d > maxAbs) { maxAbs = d; }
            if (d != 0.0) {
              bad++;
              if (firstBad < 0) {
                firstBad = static_cast<ssize_t>(n) * K + k;
              }
            }
          }
        }
      }
      std::fprintf(stderr,
                   "[convcheck] weight K=%u N=%u n_src=%zu n_dev=%zu bad=%zu "
                   "max_abs=%.6f first_bad=%zd\n",
                   K, N, w0.size(), wdev.size(), bad, maxAbs, firstBad);
      std::fflush(stderr);
    }

    // Does the activation reaching matmul still hold the input's logical data?
    // With stride 1 this is an identity: to_vector returns logical row-major
    // order whatever the physical layout, so tilizing must not change a single
    // element. Checking matmul against a reference computed FROM act cannot see
    // a corrupt act -- both sides would agree.
    if (std::getenv("TTMLIR_CONV_CHECK") && stride[0] == 1 && stride[1] == 1 &&
        act.dtype() == ::ttnn::DataType::BFLOAT16 &&
        input.dtype() == ::ttnn::DataType::BFLOAT16) {
      const std::vector<bfloat16> in0 = input.to_vector<bfloat16>();
      const std::vector<bfloat16> a0 = act.to_vector<bfloat16>();
      double maxAbs = 0.0;
      size_t firstBad = SIZE_MAX;
      const size_t n = std::min(in0.size(), a0.size());
      for (size_t i = 0; i < n; i++) {
        const double d =
            std::abs(static_cast<float>(in0[i]) - static_cast<float>(a0[i]));
        if (d > maxAbs) {
          maxAbs = d;
        }
        if (d != 0.0 && firstBad == SIZE_MAX) {
          firstBad = i;
        }
      }
      std::fprintf(stderr,
                   "[convcheck] act_vs_input n_in=%zu n_act=%zu max_abs=%.6f "
                   "first_bad=%zd\n",
                   in0.size(), a0.size(), maxAbs,
                   firstBad == SIZE_MAX ? -1 : static_cast<ssize_t>(firstBad));
      std::fflush(stderr);
    }

    // Env-gated numeric check of the 1x1 path: recompute the matmul on the host
    // from the same logical operands the device was given, and report how far
    // the device result is. This separates "the operands reaching matmul are
    // wrong" from "matmul is wrong", which shape-level reasoning cannot.
    if (std::getenv("TTMLIR_CONV_CHECK") &&
        act.dtype() == ::ttnn::DataType::BFLOAT16 &&
        flat.dtype() == ::ttnn::DataType::BFLOAT16) {
      const std::vector<bfloat16> a = act.to_vector<bfloat16>();
      const std::vector<bfloat16> w = wt.to_vector<bfloat16>();
      const std::vector<bfloat16> r = flat.to_vector<bfloat16>();
      const size_t M = (K != 0) ? a.size() / K : 0;
      if (M != 0 && w.size() == static_cast<size_t>(K) * N &&
          r.size() == M * N) {
        // The reference must include the bias: `flat` comes from linear() when a
        // bias is present, so comparing against a plain act@wt reports a bogus
        // mismatch for every convolution that has one (which is every real
        // ResNet convolution, since BatchNorm folds into it).
        std::vector<float> biasVec;
        if (bias.has_value() && bias->dtype() == ::ttnn::DataType::BFLOAT16) {
          const std::vector<bfloat16> bv = bias->to_vector<bfloat16>();
          biasVec.reserve(bv.size());
          for (const bfloat16 &e : bv) {
            biasVec.push_back(static_cast<float>(e));
          }
        }
        double num = 0.0, da = 0.0, db = 0.0, maxAbs = 0.0;
        double meanX = 0.0, meanY = 0.0;
        for (size_t i = 0; i < M * N; i++) {
          meanY += static_cast<float>(r[i]);
        }
        meanY /= static_cast<double>(M * N);
        std::vector<double> ref(M * N, 0.0);
        for (size_t m = 0; m < M; m++) {
          for (uint32_t k = 0; k < K; k++) {
            const double av = static_cast<float>(a[m * K + k]);
            if (av == 0.0) {
              continue;
            }
            for (uint32_t n = 0; n < N; n++) {
              ref[m * N + n] += av * static_cast<float>(w[k * N + n]);
            }
          }
        }
        if (biasVec.size() == static_cast<size_t>(N)) {
          for (size_t mrow = 0; mrow < M; mrow++) {
            for (uint32_t n = 0; n < N; n++) {
              ref[mrow * N + n] += biasVec[n];
            }
          }
        }
        for (size_t i = 0; i < M * N; i++) {
          meanX += ref[i];
        }
        meanX /= static_cast<double>(M * N);
        for (size_t i = 0; i < M * N; i++) {
          const double x = ref[i] - meanX;
          const double y = static_cast<float>(r[i]) - meanY;
          num += x * y;
          da += x * x;
          db += y * y;
          maxAbs = std::max(maxAbs, std::abs(ref[i] - static_cast<float>(r[i])));
        }
        const double pcc = (da > 0 && db > 0) ? num / std::sqrt(da * db) : 0.0;
        std::fprintf(stderr,
                     "[convcheck] M=%zu K=%u N=%u matmul_pcc=%.6f max_abs=%.6f\n",
                     M, K, N, pcc, maxAbs);
        std::fflush(stderr);
      } else {
        std::fprintf(stderr,
                     "[convcheck] size mismatch a=%zu w=%zu r=%zu K=%u N=%u\n",
                     a.size(), w.size(), r.size(), K, N);
        std::fflush(stderr);
      }
    }

    flat = applyFusedActivation(flat, outputMemoryConfig);

    const ::ttnn::Shape outShape =
        utils::toTTNNShape(*op->out()->desc()->shape());
    std::vector<int32_t> outDims;
    outDims.reserve(outShape.rank());
    for (size_t i = 0; i < outShape.rank(); i++) {
      outDims.push_back(static_cast<int32_t>(outShape[i]));
    }
    ::ttnn::Tensor reshaped =
        ::ttnn::operations::experimental::quasar::reshape(flat, outDims,
                                                         std::nullopt);

    // Hand back the layout the graph declares for this conv's output, not the
    // input's. The flatbuffer's tile shape is the contract the next op was
    // compiled against: returning ROW_MAJOR where the graph says TILE makes
    // downstream ops pick their ROW_MAJOR program factories, several of which
    // are unported on Quasar and refuse ("DataMovementKernel is not supported on
    // Quasar"). That is what broke the residual add in a full ResNet block while
    // every single-op conv probe still passed.
    const ::ttnn::Layout expectedLayout =
        ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
    if (reshaped.layout() != expectedLayout) {
      if (utils::quasarTilizeNeedsHostRoute(reshaped, expectedLayout,
                                            std::nullopt)) {
        const std::vector<bfloat16> d = reshaped.to_vector<bfloat16>();
        const ::tt::tt_metal::TensorSpec spec(
            reshaped.logical_shape(),
            ::tt::tt_metal::TensorLayout(reshaped.dtype(), expectedLayout,
                                         reshaped.memory_config()));
        reshaped = ::ttnn::Tensor::from_vector(std::move(d), spec,
                                               &targetDevice);
      } else {
        reshaped = ::ttnn::operations::experimental::quasar::to_layout(
            reshaped, expectedLayout, std::nullopt, std::nullopt);
      }
    }
    // Does the tensor this path hands back carry the same physical spec the
    // graph declared for it? to_vector says the values are right; a downstream
    // op reads the physical buffer, not the logical view, so a padded shape or
    // alignment that differs is invisible here and wrong there.
    // The tail: does the tensor we hand back still hold the matmul's values? The
    // matmul check above stops at `flat`; everything after it -- the reshape to
    // the declared output shape and the layout conversion -- is unverified, and a
    // correct spec says nothing about contents.
    if (std::getenv("TTMLIR_CONV_CHECK") &&
        reshaped.dtype() == ::ttnn::DataType::BFLOAT16 &&
        flat.dtype() == ::ttnn::DataType::BFLOAT16) {
      const std::vector<bfloat16> a = flat.to_vector<bfloat16>();
      const std::vector<bfloat16> b = reshaped.to_vector<bfloat16>();
      double maxAbs = 0.0;
      ssize_t firstBad = -1;
      const size_t n = std::min(a.size(), b.size());
      for (size_t i = 0; i < n; i++) {
        const double d =
            std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        if (d > maxAbs) { maxAbs = d; }
        if (d != 0.0 && firstBad < 0) { firstBad = static_cast<ssize_t>(i); }
      }
      std::fprintf(stderr,
                   "[convcheck] tail flat->out n=%zu/%zu max_abs=%.6f first_bad=%zd\n",
                   a.size(), b.size(), maxAbs, firstBad);
      std::fflush(stderr);
    }

    if (std::getenv("TTMLIR_CONV_CHECK")) {
      const ::ttnn::Tensor declared =
          utils::allocateTensorOnDevice(op->out(), targetDevice);
      const auto &got = reshaped.tensor_spec();
      const auto &want = declared.tensor_spec();
      const bool same = got == want;
      auto shapeStr = [](const ::ttnn::Shape &sh) {
        std::string out = "[";
        for (size_t i = 0; i < sh.rank(); i++) {
          out += (i ? "," : "") + std::to_string(sh[i]);
        }
        return out + "]";
      };
      std::fprintf(stderr,
                   "[convcheck] out spec %s: got padded=%s layout=%d ; "
                   "want padded=%s layout=%d\n",
                   same ? "MATCH" : "DIFFER",
                   shapeStr(got.padded_shape()).c_str(),
                   static_cast<int>(got.layout()),
                   shapeStr(want.padded_shape()).c_str(),
                   static_cast<int>(want.layout()));
      std::fflush(stderr);
    }

    tensorPool.insertTTNNTensorAndValidate(op->out(), reshaped);
    return;
  }

  // ------------------------------------------------------------------
  // Quasar general convolution: decompose into one matmul per kernel tap.
  //
  // Quasar's own conv2d hangs in a way that depends on the problem size, not the
  // stride: a 3x3 stride-1 pad-1 convolution passes at 32x32 spatial
  // (pcc 0.999882) but hangs at 16x16, writing its DFB configs and then never
  // leaving the core barrier. ResNet-50 runs its 3x3 convolutions at 56, 28, 14
  // and 7 spatial, so most of them are in the hanging range.
  //
  // A convolution is a sum of 1x1 convolutions over the kernel taps:
  //     out[n][oh][ow][co] = sum_{i,j} sum_ci pad_in[n][oh*sh+i][ow*sw+j][ci]
  //                                           * w[co][ci][i][j]
  // Each (i, j) term is a strided slice of the zero-padded activation times a
  // [C_in, C_out] matrix -- exactly the 1x1 path above, which is verified. So
  // zero-pad once, then accumulate kh*kw matmuls. This costs kh*kw matmuls
  // instead of one convolution, which is the price of not depending on
  // quasar::conv2d.
  //
  // Padding is done in ROW_MAJOR on purpose: quasar::pad routes a tiled tensor
  // through detail::invoke_tile, which calls the mainline
  // ttnn::fill_implicit_tile_padding that Quasar refuses. invoke_rm does not.
  const bool quasarTapDecomposition =
      utils::isQuasar() && op->groups() == 1 && dilation[0] == 1 &&
      dilation[1] == 1 && (kernelSize[0] > 1 || kernelSize[1] > 1);

  // Count non-finite elements at a named stage. The emulator returned inf for a
  // 4x4-spatial convolution that craq-sim computes exactly, and DRAM there is not
  // zero-initialised the way a simulator's is, so the question is which stage
  // first reads memory nobody wrote.
  auto reportNonFinite = [](const ::ttnn::Tensor &t, const char *label) {
    if (!std::getenv("TTMLIR_CONV_CHECK") ||
        t.dtype() != ::ttnn::DataType::BFLOAT16) {
      return;
    }
    const std::vector<bfloat16> v = t.to_vector<bfloat16>();
    size_t bad = 0;
    float firstBad = 0.0f;
    for (size_t i = 0; i < v.size(); i++) {
      const float f = static_cast<float>(v[i]);
      if (!std::isfinite(f)) {
        if (bad == 0) {
          firstBad = f;
        }
        bad++;
      }
    }
    std::fprintf(stderr, "[nonfinite] %-14s n=%zu bad=%zu first=%g\n", label,
                 v.size(), bad, static_cast<double>(firstBad));
    std::fflush(stderr);
  };

  if (quasarTapDecomposition) {
    const uint32_t N = op->batch_size();
    const uint32_t H = op->input_height();
    const uint32_t W = op->input_width();
    const uint32_t K = op->in_channels();
    const uint32_t Cout = op->out_channels();
    const uint32_t kh = kernelSize[0];
    const uint32_t kw = kernelSize[1];

    // 4-element padding is {top, bottom, left, right} (conv2d_utils.cpp:278
    // sums [0]+[1] for height and [2]+[3] for width).
    const auto pads = std::visit(
        [](const auto &p) -> std::array<uint32_t, 4> {
          if constexpr (std::tuple_size_v<std::decay_t<decltype(p)>> == 2) {
            return {p[0], p[0], p[1], p[1]};
          } else {
            return {p[0], p[1], p[2], p[3]};
          }
        },
        padding);
    const uint32_t Hp = H + pads[0] + pads[1];
    const uint32_t Wp = W + pads[2] + pads[3];
    const uint32_t outH = (Hp - kh) / stride[0] + 1;
    const uint32_t outW = (Wp - kw) / stride[1] + 1;

    // Activation: to ROW_MAJOR, unflatten the spatial axes, zero-pad, tilize.
    ::ttnn::Tensor act = ::ttnn::is_device_tensor(input)
                             ? input
                             : ::ttnn::operations::experimental::quasar::
                                   to_device(input, &targetDevice, std::nullopt);
    if (act.layout() != ::ttnn::Layout::ROW_MAJOR) {
      act = ::ttnn::operations::experimental::quasar::to_layout(
          act, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
    }
    const std::vector<int32_t> spatialShape = {
        static_cast<int32_t>(N), static_cast<int32_t>(H),
        static_cast<int32_t>(W), static_cast<int32_t>(K)};
    act = ::ttnn::operations::experimental::quasar::reshape(act, spatialShape,
                                                            std::nullopt);
    if (pads[0] || pads[1] || pads[2] || pads[3]) {
      const ::ttsl::SmallVector<
          ::ttnn::operations::experimental::quasar::PadSpecDim>
          padSpec = {{0u, 0u}, {pads[0], pads[1]}, {pads[2], pads[3]}, {0u, 0u}};
      act = ::ttnn::operations::experimental::quasar::pad(
          act, padSpec, 0.0f, /*use_multicore=*/true, std::nullopt,
          std::nullopt);
      reportNonFinite(act, "after_pad");
    }
    // Deliberately left ROW_MAJOR: the taps are sliced out of it below, and a
    // strided slice of a TILE tensor is wrong once the second-last dim crosses a
    // tile boundary. Measured: a 3x3 pad-1 convolution decomposes correctly at
    // 16x16 spatial (padded extent 18, inside one 32-row tile, pcc 0.999843) but
    // not at 32x32 (padded extent 34, spanning two, pcc 0.918844). Slice first,
    // then tilize each tap.

    // One [C_in, C_out] weight matrix per tap, built from the weight's logical
    // row-major data ([C_out, C_in, kh, kw]) the same way the 1x1 path does.
    const ::ttnn::MemoryConfig tapMemoryConfig =
        outputMemoryConfig.value_or(::ttnn::DRAM_MEMORY_CONFIG);


    // Collected, then summed pairwise rather than folded left. Each add rounds in
    // bf16, and a left fold of n terms accumulates error like sqrt(n) * eps: for
    // the 7x7 stem that is sqrt(49) * 0.0039 ~= 0.027, which is what a left fold
    // measured (max relative error 0.02803, just over the 0.02 gate, with pcc
    // still 0.9995). Pairwise summation makes the depth log2(n) instead of n.
    // One matmul over all taps, not one matmul per tap plus a summing tree.
    //
    // Summing the taps as separate bf16 tensor adds rounds at every step, and that
    // is the whole of the model's residual error. Measured on CPU for a 3x3
    // 256->256 convolution at 2x2 spatial: the bf16 floor is err/scale 0.00256 and
    // a bf16 pairwise tap sum is 0.00560 -- and the device measured 0.00557, a
    // match to three digits. Compounded over ResNet-50's 53 convolutions that took
    // the model from a ~1% bf16 floor to 7%.
    //
    // Building the im2col matrix instead puts every tap in one contraction, so the
    // summation happens inside the matmul's own fp32 accumulator and the result
    // lands at the bf16 floor. It is assembled on the host because Quasar has no
    // concat, and because the alternative -- slice_write per tap -- would keep the
    // per-tap program launches this is meant to remove. On the emulator that
    // matters twice over: ~35 launches per convolution become one.
    const uint32_t T = kh * kw;
    const uint32_t M = N * outH * outW;
    auto fusedIm2Col = [&](auto probe) -> ::ttnn::Tensor {
      using TT = decltype(probe);
      const std::vector<TT> src = act.to_vector<TT>();
      LOG_ASSERT(src.size() == static_cast<size_t>(N) * Hp * Wp * K,
                 "Padded activation size does not match N*Hp*Wp*C_in");
      std::vector<TT> dst(static_cast<size_t>(M) * T * K, TT{});
      for (uint32_t n = 0; n < N; n++) {
        for (uint32_t oh = 0; oh < outH; oh++) {
          for (uint32_t ow = 0; ow < outW; ow++) {
            const size_t row = (static_cast<size_t>(n) * outH + oh) * outW + ow;
            for (uint32_t i = 0; i < kh; i++) {
              const uint32_t h = i + oh * stride[0];
              for (uint32_t j = 0; j < kw; j++) {
                const uint32_t w = j + ow * stride[1];
                const size_t tap = static_cast<size_t>(i) * kw + j;
                const size_t srcBase =
                    ((static_cast<size_t>(n) * Hp + h) * Wp + w) * K;
                const size_t dstBase = row * T * K + tap * K;
                for (uint32_t ci = 0; ci < K; ci++) {
                  dst[dstBase + ci] = src[srcBase + ci];
                }
              }
            }
          }
        }
      }
      const ::tt::tt_metal::TensorSpec spec(
          ::ttnn::Shape({1, 1, M, T * K}),
          ::tt::tt_metal::TensorLayout(act.dtype(), ::ttnn::Layout::TILE,
                                       tapMemoryConfig));
      return ::ttnn::Tensor::from_vector(std::move(dst), spec, &targetDevice);
    };
    // And the matching [T*C_in, C_out] weight, laid out so row tap*C_in + ci pairs
    // with the column this tap wrote.
    auto fusedWeight = [&](auto probe) -> ::ttnn::Tensor {
      using TT = decltype(probe);
      const std::vector<TT> src = weight.to_vector<TT>();
      LOG_ASSERT(src.size() == static_cast<size_t>(Cout) * K * kh * kw,
                 "Weight element count does not match C_out * C_in * kh * kw");
      std::vector<TT> dst(static_cast<size_t>(T) * K * Cout);
      for (uint32_t co = 0; co < Cout; co++) {
        for (uint32_t ci = 0; ci < K; ci++) {
          for (uint32_t i = 0; i < kh; i++) {
            for (uint32_t j = 0; j < kw; j++) {
              const size_t tap = static_cast<size_t>(i) * kw + j;
              dst[(tap * K + ci) * Cout + co] =
                  src[((static_cast<size_t>(co) * K + ci) * kh + i) * kw + j];
            }
          }
        }
      }
      const ::tt::tt_metal::TensorSpec spec(
          ::ttnn::Shape({T * K, Cout}),
          ::tt::tt_metal::TensorLayout(weight.dtype(), ::ttnn::Layout::TILE,
                                       tapMemoryConfig));
      return ::ttnn::Tensor::from_vector(std::move(dst), spec, &targetDevice);
    };

    const bool f32 = act.dtype() == ::ttnn::DataType::FLOAT32;
    ::ttnn::Tensor im2col = f32 ? fusedIm2Col(float{}) : fusedIm2Col(bfloat16());
    ::ttnn::Tensor fusedWt = weight.dtype() == ::ttnn::DataType::FLOAT32
                                 ? fusedWeight(float{})
                                 : fusedWeight(bfloat16());
    const std::optional<
        ::ttnn::operations::experimental::quasar::matmul::MatmulProgramConfig>
        fusedConfig = ::ttnn::operations::experimental::quasar::matmul::
            MatmulMultiCoreProgramConfig{};
    ::ttnn::Tensor result =
        ::ttnn::operations::experimental::quasar::matmul::matmul(
            im2col, fusedWt, /*transpose_a=*/false, /*transpose_b=*/false,
            outputMemoryConfig, outputDtype, fusedConfig,
            /*activation=*/std::nullopt,
            /*compute_kernel_config=*/computeConfig);

    if (bias.has_value()) {
      ::ttnn::Tensor biasTensor =
          ::ttnn::is_device_tensor(*bias)
              ? *bias
              : ::ttnn::operations::experimental::quasar::to_device(
                    *bias, &targetDevice, std::nullopt);
      if (biasTensor.layout() != ::ttnn::Layout::TILE) {
        // A folded BatchNorm leaves a bias of shape [1, 1, 1, C_out]: the
        // second-last extent is 1, so it is never tile aligned and takes the
        // same host route as the taps.
        if (utils::quasarTilizeNeedsHostRoute(biasTensor, ::ttnn::Layout::TILE,
                                              std::nullopt)) {
          const std::vector<bfloat16> d = biasTensor.to_vector<bfloat16>();
          const ::tt::tt_metal::TensorSpec spec(
              biasTensor.logical_shape(),
              ::tt::tt_metal::TensorLayout(biasTensor.dtype(),
                                           ::ttnn::Layout::TILE,
                                           biasTensor.memory_config()));
          biasTensor = ::ttnn::Tensor::from_vector(std::move(d), spec,
                                                   &targetDevice);
        } else {
          biasTensor = ::ttnn::operations::experimental::quasar::to_layout(
              biasTensor, ::ttnn::Layout::TILE, std::nullopt, std::nullopt);
        }
      }
      result = ::ttnn::operations::experimental::quasar::binary::add(result,
                                                                     biasTensor);
    }

    result = applyFusedActivation(result, outputMemoryConfig);

    const ::ttnn::Shape convOutShape =
        utils::toTTNNShape(*op->out()->desc()->shape());
    std::vector<int32_t> convOutDims;
    convOutDims.reserve(convOutShape.rank());
    for (size_t i = 0; i < convOutShape.rank(); i++) {
      convOutDims.push_back(static_cast<int32_t>(convOutShape[i]));
    }
    result = ::ttnn::operations::experimental::quasar::reshape(
        result, convOutDims, std::nullopt);
    const ::ttnn::Layout expectedTapLayout =
        ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
    if (result.layout() != expectedTapLayout) {
      result = ::ttnn::operations::experimental::quasar::to_layout(
          result, expectedTapLayout, std::nullopt, std::nullopt);
    }
    tensorPool.insertTTNNTensorAndValidate(op->out(), result);
    return;
  }

  ::ttnn::Tensor out;
  if (utils::isQuasar()) {
    // Quasar's conv2d hangs for stride > 1: it writes its DFB configs and then
    // never leaves the core barrier (no watchdog output, killed at the 1800 s
    // timeout). Stride 1 works -- a 3x3 stride-1 padded convolution measures
    // pcc 0.999882 -- so run at stride 1 and subsample the output.
    //
    // The identity is exact, not an approximation. With
    //     out_s[oh][ow] = sum_ij in[oh*s + i - p][ow*s + j - p] * w[i][j]
    //     out_1[h][w]   = sum_ij in[h + i - p][w + j - p] * w[i][j]
    // substituting h = oh*s, w = ow*s gives out_s[oh][ow] == out_1[oh*s][ow*s],
    // so the strided result is every s-th row and column of the stride-1 result.
    // It computes s^2 times more than needed, which is the price of not hanging.
    const bool subsampleAfter = stride[0] > 1 || stride[1] > 1;
    const std::array<uint32_t, 2> effectiveStride =
        subsampleAfter ? std::array<uint32_t, 2>{1, 1} : stride;
    auto result = ::ttnn::operations::experimental::quasar::conv2d(
        input, weight, &targetDevice, op->in_channels(), op->out_channels(),
        op->batch_size(), op->input_height(), op->input_width(), kernelSize,
        effectiveStride, padding, dilation, op->groups(), outputDtype, bias,
        conv2dConfig, computeConfig, outputMemoryConfig, sliceConfig);
    LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result));
    out = std::get<::ttnn::Tensor>(result);

    if (subsampleAfter) {
      // Padding totals per axis. The 4-element form is {top, bottom, left,
      // right}: conv2d_utils.cpp:278 sums [0]+[1] for height and [2]+[3] for
      // width.
      const auto padTotals = std::visit(
          [](const auto &p) -> std::array<uint32_t, 2> {
            if constexpr (std::tuple_size_v<std::decay_t<decltype(p)>> == 2) {
              return {2 * p[0], 2 * p[1]};
            } else {
              return {p[0] + p[1], p[2] + p[3]};
            }
          },
          padding);
      // Stride-1 output extent: H + padTotal - dilation * (kernel - 1).
      const uint32_t h1 = op->input_height() + padTotals[0] -
                          dilation[0] * (kernelSize[0] - 1);
      const uint32_t w1 = op->input_width() + padTotals[1] -
                          dilation[1] * (kernelSize[1] - 1);
      const uint32_t batch = op->batch_size();
      const uint32_t outChannels = op->out_channels();
      const std::vector<int32_t> spatialShape = {
          static_cast<int32_t>(batch), static_cast<int32_t>(h1),
          static_cast<int32_t>(w1), static_cast<int32_t>(outChannels)};
      ::ttnn::Tensor spatial =
          ::ttnn::operations::experimental::quasar::reshape(out, spatialShape,
                                                            std::nullopt);
      const std::vector<int32_t> begins = {0, 0, 0, 0};
      const std::vector<int32_t> ends = {
          static_cast<int32_t>(batch), static_cast<int32_t>(h1),
          static_cast<int32_t>(w1), static_cast<int32_t>(outChannels)};
      const std::vector<int32_t> steps = {
          1, static_cast<int32_t>(stride[0]), static_cast<int32_t>(stride[1]),
          1};
      ::ttnn::Tensor strided =
          ::ttnn::operations::experimental::quasar::slice<int32_t>(
              spatial, ::ttsl::Span<const int32_t>(begins),
              ::ttsl::Span<const int32_t>(ends),
              ::ttsl::Span<const int32_t>(steps), std::nullopt, std::nullopt,
              std::nullopt, std::nullopt);
      // Back to whatever shape the graph expects for this conv's output.
      const ::ttnn::Shape outShape =
          utils::toTTNNShape(*op->out()->desc()->shape());
      std::vector<int32_t> outDims;
      outDims.reserve(outShape.rank());
      for (size_t i = 0; i < outShape.rank(); i++) {
        outDims.push_back(static_cast<int32_t>(outShape[i]));
      }
      out = ::ttnn::operations::experimental::quasar::reshape(strided, outDims,
                                                              std::nullopt);
    }
  } else {
    Conv2dResultWithOptions result = ::ttnn::conv2d(
        input, weight, &targetDevice, op->in_channels(), op->out_channels(),
        op->batch_size(), op->input_height(), op->input_width(), kernelSize,
        stride, padding, dilation, op->groups(), outputDtype, bias,
        conv2dConfig, computeConfig, outputMemoryConfig, sliceConfig);
    LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result));
    out = std::get<::ttnn::Tensor>(result);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
