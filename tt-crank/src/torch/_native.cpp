#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/InferSize.h>
#include <ATen/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <mlir/IR/Value.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ttmlir/Target/Common/types_generated.h>

#include "cast.hpp"
#include "config.hpp"
#include "engine/compile.hpp"
#include "engine/compile_options.hpp"
#include "engine/device.hpp"
#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/ops/fallback.hpp"
#include "torch/tensor.hpp"
#include "torch/ttir_module_builder.hpp"
#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>

namespace nb = nanobind;
using namespace nb::literals; // for `"name"_a` argument literals

namespace tk = ::tt::kurbla::torch_backend;

namespace {

// Holds a ModuleBuilder until `compile()` consumes it. Subsequent method calls
// after compile() raise — the underlying MLIR module has been moved out.
class PyModuleBuilder {
public:
    explicit PyModuleBuilder(std::vector<tk::TensorTypeSpec> specs,
                             const std::vector<mlir::tt::ttcore::ArgumentType> &argument_roles = {})
        : mb_(tk::ModuleBuilder::init(specs, argument_roles)) {}

    PyModuleBuilder(PyModuleBuilder &&) = default;
    PyModuleBuilder &operator=(PyModuleBuilder &&) = default;
    PyModuleBuilder(const PyModuleBuilder &) = delete;
    PyModuleBuilder &operator=(const PyModuleBuilder &) = delete;

    mlir::Value arg(std::size_t index) {
        assert_builder();
        const auto args = mb_->args();
        TORCH_CHECK(index < args.size(), "tt-kurbla ModuleBuilder.arg: index ", index, " out of range (", args.size(),
                    " args)");
        return args[index];
    }

    mlir::Value add(mlir::Value lhs, mlir::Value rhs, double alpha) {
        assert_builder();
        return tk::build_add(*mb_, lhs, rhs, alpha);
    }

    mlir::Value sub(mlir::Value lhs, mlir::Value rhs, double alpha) {
        assert_builder();
        return tk::build_sub(*mb_, lhs, rhs, alpha);
    }

    mlir::Value mul(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_mul(*mb_, lhs, rhs);
    }

    mlir::Value mm(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_mm(*mb_, lhs, rhs);
    }

    mlir::Value all_reduce(mlir::Value input, const std::string &reduce_op, std::uint32_t cluster_axis) {
        assert_builder();
        return tk::build_all_reduce(*mb_, input, reduce_op, cluster_axis);
    }

    mlir::Value all_gather(mlir::Value input, std::int64_t group_size, std::uint32_t cluster_axis) {
        assert_builder();
        return tk::build_all_gather(*mb_, input, group_size, cluster_axis);
    }

    mlir::Value reduce_scatter(mlir::Value input, std::int64_t group_size, std::uint32_t cluster_axis,
                               std::int64_t scatter_dim) {
        assert_builder();
        return tk::build_reduce_scatter(*mb_, input, group_size, cluster_axis, scatter_dim);
    }

    mlir::Value addmm(mlir::Value bias, mlir::Value mat1, mlir::Value mat2, double beta, double alpha) {
        assert_builder();
        return tk::build_addmm(*mb_, bias, mat1, mat2, beta, alpha);
    }

    mlir::Value t(mlir::Value input) {
        assert_builder();
        return tk::build_t(*mb_, input);
    }

    mlir::Value relu(mlir::Value input) {
        assert_builder();
        return tk::build_relu(*mb_, input);
    }

    mlir::Value rsqrt(mlir::Value input) {
        assert_builder();
        return tk::build_rsqrt(*mb_, input);
    }

    mlir::Value reshape(mlir::Value input, std::vector<std::int64_t> new_shape) {
        assert_builder();
        auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
        int64_t numel = 1;
        for (auto d : input_type.getShape()) {
            numel *= d;
        }
        new_shape = at::infer_size(new_shape, numel);
        return tk::build_reshape(*mb_, input, new_shape);
    }

    mlir::Value mean(mlir::Value input, std::vector<std::int64_t> dims, bool keepdim) {
        assert_builder();
        return tk::build_mean(*mb_, input, dims, keepdim);
    }

    mlir::Value sum(mlir::Value input, std::vector<std::int64_t> dims, bool keepdim) {
        assert_builder();
        return tk::build_sum(*mb_, input, dims, keepdim);
    }

    mlir::Value threshold_backward(mlir::Value grad_output, mlir::Value self, double threshold) {
        assert_builder();
        return tk::build_threshold_backward(*mb_, grad_output, self, threshold);
    }

    mlir::Value mse_loss(mlir::Value self, mlir::Value target, std::int64_t reduction) {
        assert_builder();
        return tk::build_mse_loss(*mb_, self, target, reduction);
    }

    mlir::Value mse_loss_backward(mlir::Value grad_output, mlir::Value self, mlir::Value target,
                                  std::int64_t reduction) {
        assert_builder();
        return tk::build_mse_loss_backward(*mb_, grad_output, self, target, reduction);
    }

    mlir::Value batch_norm_inference(mlir::Value operand, mlir::Value scale, mlir::Value offset, mlir::Value mean,
                                     mlir::Value variance, float eps) {
        assert_builder();
        return tk::build_bn_inference(*mb_, operand, scale, offset, mean, variance, eps);
    }

    mlir::Value conv2d(mlir::Value input, mlir::Value weight, std::optional<mlir::Value> bias_opt,
                       std::vector<std::int64_t> stride, std::vector<std::int64_t> padding,
                       std::vector<std::int64_t> dilation, int64_t groups) {
        assert_builder();
        return tk::build_conv2d(*mb_, input, weight, bias_opt.value_or(mlir::Value{}), stride, padding, dilation,
                                groups);
    }

    mlir::Value max_pool2d(mlir::Value input, std::vector<std::int64_t> kernel_size, std::vector<std::int64_t> stride,
                           std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation, bool ceil_mode) {
        assert_builder();
        return tk::build_max_pool2d(*mb_, input, kernel_size, stride, padding, dilation, ceil_mode);
    }

    // Lift a Python scalar to a broadcastable `ttir.constant` at `dtype`.
    mlir::Value scalar(::tt::target::DataType dtype, double value) {
        assert_builder();
        return tk::build_scalar(*mb_, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)), value);
    }

    // `shape`-shaped creation ops at `dtype` (ttir.zeros / ones / full).
    mlir::Value zeros(std::vector<int64_t> shape, ::tt::target::DataType dtype) {
        assert_builder();
        return tk::build_zeros(*mb_, shape, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)));
    }
    mlir::Value ones(std::vector<int64_t> shape, ::tt::target::DataType dtype) {
        assert_builder();
        return tk::build_ones(*mb_, shape, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)));
    }
    mlir::Value full(std::vector<int64_t> shape, double value, ::tt::target::DataType dtype) {
        assert_builder();
        return tk::build_full(*mb_, shape, value, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)));
    }

    // Same, but the element type comes from `like` (for new_* ops, which default
    // their dtype to the reference tensor's).
    mlir::Value zeros_like(mlir::Value like, std::vector<int64_t> shape) {
        assert_builder();
        return tk::build_zeros(*mb_, shape, mlir::cast<mlir::RankedTensorType>(like.getType()).getElementType());
    }
    mlir::Value ones_like(mlir::Value like, std::vector<int64_t> shape) {
        assert_builder();
        return tk::build_ones(*mb_, shape, mlir::cast<mlir::RankedTensorType>(like.getType()).getElementType());
    }
    mlir::Value full_like(mlir::Value like, std::vector<int64_t> shape, double value) {
        assert_builder();
        return tk::build_full(*mb_, shape, value, mlir::cast<mlir::RankedTensorType>(like.getType()).getElementType());
    }

    // Lift a Python scalar to a broadcastable `ttir.constant` matching the
    // element type of `like`. Use this for Tensor_Scalar ops (e.g. pow) where
    // the scalar operand must carry the same dtype as the tensor operand.
    mlir::Value scalar_like(mlir::Value like, double value) {
        assert_builder();
        auto elem_type = mlir::cast<mlir::RankedTensorType>(like.getType()).getElementType();
        return tk::build_scalar(*mb_, elem_type, value);
    }

    // Cast `value` to a tensor with `dtype`'s element type, preserving shape.
    // No-op if `value` is already at that element type.
    mlir::Value typecast(mlir::Value value, ::tt::target::DataType dtype) {
        assert_builder();
        return mb_->insert_typecast(value, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)));
    }

    // Returns the compiled program along with metadata.
    ::tt::kurbla::CompileResult compile(const std::vector<mlir::Value> &outputs,
                                        const ::tt::kurbla::CompileOptions &options, bool capture_ttir) {
        assert_builder();
        auto module_op = std::move(*mb_).finalize(outputs);
        mb_.reset();
        return tk::compile_module(std::move(module_op), options, capture_ttir);
    }

    // Unary elementwise
    mlir::Value cos(mlir::Value input) {
        assert_builder();
        return tk::build_cos(*mb_, input);
    }
    mlir::Value sin(mlir::Value input) {
        assert_builder();
        return tk::build_sin(*mb_, input);
    }
    mlir::Value neg(mlir::Value input) {
        assert_builder();
        return tk::build_neg(*mb_, input);
    }
    mlir::Value silu(mlir::Value input) {
        assert_builder();
        return tk::build_silu(*mb_, input);
    }
    mlir::Value sigmoid(mlir::Value input) {
        assert_builder();
        return tk::build_sigmoid(*mb_, input);
    }
    mlir::Value floor_divide(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_floor_divide(*mb_, lhs, rhs);
    }
    mlir::Value bitwise_and(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_bitwise_and(*mb_, lhs, rhs);
    }
    mlir::Value bitwise_or(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_bitwise_or(*mb_, lhs, rhs);
    }
    mlir::Value bitwise_not(mlir::Value input) {
        assert_builder();
        return tk::build_bitwise_not(*mb_, input);
    }
    mlir::Value logical_and(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_logical_and(*mb_, lhs, rhs);
    }
    mlir::Value logical_or(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_logical_or(*mb_, lhs, rhs);
    }
    mlir::Value logical_not(mlir::Value input) {
        assert_builder();
        return tk::build_logical_not(*mb_, input);
    }
    mlir::Value clamp(mlir::Value input, std::optional<double> min_val, std::optional<double> max_val) {
        assert_builder();
        return tk::build_clamp(*mb_, input, min_val, max_val);
    }
    mlir::Value gather(mlir::Value input, mlir::Value index, int64_t dim) {
        assert_builder();
        return tk::build_gather(*mb_, input, index, dim);
    }
    mlir::Value gelu(mlir::Value input) {
        assert_builder();
        return tk::build_gelu(*mb_, input);
    }

    // Binary elementwise
    mlir::Value div(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_div(*mb_, lhs, rhs);
    }
    mlir::Value pow(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_pow(*mb_, lhs, rhs);
    }
    mlir::Value matmul(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_matmul(*mb_, lhs, rhs);
    }
    std::pair<std::optional<mlir::Value>, std::optional<mlir::Value>>
    matmul_backward(mlir::Value grad, mlir::Value self, mlir::Value other, bool need_self, bool need_other) {
        assert_builder();
        return tk::build_matmul_backward(*mb_, grad, self, other, need_self, need_other);
    }

    // Reductions
    mlir::Value softmax(mlir::Value input, int64_t dim) {
        assert_builder();
        int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
        int64_t norm_dim = (dim + rank) % rank;
        return tk::build_softmax(*mb_, input, norm_dim);
    }
    mlir::Value argmax(mlir::Value input, std::optional<int64_t> dim, bool keepdim) {
        assert_builder();
        if (dim.has_value()) {
            int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
            dim = (dim.value() + rank) % rank;
        }
        return tk::build_argmax(*mb_, input, dim, keepdim);
    }

    // Shape ops — normalize dims using MLIR type info
    mlir::Value unsqueeze(mlir::Value input, int64_t dim) {
        assert_builder();
        int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
        int64_t norm_dim = (dim + rank + 1) % (rank + 1);
        return tk::build_unsqueeze(*mb_, input, norm_dim);
    }
    mlir::Value squeeze(mlir::Value input, int64_t dim) {
        assert_builder();
        int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
        int64_t norm_dim = (dim + rank) % rank;
        return tk::build_squeeze(*mb_, input, norm_dim);
    }
    mlir::Value transpose_dims(mlir::Value input, int64_t dim0, int64_t dim1) {
        assert_builder();
        int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
        int64_t norm0 = (dim0 + rank) % rank;
        int64_t norm1 = (dim1 + rank) % rank;
        return tk::build_transpose(*mb_, input, norm0, norm1);
    }
    mlir::Value broadcast(mlir::Value input, std::vector<int64_t> target_shape) {
        assert_builder();
        // PyTorch expand() uses -1 to mean "keep current size"; resolve before MLIR.
        auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
        auto input_shape = input_type.getShape();
        int64_t input_rank = as<int64_t>(input_shape.size());
        int64_t target_rank = as<int64_t>(target_shape.size());
        for (int64_t i = 0; i < target_rank; ++i) {
            if (target_shape[as<std::size_t>(i)] == -1) {
                int64_t input_i = i - (target_rank - input_rank);
                TORCH_CHECK(input_i >= 0, "tt-kurbla broadcast: -1 in target_shape at a prepended dim");
                target_shape[as<std::size_t>(i)] = input_shape[as<std::size_t>(input_i)];
            }
        }
        return tk::build_broadcast(*mb_, input, target_shape);
    }
    mlir::Value permute(mlir::Value input, std::vector<int64_t> permutation) {
        assert_builder();
        int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
        for (auto &d : permutation) {
            d = (d + rank) % rank;
        }
        return tk::build_permute(*mb_, input, permutation);
    }

    // Cat — typecast all inputs to the promoted (first input's) element type before concat.
    mlir::Value cat(std::vector<mlir::Value> inputs, int64_t dim) {
        assert_builder();
        auto promoted = mlir::cast<mlir::RankedTensorType>(inputs[0].getType()).getElementType();
        llvm::SmallVector<mlir::Value> casted;
        for (auto &v : inputs) {
            casted.push_back(mb_->insert_typecast(v, promoted));
        }
        return tk::build_cat(*mb_, casted, dim);
    }

    // Slice — normalize dim, resolve None start/end, build full begins/ends/steps
    mlir::Value slice(mlir::Value input, int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end,
                      int64_t step) {
        assert_builder();
        auto type = mlir::cast<mlir::RankedTensorType>(input.getType());
        int64_t rank = type.getRank();
        int64_t norm_dim = (dim + rank) % rank;
        int64_t dim_size = type.getDimSize(norm_dim);
        int64_t s = start.has_value() ? start.value() : 0;
        int64_t e = end.has_value() ? end.value() : dim_size;
        if (s < 0) {
            s += dim_size;
        }
        if (e < 0) {
            e += dim_size;
        }
        s = std::max<int64_t>(0, std::min(s, dim_size));
        e = std::max<int64_t>(0, std::min(e, dim_size));
        std::vector<int64_t> begins(as<std::size_t>(rank), 0);
        std::vector<int64_t> ends, steps(as<std::size_t>(rank), 1);
        for (int64_t i = 0; i < rank; ++i) {
            ends.push_back(type.getDimSize(i));
        }
        begins[as<std::size_t>(norm_dim)] = s;
        ends[as<std::size_t>(norm_dim)] = e;
        steps[as<std::size_t>(norm_dim)] = step;
        return tk::build_slice(*mb_, input, begins, ends, steps);
    }

    // Arange — creation op (no tensor inputs)
    mlir::Value arange(int64_t start, int64_t end, int64_t step, ::tt::target::DataType dtype) {
        assert_builder();
        return tk::build_arange(*mb_, start, end, step, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)));
    }

    // Returns a 3-tuple; a gradient the caller did not request comes back as None.
    std::tuple<std::optional<mlir::Value>, std::optional<mlir::Value>, std::optional<mlir::Value>>
    linear_backward(mlir::Value self, mlir::Value grad, mlir::Value weight, bool need_self, bool need_weight,
                    bool need_bias) {
        assert_builder();
        return tk::build_linear_backward(*mb_, self, grad, weight, need_self, need_weight, need_bias);
    }

    // aten::linear as a leaf: weight stays in stored [out, in] orientation and the transpose
    // rides on the op, so it is never materialized (and never saved for backward).
    mlir::Value linear(mlir::Value input, mlir::Value weight, std::optional<mlir::Value> bias) {
        assert_builder();
        return tk::build_linear(*mb_, input, weight, bias.has_value() ? *bias : mlir::Value{});
    }

    // Embedding — indices (int) first, weight (float) second; no typecast on either
    // (intentional dtype mismatch — build_embedding expects indices to be integer-typed)
    mlir::Value embedding(mlir::Value weight, mlir::Value indices) {
        assert_builder();
        return tk::build_embedding(*mb_, indices, weight);
    }

    // Comparison
    mlir::Value le(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_le(*mb_, lhs, rhs);
    }
    mlir::Value lt(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_lt(*mb_, lhs, rhs);
    }
    mlir::Value gt(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_gt(*mb_, lhs, rhs);
    }
    mlir::Value ge(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_ge(*mb_, lhs, rhs);
    }
    mlir::Value eq(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_eq(*mb_, lhs, rhs);
    }
    mlir::Value ne(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_ne(*mb_, lhs, rhs);
    }

    // Conditional selection: result[i] = condition[i] ? true_val[i] : false_val[i]
    mlir::Value where(mlir::Value condition, mlir::Value true_val, mlir::Value false_val) {
        assert_builder();
        return tk::build_where(*mb_, condition, true_val, false_val);
    }

    // Scaled dot-product attention (FlashAttention-2).
    mlir::Value sdpa(mlir::Value query, mlir::Value key, mlir::Value value, bool is_causal, std::optional<float> scale,
                     std::optional<mlir::Value> attn_mask) {
        assert_builder();
        return tk::build_sdpa(*mb_, query, key, value, is_causal, scale, attn_mask.value_or(mlir::Value{}));
    }

    // index_copy: result = self with source values placed at index positions along dim.
    // index must be 1D; source must have the same rank as self.
    mlir::Value index_copy(mlir::Value input, int64_t dim, mlir::Value index, mlir::Value source) {
        assert_builder();
        return tk::build_index_copy(*mb_, input, dim, index, source);
    }

    // tril: lower-triangular part of input; elements strictly above the diagonal-th diagonal are zeroed.
    mlir::Value tril(mlir::Value input, int64_t diagonal = 0) {
        assert_builder();
        return tk::build_tril(*mb_, input, diagonal);
    }

private:
    void assert_builder() const {
        TORCH_CHECK(mb_.has_value(), "tt-kurbla ModuleBuilder: builder already consumed by compile()");
    }

    std::optional<tk::ModuleBuilder> mb_;
};

// Python `torch.Tensor` -> `at::Tensor` at the binding boundary, since
// nanobind has no built-in caster for `at::Tensor`. Throws if `h` isn't a
// `torch.Tensor`. Returns an independent strong ref on the underlying
// TensorImpl; the Python wrapper's ref is unchanged.
at::Tensor unpack_torch_tensor(nb::handle h) {
    PyObject *p = h.ptr();
    TORCH_CHECK(THPVariable_Check(p) != 0, "tt-kurbla: expected a torch.Tensor");
    return THPVariable_Unpack(p);
}

// `at::Tensor` -> Python `torch.Tensor`. `THPVariable_Wrap` returns a new
// PyObject* with refcount=1; `nb::steal` adopts that ref without bumping it
// again (which `nb::borrow` would do, leaking).
nb::object wrap_torch_tensor(at::Tensor t) {
    PyObject *p = THPVariable_Wrap(std::move(t));
    if (p == nullptr) {
        throw nb::python_error();
    }
    return nb::steal<nb::object>(p);
}

nb::list run_program(::tt::kurbla::CompiledProgram &program, nb::list &inputs,
                     std::vector<::tt::target::DataType> &output_dtypes) {
    std::vector<at::Tensor> at_inputs;
    at_inputs.reserve(nb::len(inputs));
    for (auto item : inputs) {
        at_inputs.push_back(unpack_torch_tensor(item));
    }

    auto outputs = tk::run_compiled_program(program, at_inputs, output_dtypes);

    nb::list result;
    for (auto &t : outputs) {
        result.append(wrap_torch_tensor(std::move(t)));
    }
    return result;
}

} // namespace

NB_MODULE(_native, m) {
    // Establish the PrivateUse1 backend's canonical name from C++. This is the
    // c10 primitive that backs ScalarType / Device / dispatch-key lookups for
    // "tt". Calling it here keeps registration co-located with the
    // TORCH_LIBRARY_IMPL kernels that ship in this .so — if anyone ever loads
    // _native.so without going through the Python wrapper (unusual but
    // possible), the kernels are still wired correctly.
    //
    // The Python-side counterpart, torch.utils.rename_privateuse1_backend("tt")
    // in tt_kurbla.torch._device.register(), adds the framework-level
    // ergonomics on top (Tensor.tt() method, the autograd-aware dispatch-key
    // alias, etc.). It calls into the same c10 primitive internally and is a
    // no-op when the name already matches — the supported order is the one
    // enforced by python/tt_kurbla/torch/__init__.py: this C++ call first,
    // then register() afterwards.
    c10::register_privateuse1_backend("tt");

    // Op kernels (ops/tensor.cpp, ops/elementwise.cpp) and the PrivateUse1
    // DeviceGuardImpl self-register via global ctors when this .so loads —
    // TORCH_LIBRARY_IMPL / C10_REGISTER_GUARD_IMPL handle the dispatcher and
    // guard sides; we just need the allocator hook.
    tk::register_allocator();

    m.doc() = "tt-kurbla torch backend native module";
    m.def("loaded", []() { return true; });

    m.def("artifacts_dir", &artifacts_dir_config, "Directory where the artifacts should be stored.");

    m.def("runtime_device_num_chips", &::tt::kurbla::runtime_device_num_chips,
          "Number of physical chips behind the single logical tt device.");

    m.def(
        "runtime_device_arch", []() { return std::string(mlir::tt::ttcore::stringifyArch(::tt::kurbla::arch())); },
        "Architecture of the chips behind the tt device, e.g. \"wormhole_b0\".");

    m.def("open_runtime_device_mesh", &::tt::kurbla::open_runtime_device_mesh, "rows"_a, "cols"_a,
          "Open (or reopen, if already open with a different shape) the MeshDevice "
          "with mesh shape (rows, cols). rows*cols must be in [1, num_chips()].");

    m.def("runtime_device_mesh_shape", &::tt::kurbla::runtime_device_mesh_shape,
          "Current runtime mesh shape as [rows, cols].");

    m.def("close_runtime_device_mesh", &::tt::kurbla::close_runtime_device_mesh,
          "Close the process-wide MeshDevice if open.");

    // Thin binding wrappers over the distributed primitives in tensor.cpp:
    // unwrap Python objects to at::Tensor, forward, that's it. The actual
    // logic (TTIR construction, runtime calls, storage manipulation) lives
    // in src/torch/tensor.cpp.

    m.def(
        "scatter_into",
        [](nb::handle py_output, nb::list py_chunks, std::uint32_t cluster_axis) {
            at::Tensor output = unpack_torch_tensor(py_output);
            std::vector<at::Tensor> chunks;
            chunks.reserve(nb::len(py_chunks));
            for (auto h : py_chunks) {
                chunks.push_back(unpack_torch_tensor(h));
            }
            tk::scatter_into(output, chunks, cluster_axis);
        },
        "output"_a, "chunks"_a, "cluster_axis"_a);

    m.def(
        "allgather_into",
        [](nb::handle py_output, nb::handle py_input, std::uint32_t cluster_axis) {
            tk::allgather_into(unpack_torch_tensor(py_output), unpack_torch_tensor(py_input), cluster_axis);
        },
        "output"_a, "input"_a, "cluster_axis"_a);

    m.def(
        "allreduce_into",
        [](nb::handle py_tensor, std::uint32_t cluster_axis) {
            tk::allreduce_into(unpack_torch_tensor(py_tensor), cluster_axis);
        },
        "tensor"_a, "cluster_axis"_a);

    m.def(
        "reduce_scatter_into",
        [](nb::handle py_output, nb::handle py_input, std::uint32_t cluster_axis, std::int64_t scatter_dim) {
            tk::reduce_scatter_into(unpack_torch_tensor(py_output), unpack_torch_tensor(py_input), cluster_axis,
                                    scatter_dim);
        },
        "output"_a, "input"_a, "cluster_axis"_a, "scatter_dim"_a);

    // Strict-fallback toggle. Tests flip this on to assert that a code path
    // never falls back to CPU; the catch-all fallback raises instead of
    // running while strict mode is set.
    m.def("set_fallback_strict", &tk::set_fallback_strict);
    m.def("fallback_strict", &tk::fallback_strict);

    // ====== torch.compile backend bindings ======
    //
    // The Python `tt_kurbla.torch._compile` module walks an FX graph and calls
    // these to emit a single TTIR module, compile it, and run it on each
    // invocation. Phase-0 surface: add only — enough for a stacked-add model.

    nb::enum_<::tt::target::DataType>(m, "DataType")
        .value("Float32", ::tt::target::DataType::Float32)
        .value("Float64", ::tt::target::DataType::Float64)
        .value("Float16", ::tt::target::DataType::Float16)
        .value("BFloat16", ::tt::target::DataType::BFloat16)
        .value("Int32", ::tt::target::DataType::Int32)
        .value("Int64", ::tt::target::DataType::Int64)
        .value("Bool", ::tt::target::DataType::Bool)
        .value("UInt8", ::tt::target::DataType::UInt8);

    nb::enum_<mlir::tt::ttcore::ArgumentType>(m, "ArgumentType")
        .value("Input", mlir::tt::ttcore::ArgumentType::Input)
        .value("Parameter", mlir::tt::ttcore::ArgumentType::Parameter)
        .value("Constant", mlir::tt::ttcore::ArgumentType::Constant);

    nb::class_<tk::TensorTypeSpec>(m, "TensorTypeSpec")
        .def(
            "__init__",
            [](tk::TensorTypeSpec *self, std::vector<std::int64_t> shape, ::tt::target::DataType dtype) {
                new (self) tk::TensorTypeSpec{std::move(shape), dtype};
            },
            "shape"_a, "dtype"_a);

    // Opaque handle to an `mlir::Value` living inside an in-flight
    // ModuleBuilder. Python should not construct these directly — they come
    // back from ModuleBuilder.arg() / per-op methods.
    nb::class_<mlir::Value>(m, "Value").def_prop_ro("shape", [](mlir::Value self) {
        auto type = mlir::cast<mlir::RankedTensorType>(self.getType());
        auto dims = type.getShape();
        return std::vector<std::int64_t>(dims.begin(), dims.end());
    });

    // Typed enums for the dtype / math-fidelity compile options.
    nb::enum_<::tt::kurbla::CompileOptions::BfpDtype>(m, "BfpDtype")
        .value("BfpBf8", ::tt::kurbla::CompileOptions::BfpDtype::BfpBf8)
        .value("BfpBf4", ::tt::kurbla::CompileOptions::BfpDtype::BfpBf4);

    nb::enum_<::tt::kurbla::CompileOptions::MathFidelity>(m, "MathFidelity")
        .value("LoFi", ::tt::kurbla::CompileOptions::MathFidelity::LoFi)
        .value("HiFi2", ::tt::kurbla::CompileOptions::MathFidelity::HiFi2)
        .value("HiFi3", ::tt::kurbla::CompileOptions::MathFidelity::HiFi3)
        .value("HiFi4", ::tt::kurbla::CompileOptions::MathFidelity::HiFi4);

    // Compile options: `torch.compile(..., options=...)`
    nb::class_<::tt::kurbla::CompileOptions>(m, "CompileOptions")
        .def("__init__", [](::tt::kurbla::CompileOptions *self) { new (self)::tt::kurbla::CompileOptions{}; })
        .def_rw("optimization_level", &::tt::kurbla::CompileOptions::optimization_level)
        .def_rw("experimental_weight_dtype", &::tt::kurbla::CompileOptions::experimental_weight_dtype)
        .def_rw("experimental_kv_cache_dtype", &::tt::kurbla::CompileOptions::experimental_kv_cache_dtype)
        .def_rw("math_fidelity", &::tt::kurbla::CompileOptions::math_fidelity)
        .def_rw("fp32_dest_acc_en", &::tt::kurbla::CompileOptions::fp32_dest_acc_en)
        .def_rw("experimental_enable_fusing_conv2d_with_multiply_pattern",
                &::tt::kurbla::CompileOptions::experimental_enable_fusing_conv2d_with_multiply_pattern)
        .def_rw("experimental_enable_permute_matmul_fusion",
                &::tt::kurbla::CompileOptions::experimental_enable_permute_matmul_fusion)
        .def_rw("enable_trace", &::tt::kurbla::CompileOptions::enable_trace)
        .def_rw("enable_const_eval", &::tt::kurbla::CompileOptions::enable_const_eval)
        .def_rw("enable_const_eval_on_cpu", &::tt::kurbla::CompileOptions::enable_const_eval_on_cpu)
        .def_rw("enable_const_eval_inputs_to_system_memory",
                &::tt::kurbla::CompileOptions::enable_const_eval_inputs_to_system_memory)
        .def_rw("experimental_enable_dram_space_saving_optimization",
                &::tt::kurbla::CompileOptions::experimental_enable_dram_space_saving_optimization)
        .def_rw("enable_create_d2m_subgraphs", &::tt::kurbla::CompileOptions::enable_create_d2m_subgraphs)
        .def_rw("ttnn_perf_metrics_enabled", &::tt::kurbla::CompileOptions::ttnn_perf_metrics_enabled)
        .def_rw("ttnn_perf_metrics_output_file", &::tt::kurbla::CompileOptions::ttnn_perf_metrics_output_file);

    nb::class_<PyModuleBuilder>(m, "ModuleBuilder")
        .def(nb::init<std::vector<tk::TensorTypeSpec>, const std::vector<mlir::tt::ttcore::ArgumentType> &>(),
             "input_specs"_a, "argument_roles"_a = std::vector<mlir::tt::ttcore::ArgumentType>{})
        .def("arg", &PyModuleBuilder::arg, "index"_a)
        .def("add", &PyModuleBuilder::add, "lhs"_a, "rhs"_a, "alpha"_a = 1.0)
        .def("sub", &PyModuleBuilder::sub, "lhs"_a, "rhs"_a, "alpha"_a = 1.0)
        .def("mul", &PyModuleBuilder::mul, "lhs"_a, "rhs"_a)
        .def("mm", &PyModuleBuilder::mm, "lhs"_a, "rhs"_a)
        .def("all_reduce", &PyModuleBuilder::all_reduce, "input"_a, "reduce_op"_a, "cluster_axis"_a)
        .def("all_gather", &PyModuleBuilder::all_gather, "input"_a, "group_size"_a, "cluster_axis"_a)
        .def("reduce_scatter", &PyModuleBuilder::reduce_scatter, "input"_a, "group_size"_a, "cluster_axis"_a,
             "scatter_dim"_a)
        .def("addmm", &PyModuleBuilder::addmm, "bias"_a, "mat1"_a, "mat2"_a, "beta"_a = 1.0, "alpha"_a = 1.0)
        .def("t", &PyModuleBuilder::t, "input"_a)
        .def("relu", &PyModuleBuilder::relu, "input"_a)
        .def("rsqrt", &PyModuleBuilder::rsqrt, "input"_a)
        .def("reshape", &PyModuleBuilder::reshape, "input"_a, "new_shape"_a)
        .def("mean", &PyModuleBuilder::mean, "input"_a, "dims"_a, "keepdim"_a = false)
        .def("sum", &PyModuleBuilder::sum, "input"_a, "dims"_a, "keepdim"_a = false)
        .def("threshold_backward", &PyModuleBuilder::threshold_backward, "grad_output"_a, "self"_a, "threshold"_a)
        .def("mse_loss", &PyModuleBuilder::mse_loss, "self"_a, "target"_a, "reduction"_a)
        .def("mse_loss_backward", &PyModuleBuilder::mse_loss_backward, "grad_output"_a, "self"_a, "target"_a,
             "reduction"_a)
        .def("batch_norm_inference", &PyModuleBuilder::batch_norm_inference, "operand"_a, "scale"_a, "offset"_a,
             "mean"_a, "variance"_a, "eps"_a)
        .def("conv2d", &PyModuleBuilder::conv2d, "input"_a, "weight"_a, "bias"_a, "stride"_a, "padding"_a, "dilation"_a,
             "groups"_a)
        .def("max_pool2d", &PyModuleBuilder::max_pool2d, "input"_a, "kernel_size"_a, "stride"_a, "padding"_a,
             "dilation"_a, "ceil_mode"_a = false)
        .def("scalar", &PyModuleBuilder::scalar, "dtype"_a, "value"_a)
        .def("zeros", &PyModuleBuilder::zeros, "shape"_a, "dtype"_a)
        .def("ones", &PyModuleBuilder::ones, "shape"_a, "dtype"_a)
        .def("full", &PyModuleBuilder::full, "shape"_a, "value"_a, "dtype"_a)
        .def("zeros_like", &PyModuleBuilder::zeros_like, "like"_a, "shape"_a)
        .def("ones_like", &PyModuleBuilder::ones_like, "like"_a, "shape"_a)
        .def("full_like", &PyModuleBuilder::full_like, "like"_a, "shape"_a, "value"_a)
        .def("scalar_like", &PyModuleBuilder::scalar_like, "like"_a, "value"_a)
        .def("typecast", &PyModuleBuilder::typecast, "value"_a, "dtype"_a)
        .def("cos", &PyModuleBuilder::cos, "input"_a)
        .def("sin", &PyModuleBuilder::sin, "input"_a)
        .def("neg", &PyModuleBuilder::neg, "input"_a)
        .def("silu", &PyModuleBuilder::silu, "input"_a)
        .def("sigmoid", &PyModuleBuilder::sigmoid, "input"_a)
        .def("floor_divide", &PyModuleBuilder::floor_divide, "lhs"_a, "rhs"_a)
        .def("bitwise_and", &PyModuleBuilder::bitwise_and, "lhs"_a, "rhs"_a)
        .def("bitwise_or", &PyModuleBuilder::bitwise_or, "lhs"_a, "rhs"_a)
        .def("bitwise_not", &PyModuleBuilder::bitwise_not, "input"_a)
        .def("logical_and", &PyModuleBuilder::logical_and, "lhs"_a, "rhs"_a)
        .def("logical_or", &PyModuleBuilder::logical_or, "lhs"_a, "rhs"_a)
        .def("logical_not", &PyModuleBuilder::logical_not, "input"_a)
        .def("clamp", &PyModuleBuilder::clamp, "input"_a, "min"_a, "max"_a)
        .def("gather", &PyModuleBuilder::gather, "input"_a, "index"_a, "dim"_a)
        .def("gelu", &PyModuleBuilder::gelu, "input"_a)
        .def("div", &PyModuleBuilder::div, "lhs"_a, "rhs"_a)
        .def("pow", &PyModuleBuilder::pow, "lhs"_a, "rhs"_a)
        .def("matmul", &PyModuleBuilder::matmul, "lhs"_a, "rhs"_a)
        .def("matmul_backward", &PyModuleBuilder::matmul_backward, "grad"_a, "self"_a, "other"_a, "need_self"_a,
             "need_other"_a)
        .def("softmax", &PyModuleBuilder::softmax, "input"_a, "dim"_a)
        .def("argmax", &PyModuleBuilder::argmax, "input"_a, "dim"_a, "keepdim"_a = false)
        .def("unsqueeze", &PyModuleBuilder::unsqueeze, "input"_a, "dim"_a)
        .def("squeeze", &PyModuleBuilder::squeeze, "input"_a, "dim"_a)
        .def("transpose", &PyModuleBuilder::transpose_dims, "input"_a, "dim0"_a, "dim1"_a)
        .def("broadcast", &PyModuleBuilder::broadcast, "input"_a, "target_shape"_a)
        .def("permute", &PyModuleBuilder::permute, "input"_a, "permutation"_a)
        .def("cat", &PyModuleBuilder::cat, "inputs"_a, "dim"_a)
        .def("slice", &PyModuleBuilder::slice, "input"_a, "dim"_a, "start"_a, "end"_a, "step"_a = 1LL)
        .def("arange", &PyModuleBuilder::arange, "start"_a, "end"_a, "step"_a, "dtype"_a)
        .def("linear", &PyModuleBuilder::linear, "input"_a, "weight"_a, "bias"_a = nb::none())
        .def("linear_backward", &PyModuleBuilder::linear_backward, "self"_a, "grad"_a, "weight"_a, "need_self"_a,
             "need_weight"_a, "need_bias"_a)
        .def("embedding", &PyModuleBuilder::embedding, "weight"_a, "indices"_a)
        .def("le", &PyModuleBuilder::le, "lhs"_a, "rhs"_a)
        .def("lt", &PyModuleBuilder::lt, "lhs"_a, "rhs"_a)
        .def("gt", &PyModuleBuilder::gt, "lhs"_a, "rhs"_a)
        .def("ge", &PyModuleBuilder::ge, "lhs"_a, "rhs"_a)
        .def("eq", &PyModuleBuilder::eq, "lhs"_a, "rhs"_a)
        .def("ne", &PyModuleBuilder::ne, "lhs"_a, "rhs"_a)
        .def("where", &PyModuleBuilder::where, "condition"_a, "true_val"_a, "false_val"_a)
        .def("sdpa", &PyModuleBuilder::sdpa, "query"_a, "key"_a, "value"_a, "is_causal"_a = true,
             "scale"_a = nb::none(), "attn_mask"_a = nb::none())
        .def("index_copy", &PyModuleBuilder::index_copy, "input"_a, "dim"_a, "index"_a, "source"_a)
        .def("tril", &PyModuleBuilder::tril, "input"_a, "diagonal"_a = 0)
        // Consumes the builder. Subsequent calls on `self` raise.
        // `capture_ttir`: set when the TTIR string is needed.
        .def("compile", &PyModuleBuilder::compile, "outputs"_a, "options"_a, "capture_ttir"_a = false);

    nb::class_<::tt::kurbla::CompileResult>(m, "CompileResult")
        // `reference` overrides the `reference_internal` a property getter uses by
        // default. Both are non-owning — Python never frees the program either way —
        // but `reference_internal` would also keep this `CompileResult` (and its
        // TTIR string) alive for as long as the returned program object, which
        // outlives it. The compile cache owns the program, so no keepalive is needed.
        .def_prop_ro(
            "program", [](const ::tt::kurbla::CompileResult &self) { return self.program; }, nb::rv_policy::reference,
            "The compiled program, owned by the process-wide compile cache.")
        .def_ro("ttir", &::tt::kurbla::CompileResult::ttir,
                "The TTIR this program was compiled from. Empty unless `capture_ttir` was set.")
        .def_ro("cache_hit", &::tt::kurbla::CompileResult::cache_hit,
                "True when the program came from the compile cache and no pipeline ran.")
        .def_prop_ro(
            "compile_duration_ms",
            [](const ::tt::kurbla::CompileResult &self) { return self.compile_duration.count(); },
            "Wall-clock engine compile time in milliseconds. Near zero on a cache hit.");

    nb::class_<::tt::kurbla::CompiledProgram>(m, "CompiledProgram")
        // Returns a copy: `ttnn_ir()` hands out a view into the flatbuffer, and
        // nanobind has no string_view caster here.
        .def(
            "ttnn_ir", [](const ::tt::kurbla::CompiledProgram &self) { return std::string(self.ttnn_ir()); },
            "Gets the TTNN IR.");

    m.def("run_program", &run_program, "program"_a, "inputs"_a, "output_dtypes"_a,
          "Bind tt-backend torch.Tensors to the compiled program's inputs and run it. "
          "`output_dtypes` is the per-output user-facing dtype the result tensors "
          "should be wrapped as (one per program output). "
          "Returns a list of tt-backend torch.Tensors.");
}
