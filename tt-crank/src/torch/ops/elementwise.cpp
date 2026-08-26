#include "torch/ops/builders.hpp"

#include "cast.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/core/Reduction.h>
#include <c10/core/Scalar.h>
#include <c10/core/StorageImpl.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <torch/library.h>
#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/tensor.hpp"

namespace tt::kurbla::torch_backend {

namespace {

at::Tensor tt_add(const at::Tensor &a_in, const at::Tensor &b_in, const at::Scalar &alpha) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    // `promoted` is the user-facing dtype we'll stamp on the result; physical
    // storage will be the rewriter's hardware-backed alias (e.g. f32 when
    // promoted is f64).
    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);

    auto result = build_add(mb, lhs, rhs, alpha.toDouble());
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_sub(const at::Tensor &a_in, const at::Tensor &b_in, const at::Scalar &alpha) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);

    auto result = build_sub(mb, lhs, rhs, alpha.toDouble());
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_mul(const at::Tensor &a_in, const at::Tensor &b_in) {
    const auto [a, b] = align_on_tt(a_in, b_in);

    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);

    auto result = build_mul(mb, lhs, rhs);
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});

    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_relu(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::relu: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_relu(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_rsqrt(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::rsqrt: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_rsqrt(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_batch_norm_inference(const at::Tensor &input_in, const std::optional<at::Tensor> &weight_in,
                                   const std::optional<at::Tensor> &bias_in,
                                   const std::optional<at::Tensor> &running_mean_in,
                                   const std::optional<at::Tensor> &running_var_in, bool training, double /*momentum*/,
                                   double eps, bool /*cudnn_enabled*/) {
    TORCH_CHECK(is_tt(input_in), "tt-kurbla aten::batch_norm: tensor must be on tt backend");
    TORCH_CHECK(!training, "tt-kurbla aten::batch_norm: training mode not supported");
    TORCH_CHECK(running_mean_in.has_value() && running_var_in.has_value(),
                "tt-kurbla aten::batch_norm: running_mean and running_var must be provided");
    TORCH_CHECK(weight_in.has_value() && bias_in.has_value(),
                "tt-kurbla aten::batch_norm: weight and bias must be provided (affine=True)");

    const auto [input, weight, bias, running_mean, running_var] =
        align_on_tt(input_in, *weight_in, *bias_in, *running_mean_in, *running_var_in);

    auto mb = ModuleBuilder::init(
        {spec_for(input), spec_for(weight), spec_for(bias), spec_for(running_mean), spec_for(running_var)});
    auto [promoted, inp_v, w_v, b_v, m_v, v_v] = promote_inputs(mb, input, weight, bias, running_mean, running_var);

    auto result = build_bn_inference(mb, inp_v, w_v, b_v, m_v, v_v, as<float>(eps));
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {input, weight, bias, running_mean, running_var});
    return wrap_tt_tensor(std::move(outputs[0]), input.sizes(), promoted);
}

at::Tensor tt_mean(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim,
                   std::optional<at::ScalarType> /*dtype*/) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::mean.dim: tensor must be on tt backend");
    TORCH_CHECK(dim.has_value(), "tt-kurbla aten::mean.dim: dim must be specified");

    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_mean(mb, mb.args()[0], dim.value(), keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

std::tuple<at::Tensor, at::Tensor> tt_max_pool2d_with_indices(const at::Tensor &self_in, at::IntArrayRef kernel_size,
                                                              at::IntArrayRef stride, at::IntArrayRef padding,
                                                              at::IntArrayRef dilation, bool ceil_mode) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::max_pool2d_with_indices: tensor must be on tt backend");

    auto effective_stride = stride.empty() ? kernel_size : stride;

    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result = build_max_pool2d(mb, mb.args()[0], kernel_size, effective_stride, padding, dilation, ceil_mode);

    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());

    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self_in});

    auto pool_result = wrap_tt_tensor(std::move(outputs[0]), out_shape, self_in.scalar_type());
    // Indices are never used in inference; return a CPU zero tensor as placeholder.
    at::Tensor dummy_indices = at::zeros(out_shape, at::TensorOptions().dtype(c10::ScalarType::Long));
    return std::make_tuple(std::move(pool_result), std::move(dummy_indices));
}

// Move a freshly-computed result tensor's runtime buffer into a caller-provided
// `.out` tensor. We assert `out` already has the expected shape and storage
// size rather than resizing it: the structured `.out` dispatch is supposed to
// pre-size `out`, so a mismatch means an assumption broke - fail loudly so we
// can revisit before silently reshaping.
at::Tensor &write_result_into(at::Tensor &out, const at::Tensor &result) {
    TORCH_CHECK(out.sizes() == result.sizes(), "tt-kurbla .out kernel: out tensor shape ", out.sizes(),
                " does not match computed result shape ", result.sizes());
    // Equal sizes + equal storage bytes still allow a dtype mismatch when the
    // itemsizes coincide (e.g. f32 vs i32, bf16 vs f16). Replacing the storage
    // would then reinterpret the buffer's bits as out's dtype - check loudly.
    TORCH_CHECK(out.scalar_type() == result.scalar_type(), "tt-kurbla .out kernel: out dtype ", out.scalar_type(),
                " does not match computed result dtype ", result.scalar_type());
    TORCH_CHECK(out.storage().nbytes() == result.storage().nbytes(), "tt-kurbla .out kernel: out storage is ",
                out.storage().nbytes(), " bytes but result needs ", result.storage().nbytes());
    storage_of(out).replace(storage_of(result).tensor());
    return out;
}

// relu_: in-place ReLU. Runs the functional kernel and swaps self's storage for
// the result — same storage-swap pattern as the `.out` ops below. self and the
// result share shape and dtype, so write_result_into's checks always hold.
at::Tensor &tt_relu_(at::Tensor &self) {
    return write_result_into(self, tt_relu(self));
}

// mse_loss.out: forward loss. `reduction == mean` produces a rank-0 scalar.
at::Tensor &tt_mse_loss_out(const at::Tensor &self_in, const at::Tensor &target_in, int64_t reduction,
                            at::Tensor &out) {
    const auto [self, target] = align_on_tt(self_in, target_in);
    auto mb = ModuleBuilder::init({spec_for(self), spec_for(target)});
    auto [promoted, s, t] = promote_inputs(mb, self, target);
    auto result_v = build_mse_loss(mb, s, t, reduction);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self, target});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
    return write_result_into(out, result);
}

// add.out: `out = self + alpha * other`.
at::Tensor &tt_add_out(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor &out) {
    return write_result_into(out, tt_add(self, other, alpha));
}

// threshold_backward.grad_input: `grad_output * (self > threshold)`.
at::Tensor &tt_threshold_backward_out(const at::Tensor &grad_output_in, const at::Tensor &self_in,
                                      const at::Scalar &threshold, at::Tensor &grad_input) {
    const auto [grad_output, self] = align_on_tt(grad_output_in, self_in);
    auto mb = ModuleBuilder::init({spec_for(grad_output), spec_for(self)});
    auto [promoted, go, s] = promote_inputs(mb, grad_output, self);
    auto result_v = build_threshold_backward(mb, go, s, threshold.toDouble());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {grad_output, self});
    auto result = wrap_tt_tensor(std::move(outputs[0]), grad_output.sizes(), promoted);
    return write_result_into(grad_input, result);
}

// sum.IntList_out: reduce `self` over `dim`.
at::Tensor &tt_sum_out(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim,
                       std::optional<at::ScalarType> dtype, at::Tensor &out) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::sum.IntList_out: tensor must be on tt backend");
    // dim=None means reduce over all dimensions; build_reduce treats empty dims
    // the same way.
    llvm::SmallVector<int64_t> reduce_dims;
    if (dim.has_value()) {
        reduce_dims.assign(dim.value().begin(), dim.value().end());
    }

    TORCH_CHECK(!dtype.has_value() || dtype.value() == self.scalar_type(),
                "tt-kurbla aten::sum.IntList_out: dtype conversion is not yet supported (requested ", dtype.value(),
                " for a ", self.scalar_type(), " tensor)");

    const auto target_dtype = out.scalar_type();

    auto mb = ModuleBuilder::init({spec_for(self)});
    mlir::Value in = mb.args()[0];
    if (self.scalar_type() != target_dtype) {
        in = mb.insert_typecast(in, mlir_element_type_for(target_dtype));
    }
    auto result_v = build_sum(mb, in, reduce_dims, keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, target_dtype);
    return write_result_into(out, result);
}

// mse_loss_backward: `grad_output * 2 * (self - target) / N`. Functional (no
// out=), so it returns a fresh tensor. `grad_output` is the rank-0 loss grad.
at::Tensor tt_mse_loss_backward(const at::Tensor &grad_output_in, const at::Tensor &self_in,
                                const at::Tensor &target_in, int64_t reduction) {
    const auto [grad_output, self, target] = align_on_tt(grad_output_in, self_in, target_in);
    auto mb = ModuleBuilder::init({spec_for(grad_output), spec_for(self), spec_for(target)});
    auto [promoted, go, s, t] = promote_inputs(mb, grad_output, self, target);
    auto result_v = build_mse_loss_backward(mb, go, s, t, reduction);
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {grad_output, self, target});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), promoted);
}

} // namespace

at::Tensor tt_unsqueeze(const at::Tensor &self, int64_t dim) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::unsqueeze: tensor must be on tt backend");
    int64_t rank = self.dim();
    // Normalize: valid range [-(rank+1), rank]
    int64_t norm_dim = (dim + rank + 1) % (rank + 1);
    std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
    out_shape.insert(out_shape.begin() + as<std::ptrdiff_t>(norm_dim), 1);
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_unsqueeze(mb, mb.args()[0], norm_dim);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_squeeze_dim(const at::Tensor &self, int64_t dim) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::squeeze.dim: tensor must be on tt backend");
    int64_t rank = self.dim();
    int64_t norm_dim = (dim + rank) % rank;
    // squeeze on a non-size-1 dim is a no-op in shape — but we must return a
    // distinct tensor object (PyTorch's aliasing check rejects returning self).
    // Route through a reshape with the same shape to produce a fresh tensor.
    if (self.size(norm_dim) != 1) {
        llvm::SmallVector<int64_t> same_shape(self.sizes().begin(), self.sizes().end());
        auto mb = ModuleBuilder::init({spec_for(self)});
        auto result = build_reshape(mb, mb.args()[0], same_shape);
        auto module_op = std::move(mb).finalize({result});
        auto outputs = compile_and_run(std::move(module_op), {self});
        return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
    }
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != norm_dim) {
            out_shape.push_back(self.size(i));
        }
    }
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_squeeze(mb, mb.args()[0], norm_dim);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_expand(const at::Tensor &self, at::IntArrayRef size, bool /*implicit*/) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::expand: tensor must be on tt backend");
    int64_t input_rank = self.dim();
    int64_t target_rank = as<int64_t>(size.size());
    int64_t offset = target_rank - input_rank;
    std::vector<int64_t> target_shape;
    for (int64_t i = 0; i < target_rank; ++i) {
        if (size[as<std::size_t>(i)] == -1) {
            TT_FATAL(i >= offset, "tt_expand: -1 not valid for prepended dim {}", i);
            target_shape.push_back(self.size(i - offset));
        } else {
            target_shape.push_back(size[as<std::size_t>(i)]);
        }
    }
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_broadcast(mb, mb.args()[0], target_shape);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), target_shape, self.scalar_type());
}

at::Tensor tt_transpose_int(const at::Tensor &self, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::transpose.int: tensor must be on tt backend");
    int64_t rank = self.dim();
    int64_t nd0 = (dim0 + rank) % rank;
    int64_t nd1 = (dim1 + rank) % rank;
    if (nd0 == nd1) {
        return self;
    }
    std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
    std::swap(out_shape[as<std::size_t>(nd0)], out_shape[as<std::size_t>(nd1)]);
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_transpose(mb, mb.args()[0], nd0, nd1);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_to_copy(const at::Tensor &self_in, std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                      std::optional<bool> /*pin_memory*/, bool /*non_blocking*/,
                      std::optional<at::MemoryFormat> /*memory_format*/) {
    auto target_dtype = dtype.value_or(self_in.scalar_type());
    // No-op if dtype unchanged and staying on tt device (or no device specified)
    bool same_device = !device.has_value() || is_tt(*device);
    if (target_dtype == self_in.scalar_type() && same_device) {
        return self_in;
    }
    // Cross-device copies should not reach this kernel; fallback handles them.
    TT_FATAL(same_device, "tt-kurbla _to_copy: cross-device copy reached native kernel");
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::_to_copy: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto target_mlir_type = mlir_element_type_for(target_dtype);
    auto result = mb.insert_typecast(mb.args()[0], target_mlir_type);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    return wrap_tt_tensor(std::move(outputs[0]), self_in.sizes(), target_dtype);
}

at::Tensor tt_permute(const at::Tensor &self, at::IntArrayRef dims) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::permute: tensor must be on tt backend");
    int64_t rank = self.dim();
    std::vector<int64_t> perm(dims.begin(), dims.end());
    for (auto &d : perm) {
        d = (d + rank) % rank;
    }
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_permute(mb, mb.args()[0], perm);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_cat(const c10::IListRef<at::Tensor> &tensors_list, int64_t dim) {
    TORCH_CHECK(!tensors_list.empty(), "tt-kurbla aten::cat: tensors must be non-empty");
    auto aligned = align_on_tt(tensors_list);
    // CPU cat silently ignores zero-numel tensors regardless of rank; TTIR concat
    // rejects mixed-rank inputs, so we must filter them out first.
    at::Tensor fallback = aligned[0];
    std::erase_if(aligned, [](const at::Tensor &t) { return t.numel() == 0; });
    if (aligned.empty()) {
        return fallback;
    }
    if (aligned.size() == 1) {
        return aligned[0];
    }
    // Compute promoted dtype.
    at::native::ResultTypeState state{};
    for (const auto &t : aligned) {
        state = at::native::update_result_type_state(t, state);
    }
    const auto promoted = at::native::result_type(state);
    const auto promoted_mlir = mlir_element_type_for(promoted);
    std::vector<TensorTypeSpec> specs;
    for (const auto &t : aligned) {
        specs.push_back(spec_for(t));
    }
    auto mb = ModuleBuilder::init(specs);
    llvm::SmallVector<mlir::Value> inputs;
    for (auto v : mb.args()) {
        inputs.push_back(mb.insert_typecast(v, promoted_mlir));
    }
    auto result = build_cat(mb, inputs, dim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), aligned);
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_slice(const at::Tensor &self, int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end,
                    int64_t step) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::slice.Tensor: tensor must be on tt backend");
    TORCH_CHECK(step > 0, "tt-kurbla aten::slice.Tensor: step must be positive, got ", step);
    int64_t rank = self.dim();
    int64_t norm_dim = (dim + rank) % rank;
    int64_t dim_size = self.size(norm_dim);
    int64_t s = start.has_value() ? start.value() : 0;
    int64_t e = end.has_value() ? end.value() : dim_size;
    // Normalize negatives and clamp to [0, dim_size].
    if (s < 0) {
        s += dim_size;
    }
    if (e < 0) {
        e += dim_size;
    }
    s = std::max<int64_t>(0, std::min(s, dim_size));
    e = std::max<int64_t>(0, std::min(e, dim_size));
    std::vector<int64_t> begins(as<std::size_t>(rank), 0);
    std::vector<int64_t> ends;
    for (int64_t i = 0; i < rank; ++i) {
        ends.push_back(self.size(i));
    }
    std::vector<int64_t> steps(as<std::size_t>(rank), 1);
    begins[as<std::size_t>(norm_dim)] = s;
    ends[as<std::size_t>(norm_dim)] = e;
    steps[as<std::size_t>(norm_dim)] = step;
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_slice(mb, mb.args()[0], begins, ends, steps);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

at::Tensor tt_argmax(const at::Tensor &self, std::optional<int64_t> dim, bool keepdim) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::argmax.default: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_argmax(mb, mb.args()[0], dim, keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Long);
}

// max.dim_max: reduce `self` over `dim` into max values (`max`) and their indices
// (`max_values`).
std::tuple<at::Tensor &, at::Tensor &> tt_max_dim_max(const at::Tensor &self, int64_t dim, bool keepdim,
                                                      at::Tensor &max, at::Tensor &max_values) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::max.dim_max: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto values_v = build_reduce<mlir::tt::ttir::MaxOp>(mb, mb.args()[0], {dim}, keepdim);
    auto indices_v = build_argmax(mb, mb.args()[0], dim, keepdim);
    auto vshape_ref = mlir::cast<mlir::RankedTensorType>(values_v.getType()).getShape();
    std::vector<int64_t> vshape(vshape_ref.begin(), vshape_ref.end());
    auto ishape_ref = mlir::cast<mlir::RankedTensorType>(indices_v.getType()).getShape();
    std::vector<int64_t> ishape(ishape_ref.begin(), ishape_ref.end());
    auto module_op = std::move(mb).finalize({values_v, indices_v});
    auto outputs = compile_and_run(std::move(module_op), {self});
    auto values = wrap_tt_tensor(std::move(outputs[0]), vshape, self.scalar_type());
    auto indices = wrap_tt_tensor(std::move(outputs[1]), ishape, at::ScalarType::Long);
    write_result_into(max, values);
    write_result_into(max_values, indices);
    return {max, max_values};
}

at::Tensor tt_pow_tensor_scalar(const at::Tensor &self, const at::Scalar &exponent) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::pow.Tensor_Scalar: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto input = mb.args()[0];
    auto elem_type = mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
    auto exp_val = build_scalar(mb, elem_type, exponent.toDouble());
    auto result = build_pow(mb, input, exp_val);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_add_scalar(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::add.Scalar: tensor must be on tt backend");
    double effective = other.toDouble() * alpha.toDouble();
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto input = mb.args()[0];
    auto elem_type = mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
    auto scalar = build_scalar(mb, elem_type, effective);
    auto result = build_add(mb, input, scalar);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_mul_scalar(const at::Tensor &self, const at::Scalar &other) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::mul.Scalar: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = scale_tensor(mb, mb.args()[0], other.toDouble());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_div_tensor(const at::Tensor &a_in, const at::Tensor &b_in) {
    const auto [a, b] = align_on_tt(a_in, b_in);
    // div.Tensor is true division: an integral/bool result promotes to the
    // default float dtype (int64 / int64 -> float32), floats keep their type.
    // Use at::result_type on the original inputs so a wrapped Python scalar
    // (`x / 2.0` dispatches here with 2.0 as a weak f64 tensor) takes x's dtype
    // instead of promoting it — bf16 / 2.0 stays bf16.
    const auto common = at::result_type(a_in, b_in);
    const auto result_dtype = c10::isFloatingType(common) ? common : c10::typeMetaToScalarType(at::get_default_dtype());

    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    const auto target = mlir_element_type_for(result_dtype);
    auto lhs = mb.insert_typecast(mb.args()[0], target);
    auto rhs = mb.insert_typecast(mb.args()[1], target);
    auto result = build_div(mb, lhs, rhs);
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, result_dtype);
}

at::Tensor tt_floor_divide(const at::Tensor &a_in, const at::Tensor &b_in) {
    const auto [a, b] = align_on_tt(a_in, b_in);
    auto mb = ModuleBuilder::init({spec_for(a), spec_for(b)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);
    auto result = build_floor_divide(mb, lhs, rhs);
    auto out_shape = at::infer_size(a.sizes(), b.sizes());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {a, b});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor tt_div_scalar(const at::Tensor &self, const at::Scalar &other) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::div.Scalar: tensor must be on tt backend");
    TT_FATAL(other.toDouble() != 0.0, "tt_div_scalar: division by zero");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = scale_tensor(mb, mb.args()[0], 1.0 / other.toDouble());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_cos(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::cos: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_cos(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_sin(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::sin: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_sin(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_log(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::log: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_log(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_exp(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::exp: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_exp(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_log1p(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::log1p: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_log1p(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_sqrt(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::sqrt: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_sqrt(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_tanh(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::tanh: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_tanh(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

// The optional `dtype` asks for accumulation in a wider type than the input;
// ttir.cumsum accumulates in the input's element type, so only a same-dtype (or
// absent) request maps onto it.
at::Tensor tt_cumsum(const at::Tensor &self, int64_t dim, std::optional<at::ScalarType> dtype) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::cumsum: tensor must be on tt backend");
    TORCH_CHECK(!dtype.has_value() || *dtype == self.scalar_type(),
                "tt-kurbla aten::cumsum: dtype must match the input's (", self.scalar_type(), "), got ", *dtype);
    auto mb = ModuleBuilder::init({spec_for(self)});
    int64_t rank = std::max<int64_t>(self.dim(), 1);
    auto result = build_cumsum(mb, mb.args()[0], (dim % rank + rank) % rank);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

// full_like takes its shape from `self` and, unless `dtype` overrides it, its
// element type too. `self`'s contents are never read, so it is not an operand:
// the module is input-less like arange's.
at::Tensor tt_full_like(const at::Tensor &self, const at::Scalar &fill_value, std::optional<at::ScalarType> dtype,
                        std::optional<at::Layout> /*layout*/, std::optional<at::Device> /*device*/,
                        std::optional<bool> /*pin_memory*/, std::optional<at::MemoryFormat> /*memory_format*/) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::full_like: tensor must be on tt backend");
    auto out_dtype = dtype.value_or(self.scalar_type());
    auto mb = ModuleBuilder::init({});
    auto result = build_full(mb, self.sizes(), fill_value.toDouble(), mlir_element_type_for(out_dtype));
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), out_dtype);
}

at::Tensor tt_neg(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::neg: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_neg(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

namespace {

at::Tensor arange_impl(int64_t start, int64_t end, int64_t step, at::ScalarType out_dtype,
                       std::optional<at::Device> /*device*/) {
    TT_FATAL(step != 0, "tt_arange: step must be non-zero");
    int64_t n = std::max<int64_t>(0, (end - start + step - 1) / step);
    auto mb = ModuleBuilder::init({});
    auto mlir_dtype = mlir_element_type_for(out_dtype);
    auto result = build_arange(mb, start, end, step, mlir_dtype);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {});
    return wrap_tt_tensor(std::move(outputs[0]), {n}, out_dtype);
}

} // namespace

at::Tensor tt_arange(const at::Scalar &end_scalar, std::optional<at::ScalarType> dtype,
                     std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                     std::optional<bool> /*pin_memory*/) {
    auto out_dtype = dtype.value_or(at::ScalarType::Long);
    int64_t e = as<int64_t>(std::ceil(end_scalar.toDouble()));
    return arange_impl(0, e, 1, out_dtype, device);
}

at::Tensor tt_arange_start(const at::Scalar &start_scalar, const at::Scalar &end_scalar,
                           std::optional<at::ScalarType> dtype, std::optional<at::Layout> /*layout*/,
                           std::optional<at::Device> device, std::optional<bool> /*pin_memory*/) {
    auto out_dtype = dtype.value_or(at::ScalarType::Long);
    int64_t s = as<int64_t>(std::floor(start_scalar.toDouble()));
    int64_t e = as<int64_t>(std::ceil(end_scalar.toDouble()));
    return arange_impl(s, e, 1, out_dtype, device);
}

at::Tensor tt_arange_start_step(const at::Scalar &start_scalar, const at::Scalar &end_scalar,
                                const at::Scalar &step_scalar, std::optional<at::ScalarType> dtype,
                                std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                                std::optional<bool> /*pin_memory*/) {
    auto out_dtype = dtype.value_or(at::ScalarType::Long);
    int64_t s = as<int64_t>(std::floor(start_scalar.toDouble()));
    int64_t e = as<int64_t>(std::ceil(end_scalar.toDouble()));
    int64_t step = as<int64_t>(step_scalar.toDouble());
    return arange_impl(s, e, step, out_dtype, device);
}

at::Tensor tt_sigmoid(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::sigmoid: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_sigmoid(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_clamp(const at::Tensor &self, const std::optional<at::Scalar> &min,
                    const std::optional<at::Scalar> &max) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::clamp: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    std::optional<double> min_v = min.has_value() ? std::optional<double>(min->toDouble()) : std::nullopt;
    std::optional<double> max_v = max.has_value() ? std::optional<double>(max->toDouble()) : std::nullopt;
    auto result = build_clamp(mb, mb.args()[0], min_v, max_v);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_silu(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::silu: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_silu(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_gelu(const at::Tensor &self, c10::string_view approximate) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::gelu: tensor must be on tt backend");
    // tt-mlir lowers gelu to ttnn.gelu(fast_and_approximate_mode=false): the
    // exact/accurate variant, i.e. approximate="none". A "tanh" request is
    // served by this same accurate op - correct, it just doesn't get the
    // faster tanh-approx kernel.
    (void)approximate;
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result = build_gelu(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor tt_softmax(const at::Tensor &self_in, int64_t dim, bool half_to_float) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::_softmax: tensor must be on tt backend");
    auto out_dtype = half_to_float ? at::ScalarType::Float : self_in.scalar_type();
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    mlir::Value input = mb.args()[0];
    if (half_to_float) {
        input = mb.insert_typecast(input, mb.attrs().getF32Type());
    }
    auto result = build_softmax(mb, input, dim);
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    return wrap_tt_tensor(std::move(outputs[0]), self_in.sizes(), out_dtype);
}

namespace {

// Shared implementation for where.self and where.self_out.
at::Tensor tt_where(const at::Tensor &condition_in, const at::Tensor &self_in, const at::Tensor &other_in) {
    auto [condition, self, other] = align_on_tt(condition_in, self_in, other_in);
    auto promoted = at::result_type(self, other);
    auto promoted_mlir = mlir_element_type_for(promoted);
    auto mb = ModuleBuilder::init({spec_for(condition), spec_for(self), spec_for(other)});
    // condition is Bool — don't promote it; promote self and other independently.
    auto s_v = mb.insert_typecast(mb.args()[1], promoted_mlir);
    auto o_v = mb.insert_typecast(mb.args()[2], promoted_mlir);
    auto result_v = build_where(mb, mb.args()[0], s_v, o_v);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {condition, self, other});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor &tt_where_out(const at::Tensor &condition_in, const at::Tensor &self_in, const at::Tensor &other_in,
                         at::Tensor &out) {
    return write_result_into(out, tt_where(condition_in, self_in, other_in));
}

// index_copy: out-of-place scatter of `source` rows into a copy of `self` at the
// positions named by the 1-D integer `index` along `dim`. build_index_copy wants
// a non-negative dim and source rank == self rank (the aten contract guarantees
// the latter).
at::Tensor tt_index_copy(const at::Tensor &self_in, int64_t dim, const at::Tensor &index_in,
                         const at::Tensor &source_in) {
    auto [self, index, source] = align_on_tt(self_in, index_in, source_in);
    int64_t rank = self.dim();
    int64_t norm_dim = (dim + rank) % rank;
    auto mb = ModuleBuilder::init({spec_for(self), spec_for(index), spec_for(source)});
    auto result_v = build_index_copy(mb, mb.args()[0], norm_dim, mb.args()[1], mb.args()[2]);
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self, index, source});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

// index_copy.out: scatter into `out`. `index_copy_` (in-place) also routes here
// with out aliasing self.
at::Tensor &tt_index_copy_out(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &source,
                              at::Tensor &out) {
    return write_result_into(out, tt_index_copy(self, dim, index, source));
}

// le.Tensor: element-wise `self <= other`, producing a Bool tensor.
at::Tensor tt_le_tensor(const at::Tensor &self_in, const at::Tensor &other_in) {
    auto [self, other] = align_on_tt(self_in, other_in);
    auto mb = ModuleBuilder::init({spec_for(self), spec_for(other)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, self, other);
    auto result_v = build_le(mb, lhs, rhs);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self, other});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Bool);
}

at::Tensor &tt_le_tensor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out) {
    return write_result_into(out, tt_le_tensor(self, other));
}

// gt.Tensor: element-wise `self > other`, producing a Bool tensor.
at::Tensor tt_gt_tensor(const at::Tensor &self_in, const at::Tensor &other_in) {
    auto [self, other] = align_on_tt(self_in, other_in);
    auto mb = ModuleBuilder::init({spec_for(self), spec_for(other)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, self, other);
    auto result_v = build_gt(mb, lhs, rhs);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self, other});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Bool);
}

at::Tensor &tt_gt_tensor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out) {
    return write_result_into(out, tt_gt_tensor(self, other));
}

// bitwise_and.Tensor: element-wise `self & other`. On Bool operands this is the
// logical AND used to combine attention masks; the result keeps the promoted
// integer/bool element type.
at::Tensor tt_bitwise_and_tensor(const at::Tensor &self_in, const at::Tensor &other_in) {
    auto [self, other] = align_on_tt(self_in, other_in);
    auto mb = ModuleBuilder::init({spec_for(self), spec_for(other)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, self, other);
    auto result_v = build_bitwise_and(mb, lhs, rhs);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self, other});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor &tt_bitwise_and_tensor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out) {
    return write_result_into(out, tt_bitwise_and_tensor(self, other));
}

at::Tensor tt_bitwise_or_tensor(const at::Tensor &self_in, const at::Tensor &other_in) {
    auto [self, other] = align_on_tt(self_in, other_in);
    auto mb = ModuleBuilder::init({spec_for(self), spec_for(other)});
    auto [promoted, lhs, rhs] = promote_inputs(mb, self, other);
    auto result_v = build_bitwise_or(mb, lhs, rhs);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self, other});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

at::Tensor &tt_bitwise_or_tensor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out) {
    return write_result_into(out, tt_bitwise_or_tensor(self, other));
}

at::Tensor tt_bitwise_not(const at::Tensor &self) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::bitwise_not: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result_v = build_bitwise_not(mb, mb.args()[0]);
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), self.sizes(), self.scalar_type());
}

at::Tensor &tt_bitwise_not_out(const at::Tensor &self, at::Tensor &out) {
    return write_result_into(out, tt_bitwise_not(self));
}

// tril.out: lower-triangular part of self, written into `out`.
at::Tensor &tt_tril_out(const at::Tensor &self_in, int64_t diagonal, at::Tensor &out) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::tril.out: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result_v = build_tril(mb, mb.args()[0], diagonal);
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    auto result = wrap_tt_tensor(std::move(outputs[0]), self_in.sizes(), self_in.scalar_type());
    return write_result_into(out, result);
}

// isneginf.out: element-wise test for negative infinity.
at::Tensor &tt_isneginf_out(const at::Tensor &self_in, at::Tensor &out) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::isneginf.out: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result_v = build_isneginf(mb, mb.args()[0]);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Bool);
    return write_result_into(out, result);
}

// all.out: logical AND reduction along `dim`.
at::Tensor &tt_all_out(const at::Tensor &self_in, int64_t dim, bool keepdim, at::Tensor &out) {
    TORCH_CHECK(is_tt(self_in), "tt-kurbla aten::all.out: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self_in)});
    auto result_v = build_all(mb, mb.args()[0], {dim}, keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self_in});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Bool);
    return write_result_into(out, result);
}

// Shared body of the three any.* out kernels. `dims` is empty for a reduction
// over every element (rank-0 result); build_any normalizes negative dims.
at::Tensor &tt_any_reduce_out(const at::Tensor &self, llvm::ArrayRef<int64_t> dims, bool keepdim, at::Tensor &out) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::any: tensor must be on tt backend");
    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result_v = build_any(mb, mb.args()[0], dims, keepdim);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self});
    auto result = wrap_tt_tensor(std::move(outputs[0]), out_shape, at::ScalarType::Bool);
    return write_result_into(out, result);
}

// constant_pad_nd: aten lists the pad amounts from the *last* dimension
// backwards as (low, high) pairs, and only for the trailing dims it touches.
// Spread them over one low/high entry per dim, in dim order, for build_pad.
at::Tensor tt_constant_pad_nd(const at::Tensor &self, at::IntArrayRef pad, const at::Scalar &value) {
    TORCH_CHECK(is_tt(self), "tt-kurbla aten::constant_pad_nd: tensor must be on tt backend");
    TORCH_CHECK(pad.size() % 2 == 0, "tt-kurbla aten::constant_pad_nd: pad must have an even length, got ", pad.size());
    int64_t rank = self.dim();
    TORCH_CHECK(as<int64_t>(pad.size()) <= 2 * rank, "tt-kurbla aten::constant_pad_nd: pad covers ", pad.size() / 2,
                " dims but the tensor has rank ", rank);

    std::vector<int64_t> low(as<std::size_t>(rank), 0);
    std::vector<int64_t> high(as<std::size_t>(rank), 0);
    for (std::size_t i = 0; i < pad.size() / 2; ++i) {
        auto dim = as<std::size_t>(rank - 1 - as<int64_t>(i));
        low[dim] = pad[2 * i];
        high[dim] = pad[2 * i + 1];
    }

    auto mb = ModuleBuilder::init({spec_for(self)});
    auto result_v = build_pad(mb, mb.args()[0], low, high, value.toDouble());
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result_v.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result_v});
    auto outputs = compile_and_run(std::move(module_op), {self});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, self.scalar_type());
}

// any.all_out: logical OR over every element, into a rank-0 `out`.
at::Tensor &tt_any_all_out(const at::Tensor &self, at::Tensor &out) {
    return tt_any_reduce_out(self, {}, /*keepdim=*/false, out);
}

// any.out: logical OR reduction along a single `dim`.
at::Tensor &tt_any_out(const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &out) {
    return tt_any_reduce_out(self, {dim}, keepdim, out);
}

// any.dims_out: logical OR reduction along `dim`; dim=None means every dimension.
at::Tensor &tt_any_dims_out(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim, at::Tensor &out) {
    llvm::SmallVector<int64_t> dims;
    if (dim.has_value()) {
        dims.assign(dim.value().begin(), dim.value().end());
    }
    return tt_any_reduce_out(self, dims, keepdim, out);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(tt_add));
    m.impl("add.Scalar", TORCH_FN(tt_add_scalar));
    m.impl("add.out", TORCH_FN(tt_add_out));
    m.impl("sub.Tensor", TORCH_FN(tt_sub));
    m.impl("mul.Tensor", TORCH_FN(tt_mul));
    m.impl("mul.Scalar", TORCH_FN(tt_mul_scalar));
    m.impl("relu", TORCH_FN(tt_relu));
    m.impl("relu_", TORCH_FN(tt_relu_));
    m.impl("rsqrt", TORCH_FN(tt_rsqrt));
    m.impl("mean.dim", TORCH_FN(tt_mean));
    m.impl("batch_norm", TORCH_FN(tt_batch_norm_inference));
    m.impl("max_pool2d_with_indices", TORCH_FN(tt_max_pool2d_with_indices));
    m.impl("unsqueeze", TORCH_FN(tt_unsqueeze));
    m.impl("squeeze.dim", TORCH_FN(tt_squeeze_dim));
    m.impl("expand", TORCH_FN(tt_expand));
    m.impl("transpose.int", TORCH_FN(tt_transpose_int));
    // NOTE: _to_copy is intentionally NOT registered here, because it has problems.
    m.impl("permute", TORCH_FN(tt_permute));
    m.impl("cat", TORCH_FN(tt_cat));
    m.impl("slice.Tensor", TORCH_FN(tt_slice));
    m.impl("argmax", TORCH_FN(tt_argmax));
    m.impl("max.dim_max", TORCH_FN(tt_max_dim_max));
    m.impl("pow.Tensor_Scalar", TORCH_FN(tt_pow_tensor_scalar));
    m.impl("div.Tensor", TORCH_FN(tt_div_tensor));
    m.impl("div.Scalar", TORCH_FN(tt_div_scalar));
    m.impl("cos", TORCH_FN(tt_cos));
    m.impl("sin", TORCH_FN(tt_sin));
    m.impl("neg", TORCH_FN(tt_neg));
    m.impl("log", TORCH_FN(tt_log));
    m.impl("exp", TORCH_FN(tt_exp));
    m.impl("log1p", TORCH_FN(tt_log1p));
    m.impl("sqrt", TORCH_FN(tt_sqrt));
    m.impl("tanh", TORCH_FN(tt_tanh));
    m.impl("cumsum", TORCH_FN(tt_cumsum));
    m.impl("full_like", TORCH_FN(tt_full_like));
    m.impl("arange", TORCH_FN(tt_arange));
    m.impl("arange.start", TORCH_FN(tt_arange_start));
    m.impl("arange.start_step", TORCH_FN(tt_arange_start_step));
    m.impl("silu", TORCH_FN(tt_silu));
    m.impl("sigmoid", TORCH_FN(tt_sigmoid));
    m.impl("clamp", TORCH_FN(tt_clamp));
    m.impl("floor_divide", TORCH_FN(tt_floor_divide));
    m.impl("gelu", TORCH_FN(tt_gelu));
    m.impl("_softmax", TORCH_FN(tt_softmax));
    m.impl("sum.IntList_out", TORCH_FN(tt_sum_out));
    m.impl("threshold_backward.grad_input", TORCH_FN(tt_threshold_backward_out));
    m.impl("mse_loss.out", TORCH_FN(tt_mse_loss_out));
    m.impl("mse_loss_backward", TORCH_FN(tt_mse_loss_backward));
    m.impl("where.self", TORCH_FN(tt_where));
    m.impl("where.self_out", TORCH_FN(tt_where_out));
    m.impl("isneginf.out", TORCH_FN(tt_isneginf_out));
    m.impl("all.out", TORCH_FN(tt_all_out));
    m.impl("any.all_out", TORCH_FN(tt_any_all_out));
    m.impl("any.out", TORCH_FN(tt_any_out));
    m.impl("any.dims_out", TORCH_FN(tt_any_dims_out));
    m.impl("constant_pad_nd", TORCH_FN(tt_constant_pad_nd));
    m.impl("tril.out", TORCH_FN(tt_tril_out));
    m.impl("index_copy.out", TORCH_FN(tt_index_copy_out));
    m.impl("le.Tensor_out", TORCH_FN(tt_le_tensor_out));
    m.impl("gt.Tensor_out", TORCH_FN(tt_gt_tensor_out));
    m.impl("bitwise_and.Tensor_out", TORCH_FN(tt_bitwise_and_tensor_out));
    m.impl("bitwise_or.Tensor_out", TORCH_FN(tt_bitwise_or_tensor_out));
    m.impl("bitwise_not.out", TORCH_FN(tt_bitwise_not_out));
}

} // namespace tt::kurbla::torch_backend
