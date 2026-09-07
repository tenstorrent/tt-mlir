// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "engine/ttir_module_builder.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include "assert.hpp"
#include "cast.hpp"
#include "config.hpp"
#include "engine/compile.hpp"

namespace tt::crank {

namespace {

mlir::RankedTensorType to_tensor_type(mlir::MLIRContext &ctx, const TensorTypeSpec &spec) {
    return mlir::RankedTensorType::get(spec.shape, to_mlir_element_type(ctx, spec.dtype));
}

} // namespace

mlir::Type to_mlir_element_type(mlir::MLIRContext &ctx, ::tt::target::DataType dtype) {
    mlir::Builder b(&ctx);
    switch (dtype) {
        case ::tt::target::DataType::BFloat16:
            return b.getBF16Type();
        case ::tt::target::DataType::Float32:
            return b.getF32Type();
        case ::tt::target::DataType::Float64:
            return b.getF64Type();
        case ::tt::target::DataType::Int32:
            return b.getI32Type();
        case ::tt::target::DataType::Int64:
            return b.getI64Type();
        case ::tt::target::DataType::Bool:
            return b.getI1Type();
        default:
            break;
    }
    TT_THROW("tt-crank ModuleBuilder: unsupported runtime dtype for MLIR element type: {}", as<int>(dtype));
}

ModuleBuilder::ModuleBuilder(mlir::OwningOpRef<mlir::ModuleOp> module_op, mlir::func::FuncOp func,
                             mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<mlir::Value> args)
    : module_op_(std::move(module_op)), func_(func), builder_(std::move(builder)), loc_(loc), args_(std::move(args)) {}

ModuleBuilder ModuleBuilder::init(llvm::ArrayRef<TensorTypeSpec> inputs,
                                  llvm::ArrayRef<mlir::tt::ttcore::ArgumentType> arg_types) {
    TT_FATAL(arg_types.empty() || arg_types.size() == inputs.size(),
             "tt-crank ModuleBuilder: inputs and arg_types size mismatch: {} vs {}", inputs.size(), arg_types.size());

    auto &ctx = mlir_context();
    auto loc = mlir::UnknownLoc::get(&ctx);

    llvm::SmallVector<mlir::Type> input_types;
    input_types.reserve(inputs.size());
    for (const auto &spec : inputs) {
        input_types.push_back(to_tensor_type(ctx, spec));
    }

    auto module_op = mlir::ModuleOp::create(loc);
    mlir::OpBuilder module_builder(module_op.getBodyRegion());
    auto fn_type = mlir::FunctionType::get(&ctx, input_types, /*results=*/{});
    auto func = module_builder.create<mlir::func::FuncOp>(loc, "main", fn_type);

    // Tag non-Input args so tt-mlir's const-eval hoist can fold weight-only
    // subgraphs into cached funcs. Untagged args default to Input.
    if (comp_consteval_enabled()) {
        for (std::size_t i = 0; i < arg_types.size(); ++i) {
            if (arg_types[i] != mlir::tt::ttcore::ArgumentType::Input) {
                func.setArgAttr(as<unsigned>(i), mlir::tt::ttcore::ArgumentTypeAttr::name,
                                mlir::tt::ttcore::ArgumentTypeAttr::get(&ctx, arg_types[i]));
            }
        }
    }

    mlir::Block *entry = func.addEntryBlock();
    mlir::OpBuilder body_builder(&ctx);
    body_builder.setInsertionPointToStart(entry);

    llvm::SmallVector<mlir::Value> args(entry->args_begin(), entry->args_end());

    return ModuleBuilder(mlir::OwningOpRef<mlir::ModuleOp>(module_op), func, std::move(body_builder), loc,
                         std::move(args));
}

mlir::OwningOpRef<mlir::ModuleOp> ModuleBuilder::finalize(llvm::ArrayRef<mlir::Value> outputs) && {
    builder_.create<mlir::func::ReturnOp>(loc_, mlir::ValueRange(outputs));

    llvm::SmallVector<mlir::Type> result_types;
    result_types.reserve(outputs.size());
    for (auto v : outputs) {
        result_types.push_back(v.getType());
    }
    auto input_types = func_.getFunctionType().getInputs();
    func_.setFunctionType(mlir::FunctionType::get(builder_.getContext(), input_types, result_types));

    return std::move(module_op_);
}

mlir::Value ModuleBuilder::insert_typecast(mlir::Value value, mlir::Type target) {
    auto src_type = mlir::cast<mlir::RankedTensorType>(value.getType());
    if (src_type.getElementType() == target) {
        return value;
    }
    auto dst_type = mlir::RankedTensorType::get(src_type.getShape(), target);
    return create<mlir::tt::ttir::TypecastOp>(dst_type, value).getResult();
}

// ---- op builders ------------------------------------------------------------

namespace {

// Mirror at::Reduction's None=0 / Mean=1 values — part of the builder contract.
constexpr std::int64_t reduction_none = 0;
constexpr std::int64_t reduction_mean = 1;

llvm::SmallVector<int32_t> shape_to_i32(llvm::ArrayRef<int64_t> shape) {
    llvm::SmallVector<int32_t> out;
    for (int64_t d : shape) {
        out.push_back(as<int32_t>(d));
    }
    return out;
}

// Promote two element types the way PyTorch does for the cases that reach us:
// a floating type outranks any integer/bool type; otherwise the wider bit width
// wins (bool is i1, so int beats bool; f64 beats f32; i64 beats i32).
mlir::Type promote_element_types(mlir::Type a, mlir::Type b) {
    if (a == b) {
        return a;
    }
    bool a_float = mlir::isa<mlir::FloatType>(a);
    bool b_float = mlir::isa<mlir::FloatType>(b);
    if (a_float != b_float) {
        return a_float ? a : b;
    }
    return a.getIntOrFloatBitWidth() >= b.getIntOrFloatBitWidth() ? a : b;
}

// Writes `source` into the rank-4 [batch, heads, seq, head_dim] `cache` at the
// sequence positions in the 1-D `index`, emitting the purpose-built
// ttir.update_cache (decode, one new token) / ttir.fill_cache (prefill, many
// tokens) so it maps to dedicated ttnn ops rather than a generic ttir.scatter.
//
// update_cache honors the runtime update_index, so the decode rewrite is exact
// for any positions. fill_cache writes contiguously from seq 0 and carries no
// seq offset, so the prefill rewrite is correct only for a from-scratch prefill
// (index = arange from 0); an offset/chunked prefill would miswrite, and nothing
// here checks that. The caller is trusted to only reach this with seq_update > 1
// on a fresh cache.
mlir::Value build_kv_cache_write(ModuleBuilder &mb, mlir::Value cache, mlir::Value index, mlir::Value source) {
    auto cache_type = mlir::cast<mlir::RankedTensorType>(cache.getType());
    auto source_shape = mlir::cast<mlir::RankedTensorType>(source.getType()).getShape();
    int64_t batch = cache_type.getShape()[0];
    int64_t seq_update = source_shape[2];
    auto i32_type = mb.attrs().getI32Type();

    if (seq_update == 1) {
        // Decode: update_cache wants input [1, num_heads, num_users, head_dim], so
        // permute the [batch, heads, 1, head_dim] new token to [1, heads, batch,
        // head_dim] (a no-op when batch == 1, where dim 0 is already 1).
        mlir::Value updates = batch > 1 ? build_permute(mb, source, {2, 1, 0, 3}) : source;
        // The tt-metal kernel takes i32 positions; cache_position is i64.
        mlir::Value update_index = index;
        if (mlir::cast<mlir::RankedTensorType>(index.getType()).getElementType() != i32_type) {
            update_index = mb.insert_typecast(index, i32_type);
        }
        // update_cache mutates `cache` in place and produces no result; the same
        // SSA value carries the updated tensor forward.
        mb.create<mlir::tt::ttir::UpdateCacheOp>(cache, updates, update_index, mb.attrs().getI32IntegerAttr(0));
        return cache;
    }

    // Prefill: fill_cache fills a single batch slab from seq 0, so emit one op per
    // batch element (slicing it out) and chain the in-place result.
    llvm::SmallVector<int64_t> begins(4, 0), steps(4, 1);
    llvm::SmallVector<int64_t> ends(source_shape.begin(), source_shape.end());
    for (int64_t b = 0; b < batch; ++b) {
        mlir::Value slab = source;
        if (batch > 1) {
            begins[0] = b;
            ends[0] = b + 1;
            slab = build_slice(mb, source, begins, ends, steps);
        }
        // fill_cache mutates `cache` in place and produces no result; each op in
        // the loop writes a batch slab, chained by their MemWrite effect.
        mb.create<mlir::tt::ttir::FillCacheOp>(cache, slab, mb.attrs().getI32IntegerAttr(as<int32_t>(b)));
    }
    return cache;
}

// ReduceAndOp/ReduceOrOp are i1-in, i1-out, but torch's all/any take any dtype
// and treat every nonzero element as true. A typecast to i1 would truncate
// instead of testing, so compare against zero (the same trick build_logical_not
// uses on non-Bool input).
mlir::Value coerce_to_bool(ModuleBuilder &mb, mlir::Value input) {
    auto elem = mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
    if (elem.isInteger(1)) {
        return input;
    }
    return build_ne(mb, input, build_scalar(mb, elem, 0.0));
}

} // namespace

llvm::SmallVector<std::int64_t> broadcast_shape(llvm::ArrayRef<std::int64_t> lhs, llvm::ArrayRef<std::int64_t> rhs) {
    std::size_t rank = std::max(lhs.size(), rhs.size());
    llvm::SmallVector<std::int64_t> out(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        std::int64_t l = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
        std::int64_t r = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
        TT_FATAL(l == r || l == 1 || r == 1, "broadcast_shape: incompatible sizes {} and {} at dim {}", l, r, i);
        out[i] = std::max(l, r);
    }
    return out;
}

mlir::Value build_relu(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::ReluOp>(result_type, input).getResult();
}

mlir::Value build_rsqrt(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::RsqrtOp>(result_type, input).getResult();
}

mlir::Value build_sub(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "tt-crank build_sub: lhs and rhs must share element type — callers must promote first");

    if (alpha != 1.0) {
        rhs = scale_tensor(mb, rhs, alpha);
    }

    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::SubtractOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_mul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "tt-crank build_mul: lhs and rhs must share element type — callers must promote first");

    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::MultiplyOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_reshape(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> new_shape) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto result_type = mlir::RankedTensorType::get(new_shape, input_type.getElementType());
    llvm::SmallVector<int32_t> shape_i32(new_shape.begin(), new_shape.end());
    auto shape_attr = mb.attrs().getI32ArrayAttr(shape_i32);
    return mb.create<mlir::tt::ttir::ReshapeOp>(result_type, input, shape_attr).getResult();
}

mlir::Value build_mean(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim) {
    return build_reduce<mlir::tt::ttir::MeanOp>(mb, input, dims, keepdim);
}

mlir::Value build_sum(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim) {
    return build_reduce<mlir::tt::ttir::SumOp>(mb, input, dims, keepdim);
}

mlir::Value build_vector_norm(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> dims, bool keepdim) {
    return build_sqrt(mb, build_sum(mb, build_mul(mb, input, input), dims, keepdim));
}

mlir::Value build_threshold_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self, double threshold) {
    auto self_type = mlir::cast<mlir::RankedTensorType>(self.getType());
    auto grad_type = mlir::cast<mlir::RankedTensorType>(grad_output.getType());

    // mask = self > threshold, emitted as an i1 tensor of self's shape. The
    // threshold constant is [1]-shaped and broadcasts against self.
    auto threshold_const = build_scalar(mb, self_type.getElementType(), threshold);
    auto mask_type = mlir::RankedTensorType::get(self_type.getShape(), mb.attrs().getI1Type());
    auto mask = mb.create<mlir::tt::ttir::GreaterThanOp>(mask_type, self, threshold_const).getResult();

    // Cast the bool mask to the gradient's element type (1.0 / 0.0) and gate
    // the incoming gradient with a plain elementwise multiply.
    auto mask_cast = mb.insert_typecast(mask, grad_type.getElementType());
    return build_mul(mb, grad_output, mask_cast);
}

mlir::Value build_mse_loss(ModuleBuilder &mb, mlir::Value self, mlir::Value target, std::int64_t reduction) {
    auto diff = build_sub(mb, self, target);
    auto sq = build_mul(mb, diff, diff);
    if (reduction == reduction_none) {
        // elementwise squared error, no reduction.
        return sq;
    }

    // Mean / Sum: reduce over every element. Flatten first so the reduction is
    // a single dim-0 reduce that keeps a `[1]` scalar result.
    auto sq_type = mlir::cast<mlir::RankedTensorType>(sq.getType());
    std::int64_t numel = 1;
    for (auto d : sq_type.getShape()) {
        numel *= d;
    }
    auto flat = build_reshape(mb, sq, {numel});
    if (reduction == reduction_mean) {
        return build_mean(mb, flat, {0}, /*keepdim=*/false);
    }
    return build_sum(mb, flat, {0}, /*keepdim=*/false);
}

mlir::Value build_mse_loss_backward(ModuleBuilder &mb, mlir::Value grad_output, mlir::Value self, mlir::Value target,
                                    std::int64_t reduction) {
    auto diff = build_sub(mb, self, target);

    // d/dself mean((self-target)^2) = 2*(self-target)/N; Sum/None drop the /N.
    std::int64_t n = 1;
    if (reduction == reduction_mean) {
        for (auto d : mlir::cast<mlir::RankedTensorType>(self.getType()).getShape()) {
            n *= d;
        }
    }
    auto scaled = scale_tensor(mb, diff, 2.0 / as<double>(n));

    // grad_output is the upstream (scalar, `[1]`) gradient; it broadcasts over
    // `self`'s shape just like build_scalar's constants do.
    return build_mul(mb, grad_output, scaled);
}

mlir::Value build_all_reduce(ModuleBuilder &mb, mlir::Value input, const std::string &reduce_op,
                             std::uint32_t cluster_axis) {
    // Map the c10d reduce-op string to a ttcore ReduceType. Only Sum is wired
    // end-to-end today (the metal all_reduce kernel hardcodes sum); other ops
    // surface here as a clear error rather than silently summing.
    TT_FATAL(reduce_op == "sum", "tt-crank build_all_reduce: only reduce_op='sum' supported, got '{}'", reduce_op);
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto reduce_type_attr =
        ::mlir::tt::ttcore::ReduceTypeAttr::get(result_type.getContext(), ::mlir::tt::ttcore::ReduceType::Sum);
    return mb
        .create<mlir::tt::ttir::AllReduceOp>(result_type, input, reduce_type_attr,
                                             mb.attrs().getUI32IntegerAttr(cluster_axis))
        .getResult();
}

mlir::Value build_reduce_scatter(ModuleBuilder &mb, mlir::Value input, std::int64_t group_size,
                                 std::uint32_t cluster_axis, std::int64_t scatter_dim) {
    // Inverse of build_all_gather: scatter `scatter_dim` across the group,
    // summing the per-chip contributions. Sum is the only wired reduce op
    // (mirrors build_all_reduce). TTIR/TTNN reduce_scatter carry an arbitrary
    // scatter_dim, so we scatter the requested dim directly.
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    std::vector<std::int64_t> out_shape(input_type.getShape().begin(), input_type.getShape().end());
    const auto rank = as<std::int64_t>(out_shape.size());
    TT_FATAL(rank > 0, "tt-crank build_reduce_scatter: input must be at least 1-D");
    const auto dim = scatter_dim < 0 ? scatter_dim + rank : scatter_dim;
    TT_FATAL(dim >= 0 && dim < rank, "tt-crank build_reduce_scatter: scatter_dim {} out of range for rank {}",
             scatter_dim, rank);
    TT_FATAL(out_shape[as<std::size_t>(dim)] % group_size == 0,
             "tt-crank build_reduce_scatter: dim {} ({}) not divisible by group size {}", dim,
             out_shape[as<std::size_t>(dim)], group_size);
    out_shape[as<std::size_t>(dim)] /= group_size;
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    auto reduce_type_attr =
        ::mlir::tt::ttcore::ReduceTypeAttr::get(result_type.getContext(), ::mlir::tt::ttcore::ReduceType::Sum);
    return mb
        .create<mlir::tt::ttir::ReduceScatterOp>(result_type, input, reduce_type_attr,
                                                 mb.attrs().getSI32IntegerAttr(as<std::int32_t>(dim)),
                                                 mb.attrs().getUI32IntegerAttr(cluster_axis))
        .getResult();
}

mlir::Value build_all_gather(ModuleBuilder &mb, mlir::Value input, std::int64_t group_size,
                             std::uint32_t cluster_axis) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    std::vector<std::int64_t> out_shape(input_type.getShape().begin(), input_type.getShape().end());
    TT_FATAL(!out_shape.empty(), "tt-crank build_all_gather: input must be at least 1-D");
    // all_gather_dim is fixed at 0 by the all_gather_into_tensor / _allgather_base
    // contract (the result concatenates the per-rank slabs along dim 0).
    out_shape[0] *= group_size;
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb
        .create<mlir::tt::ttir::AllGatherOp>(result_type, input,
                                             /*all_gather_dim=*/mb.attrs().getSI32IntegerAttr(0),
                                             mb.attrs().getUI32IntegerAttr(cluster_axis))
        .getResult();
}

mlir::Value build_scalar(ModuleBuilder &mb, mlir::Type element_type, double value) {
    // Shape `[1]` broadcasts against any rank via numpy-style prepend-1 rules.
    auto tensor_type = mlir::RankedTensorType::get({1}, element_type);
    mlir::DenseElementsAttr value_attr;
    if (auto float_ty = mlir::dyn_cast<mlir::FloatType>(element_type)) {
        // Build the APFloat at the element type's semantics so bf16/f16/f64 keep
        // their native precision in the IR (no F32-attr downsampling hack).
        llvm::APFloat ap(value);
        bool loses_info = false;
        ap.convert(float_ty.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &loses_info);
        value_attr = mlir::DenseElementsAttr::get(tensor_type, ap);
    } else {
        auto int_ty = mlir::cast<mlir::IntegerType>(element_type);
        bool is_signed = int_ty.getWidth() > 1;
        llvm::APInt ap(int_ty.getWidth(), as_unchecked<std::uint64_t>(as_unchecked<std::int64_t>(value)),
                       is_signed); // NOLINT
        value_attr = mlir::DenseElementsAttr::get(tensor_type, ap);
    }
    auto constant = mb.create<mlir::tt::ttir::ConstantOp>(tensor_type, value_attr);
    return constant.getResult();
}

// These creation ops (ttir.zeros/ones/full) take the result shape directly and
// have no operands, so they replace a scalar constant + broadcast and handle a
// rank-0 result without a special case.
mlir::Value build_zeros(ModuleBuilder &mb, llvm::ArrayRef<int64_t> shape, mlir::Type element_type) {
    auto result_type = mlir::RankedTensorType::get(shape, element_type);
    return mb.create<mlir::tt::ttir::ZerosOp>(result_type, llvm::ArrayRef<int32_t>(shape_to_i32(shape))).getResult();
}

mlir::Value build_ones(ModuleBuilder &mb, llvm::ArrayRef<int64_t> shape, mlir::Type element_type) {
    auto result_type = mlir::RankedTensorType::get(shape, element_type);
    return mb.create<mlir::tt::ttir::OnesOp>(result_type, llvm::ArrayRef<int32_t>(shape_to_i32(shape))).getResult();
}

mlir::Value build_full(ModuleBuilder &mb, llvm::ArrayRef<int64_t> shape, double value, mlir::Type element_type) {
    auto result_type = mlir::RankedTensorType::get(shape, element_type);
    // ttir.full wants the fill value as an f32 or i32 attr regardless of the
    // result element type (the op casts it to fill the tensor).
    mlir::Attribute fill_attr;
    if (mlir::isa<mlir::IntegerType>(element_type)) {
        fill_attr = mb.attrs().getI32IntegerAttr(as<int32_t>(value));
    } else {
        fill_attr = mlir::FloatAttr::get(mb.attrs().getF32Type(), value);
    }
    return mb.create<mlir::tt::ttir::FullOp>(result_type, llvm::ArrayRef<int32_t>(shape_to_i32(shape)), fill_attr)
        .getResult();
}

mlir::Value build_bn_inference(ModuleBuilder &mb, mlir::Value operand, mlir::Value scale, mlir::Value offset,
                               mlir::Value mean, mlir::Value variance, float eps) {
    auto operand_elem = mlir::cast<mlir::RankedTensorType>(operand.getType()).getElementType();
    TT_FATAL(mlir::cast<mlir::RankedTensorType>(scale.getType()).getElementType() == operand_elem &&
                 mlir::cast<mlir::RankedTensorType>(offset.getType()).getElementType() == operand_elem &&
                 mlir::cast<mlir::RankedTensorType>(mean.getType()).getElementType() == operand_elem &&
                 mlir::cast<mlir::RankedTensorType>(variance.getType()).getElementType() == operand_elem,
             "tt-crank build_bn_inference: all inputs must share element type — callers must promote first");

    auto result_type = mlir::cast<mlir::RankedTensorType>(operand.getType());
    llvm::APFloat eps_ap(as<double>(eps));
    bool loses_info = false;
    eps_ap.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &loses_info);
    return mb
        .create<mlir::tt::ttir::BatchNormInferenceOp>(result_type, operand, scale, offset, mean, variance, eps_ap,
                                                      as<uint32_t>(1))
        .getResult();
}

mlir::Value build_layer_norm(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                             llvm::ArrayRef<std::int64_t> normalized_shape, float eps) {
    auto input_elem = mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
    TT_FATAL((!weight || mlir::cast<mlir::RankedTensorType>(weight.getType()).getElementType() == input_elem) &&
                 (!bias || mlir::cast<mlir::RankedTensorType>(bias.getType()).getElementType() == input_elem),
             "tt-crank build_layer_norm: all inputs must share element type — callers must promote first");

    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape_attr = mb.attrs().getDenseI64ArrayAttr(normalized_shape);
    llvm::APFloat eps_ap(as<double>(eps));
    bool loses_info = false;
    eps_ap.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &loses_info);
    return mb.create<mlir::tt::ttir::LayerNormOp>(result_type, input, weight, bias, shape_attr, eps_ap).getResult();
}

std::tuple<mlir::Value, mlir::Value, mlir::Value>
build_layer_norm_with_stats(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                            llvm::ArrayRef<std::int64_t> normalized_shape, float eps, bool stats_in_f32) {
    auto out = build_layer_norm(mb, input, weight, bias, normalized_shape, eps);

    auto stats_input = stats_in_f32 ? mb.insert_typecast(input, mb.attrs().getF32Type()) : input;
    auto rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
    const llvm::SmallVector<std::int64_t> dims{rank - 1};

    auto mean = build_mean(mb, stats_input, dims, /*keepdim=*/true);
    auto centered = build_sub(mb, stats_input, mean);
    auto variance = build_mean(mb, build_mul(mb, centered, centered), dims, /*keepdim=*/true);
    auto var_elem = mlir::cast<mlir::RankedTensorType>(variance.getType()).getElementType();
    auto rstd = build_rsqrt(mb, build_add(mb, variance, build_scalar(mb, var_elem, as<double>(eps))));
    return {out, mean, rstd};
}

mlir::Value build_add(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs, double alpha) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "tt-crank build_add: lhs and rhs must share element type — callers must promote first");

    // aten::add: lhs + alpha * rhs. Elide the scale when alpha == 1.
    if (alpha != 1.0) {
        rhs = scale_tensor(mb, rhs, alpha);
    }

    // ttir.add broadcasts internally; just give it the broadcasted output shape.
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    auto add = mb.create<mlir::tt::ttir::AddOp>(result_type, lhs, rhs);
    return add.getResult();
}

mlir::Value build_permute(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> permutation) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    llvm::SmallVector<int64_t> out_shape;
    for (auto p : permutation) {
        out_shape.push_back(shape[as<std::size_t>(p)]);
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    auto perm_attr = mb.attrs().getDenseI64ArrayAttr(permutation);
    return mb.create<mlir::tt::ttir::PermuteOp>(result_type, input, perm_attr).getResult();
}

mlir::Value build_max_pool2d(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> kernel_size,
                             llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                             llvm::ArrayRef<int64_t> dilation, bool ceil_mode) {
    auto nchw_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = nchw_type.getShape(); // [N, C, H, W]
    auto elem_type = nchw_type.getElementType();

    // NCHW[N,C,H,W] → NHWC[N,H,W,C]: permutation [0,2,3,1]
    auto nhwc_input = build_permute(mb, input, {0, 2, 3, 1});

    int64_t kH = kernel_size[0], kW = kernel_size[1];
    int64_t sH = stride[0], sW = stride[1];
    int64_t pH = padding[0], pW = padding[1];
    int64_t dH = dilation[0], dW = dilation[1];

    auto compute_out = [ceil_mode](int64_t in_size, int64_t k, int64_t s, int64_t p, int64_t d) -> int64_t {
        int64_t eff = in_size + 2 * p - d * (k - 1) - 1;
        return ceil_mode ? (eff + s - 1) / s + 1 : eff / s + 1;
    };

    int64_t H_out = compute_out(shape[2], kH, sH, pH, dH);
    int64_t W_out = compute_out(shape[3], kW, sW, pW, dW);

    auto nhwc_out_type = mlir::RankedTensorType::get({shape[0], H_out, W_out, shape[1]}, elem_type);
    auto kernel_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(kH), as<int32_t>(kW)});
    auto stride_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(sH), as<int32_t>(sW)});
    auto dilation_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(dH), as<int32_t>(dW)});
    auto padding_attr =
        mb.attrs().getDenseI32ArrayAttr({as<int32_t>(pH), as<int32_t>(pW), as<int32_t>(pH), as<int32_t>(pW)});
    auto ceil_mode_attr = mb.attrs().getBoolAttr(ceil_mode);

    auto nhwc_result = mb.create<mlir::tt::ttir::MaxPool2dOp>(nhwc_out_type, nhwc_input, kernel_attr, stride_attr,
                                                              dilation_attr, padding_attr, ceil_mode_attr)
                           .getResult();

    // NHWC[N,H_out,W_out,C] → NCHW[N,C,H_out,W_out]: permutation [0,3,1,2]
    return build_permute(mb, nhwc_result, {0, 3, 1, 2});
}

mlir::Value build_cos(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::CosOp>(result_type, input).getResult();
}

mlir::Value build_sin(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::SinOp>(result_type, input).getResult();
}

mlir::Value build_neg(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::NegOp>(result_type, input).getResult();
}

mlir::Value build_log(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::LogOp>(result_type, input).getResult();
}

mlir::Value build_exp(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::ExpOp>(result_type, input).getResult();
}

mlir::Value build_log1p(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::Log1pOp>(result_type, input).getResult();
}

mlir::Value build_sqrt(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::SqrtOp>(result_type, input).getResult();
}

mlir::Value build_tanh(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::TanhOp>(result_type, input).getResult();
}

mlir::Value build_reciprocal(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::ReciprocalOp>(result_type, input).getResult();
}

mlir::Value build_cumsum(ModuleBuilder &mb, mlir::Value input, int64_t dim) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t rank = as<int64_t>(result_type.getRank());
    TT_FATAL(dim >= 0 && dim < rank, "build_cumsum: dim {} out of range for rank {}", dim, rank);
    return mb.create<mlir::tt::ttir::CumSumOp>(result_type, input, dim).getResult();
}

mlir::Value build_silu(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::SiluOp>(result_type, input).getResult();
}

mlir::Value build_sigmoid(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::SigmoidOp>(result_type, input).getResult();
}

mlir::Value build_gelu(ModuleBuilder &mb, mlir::Value input) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    return mb.create<mlir::tt::ttir::GeluOp>(result_type, input).getResult();
}

mlir::Value build_div(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "build_div: lhs and rhs must share element type — callers must promote first");
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::DivOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_floor_divide(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    // floor(lhs / rhs), rounding toward -inf. Integers divide in float first
    // (ttir DivOp truncates toward zero, giving the wrong sign) then cast back.
    auto elem = mlir::cast<mlir::RankedTensorType>(lhs.getType()).getElementType();
    if (mlir::isa<mlir::IntegerType>(elem)) {
        auto f32 = mb.attrs().getF32Type();
        mlir::Value q = build_div(mb, mb.insert_typecast(lhs, f32), mb.insert_typecast(rhs, f32));
        auto q_type = mlir::cast<mlir::RankedTensorType>(q.getType());
        mlir::Value floored = mb.create<mlir::tt::ttir::FloorOp>(q_type, q).getResult();
        return mb.insert_typecast(floored, elem);
    }
    mlir::Value q = build_div(mb, lhs, rhs);
    auto q_type = mlir::cast<mlir::RankedTensorType>(q.getType());
    return mb.create<mlir::tt::ttir::FloorOp>(q_type, q).getResult();
}

mlir::Value build_clamp(ModuleBuilder &mb, mlir::Value input, std::optional<double> min_val,
                        std::optional<double> max_val) {
    // clamp_scalar is float-typed; clamp integers in float and cast back. A
    // missing bound becomes ±inf (no-op on that side).
    auto in_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto elem = in_type.getElementType();
    auto f32 = mb.attrs().getF32Type();
    auto min_attr = mlir::FloatAttr::get(f32, min_val.value_or(-std::numeric_limits<double>::infinity()));
    auto max_attr = mlir::FloatAttr::get(f32, max_val.value_or(std::numeric_limits<double>::infinity()));
    if (mlir::isa<mlir::IntegerType>(elem)) {
        mlir::Value in_f = mb.insert_typecast(input, f32);
        auto f_type = mlir::cast<mlir::RankedTensorType>(in_f.getType());
        mlir::Value clamped = mb.create<mlir::tt::ttir::ClampScalarOp>(f_type, in_f, min_attr, max_attr).getResult();
        return mb.insert_typecast(clamped, elem);
    }
    return mb.create<mlir::tt::ttir::ClampScalarOp>(in_type, input, min_attr, max_attr).getResult();
}

mlir::Value build_pow(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "build_pow: lhs and rhs must share element type — callers must promote first");
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::PowOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_softmax(ModuleBuilder &mb, mlir::Value input, int64_t dim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t rank = as<int64_t>(input_type.getRank());
    int64_t norm_dim = (dim + rank) % rank;
    return mb.create<mlir::tt::ttir::SoftmaxOp>(input_type, input, as<int32_t>(norm_dim), true).getResult();
}

mlir::Value build_argmax(ModuleBuilder &mb, mlir::Value input, std::optional<int64_t> dim, bool keepdim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    llvm::SmallVector<int64_t> out_shape;
    mlir::ArrayAttr dim_arg_attr;
    if (dim.has_value()) {
        int64_t norm_dim = (dim.value() + rank) % rank;
        for (int64_t i = 0; i < rank; ++i) {
            if (i != norm_dim) {
                out_shape.push_back(shape[as<std::size_t>(i)]);
            } else if (keepdim) {
                out_shape.push_back(1);
            }
        }
        dim_arg_attr = mb.attrs().getI32ArrayAttr({as<int32_t>(norm_dim)});
    } else {
        if (keepdim) {
            out_shape.assign(as<std::size_t>(rank), 1LL);
        }
        dim_arg_attr = nullptr;
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI32Type());
    auto keep_dim_attr = mb.attrs().getBoolAttr(keepdim);
    auto argmax = mb.create<mlir::tt::ttir::ArgMaxOp>(result_type, input, keep_dim_attr, dim_arg_attr).getResult();
    return mb.insert_typecast(argmax, mb.attrs().getI64Type());
}

mlir::Value build_unsqueeze(ModuleBuilder &mb, mlir::Value input, int64_t dim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    if (dim < 0) {
        dim += rank + 1;
    }
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i == dim) {
            out_shape.push_back(1);
        }
        out_shape.push_back(shape[as<std::size_t>(i)]);
    }
    if (dim == rank) {
        out_shape.push_back(1);
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb.create<mlir::tt::ttir::UnsqueezeOp>(result_type, input, as<int32_t>(dim)).getResult();
}

mlir::Value build_squeeze(ModuleBuilder &mb, mlir::Value input, int64_t dim) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    TT_FATAL(dim >= 0 && dim < rank, "build_squeeze: dim {} out of range for rank {}", dim, rank);
    TT_FATAL(shape[as<std::size_t>(dim)] == 1, "build_squeeze: dim {} has size {}, expected 1", dim,
             shape[as<std::size_t>(dim)]);
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) {
            out_shape.push_back(shape[as<std::size_t>(i)]);
        }
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb.create<mlir::tt::ttir::SqueezeOp>(result_type, input, as<int32_t>(dim)).getResult();
}

mlir::Value build_transpose(ModuleBuilder &mb, mlir::Value input, int64_t dim0, int64_t dim1) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    TT_FATAL(dim0 >= 0 && dim0 < rank, "build_transpose: dim0 {} out of range for rank {}", dim0, rank);
    TT_FATAL(dim1 >= 0 && dim1 < rank, "build_transpose: dim1 {} out of range for rank {}", dim1, rank);
    llvm::SmallVector<int64_t> out_shape(shape.begin(), shape.end());
    std::swap(out_shape[as<std::size_t>(dim0)], out_shape[as<std::size_t>(dim1)]);
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb.create<mlir::tt::ttir::TransposeOp>(result_type, input, as<int32_t>(dim0), as<int32_t>(dim1)).getResult();
}

mlir::Value build_broadcast(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> target_shape) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto input_shape = input_type.getShape();
    int64_t input_rank = as<int64_t>(input_shape.size());
    int64_t target_rank = as<int64_t>(target_shape.size());
    // Prepend implicit size-1 dims via reshape if target rank is higher.
    if (target_rank > input_rank) {
        llvm::SmallVector<int64_t> padded(as<std::size_t>(target_rank - input_rank), 1LL);
        padded.append(input_shape.begin(), input_shape.end());
        input = build_reshape(mb, input, padded);
        input_shape = mlir::cast<mlir::RankedTensorType>(input.getType()).getShape();
    }
    llvm::SmallVector<int64_t> broadcast_dims;
    for (int64_t i = 0; i < target_rank; ++i) {
        int64_t in_size = input_shape[as<std::size_t>(i)];
        int64_t out_size = target_shape[as<std::size_t>(i)];
        TT_FATAL(in_size == 1 || in_size == out_size,
                 "build_broadcast: incompatible sizes at dim {}: input={}, target={}", i, in_size, out_size);
        broadcast_dims.push_back(in_size == 1 ? out_size : 1LL);
    }
    auto result_type = mlir::RankedTensorType::get(target_shape, input_type.getElementType());
    auto dims_attr = mb.attrs().getDenseI64ArrayAttr(broadcast_dims);
    return mb.create<mlir::tt::ttir::BroadcastOp>(result_type, input, dims_attr).getResult();
}

mlir::Value build_cat(ModuleBuilder &mb, llvm::ArrayRef<mlir::Value> inputs, int64_t dim) {
    TT_FATAL(!inputs.empty(), "build_cat: inputs must be non-empty");
    auto first_type = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
    auto elem_type = first_type.getElementType();
    int64_t rank = as<int64_t>(first_type.getRank());
    int64_t norm_dim = (dim + rank) % rank;
    llvm::SmallVector<int64_t> out_shape(first_type.getShape().begin(), first_type.getShape().end());
    out_shape[as<std::size_t>(norm_dim)] = 0;
    for (auto v : inputs) {
        out_shape[as<std::size_t>(norm_dim)] +=
            mlir::cast<mlir::RankedTensorType>(v.getType()).getShape()[as<std::size_t>(norm_dim)];
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, elem_type);
    return mb.create<mlir::tt::ttir::ConcatOp>(result_type, inputs, as<int32_t>(norm_dim)).getResult();
}

mlir::Value build_slice(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> begins,
                        llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t rank = as<int64_t>(input_type.getRank());
    TT_FATAL(as<int64_t>(begins.size()) == rank && as<int64_t>(ends.size()) == rank && as<int64_t>(step.size()) == rank,
             "build_slice: begins/ends/step must all have length == rank ({})", rank);
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        int64_t size = (ends[as<std::size_t>(i)] - begins[as<std::size_t>(i)] + step[as<std::size_t>(i)] - 1) /
                       step[as<std::size_t>(i)];
        out_shape.push_back(std::max<int64_t>(0, size));
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    llvm::SmallVector<int32_t> begins_i32(begins.begin(), begins.end());
    llvm::SmallVector<int32_t> ends_i32(ends.begin(), ends.end());
    llvm::SmallVector<int32_t> step_i32(step.begin(), step.end());
    return mb
        .create<mlir::tt::ttir::SliceStaticOp>(result_type, input, mb.attrs().getI32ArrayAttr(begins_i32),
                                               mb.attrs().getI32ArrayAttr(ends_i32),
                                               mb.attrs().getI32ArrayAttr(step_i32))
        .getResult();
}

mlir::Value build_pad(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<std::int64_t> low,
                      llvm::ArrayRef<std::int64_t> high, double value) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    TT_FATAL(as<int64_t>(low.size()) == rank && as<int64_t>(high.size()) == rank,
             "build_pad: low/high must have one entry per dim (rank {})", rank);

    // A negative amount crops instead of pads, which ttir.pad can't express —
    // it only grows. Crop with a slice first, then pad whatever is left.
    bool crops = false;
    llvm::SmallVector<int64_t> begins, ends, steps;
    for (int64_t i = 0; i < rank; ++i) {
        int64_t lo = low[as<std::size_t>(i)];
        int64_t hi = high[as<std::size_t>(i)];
        int64_t dim = shape[as<std::size_t>(i)];
        begins.push_back(lo < 0 ? -lo : 0);
        ends.push_back(hi < 0 ? dim + hi : dim);
        steps.push_back(1);
        crops = crops || lo < 0 || hi < 0;
        TT_FATAL(ends.back() > begins.back(), "build_pad: crop of dim {} (size {}) by ({}, {}) leaves nothing", i, dim,
                 lo, hi);
    }
    if (crops) {
        input = build_slice(mb, input, begins, ends, steps);
    }

    bool pads = false;
    llvm::SmallVector<int32_t> padding;
    for (int64_t i = 0; i < rank; ++i) {
        padding.push_back(as<int32_t>(std::max<int64_t>(low[as<std::size_t>(i)], 0)));
        padding.push_back(as<int32_t>(std::max<int64_t>(high[as<std::size_t>(i)], 0)));
        pads = pads || padding[padding.size() - 2] > 0 || padding.back() > 0;
    }
    if (!pads) {
        return input;
    }

    auto cropped = mlir::cast<mlir::RankedTensorType>(input.getType()).getShape();
    llvm::SmallVector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        out_shape.push_back(cropped[as<std::size_t>(i)] + padding[as<std::size_t>(2 * i)] +
                            padding[as<std::size_t>(2 * i + 1)]);
    }
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());
    return mb
        .create<mlir::tt::ttir::PadOp>(result_type, input, mb.attrs().getDenseI32ArrayAttr(padding),
                                       mb.attrs().getF32FloatAttr(as<float>(value)))
        .getResult();
}

mlir::Value build_arange(ModuleBuilder &mb, int64_t start, int64_t end, int64_t step, mlir::Type dtype) {
    TT_FATAL(step != 0, "build_arange: step must be non-zero");
    int64_t n = std::max<int64_t>(0, (end - start + step - 1) / step);
    auto result_type = mlir::RankedTensorType::get({n}, dtype);
    return mb.create<mlir::tt::ttir::ArangeOp>(result_type, start, end, step, as<int64_t>(0)).getResult();
}

mlir::Value build_embedding(ModuleBuilder &mb, mlir::Value indices, mlir::Value weight) {
    auto indices_type = mlir::cast<mlir::RankedTensorType>(indices.getType());
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    llvm::SmallVector<int64_t> out_shape(indices_type.getShape().begin(), indices_type.getShape().end());
    out_shape.push_back(weight_type.getShape().back());
    auto result_type = mlir::RankedTensorType::get(out_shape, weight_type.getElementType());
    return mb.create<mlir::tt::ttir::EmbeddingOp>(result_type, indices, weight).getResult();
}

mlir::Value build_embedding_backward(ModuleBuilder &mb, mlir::Value indices, mlir::Value in_gradient,
                                     int64_t num_weights, int64_t padding_idx) {
    auto grad_type = mlir::cast<mlir::RankedTensorType>(in_gradient.getType());
    auto indices_type = mlir::cast<mlir::RankedTensorType>(indices.getType());

    TT_FATAL(indices_type.getRank() == 1 || indices_type.getRank() == 2,
             "build_embedding_backward: indices must be 1D or 2D ([batch, seq]), got rank {}", indices_type.getRank());
    TT_FATAL(mlir::isa<mlir::IntegerType>(indices_type.getElementType()),
             "build_embedding_backward: indices must be integer-typed — callers must not promote them");
    TT_FATAL(grad_type.getRank() == indices_type.getRank() + 1,
             "build_embedding_backward: gradient rank must be one more than the indices rank, got {} and {}",
             grad_type.getRank(), indices_type.getRank());
    auto grad_shape = grad_type.getShape();
    auto indices_shape = indices_type.getShape();
    for (std::size_t dim = 0; dim < indices_shape.size(); ++dim) {
        TT_FATAL(grad_shape[dim] == indices_shape[dim],
                 "build_embedding_backward: gradient dim {} is {}, expected {} to match the indices", dim,
                 grad_shape[dim], indices_shape[dim]);
    }
    TT_FATAL(num_weights > 0, "build_embedding_backward: num_weights must be positive, got {}", num_weights);
    // padding_idx == -1 means no padding row.
    TT_FATAL(padding_idx >= -1 && padding_idx < num_weights,
             "build_embedding_backward: padding_idx {} is out of range for {} rows", padding_idx, num_weights);
    auto element_type = grad_type.getElementType();
    int64_t embedding_dim = grad_shape.back();

    // ttnn.embedding_bw leaves rows unwritten when the row count is not a whole number of
    // tiles, so a vocabulary like 100 comes back with rows 96-99 holding old buffer contents
    // (https://github.com/tenstorrent/tt-mlir/issues/9220). Scatter into a table rounded up to
    // a tile and slice the extra rows back off: every index is < num_weights, so the rows the
    // slice drops only ever hold zeros.
    // TODO: drop the round-up and the slice once tt-mlir#9220 is fixed — for a GPT-2-sized
    // table they are a whole-table copy on every backward step.
    constexpr int64_t tile_height = 32;
    int64_t padded_weights = (num_weights + tile_height - 1) / tile_height * tile_height;
    llvm::SmallVector<int64_t, 2> table_shape{padded_weights, embedding_dim};

    // The op reads the weight's shape for the row count, never its data, and aten hands us
    // `num_weights` instead of the table. Zeros of the right shape satisfy the operand.
    auto weight = build_zeros(mb, table_shape, element_type);
    auto result_type = mlir::RankedTensorType::get(table_shape, element_type);

    // aten holds the padding row out of the update. Zero the gradient of the padded tokens so
    // they contribute nothing, leaving that row at zero. The [batch, seq, 1] mask broadcasts
    // across the embedding dimension, so the pass is over the gradient, not the whole table.
    mlir::Value gradient = in_gradient;
    if (padding_idx >= 0) {
        auto padding_token = build_scalar(mb, indices_type.getElementType(), as<double>(padding_idx));
        auto keep = build_unsqueeze(mb, build_ne(mb, indices, padding_token), -1);
        auto zero = build_scalar(mb, element_type, 0.0);
        gradient = build_where(mb, keep, in_gradient, zero);
    }
    mlir::Value result =
        mb.create<mlir::tt::ttir::EmbeddingBackwardOp>(result_type, indices, weight, gradient).getResult();
    if (padded_weights != num_weights) {
        result = build_slice(mb, result, {0, 0}, {num_weights, embedding_dim}, {1, 1});
    }
    return result;
}

mlir::Value build_gather(ModuleBuilder &mb, mlir::Value input, mlir::Value index, int64_t dim) {
    // torch.gather semantics (ttir.gather): `index` has the same rank as
    // `input`; the result has `index`'s shape and `input`'s element type.
    auto in_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto idx_type = mlir::cast<mlir::RankedTensorType>(index.getType());
    auto result_type = mlir::RankedTensorType::get(idx_type.getShape(), in_type.getElementType());
    return mb
        .create<mlir::tt::ttir::GatherOp>(result_type, input, index, mb.attrs().getI32IntegerAttr(as<int32_t>(dim)))
        .getResult();
}

mlir::Value build_tril(ModuleBuilder &mb, mlir::Value input, int64_t diagonal) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = input_type.getShape();
    int64_t rank = as<int64_t>(shape.size());
    TT_FATAL(rank >= 2, "build_tril: input must be at least 2D, got rank {}", rank);
    int64_t N = shape[as<std::size_t>(rank - 2)];
    int64_t M = shape[as<std::size_t>(rank - 1)];

    auto i32_type = mb.attrs().getI32Type();
    // Row indices: arange [N], reshaped to [N, 1] for column-wise broadcasting.
    auto rows = build_reshape(mb, build_arange(mb, 0, N, 1, i32_type), {N, 1});
    // Column indices: arange [M], reshaped to [1, M] for row-wise broadcasting.
    auto cols = build_reshape(mb, build_arange(mb, 0, M, 1, i32_type), {1, M});
    // threshold[i] = i + diagonal — shape [N, 1], broadcasts against cols [1, M].
    auto diag_cst = build_scalar(mb, i32_type, as<double>(diagonal));
    auto threshold_type = mlir::RankedTensorType::get({N, 1}, i32_type);
    auto threshold = mb.create<mlir::tt::ttir::AddOp>(threshold_type, rows, diag_cst).getResult();
    // mask[i,j] = (j <= i + diagonal): True for lower-triangle positions.
    auto bool_2d_type = mlir::RankedTensorType::get({N, M}, mb.attrs().getI1Type());
    auto mask = mb.create<mlir::tt::ttir::LessEqualOp>(bool_2d_type, cols, threshold).getResult();
    // Apply mask: keep original values where True, zero elsewhere.
    // mask [N, M] broadcasts against input [..., N, M] inside WhereOp.
    auto zero = build_scalar(mb, input_type.getElementType(), 0.0);
    return mb.create<mlir::tt::ttir::WhereOp>(input_type, mask, input, zero).getResult();
}

mlir::Value build_le(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::LessEqualOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_gt(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::GreaterThanOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_ge(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::GreaterEqualOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_lt(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::LessThanOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_eq(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::EqualOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_ne(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::NotEqualOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_bitwise_and(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto elem_type = lhs_type.getElementType();
    TT_FATAL(elem_type == rhs_type.getElementType(),
             "build_bitwise_and: lhs and rhs must share element type — callers must promote first");
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, elem_type);
    // On Bool operands, `bitwise_and` is exactly logical AND, and the FPU only
    // supports the logical kernel — ttnn.bitwise_and rejects i1. Integer bitwise
    // AND uses the genuine BitwiseAndOp.
    if (elem_type.isInteger(1)) {
        return mb.create<mlir::tt::ttir::LogicalAndOp>(result_type, lhs, rhs).getResult();
    }
    return mb.create<mlir::tt::ttir::BitwiseAndOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_bitwise_or(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto elem_type = lhs_type.getElementType();
    TT_FATAL(elem_type == rhs_type.getElementType(),
             "build_bitwise_or: lhs and rhs must share element type — callers must promote first");
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, elem_type);
    // On Bool operands this is logical OR (ttnn.bitwise_or rejects i1); integers
    // use the genuine BitwiseOrOp.
    if (elem_type.isInteger(1)) {
        return mb.create<mlir::tt::ttir::LogicalOrOp>(result_type, lhs, rhs).getResult();
    }
    return mb.create<mlir::tt::ttir::BitwiseOrOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_bitwise_not(ModuleBuilder &mb, mlir::Value input) {
    // Like build_bitwise_and: on Bool operands this is logical NOT (ttnn rejects
    // bitwise on i1); integers use the genuine BitwiseNotOp.
    auto in_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    if (in_type.getElementType().isInteger(1)) {
        return mb.create<mlir::tt::ttir::LogicalNotOp>(in_type, input).getResult();
    }
    return mb.create<mlir::tt::ttir::BitwiseNotOp>(in_type, input).getResult();
}

mlir::Value build_logical_and(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::LogicalAndOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_logical_or(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    auto out_shape = broadcast_shape(lhs_type.getShape(), rhs_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, mb.attrs().getI1Type());
    return mb.create<mlir::tt::ttir::LogicalOrOp>(result_type, lhs, rhs).getResult();
}

mlir::Value build_logical_not(ModuleBuilder &mb, mlir::Value input) {
    auto elem = mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
    if (elem.isInteger(1)) {
        return mb.create<mlir::tt::ttir::LogicalNotOp>(input.getType(), input).getResult();
    }
    return build_eq(mb, input, build_scalar(mb, elem, 0.0));
}

mlir::Value build_index_copy(ModuleBuilder &mb, mlir::Value input, int64_t dim, mlir::Value index, mlir::Value source) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto source_type = mlir::cast<mlir::RankedTensorType>(source.getType());
    auto index_type = mlir::cast<mlir::RankedTensorType>(index.getType());
    TT_FATAL(index_type.getRank() == 1, "tt-crank build_index_copy: index must be 1D");
    int64_t rank = as<int64_t>(input_type.getShape().size());
    if (dim < 0) {
        dim += rank;
    }

    // A rank-4 index_copy along dim 2 is a KV-cache write: HF's StaticCache
    // updates its [batch, kv_heads, seq, head_dim] cache in place via
    // `keys.index_copy_(2, cache_position, key_states)`. Route it to the dedicated
    // cache ops. The structural signature (rank-4, dim 2) has the same fidelity as
    // tt-mlir's StableHLO CacheFillUpdatePattern: aten's index_copy contract
    // already guarantees source matches the cache on every non-dim dimension, so
    // rank and dim are the only real discriminators.
    if (rank == 4 && dim == 2) {
        return build_kv_cache_write(mb, input, index, source);
    }

    // Reshape 1D index [n] to [1, ..., n, ..., 1] (size 1 except at dim)
    llvm::SmallVector<int64_t> reshaped_shape(as<std::size_t>(rank), 1LL);
    reshaped_shape[as<std::size_t>(dim)] = index_type.getShape()[0];
    mlir::Value expanded_index = build_reshape(mb, index, reshaped_shape);

    // Broadcast to source_shape so index and source have identical shapes
    llvm::SmallVector<int64_t> target_shape(source_type.getShape().begin(), source_type.getShape().end());
    expanded_index = build_broadcast(mb, expanded_index, target_shape);

    // ScatterOp needs i32 indices
    auto i32_type = mb.attrs().getI32Type();
    if (index_type.getElementType() != i32_type) {
        expanded_index = mb.insert_typecast(expanded_index, i32_type);
    }

    auto reduce_attr =
        mlir::tt::ttcore::ReduceTypeAttr::get(mb.attrs().getContext(), mlir::tt::ttcore::ReduceType::Invalid);
    return mb
        .create<mlir::tt::ttir::ScatterOp>(input_type, input, expanded_index, source,
                                           mb.attrs().getI32IntegerAttr(as<int32_t>(dim)), reduce_attr)
        .getResult();
}

mlir::Value build_where(ModuleBuilder &mb, mlir::Value condition, mlir::Value true_val, mlir::Value false_val) {
    auto cond_type = mlir::cast<mlir::RankedTensorType>(condition.getType());
    auto true_type = mlir::cast<mlir::RankedTensorType>(true_val.getType());
    auto false_type = mlir::cast<mlir::RankedTensorType>(false_val.getType());
    // aten.where type-promotes its two branches; promote to a common element
    // type here (e.g. masked_fill lowers to where with a fill of a different
    // dtype than the input).
    auto elem = promote_element_types(true_type.getElementType(), false_type.getElementType());
    true_val = mb.insert_typecast(true_val, elem);
    false_val = mb.insert_typecast(false_val, elem);
    auto shape01 = broadcast_shape(cond_type.getShape(), true_type.getShape());
    auto out_shape = broadcast_shape(shape01, false_type.getShape());
    auto result_type = mlir::RankedTensorType::get(out_shape, elem);
    return mb.create<mlir::tt::ttir::WhereOp>(result_type, condition, true_val, false_val).getResult();
}

mlir::Value build_isneginf(ModuleBuilder &mb, mlir::Value input) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto bool_type = mlir::RankedTensorType::get(input_type.getShape(), mb.attrs().getI1Type());
    // isinf = logical_not(isfinite)
    auto isfinite = mb.create<mlir::tt::ttir::IsFiniteOp>(bool_type, input).getResult();
    auto isinf = mb.create<mlir::tt::ttir::LogicalNotOp>(bool_type, isfinite).getResult();
    // x < 0 — scalar zero broadcasts against input shape
    auto zero = build_scalar(mb, input_type.getElementType(), 0.0);
    auto is_neg = mb.create<mlir::tt::ttir::LessThanOp>(bool_type, input, zero).getResult();
    return mb.create<mlir::tt::ttir::LogicalAndOp>(bool_type, isinf, is_neg).getResult();
}

mlir::Value build_all(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> dims, bool keepdim) {
    return build_reduce<mlir::tt::ttir::ReduceAndOp>(mb, coerce_to_bool(mb, input), dims, keepdim);
}

mlir::Value build_any(ModuleBuilder &mb, mlir::Value input, llvm::ArrayRef<int64_t> dims, bool keepdim) {
    return build_reduce<mlir::tt::ttir::ReduceOrOp>(mb, coerce_to_bool(mb, input), dims, keepdim);
}

mlir::Value scale_tensor(ModuleBuilder &mb, mlir::Value tensor, double value) {
    auto tensor_type = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    auto scalar = build_scalar(mb, tensor_type.getElementType(), value);
    auto mul = mb.create<mlir::tt::ttir::MultiplyOp>(tensor_type, tensor, scalar);
    return mul.getResult();
}

mlir::Value build_t(ModuleBuilder &mb, mlir::Value input) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    TT_FATAL(input_type.getRank() == 2, "tt-crank build_t: input must be 2D");
    auto shape = input_type.getShape();
    auto result_type = mlir::RankedTensorType::get({shape[1], shape[0]}, input_type.getElementType());
    return mb.create<mlir::tt::ttir::TransposeOp>(result_type, input, 0, 1).getResult();
}

mlir::Value build_mm(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "tt-crank build_mm: lhs and rhs must share element type — callers must promote first");
    auto result_type =
        mlir::RankedTensorType::get({lhs_type.getShape()[0], rhs_type.getShape()[1]}, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::MatmulOp>(result_type, lhs, rhs, false, false).getResult();
}

mlir::Value build_addmm(ModuleBuilder &mb, mlir::Value bias, mlir::Value mat1, mlir::Value mat2, double beta,
                        double alpha) {
    auto mat1_type = mlir::cast<mlir::RankedTensorType>(mat1.getType());
    auto mat2_type = mlir::cast<mlir::RankedTensorType>(mat2.getType());
    TT_FATAL(mat1_type.getElementType() == mat2_type.getElementType() &&
                 mat1_type.getElementType() == mlir::cast<mlir::RankedTensorType>(bias.getType()).getElementType(),
             "tt-crank build_addmm: all inputs must share element type — callers must promote first");

    auto result_type =
        mlir::RankedTensorType::get({mat1_type.getShape()[0], mat2_type.getShape()[1]}, mat1_type.getElementType());

    if (beta == 1.0 && alpha == 1.0) {
        return mb.create<mlir::tt::ttir::LinearOp>(result_type, mat1, mat2, bias, false, false).getResult();
    }

    mlir::Value result = build_mm(mb, mat1, mat2);
    if (alpha != 1.0) {
        result = scale_tensor(mb, result, alpha);
    }
    if (beta != 0.0) {
        mlir::Value scaled_bias = beta == 1.0 ? bias : scale_tensor(mb, bias, beta);
        result = mb.create<mlir::tt::ttir::AddOp>(result_type, result, scaled_bias).getResult();
    }
    return result;
}

std::tuple<std::optional<mlir::Value>, std::optional<mlir::Value>, std::optional<mlir::Value>>
build_linear_backward(ModuleBuilder &mb, mlir::Value self, mlir::Value grad, mlir::Value weight, bool need_self,
                      bool need_weight, bool need_bias) {
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    TT_FATAL(weight_type.getRank() == 2, "tt-crank build_linear_backward: weight must be 2D");
    int64_t out_features = weight_type.getShape()[0];
    int64_t in_features = weight_type.getShape()[1];

    auto grad_type = mlir::cast<mlir::RankedTensorType>(grad.getType());
    auto grad_shape = grad_type.getShape();
    int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < grad_shape.size(); ++i) {
        rows *= grad_shape[i];
    }

    // Collapse leading dims so both gradient matmuls are plain 2-D.
    mlir::Value grad_2d = build_reshape(mb, grad, {rows, out_features});

    std::optional<mlir::Value> grad_self;
    std::optional<mlir::Value> grad_weight;
    std::optional<mlir::Value> grad_bias;

    if (need_self) {
        // [rows, out] @ [out, in] -> [rows, in], then restore self's original shape.
        mlir::Value gs = build_mm(mb, grad_2d, weight);
        auto self_shape = mlir::cast<mlir::RankedTensorType>(self.getType()).getShape();
        grad_self = build_reshape(mb, gs, llvm::to_vector(self_shape));
    }
    if (need_weight) {
        mlir::Value self_2d = build_reshape(mb, self, {rows, in_features});
        grad_weight = build_mm(mb, build_transpose(mb, grad_2d, 0, 1), self_2d);
    }
    if (need_bias) {
        grad_bias = build_sum(mb, grad_2d, {0}, /*keepdim=*/false);
    }
    return {grad_self, grad_weight, grad_bias};
}

mlir::Value build_linear(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    TT_FATAL(input_type.getElementType() == weight_type.getElementType(),
             "tt-crank build_linear: input and weight must share element type");
    TT_FATAL(weight_type.getRank() == 2, "tt-crank build_linear: weight must be 2D");

    // weight is [out_features, in_features]; transpose_b makes the contraction use in_features.
    auto input_shape = input_type.getShape();
    llvm::SmallVector<int64_t> out_shape = llvm::to_vector(input_shape.drop_back());
    out_shape.push_back(weight_type.getShape()[0]);
    auto result_type = mlir::RankedTensorType::get(out_shape, input_type.getElementType());

    return mb
        .create<mlir::tt::ttir::LinearOp>(result_type, input, weight, bias, /*transpose_a=*/false,
                                          /*transpose_b=*/true)
        .getResult();
}

mlir::Value build_matmul(ModuleBuilder &mb, mlir::Value lhs, mlir::Value rhs) {
    auto lhs_type = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto rhs_type = mlir::cast<mlir::RankedTensorType>(rhs.getType());
    TT_FATAL(lhs_type.getElementType() == rhs_type.getElementType(),
             "tt-crank build_matmul: lhs and rhs must share element type — callers must promote first");
    auto lhs_shape = lhs_type.getShape();
    auto rhs_shape = rhs_type.getShape();
    int64_t lhs_rank = as<int64_t>(lhs_shape.size());
    int64_t rhs_rank = as<int64_t>(rhs_shape.size());
    TT_FATAL(lhs_rank >= 2 && rhs_rank >= 2, "tt-crank build_matmul: inputs must be at least 2D");

    // PERF: When RHS is a plain 2-D weight and LHS carries leading/batch dims, flatten all
    // of LHS's leading dims into a single M so a batch of small matmuls becomes one
    // dense [M, K] x [K, N] matmul.
    //
    // The two reshapes cost something, so flatten only when the matmul is large
    // enough (K*N past an empirical break-even) that the reshape overhead is
    // negligible next to the matmul work. The A/B sweep showed matmul size,
    // not the rows per matmul, is what separates perf gains from losses.
    constexpr int64_t k_flatten_min_kn = 1 << 20;
    int64_t k = lhs_shape[as<std::size_t>(lhs_rank - 1)];
    int64_t n = rhs_shape[as<std::size_t>(rhs_rank - 1)];
    bool flatten_pays_off = k * n >= k_flatten_min_kn;
    if (rhs_rank == 2 && lhs_rank > 2 && flatten_pays_off) {
        int64_t m = 1;
        for (int64_t i = 0; i < lhs_rank - 1; ++i) {
            m *= lhs_shape[as<std::size_t>(i)];
        }
        mlir::Value lhs_2d = build_reshape(mb, lhs, {m, k});
        auto result_2d_type = mlir::RankedTensorType::get({m, n}, lhs_type.getElementType());
        mlir::Value result_2d =
            mb.create<mlir::tt::ttir::MatmulOp>(result_2d_type, lhs_2d, rhs, false, false).getResult();
        llvm::SmallVector<int64_t> out_shape = llvm::to_vector(lhs_shape.drop_back());
        out_shape.push_back(n);
        return build_reshape(mb, result_2d, out_shape);
    }

    // ttir.matmul broadcasts the operands' batch dims numpy-style (either side
    // may dominate), and its verifier checks the result type against that — so
    // the declared batch dims must come from both inputs, not just lhs.
    llvm::SmallVector<int64_t> out_shape = broadcast_shape(lhs_shape.drop_back(2), rhs_shape.drop_back(2));
    out_shape.push_back(lhs_shape[as<std::size_t>(lhs_rank - 2)]);
    out_shape.push_back(rhs_shape[as<std::size_t>(rhs_rank - 1)]);
    auto result_type = mlir::RankedTensorType::get(out_shape, lhs_type.getElementType());
    return mb.create<mlir::tt::ttir::MatmulOp>(result_type, lhs, rhs, false, false).getResult();
}

mlir::Value build_sdpa(ModuleBuilder &mb, mlir::Value query, mlir::Value key, mlir::Value value, bool is_causal,
                       std::optional<float> scale, mlir::Value attn_mask) {
    auto result_type = mlir::cast<mlir::RankedTensorType>(query.getType());
    mlir::FloatAttr scale_attr = scale.has_value()
                                     ? mlir::FloatAttr::get(mb.attrs().getF32Type(), as<double>(scale.value()))
                                     : mlir::FloatAttr{};
    auto is_causal_attr = mb.attrs().getBoolAttr(is_causal);
    return mb
        .create<mlir::tt::ttir::ScaledDotProductAttentionOp>(result_type, query, key, value, attn_mask, is_causal_attr,
                                                             scale_attr, mlir::IntegerAttr{}, mlir::Value{})
        .getResult();
}

mlir::Value build_sum_to(ModuleBuilder &mb, mlir::Value t, llvm::ArrayRef<int64_t> target) {
    auto t_shape = mlir::cast<mlir::RankedTensorType>(t.getType()).getShape();
    if (t_shape == target) {
        return t;
    }
    int64_t leading = as<int64_t>(t_shape.size()) - as<int64_t>(target.size());
    TT_FATAL(leading >= 0, "tt-crank build_sum_to: target rank exceeds input rank");
    if (leading > 0) {
        llvm::SmallVector<int64_t> dims;
        for (int64_t i = 0; i < leading; ++i) {
            dims.push_back(i);
        }
        t = build_sum(mb, t, dims, /*keepdim=*/false);
    }
    auto cur = mlir::cast<mlir::RankedTensorType>(t.getType()).getShape();
    llvm::SmallVector<int64_t> keep_dims;
    for (std::size_t i = 0; i < target.size(); ++i) {
        if (target[i] == 1 && cur[i] != 1) {
            keep_dims.push_back(as<int64_t>(i));
        }
    }
    if (!keep_dims.empty()) {
        t = build_sum(mb, t, keep_dims, /*keepdim=*/true);
    }
    // Leading-dim + broadcast-dim reductions above should land exactly on target;
    // assert the invariant so a shape-inference bug surfaces here, not downstream.
    auto final_shape = mlir::cast<mlir::RankedTensorType>(t.getType()).getShape();
    TT_FATAL(final_shape == target, "tt-crank build_sum_to: reduction did not reach target shape");
    return t;
}

std::pair<std::optional<mlir::Value>, std::optional<mlir::Value>>
build_matmul_backward(ModuleBuilder &mb, mlir::Value grad, mlir::Value self, mlir::Value other, bool need_self,
                      bool need_other) {
    auto rank_of = [](mlir::Value v) { return mlir::cast<mlir::RankedTensorType>(v.getType()).getRank(); };
    auto shape_of = [](mlir::Value v) { return mlir::cast<mlir::RankedTensorType>(v.getType()).getShape(); };
    // The dim helpers normalise a possibly-negative dim against the (post-op) rank.
    auto unsqueeze_at = [&](mlir::Value v, int64_t dim) {
        int64_t r = rank_of(v);
        return build_unsqueeze(mb, v, (dim + r + 1) % (r + 1));
    };
    auto squeeze_at = [&](mlir::Value v, int64_t dim) {
        int64_t r = rank_of(v);
        return build_squeeze(mb, v, (dim + r) % r);
    };
    auto transpose_last2 = [&](mlir::Value v) {
        int64_t r = rank_of(v);
        return build_transpose(mb, v, r - 2, r - 1);
    };

    bool self_1d = rank_of(self) == 1;
    bool other_1d = rank_of(other) == 1;
    mlir::Value a = self_1d ? unsqueeze_at(self, 0) : self;
    mlir::Value b = other_1d ? unsqueeze_at(other, -1) : other;
    mlir::Value g = grad;
    if (other_1d) {
        g = unsqueeze_at(g, -1);
    }
    if (self_1d) {
        g = unsqueeze_at(g, -2);
    }

    std::optional<mlir::Value> grad_self;
    std::optional<mlir::Value> grad_other;
    if (need_self) {
        mlir::Value ga = build_sum_to(mb, build_matmul(mb, g, transpose_last2(b)), shape_of(a));
        grad_self = self_1d ? squeeze_at(ga, 0) : ga;
    }
    if (need_other) {
        mlir::Value gb = build_sum_to(mb, build_matmul(mb, transpose_last2(a), g), shape_of(b));
        grad_other = other_1d ? squeeze_at(gb, -1) : gb;
    }
    return {grad_self, grad_other};
}

mlir::Value build_conv1d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                         llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                         llvm::ArrayRef<int64_t> dilation, int64_t groups) {
    auto ncw_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = ncw_type.getShape();                                              // NCW: [N, C_in, L_in]
    auto wshape = mlir::cast<mlir::RankedTensorType>(weight.getType()).getShape(); // OIK: [C_out, C_in/groups, K]
    auto elem_type = ncw_type.getElementType();

    // NCW[N,C,L] → NLC[N,L,C]
    auto nlc_input = build_permute(mb, input, {0, 2, 1});

    // NOLINTBEGIN
    int64_t kW = wshape[2];
    int64_t pW = padding[0], dW = dilation[0], sW = stride[0];
    int64_t L_out = (shape[2] + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    // NOLINTEND

    // The verifier wants a rank-3 bias with the channel at channel_dim=2.
    mlir::Value bias_3d;
    if (bias) {
        bias_3d = build_reshape(mb, bias, {1, 1, wshape[0]});
    }

    auto stride_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(sW)});
    auto padding_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(pW), as<int32_t>(pW)});
    auto dilation_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(dW)});

    auto nlc_out_type = mlir::RankedTensorType::get({shape[0], L_out, wshape[0]}, elem_type);
    auto nlc_result = mb.create<mlir::tt::ttir::Conv1dOp>(nlc_out_type, nlc_input, weight, bias_3d, stride_attr,
                                                          padding_attr, dilation_attr, as<uint32_t>(groups),
                                                          /*batch_dim=*/uint64_t{0}, /*length_dim=*/uint64_t{1},
                                                          /*channel_dim=*/uint64_t{2})
                          .getResult();

    // NLC[N,L_out,C_out] → NCW[N,C_out,L_out]
    return build_permute(mb, nlc_result, {0, 2, 1});
}

mlir::Value build_conv2d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                         llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                         llvm::ArrayRef<int64_t> dilation, int64_t groups) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    auto shape = input_type.getShape();   // NCHW: [N, C_in, H, W]
    auto wshape = weight_type.getShape(); // OIHW: [C_out, C_in/groups, kH, kW]

    // NOLINTBEGIN
    int64_t kH = wshape[2], kW = wshape[3];
    int64_t pH = padding[0], pW = padding[1];
    int64_t dH = dilation[0], dW = dilation[1];
    int64_t sH = stride[0], sW = stride[1];
    int64_t H_out = (shape[2] + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (shape[3] + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    // NOLINTEND

    auto elem_type = input_type.getElementType();

    // Reshape 1D bias (C_out,) so the channel sits at channel_dim=1: (1, C_out, 1, 1).
    // The TTIR Conv2dOp verifier reads the bias output-channel count from channel_dim.
    mlir::Value bias_4d;
    if (bias) {
        bias_4d = build_reshape(mb, bias, {1, wshape[0], 1, 1});
    }

    auto stride_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(sH), as<int32_t>(sW)});
    // Symmetric padding: [top, left, bottom, right] = [pH, pW, pH, pW].
    auto padding_attr =
        mb.attrs().getDenseI32ArrayAttr({as<int32_t>(pH), as<int32_t>(pW), as<int32_t>(pH), as<int32_t>(pW)});
    auto dilation_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(dH), as<int32_t>(dW)});

    // NCHW output shape: [N, C_out, H_out, W_out]; dims: batch=0, channel=1, height=2, width=3.
    auto result_type = mlir::RankedTensorType::get({shape[0], wshape[0], H_out, W_out}, elem_type);
    return mb
        .create<mlir::tt::ttir::Conv2dOp>(result_type, input, weight, bias_4d, stride_attr, padding_attr, dilation_attr,
                                          as<uint32_t>(groups), /*batch_dim=*/uint64_t{0},
                                          /*height_dim=*/uint64_t{2}, /*width_dim=*/uint64_t{3},
                                          /*channel_dim=*/uint64_t{1})
        .getResult();
}

mlir::Value build_conv3d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                         llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                         llvm::ArrayRef<int64_t> dilation, int64_t groups) {
    auto ncdhw_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shape = ncdhw_type.getShape();                                            // NCDHW: [N, C_in, D, H, W]
    auto wshape = mlir::cast<mlir::RankedTensorType>(weight.getType()).getShape(); // OIDHW: [C_out, C_in, kD, kH, kW]
    auto elem_type = ncdhw_type.getElementType();

    // NCDHW[N,C,D,H,W] → NDHWC[N,D,H,W,C]
    auto ndhwc_input = build_permute(mb, input, {0, 2, 3, 4, 1});

    // NOLINTBEGIN
    int64_t kD = wshape[2], kH = wshape[3], kW = wshape[4];
    int64_t pD = padding[0], pH = padding[1], pW = padding[2];
    int64_t sD = stride[0], sH = stride[1], sW = stride[2];
    int64_t D_out = (shape[2] + 2 * pD - kD) / sD + 1;
    int64_t H_out = (shape[3] + 2 * pH - kH) / sH + 1;
    int64_t W_out = (shape[4] + 2 * pW - kW) / sW + 1;
    // NOLINTEND

    // The verifier wants a rank-5 bias with the channel at channel_dim=4.
    mlir::Value bias_5d;
    if (bias) {
        bias_5d = build_reshape(mb, bias, {1, 1, 1, 1, wshape[0]});
    }

    auto stride_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(sD), as<int32_t>(sH), as<int32_t>(sW)});
    auto padding_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(pD), as<int32_t>(pH), as<int32_t>(pW)});

    auto ndhwc_out_type = mlir::RankedTensorType::get({shape[0], D_out, H_out, W_out, wshape[0]}, elem_type);
    auto ndhwc_result =
        mb.create<mlir::tt::ttir::Conv3dOp>(ndhwc_out_type, ndhwc_input, weight, bias_5d, stride_attr, padding_attr,
                                            as<uint32_t>(groups), mb.attrs().getStringAttr("zeros"))
            .getResult();

    // NDHWC[N,D_out,H_out,W_out,C_out] → NCDHW[N,C_out,D_out,H_out,W_out]
    return build_permute(mb, ndhwc_result, {0, 4, 1, 2, 3});
}

} // namespace tt::crank
