// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <iomanip>
#include <string>
#include <type_traits>

// This namespace contains mock definitions of TTNN types for the purpose of
// conversion.
namespace ttsl {
template <typename T>
struct SmallVector {
  using value_type = T;
};
} // namespace ttsl

namespace ttnn {
struct Shape;

struct ShardSpec;
struct CoreRangeSet;
struct CoreRange;
struct CoreCoord;

struct DataType;
struct TensorMemoryLayout;
struct Layout;
struct MemoryConfig;
struct BufferType;

namespace types {
struct ShardOrientation;
} // namespace types

namespace distributed {
struct MeshDevice;
} // namespace distributed

struct Tensor;

namespace operations {
namespace unary {
struct UnaryWithParam;

// Mock definition of VecMode enum from tt-metal
enum class VecMode {
  None = 0,
  R = 1,
  C = 2,
  RC = 4,
  RC_custom = 6,
  Invalid = 0xFF,
};
} // namespace unary

namespace conv::conv2d {
struct Conv2dConfig;
struct Conv2dSliceConfig;
} // namespace conv::conv2d

namespace matmul {
struct MatmulMultiCoreReuseProgramConfig;
struct MatmulMultiCoreReuseMultiCastProgramConfig;
struct MatmulMultiCoreReuseMultiCast1DProgramConfig;
struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig;
} // namespace matmul
} // namespace operations

// Compute kernel config types
struct BlackholeComputeKernelConfig;
struct WormholeComputeKernelConfig;

// Math fidelity enum (mock for EmitPy conversion)
struct MathFidelity;

// LayerNorm program config types (mock for EmitPy conversion)
namespace prim {
struct LayerNormShardedMultiCoreProgramConfig;
} // namespace prim

namespace experimental::prim {
struct Conv3dConfig;
} // namespace experimental::prim

} // namespace ttnn

namespace mlir {
namespace tt {
namespace ttnn_to_emitpy {

template <typename T, typename Enable = void>
struct TypeName;

template <typename T>
const std::string TypeNameV = TypeName<T>::value;

template <typename T>
struct TypeName<T, std::enable_if_t<std::is_integral_v<T>>> {
  inline static const std::string value = "int";
};

template <typename T>
struct TypeName<T, std::enable_if_t<std::is_floating_point_v<T>, void>> {
  inline static const std::string value = "float";
};

template <>
struct TypeName<bool> {
  inline static const std::string value = "bool";
};

template <>
struct TypeName<::ttnn::Tensor> {
  inline static const std::string value = "ttnn.Tensor";
};

template <>
struct TypeName<::ttnn::operations::unary::UnaryWithParam> {
  inline static const std::string value = "ttnn.UnaryWithParam";
};

template <>
struct TypeName<::ttnn::operations::conv::conv2d::Conv2dConfig> {
  inline static const std::string value = "ttnn.Conv2dConfig";
};

template <>
struct TypeName<::ttnn::operations::conv::conv2d::Conv2dSliceConfig> {
  inline static const std::string value = "ttnn.Conv2dSliceConfig";
};

template <>
struct TypeName<::ttnn::experimental::prim::Conv3dConfig> {
  inline static const std::string value = "ttnn.Conv3dConfig";
};

template <>
struct TypeName<::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig> {
  inline static const std::string value =
      "ttnn.MatmulMultiCoreReuseProgramConfig";
};

template <>
struct TypeName<
    ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> {
  inline static const std::string value =
      "ttnn.MatmulMultiCoreReuseMultiCastProgramConfig";
};

template <>
struct TypeName<
    ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> {
  inline static const std::string value =
      "ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig";
};

template <>
struct TypeName<::ttnn::operations::matmul::
                    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> {
  inline static const std::string value =
      "ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig";
};

template <>
struct TypeName<::ttnn::WormholeComputeKernelConfig> {
  inline static const std::string value = "ttnn.WormholeComputeKernelConfig";
};

template <>
struct TypeName<::ttnn::BlackholeComputeKernelConfig> {
  inline static const std::string value = "ttnn.BlackholeComputeKernelConfig";
};

template <>
struct TypeName<::ttnn::prim::LayerNormShardedMultiCoreProgramConfig> {
  inline static const std::string value =
      "ttnn.LayerNormShardedMultiCoreProgramConfig";
};

template <typename T>
struct is_list_type : public std::false_type {};

template <typename T>
struct is_list_type<std::vector<T>> : public std::true_type {};

template <typename T>
struct is_list_type<::ttsl::SmallVector<T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_list_type_v = is_list_type<T>::value;

template <typename T>
struct TypeName<T, std::enable_if_t<is_list_type_v<T>, void>> {
  using value_type = typename T::value_type;
  inline static const std::string value = "[" + TypeNameV<value_type> + "]";
};

template <>
struct TypeName<std::string> {
  inline static const std::string value = "str";
};

template <>
struct TypeName<std::nullopt_t> {
  inline static const std::string value = "None";
};

template <typename T>
struct TypeName<std::set<T>> {
  inline static const std::string value = "{" + TypeNameV<T> + "}";
};

template <>
struct TypeName<::ttnn::CoreCoord> {
  inline static const std::string value = "ttnn.CoreCoord";
};

template <>
struct TypeName<::ttnn::CoreRange> {
  inline static const std::string value = "ttnn.CoreRange";
};

template <>
struct TypeName<::ttnn::CoreRangeSet> {
  inline static const std::string value = "ttnn.CoreRangeSet";
};

template <>
struct TypeName<::ttnn::ShardSpec> {
  inline static const std::string value = "ttnn.ShardSpec";
};

template <>
struct TypeName<::ttnn::types::ShardOrientation> {
  inline static const std::string value = "ttnn.ShardOrientation";
};

template <>
struct TypeName<::ttnn::DataType> {
  inline static const std::string value = "ttnn.DataType";
};

template <>
struct TypeName<::ttnn::TensorMemoryLayout> {
  inline static const std::string value = "ttnn.TensorMemoryLayout";
};

template <>
struct TypeName<::ttnn::Layout> {
  inline static const std::string value = "ttnn.Layout";
};

template <>
struct TypeName<::ttnn::MemoryConfig> {
  inline static const std::string value = "ttnn.MemoryConfig";
};

template <>
struct TypeName<::ttnn::BufferType> {
  inline static const std::string value = "ttnn.BufferType";
};

template <>
struct TypeName<::ttnn::Shape> {
  inline static const std::string value = "ttnn.Shape";
};

template <typename T, typename Enable = void>
struct EmitPyTypeConverter;

template <>
struct EmitPyTypeConverter<bool> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto boolAttr = mlir::dyn_cast_if_present<mlir::BoolAttr>(attr)) {
      return convert(boolAttr);
    }
    return {};
  }

  static std::string convert(mlir::APInt attr) {
    assert(attr.getBitWidth() == 1 && "Expected a 1-bit APInt");
    return convert(static_cast<bool>(attr.getZExtValue()));
  }

  static std::string convert(mlir::BoolAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(bool value) { return value ? "True" : "False"; }
};

template <>
struct EmitPyTypeConverter<std::string> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto strAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(attr)) {
      return convert(strAttr);
    }
    return {};
  }

  static std::string convert(mlir::StringAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(mlir::StringRef attr) {
    return convert(attr.str());
  }

  static std::string convert(std::string value) { return "\"" + value + "\""; }
};

template <>
struct EmitPyTypeConverter<mlir::tt::ttcore::MeshShardDirection> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto meshShardDirectionAttr =
            mlir::dyn_cast_if_present<mlir::tt::ttcore::MeshShardDirectionAttr>(
                attr)) {
      return convert(meshShardDirectionAttr);
    }
    return {};
  }

  static std::string convert(mlir::tt::ttcore::MeshShardDirectionAttr attr) {
    return convert(attr.getValue());
  }

  static std::string
  convert(::mlir::tt::ttcore::MeshShardDirection meshShardDirection) {
    switch (meshShardDirection) {
    case ::mlir::tt::ttcore::MeshShardDirection::FullToShard:
      return "ttnn.MeshShardDirection.FullToShard";
    case ::mlir::tt::ttcore::MeshShardDirection::ShardToFull:
      return "ttnn.MeshShardDirection.ShardToFull";
    }
    llvm_unreachable("Unknown ttnn.MeshShardDirection");
  }
};

template <>
struct EmitPyTypeConverter<mlir::tt::ttcore::MeshShardType> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto meshShardTypeAttr =
            mlir::dyn_cast_if_present<mlir::tt::ttcore::MeshShardTypeAttr>(
                attr)) {
      return convert(meshShardTypeAttr);
    }
    return {};
  }

  static std::string convert(mlir::tt::ttcore::MeshShardTypeAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(::mlir::tt::ttcore::MeshShardType meshShardType) {
    switch (meshShardType) {
    case ::mlir::tt::ttcore::MeshShardType::Identity:
      return "ttnn.MeshShardType.Identity";
    case ::mlir::tt::ttcore::MeshShardType::Replicate:
      return "ttnn.MeshShardType.Replicate";
    case ::mlir::tt::ttcore::MeshShardType::Maximal:
      return "ttnn.MeshShardType.Maximal";
    case ::mlir::tt::ttcore::MeshShardType::Devices:
      return "ttnn.MeshShardType.Devices";
    }
    llvm_unreachable("Unknown ttnn.MeshShardType");
  }
};

template <>
struct EmitPyTypeConverter<mlir::tt::ttcore::Topology> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto topologyAttr =
            mlir::dyn_cast_if_present<mlir::tt::ttcore::TopologyAttr>(attr)) {
      return convert(topologyAttr);
    }
    return {};
  }

  static std::string convert(mlir::tt::ttcore::TopologyAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(::mlir::tt::ttcore::Topology topology) {
    switch (topology) {
    case ::mlir::tt::ttcore::Topology::Ring:
      return "ttnn.Topology.Ring";
    case ::mlir::tt::ttcore::Topology::Linear:
      return "ttnn.Topology.Linear";
    default:
      // Mesh, Torus, and Disabled are not defined in ttnn.Topology yet, so we
      // don't need to handle them here. See ccl_pybind.cpp for the definition
      // of ttnn.Topology.
      llvm_unreachable("Unknown ttnn.Topology");
    }
  }
};

template <>
struct EmitPyTypeConverter<mlir::tt::ttcore::ReduceType> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto reduceTypeAttr =
            mlir::dyn_cast_if_present<mlir::tt::ttcore::ReduceTypeAttr>(attr)) {
      return convert(reduceTypeAttr);
    }
    return {};
  }

  static std::string convert(mlir::tt::ttcore::ReduceTypeAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(::mlir::tt::ttcore::ReduceType type) {
    std::string base = "ttnn.ReduceType";
    switch (type) {
    case ::mlir::tt::ttcore::ReduceType::Sum:
      return base + ".Sum";
    case ::mlir::tt::ttcore::ReduceType::Mean:
      return base + ".Mean";
    case ::mlir::tt::ttcore::ReduceType::Max:
      return base + ".Max";
    case ::mlir::tt::ttcore::ReduceType::Min:
      return base + ".Min";
    case ::mlir::tt::ttcore::ReduceType::Std:
      return base + ".Std";
    case ::mlir::tt::ttcore::ReduceType::Var:
      return base + ".Var";
    case ::mlir::tt::ttcore::ReduceType::Prod:
      return base + ".Prod";
    case ::mlir::tt::ttcore::ReduceType::Invalid:
      return base + ".Invalid";
    }
    llvm_unreachable("Unknown ttnn.ReduceType");
  }
};

// Converter for integral types.
template <typename T>
struct EmitPyTypeConverter<T, std::enable_if_t<std::is_integral_v<T>, void>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto intAttr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(attr)) {
      return convert(intAttr);
    }
    return {};
  }

  static std::string convert(mlir::IntegerAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(mlir::APInt value) {
    if constexpr (std::is_signed_v<T>) {
      return convert(value.getSExtValue());
    }
    return convert(value.getZExtValue());
  }

  template <typename U>
  static std::enable_if_t<std::is_integral_v<U>, std::string> convert(U value) {
    return std::to_string(static_cast<T>(value));
  }
};

// Converter for floating point types. Double is the only type that makes sense
// to convert to in Python so we always convert to double.
template <typename T>
struct EmitPyTypeConverter<
    T, std::enable_if_t<std::is_floating_point_v<T>, void>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto floatAttr = mlir::dyn_cast_if_present<mlir::FloatAttr>(attr)) {
      return convert(floatAttr);
    }
    return {};
  }

  static std::string convert(mlir::FloatAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(mlir::APFloat value) {
    return convert(value.convertToDouble());
  }

  static std::string convert(double value) {
    if (std::isfinite(value)) {
      // Convert to string with full precision.
      std::ostringstream oss;
      oss << std::setprecision(std::numeric_limits<double>::max_digits10);
      oss << value;
      std::string result = oss.str();
      // Ensure decimal point is present for Python float compatibility
      if (result.find('.') == std::string::npos &&
          result.find('e') == std::string::npos) {
        result += ".0";
      }
      return result;
    }

    if (std::isinf(value)) {
      return value > 0 ? "float('inf')" : "float('-inf')";
    }

    if (std::isnan(value)) {
      return "float('nan')";
    }

    llvm_unreachable("Unknown class of floating point value");
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::CoreCoord> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto coreCoordAttr =
            mlir::dyn_cast_if_present<ttnn::CoreCoordAttr>(attr)) {
      return convert(coreCoordAttr);
    }
    return {};
  }

  static std::string convert(ttnn::CoreCoordAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::CoreCoord>;
    rso << "(";
    rso << EmitPyTypeConverter<size_t>::convert(attr.getX()) << ", ";
    rso << EmitPyTypeConverter<size_t>::convert(attr.getY());
    rso << ")";

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::CoreRange> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto coreRangeAttr =
            mlir::dyn_cast_if_present<ttnn::CoreRangeAttr>(attr)) {
      return convert(coreRangeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::CoreRangeAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::CoreRange>;
    rso << "(";
    rso << EmitPyTypeConverter<::ttnn::CoreCoord>::convert(attr.getStartCoord())
        << ", ";
    rso << EmitPyTypeConverter<::ttnn::CoreCoord>::convert(attr.getEndCoord());
    rso << ")";

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::types::ShardOrientation> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shardOrientationAttr =
            mlir::dyn_cast_if_present<ttnn::ShardOrientationAttr>(attr)) {
      return convert(shardOrientationAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShardOrientationAttr attr) {
    assert(
        attr &&
        "expected non-null attribute, call "
        "EmitPyTypeConverter<std::optional<::ttnn::types::ShardOrientation>>:"
        ":convert(attr) if attribute is optional");
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::ShardOrientation attr) {
    switch (attr) {
    case ttnn::ShardOrientation::RowMajor:
      return TypeNameV<::ttnn::types::ShardOrientation> + ".ROW_MAJOR";
    case ttnn::ShardOrientation::ColMajor:
      return TypeNameV<::ttnn::types::ShardOrientation> + ".COL_MAJOR";
    }

    llvm_unreachable("Unknown ttnn.ShardOrientation");
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::DataType> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto dataTypeAttr =
            mlir::dyn_cast_if_present<ttcore::DataTypeAttr>(attr)) {
      return convert(dataTypeAttr);
    }
    return {};
  }

  static std::string convert(ttcore::DataType attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::DataType> << ".";
    switch (attr) {
    case ttcore::DataType::BFloat16:
      rso << "BFLOAT16";
      break;
    case ttcore::DataType::Float32:
      rso << "FLOAT32";
      break;
    case ttcore::DataType::UInt32:
      rso << "UINT32";
      break;
    case ttcore::DataType::BFP_BFloat8:
      rso << "BFLOAT8_B";
      break;
    case ttcore::DataType::BFP_BFloat4:
      rso << "BFLOAT4_B";
      break;
    case ttcore::DataType::UInt8:
      rso << "UINT8";
      break;
    case ttcore::DataType::UInt16:
      rso << "UINT16";
      break;
    case ttcore::DataType::Int32:
      rso << "INT32";
      break;
    case ttcore::DataType::Float16:
    case ttcore::DataType::BFP_Float2:
    case ttcore::DataType::BFP_Float4:
    case ttcore::DataType::BFP_Float8:
    case ttcore::DataType::BFP_BFloat2:
    case ttcore::DataType::Bool:
      llvm_unreachable("Unsupported ttnn.DataType");
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::TensorMemoryLayout> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto tensorMemoryLayoutAttr =
            mlir::dyn_cast_if_present<ttnn::TensorMemoryLayoutAttr>(attr)) {
      return convert(tensorMemoryLayoutAttr);
    }
    return {};
  }

  static std::string convert(ttnn::TensorMemoryLayoutAttr attr) {
    // TODO (azecevic): There is a dissonance between the way we model
    // TensorMemoryLayout in TTNN dialect and TTNN library. This should be fixed
    // with https://github.com/tenstorrent/tt-mlir/issues/2521. For now, we
    // default to Interleaved, which is default value in TTNN library.
    if (!attr) {
      return convert(ttnn::TensorMemoryLayout::Interleaved);
    }
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::TensorMemoryLayout attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::TensorMemoryLayout> << ".";
    switch (attr) {
    case ttnn::TensorMemoryLayout::BlockSharded:
      rso << "BLOCK_SHARDED";
      break;
    case ttnn::TensorMemoryLayout::HeightSharded:
      rso << "HEIGHT_SHARDED";
      break;
    case ttnn::TensorMemoryLayout::Interleaved:
      rso << "INTERLEAVED";
      break;
    case ttnn::TensorMemoryLayout::WidthSharded:
      rso << "WIDTH_SHARDED";
      break;
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::Layout> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto layoutAttr = mlir::dyn_cast_if_present<ttnn::LayoutAttr>(attr)) {
      return convert(layoutAttr);
    }
    return {};
  }

  static std::string convert(ttnn::LayoutAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::Layout attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::Layout> << ".";
    switch (attr) {
    case ttnn::Layout::RowMajor:
      rso << "ROW_MAJOR";
      break;
    case ttnn::Layout::Tile:
      rso << "TILE";
      break;
    case ttnn::Layout::Invalid:
      rso << "INVALID";
      break;
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::BufferType> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto bufferTypeAttr =
            mlir::dyn_cast_if_present<ttnn::BufferTypeAttr>(attr)) {
      return convert(bufferTypeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::BufferTypeAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::BufferType attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::BufferType> << ".";
    switch (attr) {
    case ttnn::BufferType::DRAM:
      rso << "DRAM";
      break;
    case ttnn::BufferType::L1:
      rso << "L1";
      break;
    case ttnn::BufferType::L1Small:
      rso << "L1_SMALL";
      break;
    case ttnn::BufferType::Trace:
      rso << "TRACE";
      break;
    case ttnn::BufferType::SystemMemory:
      llvm_unreachable("Unsupported ttnn.BufferType: SystemMemory");
      break;
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::Shape> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shapeAttr = mlir::dyn_cast_if_present<ttnn::ShapeAttr>(attr)) {
      return convert(shapeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShapeAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    auto shape = attr.getShape();
    rso << TypeNameV<::ttnn::Shape> << "([";
    llvm::interleaveComma(shape, rso);
    rso << "])";

    return buf;
  }
};

// Convert container types (std::vector, ttnn::SmallVector, etc.).
template <typename T>
struct EmitPyContainerTypeConverter {
  using value_type = typename T::value_type;

  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      if (arrayAttr.empty() ||
          EmitPyTypeConverter<value_type>::convert(arrayAttr[0])) {
        return convert(arrayAttr);
      }
      return {};
    }

    if constexpr (std::is_integral_v<value_type>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseBoolArrayAttr, mlir::DenseI8ArrayAttr,
                mlir::DenseI16ArrayAttr, mlir::DenseI32ArrayAttr,
                mlir::DenseI64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseIntElementsAttr>(
              [](mlir::DenseIntElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    if constexpr (std::is_floating_point_v<value_type>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseF32ArrayAttr, mlir::DenseF64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseFPElementsAttr>(
              [](mlir::DenseFPElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    return {};
  }

  static std::string convert(mlir::ArrayAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(*EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  template <typename U>
  static std::enable_if_t<
      std::is_constructible_v<mlir::detail::DenseArrayAttrImpl<U>>, std::string>
  convert(mlir::detail::DenseArrayAttrImpl<U> attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr.asArrayRef()) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseIntElementsAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseFPElementsAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  template <typename U>
  static std::string convert(llvm::ArrayRef<U> attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

private:
  static std::string convert(const llvm::SmallVector<std::string> &values) {
    constexpr char openingParen =
        std::is_same_v<T, std::set<value_type>> ? '{' : '[';
    constexpr char closingParen =
        std::is_same_v<T, std::set<value_type>> ? '}' : ']';

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << openingParen;
    llvm::interleaveComma(values, rso);
    rso << closingParen;
    return buf;
  }
};

template <typename T>
struct EmitPyTypeConverter<std::vector<T>>
    : public EmitPyContainerTypeConverter<std::vector<T>> {};

template <typename T>
struct EmitPyTypeConverter<::ttsl::SmallVector<T>>
    : public EmitPyContainerTypeConverter<::ttsl::SmallVector<T>> {};

template <typename T>
struct EmitPyTypeConverter<std::set<T>>
    : public EmitPyContainerTypeConverter<std::set<T>> {};

template <typename T>
struct EmitPyTypeConverter<llvm::ArrayRef<T>>
    : public EmitPyContainerTypeConverter<::ttsl::SmallVector<T>> {};

template <typename T>
struct EmitPyTypeConverter<mlir::detail::DenseArrayAttrImpl<T>>
    : public EmitPyContainerTypeConverter<::ttsl::SmallVector<T>> {};

template <typename T, size_t k>
struct EmitPyTypeConverter<std::array<T, k>> {

  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      if (arrayAttr.empty() || EmitPyTypeConverter<T>::convert(arrayAttr[0])) {
        return convert(arrayAttr);
      }
      return {};
    }

    if constexpr (std::is_integral_v<T>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseBoolArrayAttr, mlir::DenseI8ArrayAttr,
                mlir::DenseI16ArrayAttr, mlir::DenseI32ArrayAttr,
                mlir::DenseI64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseIntElementsAttr>(
              [](mlir::DenseIntElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    if constexpr (std::is_floating_point_v<T>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseF32ArrayAttr, mlir::DenseF64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseFPElementsAttr>(
              [](mlir::DenseFPElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    return {};
  }

  static std::optional<std::string> convert(mlir::ArrayAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (size_t i = 0; i < attr.size(); ++i) {
      auto element = EmitPyTypeConverter<T>::convert(attr[i]);
      if (!element) {
        return {};
      }
      result[i] = *element;
    }
    return convert(result);
  }

  template <typename U>
  static std::enable_if_t<
      std::is_constructible_v<mlir::detail::DenseArrayAttrImpl<U>>,
      std::optional<std::string>>
  convert(mlir::detail::DenseArrayAttrImpl<U> attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(attr[i]);
    }
    return convert(result);
  }

  static std::optional<std::string> convert(mlir::DenseIntElementsAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return convert(result);
  }

  static std::optional<std::string> convert(mlir::DenseFPElementsAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return convert(result);
  }

  template <typename U>
  static std::optional<std::string> convert(llvm::ArrayRef<U> attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (size_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(attr[i]);
    }
    return convert(result);
  }

  static std::string convert(const std::array<T, k> &values) {
    std::array<std::string, k> result;
    for (size_t i = 0; i < k; ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(values[i]);
    }
    return convert(result);
  }

private:
  static std::string convert(const std::array<std::string, k> &values) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << "[";
    llvm::interleaveComma(values, rso);
    rso << "]";
    return buf;
  }
};

template <>
struct EmitPyTypeConverter<mlir::ElementsAttr> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
        .Case<mlir::DenseIntElementsAttr>(
            [](mlir::DenseIntElementsAttr denseIntAttr)
                -> std::optional<std::string> {
              // Determine the element type and delegate to appropriate
              // converter. If it's a bool, convert to a vector of bools,
              // otherwise convert to a vector of int64_t (as the widest
              // standard integer type in C++).
              auto elementType = denseIntAttr.getElementType();
              if (elementType.isInteger(1)) {
                return EmitPyTypeConverter<std::vector<bool>>::convert(
                    denseIntAttr);
              }
              return EmitPyTypeConverter<std::vector<int64_t>>::convert(
                  denseIntAttr);
            })
        .Case<mlir::DenseFPElementsAttr>(
            [](mlir::DenseFPElementsAttr denseFPAttr)
                -> std::optional<std::string> {
              // Convert to a vector of doubles (as the widest standard floating
              // point type in C++).
              return EmitPyTypeConverter<std::vector<double>>::convert(
                  denseFPAttr);
            })
        .Default([](auto) { return std::optional<std::string>{}; });
  }
};

template <typename First, typename... Rest>
struct EmitPyTypeConverter<std::variant<First, Rest...>> {
  static std::string convert(mlir::Attribute attr) {
    auto tryFirst = EmitPyTypeConverter<First>::convert(attr);
    if constexpr (std::is_same_v<decltype(tryFirst), std::string>) {
      return tryFirst;
    } else {
      static_assert(
          std::is_same_v<decltype(tryFirst), std::optional<std::string>>);
      if (tryFirst) {
        return *tryFirst;
      }
    }

    if constexpr (sizeof...(Rest) > 0) {
      return EmitPyTypeConverter<std::variant<Rest...>>::convert(attr);
    }

    llvm_unreachable("Invalid variant type");
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::CoreRangeSet> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto coreRangeSetAttr =
            mlir::dyn_cast_if_present<ttnn::CoreRangeSetAttr>(attr)) {
      return convert(coreRangeSetAttr);
    }
    return {};
  }

  static std::string convert(ttnn::CoreRangeSetAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<::ttnn::CoreRangeSet>;
    rso << "(";
    rso << EmitPyTypeConverter<std::vector<::ttnn::CoreRange>>::convert(
        attr.getCoreRanges());
    rso << ")";
    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::ShardSpec> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shardSpecAttr =
            mlir::dyn_cast_if_present<ttnn::ShardSpecAttr>(attr)) {
      return convert(shardSpecAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShardSpecAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::ShardSpec>;
    rso << "(";
    rso << EmitPyTypeConverter<::ttnn::CoreRangeSet>::convert(
               attr.getCoreRangeSet())
        << ", ";
    rso << EmitPyTypeConverter<std::vector<uint32_t>>::convert(
               attr.getShape().getShape())
        << ", ";
    rso << EmitPyTypeConverter<::ttnn::types::ShardOrientation>::convert(
        attr.getShardOrientation());
    rso << ")";

    return buf;
  }
};

template <typename T>
struct EmitPyTypeConverter<std::optional<T>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }
    return EmitPyTypeConverter<T>::convert(attr);
  }

  template <typename U>
  static std::string convert(std::optional<U> &&attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    return EmitPyTypeConverter<T>::convert(*attr);
  }

  template <typename U>
  static std::string convert(U &&attr) {
    return EmitPyTypeConverter<T>::convert(std::forward<U>(attr));
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::MemoryConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto memoryConfigAttr =
            mlir::dyn_cast_if_present<ttnn::MemoryConfigAttr>(attr)) {
      return convert(memoryConfigAttr);
    }
    return {};
  }

  static std::string convert(ttnn::MemoryConfigAttr attr) {
    // If BufferType is SystemMemory, there is no need for a MemoryConfig as it
    // is only used to represent memory of tensors on device.
    //
    if (!attr ||
        attr.getBufferType().getValue() == ttnn::BufferType::SystemMemory) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<::ttnn::MemoryConfig> << "(";
    rso << EmitPyTypeConverter<::ttnn::TensorMemoryLayout>::convert(
               attr.getTensorMemoryLayout())
        << ", ";
    rso << EmitPyTypeConverter<::ttnn::BufferType>::convert(
               attr.getBufferType())
        << ", ";
    rso << EmitPyTypeConverter<std::optional<::ttnn::ShardSpec>>::convert(
        attr.getShardSpec());
    rso << ")";
    return buf;
  }
};

inline std::string convert(ttnn::UnaryOpType opType) {
  static const std::unordered_map<ttnn::UnaryOpType, std::string> opTypeMap = {
      {ttnn::UnaryOpType::Exp, "ttnn.UnaryOpType.EXP"},
      {ttnn::UnaryOpType::Recip, "ttnn.UnaryOpType.RECIP"},
      {ttnn::UnaryOpType::Gelu, "ttnn.UnaryOpType.GELU"},
      {ttnn::UnaryOpType::Relu, "ttnn.UnaryOpType.RELU"},
      {ttnn::UnaryOpType::Sqrt, "ttnn.UnaryOpType.SQRT"},
      {ttnn::UnaryOpType::Sigmoid, "ttnn.UnaryOpType.SIGMOID"},
      {ttnn::UnaryOpType::Log, "ttnn.UnaryOpType.LOG"},
      {ttnn::UnaryOpType::Tanh, "ttnn.UnaryOpType.TANH"},
      {ttnn::UnaryOpType::Log2, "ttnn.UnaryOpType.LOG2"},
      {ttnn::UnaryOpType::Log10, "ttnn.UnaryOpType.LOG10"},
      {ttnn::UnaryOpType::Sin, "ttnn.UnaryOpType.SIN"},
      {ttnn::UnaryOpType::Cos, "ttnn.UnaryOpType.COS"},
      {ttnn::UnaryOpType::Abs, "ttnn.UnaryOpType.ABS"},
      {ttnn::UnaryOpType::AbsInt32, "ttnn.UnaryOpType.ABS_INT32"},
      {ttnn::UnaryOpType::Sign, "ttnn.UnaryOpType.SIGN"},
      {ttnn::UnaryOpType::Square, "ttnn.UnaryOpType.SQUARE"},
      {ttnn::UnaryOpType::Eqz, "ttnn.UnaryOpType.EQZ"},
      {ttnn::UnaryOpType::Nez, "ttnn.UnaryOpType.NEZ"},
      {ttnn::UnaryOpType::Gtz, "ttnn.UnaryOpType.GTZ"},
      {ttnn::UnaryOpType::Ltz, "ttnn.UnaryOpType.LTZ"},
      {ttnn::UnaryOpType::Gez, "ttnn.UnaryOpType.GEZ"},
      {ttnn::UnaryOpType::Lez, "ttnn.UnaryOpType.LEZ"},
      {ttnn::UnaryOpType::ReluMax, "ttnn.UnaryOpType.RELU_MAX"},
      {ttnn::UnaryOpType::ReluMin, "ttnn.UnaryOpType.RELU_MIN"},
      {ttnn::UnaryOpType::Power, "ttnn.UnaryOpType.POWER"},
      {ttnn::UnaryOpType::LeakyRelu, "ttnn.UnaryOpType.LEAKY_RELU"},
      {ttnn::UnaryOpType::Elu, "ttnn.UnaryOpType.ELU"},
      {ttnn::UnaryOpType::Exp2, "ttnn.UnaryOpType.EXP2"},
      {ttnn::UnaryOpType::Heaviside, "ttnn.UnaryOpType.HEAVISIDE"},
      {ttnn::UnaryOpType::Expm1, "ttnn.UnaryOpType.EXPM1"},
      {ttnn::UnaryOpType::Signbit, "ttnn.UnaryOpType.SIGNBIT"},
      {ttnn::UnaryOpType::Asin, "ttnn.UnaryOpType.ASIN"},
      {ttnn::UnaryOpType::Acos, "ttnn.UnaryOpType.ACOS"},
      {ttnn::UnaryOpType::Rsqrt, "ttnn.UnaryOpType.RSQRT"},
      {ttnn::UnaryOpType::Relu6, "ttnn.UnaryOpType.RELU6"},
      {ttnn::UnaryOpType::Atan, "ttnn.UnaryOpType.ATAN"},
      {ttnn::UnaryOpType::Erf, "ttnn.UnaryOpType.ERF"},
      {ttnn::UnaryOpType::Erfc, "ttnn.UnaryOpType.ERFC"},
      {ttnn::UnaryOpType::IsInf, "TTNNUnaryOpType::ISINF"},
      {ttnn::UnaryOpType::IsPosInf, "TTNNUnaryOpType::ISPOSINF"},
      {ttnn::UnaryOpType::IsNegInf, "TTNNUnaryOpType::ISNEGINF"},
      {ttnn::UnaryOpType::IsNan, "TTNNUnaryOpType::ISNAN"},
      {ttnn::UnaryOpType::LogicalNotUnary,
       "ttnn.UnaryOpType.LOGICAL_NOT_UNARY"},
      {ttnn::UnaryOpType::IsFinite, "TTNNUnaryOpType::ISFINITE"},
      {ttnn::UnaryOpType::Erfinv, "ttnn.UnaryOpType.ERFINV"},
      {ttnn::UnaryOpType::I0, "ttnn.UnaryOpType.I0"},
      {ttnn::UnaryOpType::I1, "ttnn.UnaryOpType.I1"},
      {ttnn::UnaryOpType::Tan, "ttnn.UnaryOpType.TAN"},
      {ttnn::UnaryOpType::Rsub, "ttnn.UnaryOpType.RSUB"},
      {ttnn::UnaryOpType::Rdiv, "ttnn.UnaryOpType.RDIV"},
      {ttnn::UnaryOpType::Silu, "ttnn.UnaryOpType.SILU"},
      {ttnn::UnaryOpType::SoftPlus, "ttnn.UnaryOpType.SOFTPLUS"},
      {ttnn::UnaryOpType::Identity, "ttnn.UnaryOpType.IDENTITY"},
      {ttnn::UnaryOpType::Neg, "ttnn.UnaryOpType.NEG"},
      {ttnn::UnaryOpType::AddUnarySfpu, "ttnn.UnaryOpType.ADD_UNARY_SFPU"},
      {ttnn::UnaryOpType::SubUnarySfpu, "ttnn.UnaryOpType.SUB_UNARY_SFPU"},
      {ttnn::UnaryOpType::MulUnarySfpu, "ttnn.UnaryOpType.MUL_UNARY_SFPU"},
      {ttnn::UnaryOpType::DivUnarySfpu, "ttnn.UnaryOpType.DIV_UNARY_SFPU"},
      {ttnn::UnaryOpType::IdentityUint32, "ttnn.UnaryOpType.IDENTITY"},
      {ttnn::UnaryOpType::UnaryNe, "ttnn.UnaryOpType.UNARY_NE"},
      {ttnn::UnaryOpType::UnaryGt, "ttnn.UnaryOpType.UNARY_GT"},
      {ttnn::UnaryOpType::UnaryLt, "ttnn.UnaryOpType.UNARY_LT"},
      {ttnn::UnaryOpType::TiledProd, "ttnn.UnaryOpType.TILED_PROD"},
      {ttnn::UnaryOpType::Typecast, "ttnn.UnaryOpType.TYPECAST"},
      {ttnn::UnaryOpType::BitwiseXor, "ttnn.UnaryOpType.BITWISE_XOR"},
      {ttnn::UnaryOpType::BitwiseNot, "ttnn.UnaryOpType.BITWISE_NOT"},
      {ttnn::UnaryOpType::BitwiseAnd, "ttnn.UnaryOpType.BITWISE_AND"},
      {ttnn::UnaryOpType::BitwiseOr, "ttnn.UnaryOpType.BITWISE_OR"},
      {ttnn::UnaryOpType::RightShift, "ttnn.UnaryOpType.RIGHT_SHIFT"},
      {ttnn::UnaryOpType::Floor, "ttnn.UnaryOpType.FLOOR"},
      {ttnn::UnaryOpType::Ceil, "ttnn.UnaryOpType.CEIL"},
      {ttnn::UnaryOpType::Round, "ttnn.UnaryOpType.ROUND"},
      {ttnn::UnaryOpType::LeftShift, "ttnn.UnaryOpType.LEFT_SHIFT"},
      {ttnn::UnaryOpType::Remainder, "ttnn.UnaryOpType.REMAINDER"},
      {ttnn::UnaryOpType::Fmod, "ttnn.UnaryOpType.FMOD"},
      {ttnn::UnaryOpType::Dropout, "ttnn.UnaryOpType.DROPOUT"},
      {ttnn::UnaryOpType::Fill, "ttnn.UnaryOpType.FILL"},
      {ttnn::UnaryOpType::PreluSfpu, "ttnn.UnaryOpType.PRELU_SFPU"},
      {ttnn::UnaryOpType::ZeroPoint, "ttnn.UnaryOpType.ZERO_POINT"},
  };

  return opTypeMap.at(opType);
}

template <>
struct EmitPyTypeConverter<::ttnn::operations::unary::UnaryWithParam> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto unaryWithParamAttr =
            mlir::dyn_cast_if_present<ttnn::UnaryWithParamAttr>(attr)) {
      return convert(unaryWithParamAttr);
    }
    return {};
  }

  static std::string convert(ttnn::UnaryWithParamAttr attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::operations::unary::UnaryWithParam> << "(";
    rso << ttnn_to_emitpy::convert(attr.getOpType());
    if (!attr.getParams().empty()) {
      rso << ", ";
      rso << EmitPyTypeConverter<std::vector<float>>::convert(attr.getParams());
    }
    rso << ")";

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<mlir::tt::ttnn::MeshShapeAttr> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto meshShapeAttr =
            mlir::dyn_cast_if_present<mlir::tt::ttnn::MeshShapeAttr>(attr)) {
      return convert(meshShapeAttr);
    }
    return {};
  }

  static std::string convert(mlir::tt::ttnn::MeshShapeAttr meshShapeAttr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << "(";
    rso << EmitPyTypeConverter<int64_t>::convert(meshShapeAttr.getY());
    rso << ", ";
    rso << EmitPyTypeConverter<int64_t>::convert(meshShapeAttr.getX());
    rso << ")";
    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::operations::conv::conv2d::Conv2dConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto conv2dConfigAttr =
            mlir::dyn_cast_if_present<ttnn::Conv2dConfigAttr>(attr)) {
      return convert(conv2dConfigAttr);
    }
    return {};
  }

  static std::string convert(ttnn::Conv2dConfigAttr attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    bool firstElement = true;
    rso << TypeNameV<::ttnn::operations::conv::conv2d::Conv2dConfig> << "(";
    if (attr.getWeightsDtype()) {
      rso << (firstElement ? "" : ", ") << "weights_dtype="
          << EmitPyTypeConverter<::ttnn::DataType>::convert(
                 *attr.getWeightsDtype());
      firstElement = false;
    }
    if (attr.getActivation()) {
      rso << (firstElement ? "" : ", ") << "activation="
          << EmitPyTypeConverter<::ttnn::operations::unary::UnaryWithParam>::
                 convert(attr.getActivation());
      firstElement = false;
    }
    if (attr.getDeallocateActivation()) {
      rso << (firstElement ? "" : ", ") << "deallocate_activation="
          << EmitPyTypeConverter<bool>::convert(attr.getDeallocateActivation());
      firstElement = false;
    }
    if (attr.getReallocateHaloOutput()) {
      rso << (firstElement ? "" : ", ") << "reallocate_halo_output="
          << EmitPyTypeConverter<bool>::convert(attr.getReallocateHaloOutput());
      firstElement = false;
    }
    if (attr.getConfigTensorsInDram()) {
      rso << (firstElement ? "" : ", ") << "config_tensors_in_dram="
          << EmitPyTypeConverter<bool>::convert(attr.getConfigTensorsInDram());
      firstElement = false;
    }
    if (attr.getActBlockHOverride()) {
      rso << (firstElement ? "" : ", ") << "act_block_h_override="
          << EmitPyTypeConverter<uint32_t>::convert(
                 *attr.getActBlockHOverride());
      firstElement = false;
    }
    if (attr.getActBlockWDiv()) {
      rso << (firstElement ? "" : ", ") << "act_block_w_div="
          << EmitPyTypeConverter<uint32_t>::convert(*attr.getActBlockWDiv());
      firstElement = false;
    }
    if (attr.getReshardIfNotOptimal()) {
      rso << (firstElement ? "" : ", ") << "reshard_if_not_optimal="
          << EmitPyTypeConverter<bool>::convert(attr.getReshardIfNotOptimal());
      firstElement = false;
    }
    if (attr.getOverrideShardingConfig()) {
      rso << (firstElement ? "" : ", ") << "override_sharding_config="
          << EmitPyTypeConverter<bool>::convert(
                 attr.getOverrideShardingConfig());
      firstElement = false;
    }
    if (attr.getShardLayout()) {
      rso << (firstElement ? "" : ", ") << "shard_layout="
          << EmitPyTypeConverter<::ttnn::TensorMemoryLayout>::convert(
                 *attr.getShardLayout());
      firstElement = false;
    }
    if (attr.getCoreGrid()) {
      rso << (firstElement ? "" : ", ") << "core_grid="
          << EmitPyTypeConverter<::ttnn::CoreRangeSet>::convert(
                 attr.getCoreGrid());
      firstElement = false;
    }
    if (attr.getTransposeShards()) {
      rso << (firstElement ? "" : ", ") << "transpose_shards="
          << EmitPyTypeConverter<bool>::convert(attr.getTransposeShards());
      firstElement = false;
    }
    if (attr.getOutputLayout()) {
      rso << (firstElement ? "" : ", ") << "output_layout="
          << EmitPyTypeConverter<::ttnn::Layout>::convert(
                 *attr.getOutputLayout());
      firstElement = false;
    }
    if (attr.getEnableActDoubleBuffer()) {
      rso << (firstElement ? "" : ", ") << "enable_act_double_buffer = "
          << EmitPyTypeConverter<bool>::convert(
                 attr.getEnableActDoubleBuffer());
      firstElement = false;
    }
    if (attr.getEnableWeightsDoubleBuffer()) {
      rso << (firstElement ? "" : ", ") << "enable_weights_double_buffer = "
          << EmitPyTypeConverter<bool>::convert(
                 attr.getEnableWeightsDoubleBuffer());
      firstElement = false;
    }
    if (attr.getEnableKernelStrideFolding()) {
      rso << (firstElement ? "" : ", ") << "enable_kernel_stride_folding="
          << EmitPyTypeConverter<bool>::convert(
                 attr.getEnableKernelStrideFolding());
    }
    rso << ")";
    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::experimental::prim::Conv3dConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto conv3dConfigAttr =
            mlir::dyn_cast_if_present<ttnn::Conv3dConfigAttr>(attr)) {
      return convert(conv3dConfigAttr);
    }
    // Ensure Conv3dConfig is always materialized so runtime receives a valid
    // config object even when Conv3dConfigAttr is absent.
    return convert(ttnn::Conv3dConfigAttr{});
  }

  static std::string convert(ttnn::Conv3dConfigAttr attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    bool firstElement = true;
    rso << TypeNameV<::ttnn::experimental::prim::Conv3dConfig> << "(";
    if (attr) {
      if (attr.getWeightsDtype()) {
        rso << (firstElement ? "" : ", ") << "weights_dtype="
            << EmitPyTypeConverter<::ttnn::DataType>::convert(
                   *attr.getWeightsDtype());
        firstElement = false;
      }
      if (attr.getTOutBlock()) {
        rso << (firstElement ? "" : ", ") << "T_out_block="
            << *attr.getTOutBlock();
        firstElement = false;
      }
      if (attr.getWOutBlock()) {
        rso << (firstElement ? "" : ", ") << "W_out_block="
            << *attr.getWOutBlock();
        firstElement = false;
      }
      if (attr.getHOutBlock()) {
        rso << (firstElement ? "" : ", ") << "H_out_block="
            << *attr.getHOutBlock();
        firstElement = false;
      }
      if (attr.getCOutBlock()) {
        rso << (firstElement ? "" : ", ") << "C_out_block="
            << *attr.getCOutBlock();
        firstElement = false;
      }
      if (attr.getCInBlock()) {
        rso << (firstElement ? "" : ", ") << "C_in_block="
            << *attr.getCInBlock();
        firstElement = false;
      }
      if (attr.getComputeWithStorageGridSize()) {
        auto gridAttr = *attr.getComputeWithStorageGridSize();
        rso << (firstElement ? "" : ", ")
            << "compute_with_storage_grid_size=ttnn.CoreCoord("
            << gridAttr.getShape()[0] << ", " << gridAttr.getShape()[1] << ")";
      }
    }

    rso << ")";
    return buf;
  }
};

template <>
struct EmitPyTypeConverter<
    ::ttnn::operations::conv::conv2d::Conv2dSliceConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto conv2dSliceConfigAttr =
            mlir::dyn_cast_if_present<ttnn::Conv2dSliceConfigAttr>(attr)) {
      return convert(conv2dSliceConfigAttr);
    }
    return {};
  }

  static std::string convert(ttnn::Conv2dSliceConfigAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<
               ::ttnn::operations::conv::conv2d::Conv2dSliceConfig> << "(";
    rso << "slice_type=";
    // Convert enum to proper C++ enum value instead of integer
    switch (attr.getSliceType()) {
    case ttnn::Conv2dSliceType::DramHeight:
      rso << "ttnn.Conv2dDRAMSliceHeight";
      break;
    case ttnn::Conv2dSliceType::DramWidth:
      rso << "ttnn.Conv2dDRAMSliceWidth";
      break;
    case ttnn::Conv2dSliceType::L1Full:
      rso << "ttnn.Conv2dL1Full";
      break;
    }
    rso << ", ";
    rso << "num_slices="
        << EmitPyTypeConverter<uint32_t>::convert(attr.getNumSlices());
    rso << ")";

    return buf;
  }
};

// MatmulMultiCoreReuseProgramConfig converter
template <>
struct EmitPyTypeConverter<
    ::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto configAttr = mlir::dyn_cast_if_present<
            ttnn::MatmulMultiCoreReuseProgramConfigAttr>(attr)) {
      return convert(configAttr);
    }
    return {};
  }

  static std::string convert(ttnn::MatmulMultiCoreReuseProgramConfigAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::operations::matmul::
                         MatmulMultiCoreReuseProgramConfig> << "(";
    rso << "in0_block_w="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getIn0BlockW());
    rso << ", out_subblock_h="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getOutSubblockH());
    rso << ", out_subblock_w="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getOutSubblockW());
    rso << ", per_core_M="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreM());
    rso << ", per_core_N="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreN());
    rso << ")";

    return buf;
  }
};

// MatmulMultiCoreReuseMultiCastProgramConfig converter
template <>
struct EmitPyTypeConverter<
    ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto configAttr = mlir::dyn_cast_if_present<
            ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr>(attr)) {
      return convert(configAttr);
    }
    return {};
  }

  static std::string
  convert(ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::operations::matmul::
                         MatmulMultiCoreReuseMultiCastProgramConfig> << "(";
    rso << "compute_with_storage_grid_size="
        << EmitPyTypeConverter<::ttnn::CoreCoord>::convert(
               attr.getComputeWithStorageGridSize());
    rso << ", in0_block_w="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getIn0BlockW());
    rso << ", out_subblock_h="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getOutSubblockH());
    rso << ", out_subblock_w="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getOutSubblockW());
    rso << ", per_core_M="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreM());
    rso << ", per_core_N="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreN());
    rso << ", transpose_mcast="
        << EmitPyTypeConverter<bool>::convert(attr.getTransposeMcast());
    rso << ", fused_activation=";
    if (attr.getFusedActivation()) {
      rso << EmitPyTypeConverter<::ttnn::operations::unary::UnaryWithParam>::
              convert(attr.getFusedActivation());
    } else {
      rso << "None";
    }
    rso << ", fuse_batch="
        << EmitPyTypeConverter<bool>::convert(attr.getFuseBatch());
    rso << ")";

    return buf;
  }
};

// MatmulMultiCoreReuseMultiCast1DProgramConfig converter
template <>
struct EmitPyTypeConverter<
    ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto configAttr = mlir::dyn_cast_if_present<
            ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(attr)) {
      return convert(configAttr);
    }
    return {};
  }

  static std::string
  convert(ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::operations::matmul::
                         MatmulMultiCoreReuseMultiCast1DProgramConfig> << "(";
    rso << "compute_with_storage_grid_size="
        << EmitPyTypeConverter<::ttnn::CoreCoord>::convert(
               attr.getComputeWithStorageGridSize());
    rso << ", in0_block_w="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getIn0BlockW());
    rso << ", out_subblock_h="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getOutSubblockH());
    rso << ", out_subblock_w="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getOutSubblockW());
    rso << ", per_core_M="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreM());
    rso << ", per_core_N="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreN());
    rso << ", fuse_batch="
        << EmitPyTypeConverter<bool>::convert(attr.getFuseBatch());
    rso << ", fused_activation=";
    if (attr.getFusedActivation()) {
      rso << EmitPyTypeConverter<::ttnn::operations::unary::UnaryWithParam>::
              convert(attr.getFusedActivation());
    } else {
      rso << "None";
    }
    rso << ", mcast_in0="
        << EmitPyTypeConverter<bool>::convert(attr.getMcastIn0());
    rso << ", gather_in0="
        << EmitPyTypeConverter<bool>::convert(attr.getGatherIn0());
    rso << ", hop_cores="
        << EmitPyTypeConverter<::ttnn::CoreRangeSet>::convert(
               attr.getHopCores());
    rso << ", num_global_cb_receivers="
        << EmitPyTypeConverter<uint64_t>::convert(
               attr.getNumGlobalCbReceivers());
    rso << ", untilize_out="
        << EmitPyTypeConverter<bool>::convert(attr.getUntilizeOut());
    rso << ")";

    return buf;
  }
};

// MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig converter
template <>
struct EmitPyTypeConverter<
    ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto configAttr = mlir::dyn_cast_if_present<
            ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
            attr)) {
      return convert(configAttr);
    }
    return {};
  }

  static std::string convert(
      ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<
               ::ttnn::operations::matmul::
                   MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> << "(";
    rso << "in0_block_w="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getIn0BlockW());
    rso << ", per_core_M="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreM());
    rso << ", per_core_N="
        << EmitPyTypeConverter<uint64_t>::convert(attr.getPerCoreN());
    rso << ", fused_activation=";
    if (attr.getFusedActivation()) {
      rso << EmitPyTypeConverter<::ttnn::operations::unary::UnaryWithParam>::
              convert(attr.getFusedActivation());
    } else {
      rso << "None";
    }
    rso << ")";

    return buf;
  }
};

// Specialization for MathFidelity enum
template <>
struct EmitPyTypeConverter<::ttnn::MathFidelity> {
  static std::string convert(ttnn::MathFidelity mathFidelity) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << "ttnn.MathFidelity.";
    switch (mathFidelity) {
    case ttnn::MathFidelity::LoFi:
      rso << "LoFi";
      break;
    case ttnn::MathFidelity::HiFi2:
      rso << "HiFi2";
      break;
    case ttnn::MathFidelity::HiFi3:
      rso << "HiFi3";
      break;
    case ttnn::MathFidelity::HiFi4:
      rso << "HiFi4";
      break;
    }
    return buf;
  }
};

template <typename ComputeKernelConfigTy>
static std::optional<std::string> convertDeviceComputeKernelConfig(
    ttnn::DeviceComputeKernelConfigAttr attr) {
  if (!attr) {
    return std::nullopt;
  }

  std::string buf;
  llvm::raw_string_ostream rso(buf);
  rso << TypeNameV<ComputeKernelConfigTy> << "(";

  bool first = true;

  // math_fidelity
  if (auto mathFidelity = attr.getMathFidelity()) {
    if (!first) {
      rso << ", ";
    }
    first = false;
    rso << "math_fidelity="
        << EmitPyTypeConverter<::ttnn::MathFidelity>::convert(*mathFidelity);
  }

  // math_approx_mode
  if (auto mathApproxMode = attr.getMathApproxMode()) {
    if (!first) {
      rso << ", ";
    }
    first = false;
    rso << "math_approx_mode=" << (mathApproxMode.getValue() ? "True" : "False");
  }

  // fp32_dest_acc_en
  if (auto fp32DestAccEn = attr.getFp32DestAccEn()) {
    if (!first) {
      rso << ", ";
    }
    first = false;
    rso << "fp32_dest_acc_en=" << (fp32DestAccEn.getValue() ? "True" : "False");
  }

  // packer_l1_acc
  if (auto packerL1Acc = attr.getPackerL1Acc()) {
    if (!first) {
      rso << ", ";
    }
    first = false;
    rso << "packer_l1_acc=" << (packerL1Acc.getValue() ? "True" : "False");
  }

  // dst_full_sync_en
  if (auto dstFullSyncEn = attr.getDstFullSyncEn()) {
    if (!first) {
      rso << ", ";
    }
    rso << "dst_full_sync_en=" << (dstFullSyncEn.getValue() ? "True" : "False");
  }

  rso << ")";
  return buf;
}

// Specialization for Wormhole DeviceComputeKernelConfig.
template <>
struct EmitPyTypeConverter<::ttnn::WormholeComputeKernelConfig> {
  static std::optional<std::string>
  convert(mlir::Attribute attr) {
    if (auto computeKernelConfigAttr =
            mlir::dyn_cast_if_present<ttnn::DeviceComputeKernelConfigAttr>(
                attr)) {
      return convert(computeKernelConfigAttr);
    }
    return {};
  }

  static std::optional<std::string>
  convert(ttnn::DeviceComputeKernelConfigAttr attr) {
    return convertDeviceComputeKernelConfig<::ttnn::WormholeComputeKernelConfig>(
        attr);
  }
};

// Specialization for Blackhole DeviceComputeKernelConfig.
template <>
struct EmitPyTypeConverter<::ttnn::BlackholeComputeKernelConfig> {
  static std::optional<std::string>
  convert(mlir::Attribute attr) {
    if (auto computeKernelConfigAttr =
            mlir::dyn_cast_if_present<ttnn::DeviceComputeKernelConfigAttr>(
                attr)) {
      return convert(computeKernelConfigAttr);
    }
    return {};
  }

  static std::optional<std::string>
  convert(ttnn::DeviceComputeKernelConfigAttr attr) {
    return convertDeviceComputeKernelConfig<::ttnn::BlackholeComputeKernelConfig>(
        attr);
  }
};

// Specialization for LayerNormShardedMultiCoreProgramConfig
template <>
struct EmitPyTypeConverter<
    ::ttnn::prim::LayerNormShardedMultiCoreProgramConfig> {
  static std::optional<std::string>
  convert(ttnn::LayerNormShardedMultiCoreProgramConfigAttr attr) {
    if (!attr) {
      return std::string("ttnn.LayerNormDefaultProgramConfig()");
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<
               ::ttnn::prim::LayerNormShardedMultiCoreProgramConfig> << "(";

    auto gridSize = attr.getComputeWithStorageGridSize();
    rso << "compute_with_storage_grid_size=(" << gridSize.getX() << ", "
        << gridSize.getY() << ")";
    rso << ", subblock_w=" << attr.getSubblockW();
    rso << ", block_h=" << attr.getBlockH();
    rso << ", block_w=" << attr.getBlockW();
    rso << ", inplace=" << (attr.getInplace() ? "True" : "False");

    rso << ")";
    return buf;
  }
};

// This template struct retrieves the most relevant C++ type with a one-to-one
// Python type correspondence for a given template type.
template <typename T>
struct TTNNTarget {
  using type = T;
};

template <typename T>
using TTNNTargetT = typename TTNNTarget<T>::type;

template <>
struct TTNNTarget<llvm::StringRef> {
  using type = std::string;
};

template <>
struct TTNNTarget<llvm::APFloat> {
  using type = double;
};

template <>
struct TTNNTarget<tt::ttnn::ShapeAttr> {
  using type = ::ttnn::Shape;
};

template <>
struct TTNNTarget<tt::ttcore::DataType> {
  using type = ::ttnn::DataType;
};

template <>
struct TTNNTarget<tt::ttcore::DataTypeAttr> {
  using type = ::ttnn::DataType;
};

template <>
struct TTNNTarget<tt::ttnn::BufferType> {
  using type = ::ttnn::BufferType;
};

template <>
struct TTNNTarget<tt::ttnn::BufferTypeAttr> {
  using type = ::ttnn::BufferType;
};

template <>
struct TTNNTarget<tt::ttnn::Layout> {
  using type = ::ttnn::Layout;
};

template <>
struct TTNNTarget<tt::ttnn::LayoutAttr> {
  using type = ::ttnn::Layout;
};

template <>
struct TTNNTarget<tt::ttnn::ShardSpecAttr> {
  using type = ::ttnn::ShardSpec;
};

template <>
struct TTNNTarget<tt::ttnn::TensorMemoryLayout> {
  using type = ::ttnn::TensorMemoryLayout;
};

template <>
struct TTNNTarget<tt::ttnn::TensorMemoryLayoutAttr> {
  using type = ::ttnn::TensorMemoryLayout;
};

template <>
struct TTNNTarget<tt::ttnn::MemoryConfigAttr> {
  using type = ::ttnn::MemoryConfig;
};

template <>
struct TTNNTarget<tt::ttnn::UnaryWithParamAttr> {
  using type = ::ttnn::operations::unary::UnaryWithParam;
};

template <>
struct TTNNTarget<mlir::tt::ttnn::Conv2dConfigAttr> {
  using type = ::ttnn::operations::conv::conv2d::Conv2dConfig;
};

template <>
struct TTNNTarget<mlir::tt::ttnn::Conv3dConfigAttr> {
  using type = ::ttnn::experimental::prim::Conv3dConfig;
};

template <>
struct TTNNTarget<tt::ttnn::Conv2dSliceConfigAttr> {
  using type = ::ttnn::operations::conv::conv2d::Conv2dSliceConfig;
};

template <>
struct TTNNTarget<tt::ttnn::DeviceComputeKernelConfigAttr> {
  using type = ::ttnn::WormholeComputeKernelConfig;
};

template <>
struct TTNNTarget<tt::ttnn::LayerNormShardedMultiCoreProgramConfigAttr> {
  using type = ::ttnn::prim::LayerNormShardedMultiCoreProgramConfig;
};

// Marker type for matmul program config union (AnyAttrOf<[...]>)
// Used with emit<MatmulProgramConfig>(attr, ...) to convert matmul program
// configs
struct MatmulProgramConfig {};

template <>
struct EmitPyTypeConverter<MatmulProgramConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return std::nullopt;
    }

    if (auto configAttr =
            mlir::dyn_cast<ttnn::MatmulMultiCoreReuseProgramConfigAttr>(attr)) {
      return EmitPyTypeConverter<
          ::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig>::
          convert(configAttr);
    }
    if (auto configAttr = mlir::dyn_cast<
            ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr>(attr)) {
      return EmitPyTypeConverter<
          ::ttnn::operations::matmul::
              MatmulMultiCoreReuseMultiCastProgramConfig>::convert(configAttr);
    }
    if (auto configAttr = mlir::dyn_cast<
            ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(attr)) {
      return EmitPyTypeConverter<
          ::ttnn::operations::matmul::
              MatmulMultiCoreReuseMultiCast1DProgramConfig>::
          convert(configAttr);
    }
    if (auto configAttr = mlir::dyn_cast<
            ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
            attr)) {
      return EmitPyTypeConverter<
          ::ttnn::operations::matmul::
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>::
          convert(configAttr);
    }

    return std::nullopt;
  }
};

template <typename T>
struct IsMLIRType {
  static constexpr bool value = std::is_convertible_v<T, mlir::Attribute> ||
                                std::is_convertible_v<T, mlir::Value>;
};

template <typename T>
static constexpr bool IsMLIRTypeV = IsMLIRType<T>::value;

// Name for the function that creates a list from a variadic number of
// `ttnn::Tensor`s.
inline constexpr const char *kCreateListFunctionName = "util_create_list";

// Name for the function that gets a scalar (int) from a `ttnn.Tensor`.
inline constexpr const char *kGetScalarFromTensorFunctionName =
    "utils.get_scalar_from_tensor";

template <typename TTNNOp>
class EmitPyTTNNEmitter {
public:
  using OpAdaptor = typename TTNNOp::Adaptor;

  EmitPyTTNNEmitter(TTNNOp op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter,
                    bool enableGoldenMode)
      : op{op}, adaptor{adaptor}, rewriter{rewriter},
        enableGoldenMode{enableGoldenMode} {}

  EmitPyTTNNEmitter(const EmitPyTTNNEmitter &) = delete;
  EmitPyTTNNEmitter &operator=(const EmitPyTTNNEmitter &) = delete;

  template <typename TargetTy = void, typename SourceTy>
  mlir::Attribute emit(std::optional<SourceTy> attr,
                       std::string attrName = "") {
    if (!attr) {
      return emit(std::nullopt, attrName);
    }

    if constexpr (std::is_void_v<TargetTy>) {
      return emit(*attr, attrName);
    } else {
      return emit<TargetTy>(*attr, attrName);
    }
  }

  mlir::Attribute emit(std::nullopt_t, std::string attrName = "") {
    addKeywordArgument(attrName);
    return rewriter.getType<emitpy::OpaqueAttr>(TypeNameV<std::nullopt_t>);
  }

  // The `val` should be either an operand of the current source operation, in
  // which case `index` should be nullopt, and the index it's found in the
  // operands list. If `index` is provided, it means that the `val` is not an
  // operand of the current source operation, and it is added as-is. Note that
  // providing an `index` for an operand of the current source operation will
  // result in an error.
  mlir::Attribute emit(mlir::Value val, std::string attrName = "",
                       std::optional<uint32_t> index = std::nullopt) {
    if (!val) {
      return emit(std::nullopt, attrName);
    }
    if (index) {
      operands.push_back(val);
      addKeywordArgument(attrName);
      return rewriter.getIndexAttr(*index);
    }

    unsigned trueIndex = getOperandIndex(val);
    operands.push_back(adaptor.getOperands()[trueIndex]);
    addKeywordArgument(attrName);
    return rewriter.getIndexAttr(operands.size() - 1);
  }

  mlir::Attribute emit(mlir::Operation::operand_range operandRange,
                       std::string attrName = "") {
    for (mlir::OpOperand &opOperand : op->getOpOperands()) {
      auto begin =
          std::next(op->getOperands().begin(), opOperand.getOperandNumber());
      if (mlir::Operation::operand_range(
              begin, std::next(begin, operandRange.size())) != operandRange) {
        continue;
      }
      unsigned index = opOperand.getOperandNumber();
      llvm::SmallVector<mlir::Value> values(
          adaptor.getOperands().begin() + index,
          adaptor.getOperands().begin() + index + operandRange.size());
      this->operands.push_back(createList(values));
      addKeywordArgument(attrName);
      return rewriter.getIndexAttr(this->operands.size() - 1);
    }
    llvm_unreachable("Invalid operand range");
  }

  mlir::Attribute emitMeshCoordinate(const mlir::ArrayRef<int64_t> &coords,
                                     std::string attrName = "") {
    std::string code = "ttnn.MeshCoordinate((";
    llvm::raw_string_ostream rso(code);
    llvm::interleaveComma(coords, rso);
    rso << "))";

    addKeywordArgument(attrName);
    return rewriter.getAttr<emitpy::OpaqueAttr>(rso.str());
  }

  mlir::Attribute emitSubDeviceId(std::optional<uint32_t> subDeviceId,
                                  std::string attrName = "") {
    if (!subDeviceId) {
      return emit(std::nullopt, attrName);
    }

    std::string code = "ttnn.SubDeviceId(";
    code += std::to_string(*subDeviceId);
    code += ")";

    addKeywordArgument(attrName);
    return rewriter.getAttr<emitpy::OpaqueAttr>(code);
  }

  // Emits Python code for ttnn.MeshShape from an array of IntegerAttr.
  // Returns empty string if meshShape is empty.
  std::string
  emitMeshShape(const mlir::ArrayRef<mlir::IntegerAttr> &meshShape) {
    if (meshShape.empty()) {
      return "";
    }
    std::string code = "ttnn.MeshShape([";
    llvm::raw_string_ostream rso(code);
    llvm::SmallVector<uint32_t> dims;
    for (const auto &intAttr : meshShape) {
      dims.push_back(intAttr.getValue().getZExtValue());
    }
    llvm::interleaveComma(dims, rso);
    rso << "])";
    return rso.str();
  }

  std::string
  emitComputeKernelConfig(std::optional<ttnn::DeviceComputeKernelConfigAttr> attr, std::string attrName = "") {
    if (!attr) {
      return emit(std::nullopt, attrName);
    }
    return emitComputeKernelConfig(*attr, attrName);
  }

  std::string
  emitComputeKernelConfig(ttnn::DeviceComputeKernelConfigAttr attr, std::string attrName = "") {
    auto moduleOp = op->template getParentOfType<mlir::ModuleOp>();
    auto systemDesc = moduleOp
                          ? moduleOp->template getAttrOfType<ttcore::SystemDescAttr>(
                                ttcore::SystemDescAttr::name)
                          : ttcore::SystemDescAttr();
    assert(systemDesc && "expected system desc to be present on the module");
    assert(!systemDesc.getChipDescs().empty() && "expected at least one chip desc");
    auto arch = systemDesc.getChipDesc(0).getArch().getValue();
    if (arch == ttcore::Arch::Blackhole) {
      return *emit<::ttnn::BlackholeComputeKernelConfig>(attr, attrName);
    }
    if (arch == ttcore::Arch::WormholeB0) {
      return *emit<::ttnn::WormholeComputeKernelConfig>(attr, attrName);
    }
    llvm_unreachable("unsupported architecture");
  }

  // Handles the case when a source type is convertible to `mlir::Attribute` and
  // there exists a `EmitPyTypeConverter` specialization for the TTNN target
  // type of the attribute.
  template <
      typename MLIRAttrTy, typename = std::void_t<TTNNTargetT<MLIRAttrTy>>,
      typename =
          std::enable_if_t<std::is_convertible_v<MLIRAttrTy, mlir::Attribute>>>
  mlir::Attribute emit(MLIRAttrTy attr, std::string attrName = "") {
    auto convertedValue =
        EmitPyTypeConverter<TTNNTargetT<MLIRAttrTy>>::convert(attr);

    addKeywordArgument(attrName);
    if constexpr (std::is_same_v<decltype(convertedValue), std::string>) {
      return rewriter.getType<emitpy::OpaqueAttr>(convertedValue);
    } else if (convertedValue) {
      return rewriter.getType<emitpy::OpaqueAttr>(*convertedValue);
    }
    // It's assumed that the conversion might fail, in which case the result
    // will be `emitpy::OpaqueAttr("::std::nullopt")`.
    return emit(std::nullopt, attrName);
  }

  // This is a special handling for cases when there is a many-to-many
  // relationship between source and target type (i.e. {mlir::ArrayAttr,
  // mlir::DenseI32ArrayAttr} to {std::vector<uint32_t>,
  // ttnn::SmallVector<int32_t>}).
  template <typename TargetTy>
  mlir::Attribute emit(mlir::Attribute attr, std::string attrName = "") {
    auto convertedValue = EmitPyTypeConverter<TargetTy>::convert(attr);

    addKeywordArgument(attrName);
    if constexpr (std::is_same_v<decltype(convertedValue), std::string>) {
      return rewriter.getType<emitpy::OpaqueAttr>(convertedValue);
    } else if (convertedValue) {
      return rewriter.getType<emitpy::OpaqueAttr>(*convertedValue);
    }
    // It's assumed that the conversion might fail, in which case the result
    // will be `emitpy::OpaqueAttr("::std::nullopt")`.
    return emit(std::nullopt, attrName);
  }

  // Handles the case when a source type is not convertible to
  // mlir::Attribute.
  template <typename SourceTy>
  std::enable_if_t<!IsMLIRTypeV<SourceTy>, mlir::Attribute>
  emit(SourceTy attr, std::string attrName = "") {
    using TargetTy = TTNNTargetT<llvm::remove_cvref_t<SourceTy>>;
    auto result = EmitPyTypeConverter<TargetTy>::convert(attr);
    // It's assumed that the conversion will always succeed, if the result is
    // `std::optional<std::string>` we assume that it contains the converted
    // value.
    addKeywordArgument(attrName);
    if constexpr (std::is_same_v<decltype(result),
                                 std::optional<std::string>>) {
      return rewriter.getType<emitpy::OpaqueAttr>(*result);
    }
    return rewriter.getType<emitpy::OpaqueAttr>(result);
  }

  // This is a temporary solution for handling the case when
  // the value of the MemoryConfigAttr is nullptr. This should be removed once
  // https://github.com/tenstorrent/tt-mlir/issues/2415 lands.
  ttnn::MemoryConfigAttr getMemoryConfig(mlir::Value val) {
    auto deviceOp = ttcore::lookupDeviceOp(op);

    if (!deviceOp) {
      // We're inside a CPU module, so no memory config is needed.
      return ttnn::MemoryConfigAttr{};
    }

    auto layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(val.getType()).getEncoding());

    ttnn::BufferTypeAttr bufferTypeAttr = ttnn::BufferTypeAttr::get(
        layoutAttr.getContext(), layoutAttr.getBufferType());
    ttnn::TensorMemoryLayoutAttr tensorMemoryLayout = layoutAttr.getMemLayout();

    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        layoutAttr.getContext(), tensorMemoryLayout, bufferTypeAttr,
        ttnn::utils::createShardSpecIfNeeded(
            layoutAttr, deviceOp.getDeviceAttr().getWorkerGrid()));

    return memoryConfigAttr;
  }

  template <typename OpConversionPatternTy>
  mlir::Value replaceOp(OpConversionPatternTy &&opConversionPattern,
                        llvm::ArrayRef<mlir::Attribute> args) {
    auto resultTypes = llvm::to_vector(
        llvm::map_to_vector(op->getResultTypes(), [&](Type type) -> Type {
          return opConversionPattern.getTypeConverter()->convertType(type);
        }));

    auto callee = opConversionPattern.convertOpName(op);

    if (enableGoldenMode) {
      callee += ".golden_function";
    }

    auto callOpaqueOp = rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultTypes, callee, operands, rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywordArgs));

    if (callOpaqueOp.getNumResults() == 0) {
      return {};
    }
    return callOpaqueOp.getResult(0);
  }

private:
  void addKeywordArgument(std::string attrName) {
    StringAttr keywordArg = rewriter.getStringAttr(attrName);
    keywordArgs.push_back(keywordArg);
  }

  mlir::Value createList(ValueRange operands) {
    return rewriter
        .create<emitpy::CallOpaqueOp>(
            op.getLoc(),
            emitpy::OpaqueType::get(rewriter.getContext(),
                                    TypeNameV<std::vector<::ttnn::Tensor>>),
            kCreateListFunctionName, operands, nullptr, nullptr)
        ->getResult(0);
  }

  unsigned getOperandIndex(mlir::Value value) {
    mlir::OpOperand *opOperand = std::find_if(
        op->getOpOperands().begin(), op->getOpOperands().end(),
        [&](OpOperand &operand) { return operand.get() == value; });

    return opOperand->getOperandNumber();
  }

  TTNNOp op;
  OpAdaptor adaptor;
  ConversionPatternRewriter &rewriter;
  llvm::SmallVector<mlir::Value> operands;
  llvm::SmallVector<mlir::Attribute> keywordArgs;
  bool enableGoldenMode;
};

// Helper function to secure memory config attribute.
// Currently, memory config is an optional attribute. If the attribute is
// explicitly provided by an op, it is used directly. Otherwise, the attribute
// is deduced from a tensor output layout attribute in the `getMemoryConfig`
// function.
inline ttnn::MemoryConfigAttr
operator|(std::optional<ttnn::MemoryConfigAttr> lhs,
          ttnn::MemoryConfigAttr rhs) {
  if (!lhs) {
    return rhs;
  }
  return *lhs;
}

// Helper function that serves as an alternative to the
// `emit<std::variant<...>>` member function of the `EmitPyTTNNEmitter` class.
// For example, instead of calling `emit<std::variant<int32_t, float>>(attr)`,
// one can call `emit<int32_t>(attr) | emit<float>(attr)`.
inline mlir::Attribute operator|(mlir::Attribute lhs, mlir::Attribute rhs) {
  const mlir::Attribute nulloptAttr = emitpy::OpaqueAttr::get(
      lhs.getContext(), tt::ttnn_to_emitpy::TypeNameV<std::nullopt_t>);
  if (!lhs || lhs == nulloptAttr) {
    return rhs;
  }
  return lhs;
}

} // namespace ttnn_to_emitpy
} // namespace tt
} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H
