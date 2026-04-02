
// #include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
// #include "ttmlir/Target/TTNN/operations/conv_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#pragma clang diagnostic pop
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
// #include "ttmlir/Target/Common/types_generated.h"

namespace unifiedOpLib::operations::utils {

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType);

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout);

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout);

MathFidelity toTTNNMathFidelity(::tt::target::MathFidelity mathFidelity);

tt::tt_metal::CoreCoord
toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord);

tt::tt_metal::CoreRange
toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange);

tt::tt_metal::CoreRangeSet
toTTNNCoreRangeSet(const tt::target::ttnn::CoreRangeSetT &coreRangeSet);

::ttnn::operations::unary::UnaryOpType
toTTNNUnaryOpType(::tt::target::ttnn::UnaryOpType unaryOpType);

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType);

::ttnn::operations::unary::UnaryOpType
toTTNNUnaryOpType(::tt::target::ttnn::EltwiseUnaryOpType unaryOpType);

::ttnn::operations::unary::UnaryWithParam
toTTNNUnaryWithParam(const ::tt::target::ttnn::UnaryWithParamT &unaryWithParam);

bool inSystemMemory(const ::tt::target::ttnn::TensorRefT &tensorRef);

const ::tt::target::ttnn::MemoryConfigT
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRefT &tensorRef);

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfigT &memcfg);

::ttnn::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfigT &config);

::ttnn::Conv2dSliceConfig
createConv2dSliceConfig(const ::tt::target::ttnn::Conv2dSliceConfigT &config);

::ttnn::DeviceComputeKernelConfig createDeviceComputeKernelConfig(
    const ::tt::target::ttnn::DeviceComputeKernelConfigT &config);

} // namespace unifiedOpLib::operations::utils