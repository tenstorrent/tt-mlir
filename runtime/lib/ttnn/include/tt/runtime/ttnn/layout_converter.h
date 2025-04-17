// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_LAYOUT_CONVERTER_H
#define TT_RUNTIME_TTNN_LAYOUT_CONVERTER_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"

namespace tt::runtime::ttnn {

class LayoutConverter {
public:
  LayoutDesc inputDesc;
  LayoutDesc outputDesc;
  bool shouldTilize = false;
  bool shouldUntilize = false;
  bool shouldTypecast = false;
  bool shouldToDevice = false;
  bool shouldToMemoryConfig = false;
  bool shouldFromDevice = false;

  LayoutConverter(const LayoutDesc &inputDesc, const LayoutDesc &outputDesc);
  ::ttnn::Tensor convertTensorLayout(const ::ttnn::Tensor &input,
                                     std::optional<DeviceVariant> targetDevice);

private:
  bool canTilizeDataTypeOnDevice(const ::ttnn::DataType &dataType) const;
  bool canUntilizeDataTypeOnDevice(const ::ttnn::DataType &dataType) const;
  ::ttnn::Tensor toLayoutIfNeeded(const ::ttnn::Tensor &input);
  ::ttnn::Tensor typecastIfNeeded(const ::ttnn::Tensor &input);
  ::ttnn::Tensor toDeviceIfNeeded(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice,
                                  bool force = false);
  ::ttnn::Tensor toMemoryConfigIfNeeded(const ::ttnn::Tensor &input);
  ::ttnn::Tensor fromDeviceIfNeeded(const ::ttnn::Tensor &input);

  ::ttnn::Tensor
  handleHostInputNoLayoutNoTypecast(const ::ttnn::Tensor &input,
                                    std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  handleHostInputLayoutNoTypecast(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  handleHostInputNoLayoutTypecast(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  handleHostInputLayoutTypecast(const ::ttnn::Tensor &input,
                                std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  convertHostTensorLayout(const ::ttnn::Tensor &input,
                          std::optional<DeviceVariant> targetDevice);

  ::ttnn::Tensor
  handleDeviceInputNoLayoutNoTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor handleDeviceInputLayoutNoTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor handleDeviceInputNoLayoutTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor handleDeviceInputLayoutTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor convertDeviceTensorLayout(const ::ttnn::Tensor &input);
};
} // namespace tt::runtime::ttnn

#endif // TT_RUNTIME_TTNN_TYPES_H
