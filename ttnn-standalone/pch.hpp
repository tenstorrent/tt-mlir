// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "tensor/types.hpp"
#include "ttnn/core.hpp"
#include "ttnn/cpp/ttnn/tensor/tensor.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
// #include "ttnn/cpp/ttnn/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/types.hpp"

#include <cstddef>
#include <iostream>
#include <vector>
