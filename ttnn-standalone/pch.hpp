// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "tensor/types.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/binary.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/types.hpp"

#include <cstddef>
#include <iostream>
#include <vector>
