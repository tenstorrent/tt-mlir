#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
namespace NAMESPACE {
void kernel_main() {
  ::tt::CB v1 = ::tt::CB::c_in0;
  ::tt::CB v2 = ::tt::CB::c_in1;
  ::tt::CB v3 = ::tt::CB::c_out0;
  int32_t v4 = 4;
  int32_t v5 = 1;
  int32_t v6 = 2;
  int32_t v7 = 0;
  binary_op_init_common(v1, v2, v3);
  mul_tiles_init(v1, v2);
  int32_t v8;
  v8 = v7;
  for (int32_t v9 = v7; v9 < v6; v9 += v5) {
    int32_t v10;
    v10 = v8;
    for (int32_t v11 = v7; v11 < v4; v11 += v5) {
      tile_regs_acquire();
      mul_tiles(v1, v2, v10, v10, v7);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile(v7, v3, v10);
      tile_regs_release();
      uint32_t v12 = (uint32_t) v10;
      uint32_t v13 = (uint32_t) v5;
      uint32_t v14 = v12 + v13;
      int32_t v15 = (int32_t) v14;
      v10 = v15;
    };
    v8 = v10;
  }
  return;
}

void MAIN { kernel_main(); }
}

