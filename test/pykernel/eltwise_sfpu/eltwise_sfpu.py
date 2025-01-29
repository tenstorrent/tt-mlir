# // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# //
# // SPDX-License-Identifier: Apache-2.0

# #include <cstdint>
# #include "compute_kernel_api/common.h"
# #include "compute_kernel_api/tile_move_copy.h"
# #include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
# #include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

# namespace NAMESPACE {
# void MAIN {
#     uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
#     uint32_t per_core_block_dim = get_compile_time_arg_val(1);

#     init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
#     for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
#         cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
#         for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
#             acquire_dst();

#             // Pop tile after tile, copy to DST and pack
#             cb_wait_front(tt::CBIndex::c_0, 1);

#             copy_tile(tt::CBIndex::c_0, 0, 0);

# #ifdef SFPU_OP_CHAIN_0
#             SFPU_OP_CHAIN_0
# #endif

#             pack_tile(0, tt::CBIndex::c_16);

#             cb_pop_front(tt::CBIndex::c_0, 1);

#             release_dst();
#         }
#         cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
#     }
# }
# }  // namespace NAMESPACE

from pykernel.pykernel_ast import *


@ttkernel_compile
def eltwise_sfpu(cb_in: int, cb_out: int):
    per_core_block_cnt = 1  # get_compile_time_arg_val(0)                 # get_compile_time_arg_val not implemented
    per_core_block_dim = 1  # get_compile_time_arg_val(1)

    unary_op_init_common(
        cb_in, cb_out
    )  # init_sfpu is wrapper around unary_op_init_ocmmon - these are both CBIndex
    for i in range(0, per_core_block_cnt, 1):
        cb_reserve_back(cb_out, per_core_block_dim)
        for j in range(0, per_core_block_dim, 1):
            tile_regs_acquire()  # acquire_dst() is deprecated, use tile_regs_acquire instead
            cb_wait_front(cb_in, 1)

            copy_tile(cb_in, 0, 0)
            pack_tile(
                0, cb_out, 0
            )  # last parameter is a optional index in c++ but not optional in pybind

            cb_pop_front(cb_in, 1)
            tile_regs_release()

        cb_push_back(cb_out, per_core_block_dim)

    return


eltwise_sfpu(0, 16)
