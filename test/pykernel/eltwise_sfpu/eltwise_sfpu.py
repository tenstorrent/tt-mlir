# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *


@ttkernel_compile
def eltwise_sfpu(cb_in: int, cb_out: int):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%[[arg0:.*]]: !ttkernel.cb<{{.*}}>, %[[arg1:.*]]: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}"ttkernel.get_compile_time_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_compile_time_arg_val"{{.*}}
    per_core_block_cnt = get_compile_time_arg_val(int, 0)
    per_core_block_dim = get_compile_time_arg_val(int, 1)

    # CHECK: "ttkernel.unary_op_init_common"(%[[arg0]], %[[arg1]]){{.*}}
    unary_op_init_common(cb_in, cb_out)

    for i in range(0, per_core_block_cnt, 1):
        # CHECK: "ttkernel.cb_reserve_back"{{.*}}
        cb_reserve_back(cb_out, per_core_block_dim)
        for j in range(0, per_core_block_dim, 1):
            # CHECK: "ttkernel.tile_regs_acquire"(){{.*}}
            # CHECK: "ttkernel.cb_wait_front"{{.*}}
            tile_regs_acquire()
            cb_wait_front(cb_in, 1)

            # CHECK: "ttkernel.copy_tile"{{.*}}
            # CHECK: "ttkernel.pack_tile"{{.*}}
            copy_tile(cb_in, 0, 0)
            pack_tile(0, cb_out, 0)
            # last parameter is a optional index in C++ but not optional in pybind

            # CHECK: "ttkernel.cb_pop_front"{{.*}}
            # CHECK: "ttkernel.tile_regs_release"(){{.*}}
            cb_pop_front(cb_in, 1)
            tile_regs_release()

        # CHECK: "ttkernel.cb_push_back"{{.*}}
        cb_push_back(cb_out, per_core_block_dim)

    return


eltwise_sfpu(0, 16)
