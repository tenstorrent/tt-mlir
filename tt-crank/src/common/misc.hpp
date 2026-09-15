// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

// Monitor condition slots.
constexpr size_t mc_dbg = 0;
constexpr size_t mc_comp_cache = 1;
constexpr size_t mc_comp_cache_on_disk = 2;

bool monitor(size_t slot, bool cond);
void print_monitor(size_t slot);
void print_monitor();

template <size_t slot = mc_dbg> bool monitor(bool cond = true) {
    return monitor(slot, cond);
}
