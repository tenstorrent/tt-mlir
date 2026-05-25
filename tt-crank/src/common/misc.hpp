#pragma once

#include <cstddef>

// Monitor condition slots.
constexpr size_t mc_dbg = 0;

bool monitor(size_t slot, bool cond);
void print_monitor(size_t slot);
void print_monitor();

template <size_t slot = mc_dbg> bool monitor(bool cond = true) {
    return monitor(slot, cond);
}
