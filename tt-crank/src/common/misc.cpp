#include "misc.hpp"
#include "assert.hpp"
#include "cast.hpp"
#include "config.hpp"

#include <atomic>
#include <cstdlib>
#include <tt-logger/tt-logger.hpp>

std::string slot_str(size_t slot) {
    switch (slot) {
        case mc_dbg:
            return "dbg";
        default:
            return std::format("slot {}", slot);
    }
}

constexpr size_t max_slots = cfg_monitor_max_slots;
using MonitorEntry = std::array<std::atomic<size_t>, 2>;
using Monitor = std::array<MonitorEntry, max_slots>;

Monitor global_monitor{};

bool monitor(size_t slot, bool cond) {
    TT_ASSERT(slot < max_slots, "Monitor slot out of bounds: {}.", slot);
    global_monitor[slot][0].fetch_add(1, std::memory_order_relaxed); // NOLINT
    if (cond) {
        global_monitor[slot][1].fetch_add(1, std::memory_order_relaxed); // NOLINT
    }

    if (monitor_verbose_print_enabled()) {
        print_monitor(slot);
    }

    return cond;
}

void print_monitor(size_t slot) {
    TT_ASSERT(slot < max_slots, "Monitor slot out of bounds: {}.", slot);
    size_t total = global_monitor[slot][0].load(std::memory_order_relaxed); // NOLINT
    size_t hits = global_monitor[slot][1].load(std::memory_order_relaxed);  // NOLINT

    if (total > 0) {
        log_info(tt::LogAlways, "Monitor {}: Total {} Hits {} Hit Rate {:.2f}%", slot_str(slot), total, hits,
                 100.0 * as<double>(hits) / as<double>(total));
    }
}

void print_monitor() {
    for (size_t i = 0; i < max_slots; ++i) {
        print_monitor(i);
    }
}
