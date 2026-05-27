#pragma once

#include "assert.hpp"
#include "preproc.hpp"
#include <type_traits>

// Checked cast in debug build.
template <class To, class From> [[nodiscard]] constexpr To checked_cast(const From &from) {
    if constexpr (build_debug) {
        if constexpr (std::is_signed_v<To> && std::is_unsigned_v<From>) {
            TT_ASSERT(static_cast<To>(from) >= 0, "Overflow error in cast (unsigned to signed): {} -> {}", from,
                      static_cast<To>(from));
        }

        if constexpr (std::is_unsigned_v<To> && std::is_signed_v<From>) {
            TT_ASSERT(from >= 0, "Overflow error in cast (signed to unsigned): {} -> {}", from, static_cast<To>(from));
        }

        TT_ASSERT(static_cast<From>(static_cast<To>(from)) == from, "Data lost after cast: {} -> {}", from,
                  static_cast<To>(from));
    }

    return static_cast<To>(from);
}

// Create a value of type To from the bits of from.
template <class To, class From> [[nodiscard]] constexpr To as(const From &from) {
    if constexpr (std::is_same_v<From, To>) {
        return from;
    } else if constexpr (std::is_pointer_v<From> || std::is_pointer_v<To> || std::is_reference_v<From> ||
                         std::is_reference_v<To>) {
        return std::bit_cast<To>(from);
    } else if constexpr (std::is_floating_point_v<From> && std::is_floating_point_v<To>) {
        // FP-to-FP narrowing conversions are lossy by design - skip the data loss check.
        return static_cast<To>(from);
    } else {
        return checked_cast<To>(from);
    }
}
