#pragma once

#ifdef DEBUG
inline constexpr bool build_debug = true;
inline constexpr bool build_release = false;
#else
inline constexpr bool build_debug = false;
inline constexpr bool build_release = true;
#endif
