# Source third_party/tt-mlir/env/activate so callers don't have to. Idempotent —
# short-circuits if TTMLIR_ENV_ACTIVATED is already set (e.g. scripts/build).
#
# We capture env/activate's variables in *two* places:
#
#   1. set(ENV{...}) — visible to the parent CMake process. Lets the existing
#      `if(NOT DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})` check in third_party/
#      CMakeLists.txt pass at our configure time.
#
#   2. TT_KURBLA_MLIR_ENV — a list of KEY=VALUE entries that third_party/
#      CMakeLists.txt prepends to ExternalProject's subprocess commands via
#      `${CMAKE_COMMAND} -E env`. This is the load-bearing part: ExternalProject
#      spawns tt-mlir's own cmake at *build time* in a process that inherits
#      env from ninja, not from our configure-time cmake — so set(ENV) alone
#      is invisible there.

set(TT_KURBLA_MLIR_ENV "" CACHE INTERNAL
    "env/activate's KEY=VALUE entries, forwarded to tt-mlir-ep subprocesses")

if(DEFINED ENV{TTMLIR_ENV_ACTIVATED})
    # Already activated by the caller (scripts/build, CI). Forward what's there
    # so ExternalProject still sees consistent env at build time.
    foreach(_var TTMLIR_TOOLCHAIN_DIR TTMLIR_VENV_DIR TTMLIR_ENV_ACTIVATED
                 TT_MLIR_HOME TT_METAL_HOME TT_METAL_RUNTIME_ROOT
                 TT_METAL_BUILD_HOME PATH)
        if(DEFINED ENV{${_var}})
            list(APPEND TT_KURBLA_MLIR_ENV "${_var}=$ENV{${_var}}")
        endif()
    endforeach()
    set(TT_KURBLA_MLIR_ENV "${TT_KURBLA_MLIR_ENV}" CACHE INTERNAL "" FORCE)
    return()
endif()

set(_activate "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/env/activate")
if(NOT EXISTS "${_activate}")
    message(FATAL_ERROR
        "tt-mlir submodule not initialized — `${_activate}` is missing. "
        "Run `git submodule update --init --recursive`.")
endif()

message(STATUS "Sourcing tt-mlir env/activate")
execute_process(
    COMMAND bash -c "
        cd '${CMAKE_SOURCE_DIR}/third_party/tt-mlir' || exit 1
        set +u
        _ACTIVATE_SUPPRESS_INIT_WARNING=1 source env/activate >&2
        env
    "
    OUTPUT_VARIABLE _ENV_DUMP
    RESULT_VARIABLE _RC
)
if(NOT _RC EQUAL 0)
    message(FATAL_ERROR "Sourcing third_party/tt-mlir/env/activate failed.")
endif()

# env emits one KEY=VALUE per line. env/activate doesn't produce multi-line
# values, so a simple newline split is enough.
string(REPLACE "\n" ";" _entries "${_ENV_DUMP}")
foreach(_kv IN LISTS _entries)
    if(_kv MATCHES "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
        set(_name  "${CMAKE_MATCH_1}")
        set(_value "${CMAKE_MATCH_2}")
        # Skip vars that would override the caller's Python env (env/activate
        # sources tt-mlir's own toolchain venv, which sets VIRTUAL_ENV/PYTHONPATH).
        if(_name MATCHES "^(VIRTUAL_ENV|PYTHONPATH|_|PS1|OLDPWD|PWD|SHLVL)$")
            continue()
        endif()
        set(ENV{${_name}} "${_value}")
        list(APPEND TT_KURBLA_MLIR_ENV "${_name}=${_value}")
    endif()
endforeach()

set(TT_KURBLA_MLIR_ENV "${TT_KURBLA_MLIR_ENV}" CACHE INTERNAL "" FORCE)
