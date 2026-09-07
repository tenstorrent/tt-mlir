# Bundle tt-metal's Python modules (tracy CLI and ttnn) into tt-crank.
#
# tt-mlir's SharedLib install (third_party/CMakeLists.txt) copies the tt-metal
# source tree under <install>/tt-metal/, which gives us on-disk copies of:
#   - tools/tracy/  (the tracy CLI module)
#   - ttnn/ttnn/    (the ttnn Python package, including _ttnn.so)
# The wrappers at python/tracy/__init__.py and python/ttnn/__init__.py expect
# those directories to be reachable via sibling `_original/` symlinks
# (mirroring tt-xla's layout). The custom targets below create and refresh
# the symlinks on every build.
#
# Unconditional: bundling is always-on, independent of TT_CRANK_TRACY_ZONES.

function(_tt_crank_add_bundle_target target_name src lnk)
    # `cmake -E create_symlink` fails if the link already exists, and these
    # targets run on every build (ALL). The rm step keeps the operation
    # idempotent.
    add_custom_target(${target_name} ALL
        COMMAND ${CMAKE_COMMAND} -E rm -f "${lnk}"
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${src}" "${lnk}"
        BYPRODUCTS "${lnk}"
        COMMENT "Symlinking ${lnk} -> ${src}"
        VERBATIM
    )
    add_dependencies(${target_name} tt-mlir-ep)
endfunction()

_tt_crank_add_bundle_target(bundle-tracy
    "${TT_CRANK_TTMLIR_INSTALL_DIR}/tt-metal/tools/tracy"
    "${PROJECT_SOURCE_DIR}/python/tracy/_original"
)

_tt_crank_add_bundle_target(bundle-ttnn
    "${TT_CRANK_TTMLIR_INSTALL_DIR}/tt-metal/ttnn/ttnn"
    "${PROJECT_SOURCE_DIR}/python/ttnn/_original"
)
