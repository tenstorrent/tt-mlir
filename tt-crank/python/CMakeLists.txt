# The python side of tt-crank: what the packages in this directory need from the
# C++ build, in both installation modes (editable/wheel).
#
# Structure of the wheel (*.py files are handled by `sci-kit`, `pyproject.toml`):
#   tt_crank/
#     __init__.py, _runtime_env.py, _wrapper.py, install_sfpi.py
#     torch/        *.py, _native.cpython-*.so
#     lib/          all shared libs needed from `tt-mlir` and `tt-metal`
#     bin/          tracy-capture, tracy-csvexport
#     tt-metal/     all files needed at runtime from `tt-metal`
#   ttnn/           bundled `ttnn` python package
#   tracy/          bundled `tracy` python package

# editable: the wrappers at tracy/__init__.py and ttnn/__init__.py re-export tt-metal's
# python modules through sibling `_original/` symlinks, refreshed on every build.
# `cmake -E create_symlink` fails if the link already exists, hence the rm.
function(_tt_crank_add_bundle_target target_name src lnk)
    add_custom_target(${target_name} ALL
        COMMAND ${CMAKE_COMMAND} -E rm -f "${lnk}"
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${src}" "${lnk}"
        BYPRODUCTS "${lnk}"
        COMMENT "Symlinking ${lnk} -> ${src}"
        VERBATIM
    )
    add_dependencies(${target_name} TTMLIRCompiler TTMLIRRuntime)
endfunction()

_tt_crank_add_bundle_target(bundle-tracy
    "${TT_CRANK_TT_METAL_HOME}/tools/tracy" "${CMAKE_CURRENT_SOURCE_DIR}/tracy/_original")
_tt_crank_add_bundle_target(bundle-ttnn
    "${TT_CRANK_TT_METAL_HOME}/ttnn/ttnn" "${CMAKE_CURRENT_SOURCE_DIR}/ttnn/_original")

if(TT_CRANK_BUILD_PYTHON_EXT)
    add_custom_target(tt_crank_python)
    add_dependencies(tt_crank_python _tt_crank_torch_native bundle-tracy bundle-ttnn)

    # editable only: create a symlink to our native lib in the directory where
    # the python sources are located, so that python can find it.
    install(CODE "file(CREATE_LINK \"$<TARGET_FILE:_tt_crank_torch_native>\"
        \"${CMAKE_CURRENT_SOURCE_DIR}/tt_crank/torch/$<TARGET_FILE_NAME:_tt_crank_torch_native>\" SYMBOLIC)"
        COMPONENT tt_crank_python
        EXCLUDE_FROM_ALL
    )

    install(TARGETS _tt_crank_torch_native
        LIBRARY
            DESTINATION tt_crank/torch
            COMPONENT wheel_payload
            EXCLUDE_FROM_ALL
    )
endif()

# wheel: tt-crank's and tt-mlir's libs; install(TARGETS) rewrites their RPATH to
# $ORIGIN. TTMLIRRuntime carries PUBLIC_HEADERs, so we include those as well.
install(TARGETS tt_crank tt_crank_common TTMLIRCompiler TTMLIRRuntime
    LIBRARY
        DESTINATION tt_crank/lib
        COMPONENT wheel_payload
        EXCLUDE_FROM_ALL
    PUBLIC_HEADER
        DESTINATION tt_crank/include
        EXCLUDE_FROM_ALL
)

# wheel: tt-metal's libs and tracy CLI binaries, the same set tt-mlir installs
# (third_party/CMakeLists.txt).
install(CODE "file(INSTALL ${TTNN_LIBRARY_PATH} ${TTNN_PY_LIBRARY_PATH}
    ${TTMETAL_LIBRARY_PATH} ${DEVICE_LIBRARY_PATH} ${TT_STL_LIBRARY_PATH} ${TRACY_LIBRARY_PATH}
    DESTINATION \${CMAKE_INSTALL_PREFIX}/tt_crank/lib TYPE SHARED_LIBRARY FOLLOW_SYMLINK_CHAIN)"
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
)
install(CODE "file(INSTALL ${TRACY_BINARY_DIR}/tracy-capture ${TRACY_BINARY_DIR}/tracy-csvexport
    DESTINATION \${CMAKE_INSTALL_PREFIX}/tt_crank/bin TYPE SHARED_LIBRARY USE_SOURCE_PERMISSIONS FOLLOW_SYMLINK_CHAIN)"
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
)

# wheel: tt-metal part of the tree needed in runtime.
install(DIRECTORY ${TT_CRANK_TT_METAL_HOME}/ttnn ${TT_CRANK_TT_METAL_HOME}/tools ${TT_CRANK_TT_METAL_HOME}/runtime
    DESTINATION tt_crank/tt-metal
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
    PATTERN "sfpi" EXCLUDE
    PATTERN "_ttnn.so" EXCLUDE
)
install(DIRECTORY ${TT_CRANK_TT_METAL_HOME}/tt_metal
    DESTINATION tt_crank/tt-metal
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
    PATTERN "third_party" EXCLUDE
)
install(DIRECTORY ${TT_CRANK_TT_METAL_HOME}/tt_metal
    DESTINATION tt_crank/tt-metal
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
    FILES_MATCHING REGEX "third_party/umd/"
)
file(RELATIVE_PATH _ttml_rel ${TT_CRANK_TT_METAL_HOME} ${TTML_SOURCE_DIR})
install(DIRECTORY ${TTML_SOURCE_DIR}/metal
    DESTINATION tt_crank/tt-metal/${_ttml_rel}
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
)

# wheel: copy the bundled `ttnn` and `tracy` python packages from `tt-metal`.
install(DIRECTORY ${TT_CRANK_TT_METAL_HOME}/ttnn/ttnn/
    DESTINATION ttnn/_original
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
    PATTERN "_ttnn.so" EXCLUDE
)
install(DIRECTORY ${TT_CRANK_TT_METAL_HOME}/tools/tracy/
    DESTINATION tracy/_original
    USE_SOURCE_PERMISSIONS
    COMPONENT wheel_payload
    EXCLUDE_FROM_ALL
)
