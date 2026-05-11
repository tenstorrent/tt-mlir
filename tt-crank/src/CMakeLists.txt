configure_file(
    version.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    @ONLY
)

add_library(tt_kurbla
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    engine/mlir_smoke.cpp
)
add_library(tt_kurbla::tt_kurbla ALIAS tt_kurbla)

target_include_directories(tt_kurbla
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

target_compile_features(tt_kurbla PUBLIC cxx_std_20)

target_link_libraries(tt_kurbla
    PUBLIC
        tt_mlir::compiler
    PRIVATE
        tt_kurbla_warnings
        tt_kurbla_sanitizers
)
