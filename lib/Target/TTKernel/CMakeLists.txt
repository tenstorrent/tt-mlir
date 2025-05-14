# Include the hex header generation script
include(GenerateLLKHeaders)

# Define the LLK header files to convert to hex
set(LLK_HEADERS
    ${CMAKE_SOURCE_DIR}/include/ttmlir/LLKs/experimental_tilize_llks.h
    ${CMAKE_SOURCE_DIR}/include/ttmlir/LLKs/experimental_untilize_llks.h
)

# Set the output directory for generated headers
set(GENERATED_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Generate hex header files
set(GENERATED_LLK_HEADERS)
foreach(llk_header ${LLK_HEADERS})
    get_filename_component(header_name ${llk_header} NAME_WE)
    set(output_file ${GENERATED_HEADERS_DIR}/${header_name}_generated.h)
    generate_llk_header(${llk_header} ${output_file} ${header_name}_generated)
    list(APPEND GENERATED_LLK_HEADERS ${output_file})
endforeach()

# Create a target for the generated headers
add_custom_target(TTKernelGeneratedLLKHeaders DEPENDS ${GENERATED_LLK_HEADERS})

add_mlir_translation_library(TTKernelTargetCpp
    TTKernelToCpp.cpp
    TTKernelToCppRegistration.cpp
    ${GENERATED_LLK_HEADERS}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/TTKernel

    LINK_LIBS PUBLIC
    MLIRTTKernelDialect
    MLIRTTIRDialect
)

# Add dependency to ensure headers are generated before the library is built
add_dependencies(TTKernelTargetCpp TTKernelGeneratedLLKHeaders)

# Add the generated headers directory to the include path
target_include_directories(TTKernelTargetCpp PRIVATE ${GENERATED_HEADERS_DIR})
