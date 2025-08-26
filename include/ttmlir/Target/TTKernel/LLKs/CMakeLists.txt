# Include the hex header generation script
include(GenerateRawStringHeader)

# Define the LLK header files to convert to hex
set(LLK_HEADERS
    ${CMAKE_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_tilize_llks.h
    ${CMAKE_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_untilize_llks.h
    ${CMAKE_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_invoke_sfpi_llks.h
    ${CMAKE_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_dataflow_api.h
    ${CMAKE_SOURCE_DIR}/include/ttmlir/Target/TTKernel/LLKs/experimental_matmul_llks.h
)

# Set the output directory for generated headers
set(GENERATED_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Generate hex header files
set(GENERATED_LLK_HEADERS)
foreach(llk_header ${LLK_HEADERS})
    get_filename_component(header_name ${llk_header} NAME_WE)
    set(output_file ${GENERATED_HEADERS_DIR}/${header_name}_generated.h)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${llk_header}
            -DOUTPUT_FILE=${output_file}
            -DVARIABLE_NAME=${header_name}_generated
            -P ${PROJECT_SOURCE_DIR}/cmake/modules/GenerateRawStringHeader.cmake
        DEPENDS ${llk_header}
        COMMENT "Generating header ${header_name}_generated.h"
        VERBATIM
    )
    list(APPEND GENERATED_LLK_HEADERS ${output_file})
endforeach()

#Create a target for the generated headers
add_custom_target(TTKernelGeneratedLLKHeaders DEPENDS ${GENERATED_LLK_HEADERS})
