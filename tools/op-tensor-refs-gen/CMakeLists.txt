find_package(flatbuffers CONFIG REQUIRED)

add_executable(op-tensor-refs-gen main.cpp)
target_link_libraries(op-tensor-refs-gen PRIVATE flatbuffers::flatbuffers)
set_target_properties(op-tensor-refs-gen PROPERTIES CXX_STANDARD 17)
