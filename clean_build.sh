rm build/ third_party/tt-metal/src/tt-metal-build/ -rf
source venv/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DTTMLIR_ENABLE_RUNTIME=ON
cmake --build build
