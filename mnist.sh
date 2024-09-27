#!/bin/bash

./build/bin/ttmlir-opt --ttir-load-system-desc="path=/localdev/odjuricic/repos/tt-mlir/ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline="sharding-pass-enabled=true override-output-layout=\
matmul_1=1x8:l1:width_sharded,\
add_2=1x8:l1:width_sharded,\
relu_3=1x8:l1:width_sharded,\
matmul_5=8x8:dram:interleaved,\
add_6=8x8:dram:interleaved\
" test/ttmlir/Dialect/TTNN/mnist_sharding.mlir -o mnist_ttnn.mlir && \
cat mnist_ttnn.mlir && \
./build/bin/ttmlir-translate --ttnn-to-flatbuffer mnist_ttnn.mlir -o mnist.ttnn && \
ttrt run mnist.ttnn
# ttrt --disable-async run mnist.ttnn

