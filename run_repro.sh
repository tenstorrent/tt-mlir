./build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=./ttrt-artifacts/system_desc.ttsys" test/ttmlir/Silicon/TTNN/n150/optimizer/resnet50_layer1_module2.mlir > rn50.mlir.tmp
./build/bin/ttmlir-translate --ttnn-to-flatbuffer rn50.mlir.tmp > out.ttnn
ttrt run out.ttnn |& tee rn50_dram_sample.log
