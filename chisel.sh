model="linear"
folder="/localdev/ndrakulic/chisel/$model/"

ttmlir-opt --ttir-to-ttir-decomposition $folder/ttir_wide.mlir -o $folder/ttir.mlir
ttmlir-opt --mlir-print-debuginfo --ttir-to-ttnn-backend-pipeline="system-desc-path=/localdev/ndrakulic/tt-mlir/ttrt-artifacts/system_desc.ttsys" $folder/ttir.mlir -o $folder/ttnn.mlir
ttmlir-translate --ttnn-to-flatbuffer $folder/ttnn.mlir -o $folder/fb.ttnn