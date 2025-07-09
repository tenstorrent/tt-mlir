
model="llama3_1layer"

if [ "$1" ]; then
    model=$1
fi
folder=$model

echo "Running Chisel on $model"
echo "Running ttir_wide -> ttir"
ttmlir-opt --allow-unregistered-dialect --ttir-to-ttir-decomposition --canonicalize $folder/ttir_wide.mlir -o $folder/ttir.mlir
echo "Done ttir_wide -> ttir"
echo "Running ttir -> ttnn"
ttmlir-opt --allow-unregistered-dialect --mlir-print-debuginfo --undo-const-eval --ttir-to-ttnn-backend-pipeline="enable-erase-inverse-ops-pass=false enable-const-eval=false system-desc-path=/localdev/ndrakulic/tt-mlir/ttrt-artifacts/system_desc.ttsys" $folder/ttir.mlir -o $folder/ttnn.mlir
echo "Done ttir -> ttnn"
echo "Running ttnn -> flatbuffer"
ttmlir-translate --ttnn-to-flatbuffer $folder/ttnn.mlir -o $folder/fb.ttnn
echo "Done ttnn -> flatbuffer"