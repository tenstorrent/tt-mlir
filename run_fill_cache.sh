ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=${SYSTEM_DESC_PATH}" test/ttmlir/Silicon/TTNN/n150/kv_cache/fill_cache.mlir > temp_output.mlir
ttmlir-translate --ttnn-to-flatbuffer temp_output.mlir > temp_output.ttnn
TTRT_LOGGER_LEVEL=DEBUG ttrt run temp_output.ttnn