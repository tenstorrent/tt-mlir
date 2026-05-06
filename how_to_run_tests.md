```
TT_VISIBLE_DEVICES=0 ttrt query --save-artifacts
export SYSTEM_DESC_PATH=/localdev/ndrakulic/tt-mlir/ttrt-artifacts/system_desc.ttsys
TT_VISIBLE_DEVICES=0 llvm-lit test/ttmlir/Silicon/TTNN/n150/
TT_VISIBLE_DEVICES=0 pytest tools/chisel/tests/test_device_execution.py --binary build/test/ttmlir/Silicon/TTNN/n150/ -svv
```
