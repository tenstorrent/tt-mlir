TODO: Potentially see how to link this demo in with the new

Need to have tt-mlir built with runtime & pykernel enabled.

```
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_PYKERNEL=ON
cmake --build build
```

In order to `import ttnn` and use the `ttnn::generic_op` we need to install the necessary pip dependencies

As as temporary solution, install these dependences into `env/`.

```
source env/activate
pip install -r test/pykernel/demo/requirements.txt
```

To run the demo:
```
python3 test/pykernel/demo/eltwise_sfpu_demo.py
```
