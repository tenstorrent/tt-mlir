Okay in `tools/chisel/docs` there is a lot of good materials for the implementation but now we need to break it down more nicely.

We will assume that we have implemented preProgram/postProgram callbacks for the chisel implementation. So lets write everything that needs to be done.

## Questions

- This logic for now works great for multi binary setups, how to handle load_cache and funcCall ops?

## Chisel Initialization

this just inits the singleton, and binds all 4 callback functions.
Here we need:
- globalTensorPool map - key Tensor::globalID -> returns golden Tensor or wrapper for the golden/device Tensor pair

## PreProgram

- Prepare the MLIR program execution
    - Parse the MLIR of the Binary
    - check the binary and program id

- Gather program inputs and manipulate them if needed
    - check if there exists already map between Tensor::globalId and golden tensor
        - if not copy the input tensor into golden tensor pool for the golden exectuion
- start new report section

## PreOp

- If op should be skipped
    - copy the inputs to host **before** the device op runs — the device op may overwrite input buffers in-place, so we need the original values for the golden op in postop to produce a correct replacement output.

## PostOp

- Get the op outputs from the device
- for each output
    - Execute golden operation store it into goldenTensor pool key in this case is just ssa value
    - Calculate metrics
    - Dump new line of csv report 
- if op should be skipped
    - Execute the golden op with the inputs extracted from the device in preop before
    - overwrite the device tensors with the golden calculated one


## PostProgram

- Finish the report
- Prepare the globalTensor pool map