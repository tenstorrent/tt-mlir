experiment="albert"

python3 chisel.py \
    --input_dir /localdev/pglusac/chisel/${experiment}/ \
    --op_config /localdev/pglusac/chisel/${experiment}/op_config.json \
    --output_dir /localdev/pglusac/chisel/${experiment}/output
