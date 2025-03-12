#!/bin/bash


mkdir logs
mkdir logs/reshape_rep
for i in {1..10000}
do
    ttrt run out.ttnn |& tee logs/reshape_rep/run_$i.log
    echo "Done with run $i" > logs/reshape_rep/status.log
done
