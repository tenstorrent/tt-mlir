softmax_mlir_check:
	mkdir im_mlirs/
	./build/bin/ttmlir-opt --ttir-load-system-desc="path=/proj_sw/user_dev/akannan/forge/tt-forge-fe/third_party/tt-mlir/ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline input_mlirs/softmax_dim_2.mlir -o im_mlirs/softmax_dim_2_ttnn.mlir
	./build/bin/ttmlir-opt --ttir-load-system-desc="path=/proj_sw/user_dev/akannan/forge/tt-forge-fe/third_party/tt-mlir/ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline input_mlirs/softmax_dim_1.mlir -o im_mlirs/softmax_dim_1_ttnn.mlir
	./build/bin/ttmlir-opt --ttir-load-system-desc="path=/proj_sw/user_dev/akannan/forge/tt-forge-fe/third_party/tt-mlir/ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline input_mlirs/softmax_dim_1.mlir -o im_mlirs/softmax_dim_0_ttnn.mlir

	./build/bin/ttmlir-translate --ttnn-to-flatbuffer im_mlirs/softmax_dim_2_ttnn.mlir -o im_mlirs/softmax_dim_2.ttnn
	./build/bin/ttmlir-translate --ttnn-to-flatbuffer im_mlirs/softmax_dim_1_ttnn.mlir -o im_mlirs/softmax_dim_1.ttnn
	./build/bin/ttmlir-translate --ttnn-to-flatbuffer im_mlirs/softmax_dim_0_ttnn.mlir -o im_mlirs/softmax_dim_0.ttnn

	ttrt run im_mlirs/softmax_dim_2.ttnn --log-file softmax_dim_2_ttrt_run.log
	mv inputs_1.pt softmax_dim_2_ttrt_run_inputs.pt

	ttrt run im_mlirs/softmax_dim_1.ttnn --log-file softmax_dim_1_ttrt_run.log
	mv inputs_1.pt softmax_dim_1_ttrt_run_inputs.pt

	ttrt run im_mlirs/softmax_dim_0.ttnn --log-file softmax_dim_0_ttrt_run.log
	mv inputs_1.pt softmax_dim_0_ttrt_run_inputs.pt


softmax_mlir_check_clean:
	rm *.pt
	rm *.log
	rm -rf im_mlirs/
	rm query_results.json
	rm run_results.json
