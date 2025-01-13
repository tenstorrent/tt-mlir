def test_llama_attention()
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 12, 3200),      # arg0
                                (1, 1, 12, 12),     # arg1
                                (1, 12),            # arg2
                                (1, 50, 1),         # arg3
                                (1, 32, 50, 100),   # arg4
                                (1),                # arg5
                                (1, 32, 50, 100),   # arg6
                                (1, 32, 50, 100),   # arg7
                                (1),                # arg8
                                (1, 32, 50, 100),   # arg9
                                (1),                # arg10
                                (3200, 3200),       # arg11
                                (3200, 3200),       # arg12
                                (3200, 3200),       # arg13
                                (3200, 3200)]       # arg14

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for shape in input_shape_list:
                golden_inputs.append(torch.randn(shape, dtype=torch.float32))

            @func.func(*input_operands, name=f"{function_name}")
            def llama_attention(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14):
                
                output1, golden_dict = create_squeeze(arg0, [(12, 3200)], 0, golden_inputs)
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output3, golden_dict = create_matmul(output1, arg11, [(12, 3200)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output5, golden_dict = create_reshape(output3, [(1, 12, 32, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output7, golden_dict = create_transpose(output5, [(1, 32, 12, 100)], -3, -2, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output9, golden_dict = create_unsqueeze(arg2, [(1, 1, 12)], 1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output11, golden_dict = create_matmul(arg3, output9, [(1, 50, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output13, golden_dict = create_transpose(arg11, [(1, 12, 50)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output15, golden_dict = create_concat(output13, output13, [(1, 12, 100)], -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output17, golden_dict = create_cos(output15, [(1, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output19, golden_dict = create_unsqueeze(output17, [(1, 1, 12, 100)], 1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output21, golden_dict = create_multiply(output7, output19, [(1, 32, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output23, golden_dict = create_transpose(output7, [(1, 32, 100, 12)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output25, golden_dict = create_matmul(arg4, output23, [(1, 32, 50, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output27, golden_dict = create_transpose(output25, [(1, 32, 12, 50)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output29, golden_dict = create_multiply(output27, arg5, [(1, 32, 12, 50)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                
                output31, golden_dict = create_transpose(output7, [(1, 32, 100, 12)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output33, golden_dict = create_matmul(arg6, output32, [(1, 32, 50, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output35, golden_dict = create_transpose(output33, [(1, 32, 12, 50)], -2, -1)
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output37, golden_dict = create_concat(output29, output35, [(1, 32, 12, 100)], -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output39, golden_dict = create_sin(output15, [(1, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output41, golden_dict = create_unsqueeze(output39, [(1, 1, 12, 100)], 1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output43, golden_dict = create_multiply(output37, output41, [(1, 32, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output45, golden_dict = create_add(output21, output43, [(1, 32, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output47, golden_dict = create_squeeze(output45, [(32, 12, 100)], 0, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output49, golden_dict = create_matmul(output1, arg12, [(12, 3200)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output51, golden_dict = create_reshape(output49, [(1, 12, 32, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output53, golden_dict = create_transpose(output51, [(1, 32, 12, 100)], -3, -2, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output55, golden_dict = create_multiply(output53, output19, [(1, 32, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output57, golden_dict = create_transpose(output53, [(1, 32, 100, 12)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output59, golden_dict = create_matmul(arg7, output57, [(1, 32, 50, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output61, golden_dict = create_transpose(output59, [(1, 32, 12, 50)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output63, golden_dict = create_multiply(output61, arg8, [(1, 32, 12, 50)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output65, golden_dict = create_transpose(output53, [(1, 32, 100, 12)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output67, golden_dict = create_matmul(arg9, output65, [(1, 32, 50, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output69, golden_dict = create_transpose(output67, [(1, 32, 12, 50)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output71, golden_dict = create_concat(output63, output69, [(1, 32, 12, 100)], -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output73, golden_dict = create_multiply(output71, output41, [(1, 32, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output75, golden_dict = create_add(output55, output73, [(1, 32, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output77, golden_dict = create_squeeze(output75, [(32, 12, 100)], 0, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output79, golden_dict = create_transpose(output77, [(32, 100, 12)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output81, golden_dict = create_matmul(output47, output79, [(32, 12, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output83, golden_dict = create_unsqueeze(output81, [(1, 32, 12, 12)], 0, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output85, golden_dict = create_multiply(output83, arg10, [(1, 32, 12, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output87, golden_dict = create_add(output85, arg1, [(1, 32, 12, 12)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output89, golden_dict = create_softmax(output87, [(1, 32, 12, 12)], -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output91, golden_dict = create_squeeze(output89, [(32, 12, 12)], 0, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output93, golden_dict = create_matmul(output1, arg13, [(12, 3200)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output95, golden_dict = create_reshape(output93, [(1, 12, 32, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output97, golden_dict = create_transpose(output95, [(1, 32, 12, 100)], -3, -2, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output99, golden_dict = create_transpose(output97, [(1, 32, 100, 12)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output101, golden_dict = create_squeeze(output99, [(32, 100, 12)], 0, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output103, golden_dict = create_transpose(output101, [(32, 12, 100)], -2, -1, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output105, golden_dict = create_matmul(output91, output103, [(32, 12, 100)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output107, golden_dict = create_unsqueeze(output105, [(1, 32, 12, 100)], 0, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output109, golden_dict = create_transpose(output107, [(1, 12, 32, 100)], -3, -2, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output111, golden_dict = create_reshape(output109, [(12, 3200)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output113, golden_dict = create_matmul(output111, arg14, [(12, 3200)], golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                output115, golden_dict = create_unsqueeze(output113, [(1, 12, 3200)], 0, golden_dict["golden_output"])
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]

                return output115

        module_post_processing(module, function_name, golden_map)

