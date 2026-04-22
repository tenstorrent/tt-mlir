module {
  "func.func"() <{arg_attrs = [{hw.port_name = "a4"}, {hw.port_name = "a7"}, {hw.port_name = "b12"}, {hw.port_name = "b16"}, {hw.port_name = "c24"}, {hw.port_name = "c32"}, {hw.port_name = "d48"}, {hw.port_name = "d64"}, {hw.port_name = "e96"}, {hw.port_name = "e128"}, {hw.port_name = "f33"}, {hw.port_name = "f40"}, {hw.port_name = "g65"}, {hw.port_name = "g100"}], function_type = (tensor<1xi8>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<3xi32>, tensor<4xi32>, tensor<2xi32>, tensor<2xi32>, tensor<3xi32>, tensor<4xi32>) -> (tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<1xi32>, tensor<2xi32>, tensor<4xi32>, tensor<3xi32>, tensor<1xi8>, tensor<3xi32>, tensor<1xi8>, tensor<3xi32>, tensor<1xi32>, tensor<4xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>, tensor<3xi32>, tensor<1xi8>), res_attrs = [{hw.port_name = "add_i8"}, {hw.port_name = "and_i16"}, {hw.port_name = "or_i32"}, {hw.port_name = "xor_i64"}, {hw.port_name = "add_cross"}, {hw.port_name = "and_cross"}, {hw.port_name = "xor_wide"}, {hw.port_name = "add_wide"}, {hw.port_name = "extract_lo"}, {hw.port_name = "concat_wide"}, {hw.port_name = "sub_i8"}, {hw.port_name = "sub_wide"}, {hw.port_name = "not_i32"}, {hw.port_name = "not_wide"}, {hw.port_name = "eq_i32"}, {hw.port_name = "ne_wide"}, {hw.port_name = "lt_i16"}, {hw.port_name = "ult_wide"}, {hw.port_name = "slt_i32"}, {hw.port_name = "sge_wide"}, {hw.port_name = "shl_i32"}, {hw.port_name = "shr_i32"}, {hw.port_name = "mux_i32"}, {hw.port_name = "mux_wide"}, {hw.port_name = "add_odd"}, {hw.port_name = "xor_odd"}, {hw.port_name = "sub_odd"}, {hw.port_name = "eq_odd"}], sym_name = "wide_datapath_impl", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<1xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>, %arg4: tensor<1xi32>, %arg5: tensor<1xi32>, %arg6: tensor<2xi32>, %arg7: tensor<2xi32>, %arg8: tensor<3xi32>, %arg9: tensor<4xi32>, %arg10: tensor<2xi32>, %arg11: tensor<2xi32>, %arg12: tensor<3xi32>, %arg13: tensor<4xi32>):
    %0 = "ttir.constant"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "ttir.constant"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "ttir.constant"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "ttir.constant"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "ttir.constant"() <{value = dense<-1> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5 = "ttir.constant"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "ttir.constant"() <{value = dense<0> : tensor<2xi32>}> : () -> tensor<2xi32>
    %10 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %11 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %14 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %15 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %16 = "ttir.typecast"(%arg0) : (tensor<1xi8>) -> tensor<1xi8>
    %17 = "ttir.typecast"(%15) : (tensor<1xi8>) -> tensor<1xi8>
    %18 = "ttir.constant"() <{value = dense<4> : tensor<1xi8>}> : () -> tensor<1xi8>
    %19 = "ttir.logical_left_shift"(%17, %18) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %20 = "ttir.bitwise_or"(%16, %19) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %21 = "ttir.typecast"(%arg1) : (tensor<1xi8>) -> tensor<1xi8>
    %22 = "ttir.typecast"(%14) : (tensor<1xi8>) -> tensor<1xi8>
    %23 = "ttir.constant"() <{value = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
    %24 = "ttir.logical_left_shift"(%22, %23) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %25 = "ttir.bitwise_or"(%21, %24) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %26 = "ttir.add"(%20, %25) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %27 = "ttir.typecast"(%arg2) : (tensor<1xi32>) -> tensor<1xi32>
    %28 = "ttir.typecast"(%15) : (tensor<1xi8>) -> tensor<1xi32>
    %29 = "ttir.constant"() <{value = dense<12> : tensor<1xi32>}> : () -> tensor<1xi32>
    %30 = "ttir.logical_left_shift"(%28, %29) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %31 = "ttir.bitwise_or"(%27, %30) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %32 = "ttir.bitwise_and"(%31, %arg3) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %33 = "ttir.bitwise_xor"(%arg5, %5) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %34 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %35 = "ttir.slice_static"(%arg9) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %36 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %37 = "ttir.bitwise_xor"(%35, %0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %38 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %39 = "ttir.slice_static"(%arg9) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %40 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %41 = "ttir.bitwise_xor"(%39, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %42 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %43 = "ttir.slice_static"(%arg9) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %44 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %45 = "ttir.bitwise_xor"(%43, %2) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %46 = "ttir.constant"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %47 = "ttir.slice_static"(%arg9) <{begins = [3 : i32], ends = [4 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %48 = "ttir.constant"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %49 = "ttir.bitwise_xor"(%47, %3) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %50 = "ttir.concat"(%37, %41, %45, %49) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %51 = "ttir.typecast"(%arg4) : (tensor<1xi32>) -> tensor<1xi32>
    %52 = "ttir.typecast"(%13) : (tensor<1xi8>) -> tensor<1xi32>
    %53 = "ttir.constant"() <{value = dense<24> : tensor<1xi32>}> : () -> tensor<1xi32>
    %54 = "ttir.logical_left_shift"(%52, %53) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %55 = "ttir.bitwise_or"(%51, %54) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %56 = "ttir.bitwise_xor"(%55, %arg5) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %57 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %58 = "ttir.logical_right_shift"(%56, %57) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %59 = "ttir.bitwise_or"(%56, %58) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %60 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %61 = "ttir.logical_right_shift"(%59, %60) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %62 = "ttir.bitwise_or"(%59, %61) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %63 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %64 = "ttir.logical_right_shift"(%62, %63) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %65 = "ttir.bitwise_or"(%62, %64) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %66 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %67 = "ttir.logical_right_shift"(%65, %66) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %68 = "ttir.bitwise_or"(%65, %67) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %69 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %70 = "ttir.logical_right_shift"(%68, %69) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %71 = "ttir.bitwise_or"(%68, %70) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %72 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %73 = "ttir.bitwise_and"(%71, %72) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %74 = "ttir.typecast"(%73) : (tensor<1xi32>) -> tensor<1xi8>
    %75 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %76 = "ttir.subtract"(%75, %74) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %77 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %78 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %79 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %80 = "ttir.slice_static"(%arg6) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %81 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %82 = "ttir.slice_static"(%arg6) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %83 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %84 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %85 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %86 = "ttir.logical_left_shift"(%84, %85) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %87 = "ttir.bitwise_or"(%82, %86) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %88 = "ttir.concat"(%80, %87) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %89 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %90 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %91 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %92 = "ttir.slice_static"(%arg7) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %93 = "ttir.bitwise_xor"(%80, %92) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %94 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %95 = "ttir.logical_right_shift"(%93, %94) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %96 = "ttir.bitwise_or"(%93, %95) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %97 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %98 = "ttir.logical_right_shift"(%96, %97) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %99 = "ttir.bitwise_or"(%96, %98) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %100 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %101 = "ttir.logical_right_shift"(%99, %100) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %102 = "ttir.bitwise_or"(%99, %101) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %103 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %104 = "ttir.logical_right_shift"(%102, %103) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %105 = "ttir.bitwise_or"(%102, %104) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %106 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %107 = "ttir.logical_right_shift"(%105, %106) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %108 = "ttir.bitwise_or"(%105, %107) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %109 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %110 = "ttir.bitwise_and"(%108, %109) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %111 = "ttir.typecast"(%110) : (tensor<1xi32>) -> tensor<1xi8>
    %112 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %113 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %114 = "ttir.slice_static"(%arg7) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %115 = "ttir.bitwise_xor"(%87, %114) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %116 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %117 = "ttir.logical_right_shift"(%115, %116) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %118 = "ttir.bitwise_or"(%115, %117) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %119 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %120 = "ttir.logical_right_shift"(%118, %119) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %121 = "ttir.bitwise_or"(%118, %120) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %122 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %123 = "ttir.logical_right_shift"(%121, %122) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %124 = "ttir.bitwise_or"(%121, %123) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %125 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %126 = "ttir.logical_right_shift"(%124, %125) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %127 = "ttir.bitwise_or"(%124, %126) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %128 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %129 = "ttir.logical_right_shift"(%127, %128) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %130 = "ttir.bitwise_or"(%127, %129) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %131 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %132 = "ttir.bitwise_and"(%130, %131) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %133 = "ttir.typecast"(%132) : (tensor<1xi32>) -> tensor<1xi8>
    %134 = "ttir.bitwise_or"(%111, %133) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %135 = "ttir.constant"() <{value = dense<-2147483648> : tensor<1xi32>}> : () -> tensor<1xi32>
    %136 = "ttir.bitwise_xor"(%31, %135) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %137 = "ttir.bitwise_xor"(%arg3, %135) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %138 = "ttir.typecast"(%136) : (tensor<1xi32>) -> tensor<1xsi32>
    %139 = "ttir.typecast"(%137) : (tensor<1xi32>) -> tensor<1xsi32>
    %140 = "ttir.lt"(%138, %139) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xi8>
    %141 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %142 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %143 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %144 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %145 = "ttir.slice_static"(%arg7) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %146 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %147 = "ttir.slice_static"(%arg7) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %148 = "ttir.concat"(%145, %147, %143) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %149 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %150 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %151 = "ttir.slice_static"(%arg8) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %152 = "ttir.constant"() <{value = dense<-2147483648> : tensor<1xi32>}> : () -> tensor<1xi32>
    %153 = "ttir.bitwise_xor"(%145, %152) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %154 = "ttir.bitwise_xor"(%151, %152) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %155 = "ttir.typecast"(%153) : (tensor<1xi32>) -> tensor<1xsi32>
    %156 = "ttir.typecast"(%154) : (tensor<1xi32>) -> tensor<1xsi32>
    %157 = "ttir.lt"(%155, %156) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xi8>
    %158 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %159 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %160 = "ttir.slice_static"(%arg8) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %161 = "ttir.constant"() <{value = dense<-2147483648> : tensor<1xi32>}> : () -> tensor<1xi32>
    %162 = "ttir.bitwise_xor"(%147, %161) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %163 = "ttir.bitwise_xor"(%160, %161) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %164 = "ttir.typecast"(%162) : (tensor<1xi32>) -> tensor<1xsi32>
    %165 = "ttir.typecast"(%163) : (tensor<1xi32>) -> tensor<1xsi32>
    %166 = "ttir.lt"(%164, %165) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xi8>
    %167 = "ttir.bitwise_xor"(%147, %160) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %168 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %169 = "ttir.logical_right_shift"(%167, %168) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %170 = "ttir.bitwise_or"(%167, %169) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %171 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %172 = "ttir.logical_right_shift"(%170, %171) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %173 = "ttir.bitwise_or"(%170, %172) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %174 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %175 = "ttir.logical_right_shift"(%173, %174) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %176 = "ttir.bitwise_or"(%173, %175) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %177 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %178 = "ttir.logical_right_shift"(%176, %177) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %179 = "ttir.bitwise_or"(%176, %178) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %180 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %181 = "ttir.logical_right_shift"(%179, %180) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %182 = "ttir.bitwise_or"(%179, %181) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %183 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %184 = "ttir.bitwise_and"(%182, %183) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %185 = "ttir.typecast"(%184) : (tensor<1xi32>) -> tensor<1xi8>
    %186 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %187 = "ttir.subtract"(%186, %185) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %188 = "ttir.bitwise_and"(%187, %157) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %189 = "ttir.bitwise_or"(%166, %188) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %190 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %191 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %192 = "ttir.slice_static"(%arg8) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %193 = "ttir.constant"() <{value = dense<-2147483648> : tensor<1xi32>}> : () -> tensor<1xi32>
    %194 = "ttir.bitwise_xor"(%143, %193) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %195 = "ttir.bitwise_xor"(%192, %193) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %196 = "ttir.typecast"(%194) : (tensor<1xi32>) -> tensor<1xsi32>
    %197 = "ttir.typecast"(%195) : (tensor<1xi32>) -> tensor<1xsi32>
    %198 = "ttir.lt"(%196, %197) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xi8>
    %199 = "ttir.bitwise_xor"(%143, %192) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %200 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %201 = "ttir.logical_right_shift"(%199, %200) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %202 = "ttir.bitwise_or"(%199, %201) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %203 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %204 = "ttir.logical_right_shift"(%202, %203) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %205 = "ttir.bitwise_or"(%202, %204) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %206 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %207 = "ttir.logical_right_shift"(%205, %206) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %208 = "ttir.bitwise_or"(%205, %207) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %209 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %210 = "ttir.logical_right_shift"(%208, %209) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %211 = "ttir.bitwise_or"(%208, %210) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %212 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %213 = "ttir.logical_right_shift"(%211, %212) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %214 = "ttir.bitwise_or"(%211, %213) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %215 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %216 = "ttir.bitwise_and"(%214, %215) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %217 = "ttir.typecast"(%216) : (tensor<1xi32>) -> tensor<1xi8>
    %218 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %219 = "ttir.subtract"(%218, %217) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %220 = "ttir.bitwise_and"(%219, %189) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %221 = "ttir.bitwise_or"(%198, %220) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %222 = "ttir.constant"() <{value = dense<23> : tensor<1xi32>}> : () -> tensor<1xi32>
    %223 = "ttir.logical_right_shift"(%arg4, %222) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %224 = "ttir.typecast"(%223) : (tensor<1xi32>) -> tensor<1xi8>
    %225 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %226 = "ttir.bitwise_and"(%224, %225) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %227 = "ttir.typecast"(%226) : (tensor<1xi8>) -> tensor<1xi8>
    %228 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %229 = "ttir.logical_left_shift"(%227, %228) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %230 = "ttir.bitwise_or"(%227, %229) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %231 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %232 = "ttir.logical_left_shift"(%230, %231) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %233 = "ttir.bitwise_or"(%230, %232) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %234 = "ttir.constant"() <{value = dense<4> : tensor<1xi8>}> : () -> tensor<1xi8>
    %235 = "ttir.logical_left_shift"(%233, %234) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %236 = "ttir.bitwise_or"(%233, %235) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %237 = "ttir.typecast"(%arg4) : (tensor<1xi32>) -> tensor<1xi32>
    %238 = "ttir.typecast"(%236) : (tensor<1xi8>) -> tensor<1xi32>
    %239 = "ttir.constant"() <{value = dense<24> : tensor<1xi32>}> : () -> tensor<1xi32>
    %240 = "ttir.logical_left_shift"(%238, %239) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %241 = "ttir.bitwise_or"(%237, %240) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %242 = "ttir.typecast"(%241) : (tensor<1xi32>) -> tensor<1xsi32>
    %243 = "ttir.typecast"(%arg5) : (tensor<1xi32>) -> tensor<1xsi32>
    %244 = "ttir.lt"(%242, %243) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xi8>
    %245 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %246 = "ttir.slice_static"(%arg6) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %247 = "ttir.constant"() <{value = dense<15> : tensor<1xi32>}> : () -> tensor<1xi32>
    %248 = "ttir.logical_right_shift"(%246, %247) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %249 = "ttir.typecast"(%248) : (tensor<1xi32>) -> tensor<1xi8>
    %250 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %251 = "ttir.bitwise_and"(%249, %250) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %252 = "ttir.typecast"(%251) : (tensor<1xi8>) -> tensor<1xi32>
    %253 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %254 = "ttir.logical_left_shift"(%252, %253) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %255 = "ttir.bitwise_or"(%252, %254) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %256 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %257 = "ttir.logical_left_shift"(%255, %256) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %258 = "ttir.bitwise_or"(%255, %257) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %259 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %260 = "ttir.logical_left_shift"(%258, %259) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %261 = "ttir.bitwise_or"(%258, %260) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %262 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %263 = "ttir.logical_left_shift"(%261, %262) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %264 = "ttir.bitwise_or"(%261, %263) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %265 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %266 = "ttir.logical_left_shift"(%264, %265) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %267 = "ttir.bitwise_or"(%264, %266) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %268 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %269 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %270 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %271 = "ttir.slice_static"(%arg6) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %272 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %273 = "ttir.slice_static"(%arg6) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %274 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %275 = "ttir.typecast"(%267) : (tensor<1xi32>) -> tensor<1xi32>
    %276 = "ttir.typecast"(%274) : (tensor<1xi32>) -> tensor<1xi32>
    %277 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %278 = "ttir.logical_left_shift"(%276, %277) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %279 = "ttir.bitwise_or"(%275, %278) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %280 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %281 = "ttir.logical_left_shift"(%279, %280) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %282 = "ttir.bitwise_or"(%273, %281) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %283 = "ttir.concat"(%271, %282) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %284 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %285 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %286 = "ttir.slice_static"(%arg7) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %287 = "ttir.constant"() <{value = dense<-2147483648> : tensor<1xi32>}> : () -> tensor<1xi32>
    %288 = "ttir.bitwise_xor"(%271, %287) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %289 = "ttir.bitwise_xor"(%286, %287) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %290 = "ttir.typecast"(%288) : (tensor<1xi32>) -> tensor<1xsi32>
    %291 = "ttir.typecast"(%289) : (tensor<1xi32>) -> tensor<1xsi32>
    %292 = "ttir.lt"(%290, %291) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xi8>
    %293 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %294 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %295 = "ttir.slice_static"(%arg7) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %296 = "ttir.typecast"(%282) : (tensor<1xi32>) -> tensor<1xsi32>
    %297 = "ttir.typecast"(%295) : (tensor<1xi32>) -> tensor<1xsi32>
    %298 = "ttir.lt"(%296, %297) : (tensor<1xsi32>, tensor<1xsi32>) -> tensor<1xi8>
    %299 = "ttir.bitwise_xor"(%282, %295) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %300 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %301 = "ttir.logical_right_shift"(%299, %300) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %302 = "ttir.bitwise_or"(%299, %301) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %303 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %304 = "ttir.logical_right_shift"(%302, %303) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %305 = "ttir.bitwise_or"(%302, %304) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %306 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %307 = "ttir.logical_right_shift"(%305, %306) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %308 = "ttir.bitwise_or"(%305, %307) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %309 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %310 = "ttir.logical_right_shift"(%308, %309) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %311 = "ttir.bitwise_or"(%308, %310) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %312 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %313 = "ttir.logical_right_shift"(%311, %312) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %314 = "ttir.bitwise_or"(%311, %313) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %315 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %316 = "ttir.bitwise_and"(%314, %315) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %317 = "ttir.typecast"(%316) : (tensor<1xi32>) -> tensor<1xi8>
    %318 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %319 = "ttir.subtract"(%318, %317) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %320 = "ttir.bitwise_and"(%319, %292) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %321 = "ttir.bitwise_or"(%298, %320) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %322 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %323 = "ttir.bitwise_xor"(%321, %322) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %324 = "ttir.bitwise_or"(%55, %arg5) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %325 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %326 = "ttir.bitwise_and"(%arg0, %325) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %327 = "ttir.typecast"(%326) : (tensor<1xi8>) -> tensor<1xi32>
    %328 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %329 = "ttir.subtract"(%328, %327) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %330 = "ttir.bitwise_not"(%329) : (tensor<1xi32>) -> tensor<1xi32>
    %331 = "ttir.bitwise_and"(%329, %55) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %332 = "ttir.bitwise_and"(%330, %arg5) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %333 = "ttir.bitwise_or"(%331, %332) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %334 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %335 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %336 = "ttir.slice_static"(%arg7) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %337 = "ttir.typecast"(%326) : (tensor<1xi8>) -> tensor<1xi32>
    %338 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %339 = "ttir.subtract"(%338, %337) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %340 = "ttir.bitwise_not"(%339) : (tensor<1xi32>) -> tensor<1xi32>
    %341 = "ttir.bitwise_and"(%339, %80) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %342 = "ttir.bitwise_and"(%340, %336) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %343 = "ttir.bitwise_or"(%341, %342) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %344 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %345 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %346 = "ttir.slice_static"(%arg7) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %347 = "ttir.typecast"(%326) : (tensor<1xi8>) -> tensor<1xi32>
    %348 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %349 = "ttir.subtract"(%348, %347) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %350 = "ttir.bitwise_not"(%349) : (tensor<1xi32>) -> tensor<1xi32>
    %351 = "ttir.bitwise_and"(%349, %87) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %352 = "ttir.bitwise_and"(%350, %346) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %353 = "ttir.bitwise_or"(%351, %352) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %354 = "ttir.concat"(%343, %353) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %355 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %356 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %357 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %358 = "ttir.slice_static"(%arg10) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %359 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %360 = "ttir.slice_static"(%arg10) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %361 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %362 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %363 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %364 = "ttir.logical_left_shift"(%362, %363) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %365 = "ttir.bitwise_or"(%360, %364) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %366 = "ttir.constant"() <{value = dense<255> : tensor<1xi32>}> : () -> tensor<1xi32>
    %367 = "ttir.bitwise_and"(%365, %366) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %368 = "ttir.concat"(%358, %367) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %369 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %370 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %371 = "ttir.slice_static"(%arg11) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %372 = "ttir.add"(%358, %371) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %373 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %374 = "ttir.logical_right_shift"(%358, %373) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %375 = "ttir.typecast"(%374) : (tensor<1xi32>) -> tensor<1xi8>
    %376 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %377 = "ttir.bitwise_and"(%375, %376) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %378 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %379 = "ttir.logical_right_shift"(%371, %378) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %380 = "ttir.typecast"(%379) : (tensor<1xi32>) -> tensor<1xi8>
    %381 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %382 = "ttir.bitwise_and"(%380, %381) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %383 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %384 = "ttir.logical_right_shift"(%372, %383) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %385 = "ttir.typecast"(%384) : (tensor<1xi32>) -> tensor<1xi8>
    %386 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %387 = "ttir.bitwise_and"(%385, %386) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %388 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %389 = "ttir.bitwise_xor"(%387, %388) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %390 = "ttir.bitwise_and"(%382, %389) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %391 = "ttir.bitwise_and"(%377, %389) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %392 = "ttir.bitwise_and"(%377, %382) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %393 = "ttir.bitwise_or"(%392, %391) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %394 = "ttir.bitwise_or"(%393, %390) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %395 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %396 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %397 = "ttir.slice_static"(%arg11) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %398 = "ttir.add"(%367, %397) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %399 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %400 = "ttir.logical_right_shift"(%367, %399) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %401 = "ttir.typecast"(%400) : (tensor<1xi32>) -> tensor<1xi8>
    %402 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %403 = "ttir.bitwise_and"(%401, %402) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %404 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %405 = "ttir.logical_right_shift"(%397, %404) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %406 = "ttir.typecast"(%405) : (tensor<1xi32>) -> tensor<1xi8>
    %407 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %408 = "ttir.bitwise_and"(%406, %407) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %409 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %410 = "ttir.logical_right_shift"(%398, %409) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %411 = "ttir.typecast"(%410) : (tensor<1xi32>) -> tensor<1xi8>
    %412 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %413 = "ttir.bitwise_and"(%411, %412) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %414 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %415 = "ttir.bitwise_xor"(%413, %414) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %416 = "ttir.bitwise_and"(%408, %415) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %417 = "ttir.bitwise_and"(%403, %415) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %418 = "ttir.bitwise_and"(%403, %408) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %419 = "ttir.bitwise_or"(%418, %417) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %420 = "ttir.bitwise_or"(%419, %416) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %421 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %422 = "ttir.typecast"(%394) : (tensor<1xi8>) -> tensor<1xi32>
    %423 = "ttir.typecast"(%421) : (tensor<1xi32>) -> tensor<1xi32>
    %424 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %425 = "ttir.logical_left_shift"(%423, %424) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %426 = "ttir.bitwise_or"(%422, %425) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %427 = "ttir.add"(%398, %426) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %428 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %429 = "ttir.logical_right_shift"(%427, %428) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %430 = "ttir.typecast"(%429) : (tensor<1xi32>) -> tensor<1xi8>
    %431 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %432 = "ttir.bitwise_and"(%430, %431) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %433 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %434 = "ttir.bitwise_xor"(%432, %433) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %435 = "ttir.bitwise_and"(%413, %434) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %436 = "ttir.bitwise_or"(%420, %435) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %437 = "ttir.constant"() <{value = dense<255> : tensor<1xi32>}> : () -> tensor<1xi32>
    %438 = "ttir.bitwise_and"(%427, %437) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %439 = "ttir.concat"(%372, %438) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %440 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %441 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %442 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %443 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %444 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %445 = "ttir.slice_static"(%arg12) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %446 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %447 = "ttir.slice_static"(%arg12) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %448 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %449 = "ttir.slice_static"(%arg12) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %450 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %451 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %452 = "ttir.logical_left_shift"(%7, %451) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %453 = "ttir.bitwise_or"(%449, %452) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %454 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %455 = "ttir.logical_right_shift"(%7, %454) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %456 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %457 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %458 = "ttir.logical_left_shift"(%8, %457) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %459 = "ttir.bitwise_or"(%455, %458) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %460 = "ttir.constant"() <{value = dense<15> : tensor<1xi32>}> : () -> tensor<1xi32>
    %461 = "ttir.bitwise_and"(%459, %460) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %462 = "ttir.concat"(%445, %447, %453, %461) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %463 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %464 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %465 = "ttir.slice_static"(%arg13) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %466 = "ttir.bitwise_xor"(%445, %465) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %467 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %468 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %469 = "ttir.slice_static"(%arg13) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %470 = "ttir.bitwise_xor"(%447, %469) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %471 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %472 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %473 = "ttir.slice_static"(%arg13) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %474 = "ttir.bitwise_xor"(%453, %473) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %475 = "ttir.constant"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %476 = "ttir.constant"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %477 = "ttir.slice_static"(%arg13) <{begins = [3 : i32], ends = [4 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %478 = "ttir.bitwise_xor"(%461, %477) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %479 = "ttir.constant"() <{value = dense<15> : tensor<1xi32>}> : () -> tensor<1xi32>
    %480 = "ttir.bitwise_and"(%478, %479) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %481 = "ttir.concat"(%466, %470, %474, %480) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %482 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %483 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %484 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %485 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %486 = "ttir.slice_static"(%arg10) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %487 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %488 = "ttir.slice_static"(%arg10) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %489 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %490 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %491 = "ttir.bitwise_or"(%488, %490) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %492 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %493 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %494 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %495 = "ttir.bitwise_and"(%493, %494) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %496 = "ttir.concat"(%486, %491, %495) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %497 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %498 = "ttir.slice_static"(%arg12) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %499 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %500 = "ttir.subtract"(%498, %486) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %501 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %502 = "ttir.logical_right_shift"(%498, %501) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %503 = "ttir.typecast"(%502) : (tensor<1xi32>) -> tensor<1xi8>
    %504 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %505 = "ttir.bitwise_and"(%503, %504) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %506 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %507 = "ttir.logical_right_shift"(%486, %506) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %508 = "ttir.typecast"(%507) : (tensor<1xi32>) -> tensor<1xi8>
    %509 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %510 = "ttir.bitwise_and"(%508, %509) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %511 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %512 = "ttir.logical_right_shift"(%500, %511) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %513 = "ttir.typecast"(%512) : (tensor<1xi32>) -> tensor<1xi8>
    %514 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %515 = "ttir.bitwise_and"(%513, %514) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %516 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %517 = "ttir.bitwise_xor"(%505, %516) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %518 = "ttir.bitwise_xor"(%505, %510) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %519 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %520 = "ttir.bitwise_xor"(%518, %519) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %521 = "ttir.bitwise_and"(%520, %515) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %522 = "ttir.bitwise_and"(%517, %510) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %523 = "ttir.bitwise_or"(%522, %521) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %524 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %525 = "ttir.slice_static"(%arg12) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %526 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %527 = "ttir.subtract"(%525, %491) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %528 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %529 = "ttir.logical_right_shift"(%525, %528) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %530 = "ttir.typecast"(%529) : (tensor<1xi32>) -> tensor<1xi8>
    %531 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %532 = "ttir.bitwise_and"(%530, %531) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %533 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %534 = "ttir.logical_right_shift"(%491, %533) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %535 = "ttir.typecast"(%534) : (tensor<1xi32>) -> tensor<1xi8>
    %536 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %537 = "ttir.bitwise_and"(%535, %536) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %538 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %539 = "ttir.logical_right_shift"(%527, %538) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %540 = "ttir.typecast"(%539) : (tensor<1xi32>) -> tensor<1xi8>
    %541 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %542 = "ttir.bitwise_and"(%540, %541) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %543 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %544 = "ttir.bitwise_xor"(%532, %543) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %545 = "ttir.bitwise_xor"(%532, %537) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %546 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %547 = "ttir.bitwise_xor"(%545, %546) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %548 = "ttir.bitwise_and"(%547, %542) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %549 = "ttir.bitwise_and"(%544, %537) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %550 = "ttir.bitwise_or"(%549, %548) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %551 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %552 = "ttir.typecast"(%523) : (tensor<1xi8>) -> tensor<1xi32>
    %553 = "ttir.typecast"(%551) : (tensor<1xi32>) -> tensor<1xi32>
    %554 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %555 = "ttir.logical_left_shift"(%553, %554) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %556 = "ttir.bitwise_or"(%552, %555) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %557 = "ttir.subtract"(%527, %556) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %558 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %559 = "ttir.logical_right_shift"(%557, %558) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %560 = "ttir.typecast"(%559) : (tensor<1xi32>) -> tensor<1xi8>
    %561 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %562 = "ttir.bitwise_and"(%560, %561) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %563 = "ttir.bitwise_and"(%544, %562) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %564 = "ttir.bitwise_or"(%550, %563) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %565 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %566 = "ttir.slice_static"(%arg12) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %567 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %568 = "ttir.subtract"(%566, %495) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %569 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %570 = "ttir.logical_right_shift"(%566, %569) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %571 = "ttir.typecast"(%570) : (tensor<1xi32>) -> tensor<1xi8>
    %572 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %573 = "ttir.bitwise_and"(%571, %572) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %574 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %575 = "ttir.logical_right_shift"(%495, %574) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %576 = "ttir.typecast"(%575) : (tensor<1xi32>) -> tensor<1xi8>
    %577 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %578 = "ttir.bitwise_and"(%576, %577) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %579 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %580 = "ttir.logical_right_shift"(%568, %579) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %581 = "ttir.typecast"(%580) : (tensor<1xi32>) -> tensor<1xi8>
    %582 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %583 = "ttir.bitwise_and"(%581, %582) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %584 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %585 = "ttir.bitwise_xor"(%573, %584) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %586 = "ttir.bitwise_xor"(%573, %578) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %587 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %588 = "ttir.bitwise_xor"(%586, %587) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %589 = "ttir.bitwise_and"(%588, %583) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %590 = "ttir.bitwise_and"(%585, %578) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %591 = "ttir.bitwise_or"(%590, %589) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %592 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %593 = "ttir.typecast"(%564) : (tensor<1xi8>) -> tensor<1xi32>
    %594 = "ttir.typecast"(%592) : (tensor<1xi32>) -> tensor<1xi32>
    %595 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %596 = "ttir.logical_left_shift"(%594, %595) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %597 = "ttir.bitwise_or"(%593, %596) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %598 = "ttir.subtract"(%568, %597) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %599 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %600 = "ttir.logical_right_shift"(%598, %599) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %601 = "ttir.typecast"(%600) : (tensor<1xi32>) -> tensor<1xi8>
    %602 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %603 = "ttir.bitwise_and"(%601, %602) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %604 = "ttir.bitwise_and"(%585, %603) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %605 = "ttir.bitwise_or"(%591, %604) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %606 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %607 = "ttir.bitwise_and"(%598, %606) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %608 = "ttir.concat"(%500, %557, %607) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %609 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %610 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %611 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %612 = "ttir.slice_static"(%arg11) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %613 = "ttir.bitwise_xor"(%358, %612) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %614 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %615 = "ttir.logical_right_shift"(%613, %614) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %616 = "ttir.bitwise_or"(%613, %615) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %617 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %618 = "ttir.logical_right_shift"(%616, %617) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %619 = "ttir.bitwise_or"(%616, %618) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %620 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %621 = "ttir.logical_right_shift"(%619, %620) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %622 = "ttir.bitwise_or"(%619, %621) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %623 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %624 = "ttir.logical_right_shift"(%622, %623) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %625 = "ttir.bitwise_or"(%622, %624) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %626 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %627 = "ttir.logical_right_shift"(%625, %626) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %628 = "ttir.bitwise_or"(%625, %627) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %629 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %630 = "ttir.bitwise_and"(%628, %629) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %631 = "ttir.typecast"(%630) : (tensor<1xi32>) -> tensor<1xi8>
    %632 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %633 = "ttir.subtract"(%632, %631) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %634 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %635 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %636 = "ttir.slice_static"(%arg11) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %637 = "ttir.bitwise_xor"(%367, %636) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %638 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %639 = "ttir.logical_right_shift"(%637, %638) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %640 = "ttir.bitwise_or"(%637, %639) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %641 = "ttir.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %642 = "ttir.logical_right_shift"(%640, %641) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %643 = "ttir.bitwise_or"(%640, %642) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %644 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %645 = "ttir.logical_right_shift"(%643, %644) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %646 = "ttir.bitwise_or"(%643, %645) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %647 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %648 = "ttir.logical_right_shift"(%646, %647) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %649 = "ttir.bitwise_or"(%646, %648) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %650 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %651 = "ttir.logical_right_shift"(%649, %650) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %652 = "ttir.bitwise_or"(%649, %651) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %653 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %654 = "ttir.bitwise_and"(%652, %653) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %655 = "ttir.typecast"(%654) : (tensor<1xi32>) -> tensor<1xi8>
    %656 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %657 = "ttir.subtract"(%656, %655) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %658 = "ttir.bitwise_and"(%633, %657) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %659 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %660 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %661 = "ttir.slice_static"(%arg7) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %662 = "ttir.bitwise_xor"(%80, %661) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %663 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %664 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %665 = "ttir.slice_static"(%arg7) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %666 = "ttir.bitwise_xor"(%87, %665) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %667 = "ttir.concat"(%662, %666) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %668 = "ttir.typecast"(%arg1) : (tensor<1xi8>) -> tensor<1xi32>
    %669 = "ttir.typecast"(%6) : (tensor<1xi32>) -> tensor<1xi32>
    %670 = "ttir.constant"() <{value = dense<7> : tensor<1xi32>}> : () -> tensor<1xi32>
    %671 = "ttir.logical_left_shift"(%669, %670) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %672 = "ttir.bitwise_or"(%668, %671) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %673 = "ttir.add"(%672, %31) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %674 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %675 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %676 = "ttir.concat"(%arg5, %675) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %677 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %678 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %679 = "ttir.bitwise_and"(%arg5, %80) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %680 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %681 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %682 = "ttir.bitwise_and"(%675, %87) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %683 = "ttir.concat"(%679, %682) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %684 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %685 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %686 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %687 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %688 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %689 = "ttir.slice_static"(%arg8) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %690 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %691 = "ttir.slice_static"(%arg8) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %692 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %693 = "ttir.slice_static"(%arg8) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %694 = "ttir.concat"(%689, %691, %693, %687) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %695 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %696 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %697 = "ttir.slice_static"(%arg9) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %698 = "ttir.bitwise_xor"(%689, %697) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %699 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %700 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %701 = "ttir.slice_static"(%arg9) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %702 = "ttir.bitwise_xor"(%691, %701) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %703 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %704 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %705 = "ttir.slice_static"(%arg9) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %706 = "ttir.bitwise_xor"(%693, %705) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %707 = "ttir.constant"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %708 = "ttir.constant"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %709 = "ttir.slice_static"(%arg9) <{begins = [3 : i32], ends = [4 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %710 = "ttir.bitwise_xor"(%687, %709) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %711 = "ttir.concat"(%698, %702, %706, %710) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %712 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %713 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %714 = "ttir.slice_static"(%arg8) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %715 = "ttir.add"(%145, %714) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %716 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %717 = "ttir.logical_right_shift"(%145, %716) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %718 = "ttir.typecast"(%717) : (tensor<1xi32>) -> tensor<1xi8>
    %719 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %720 = "ttir.bitwise_and"(%718, %719) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %721 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %722 = "ttir.logical_right_shift"(%714, %721) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %723 = "ttir.typecast"(%722) : (tensor<1xi32>) -> tensor<1xi8>
    %724 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %725 = "ttir.bitwise_and"(%723, %724) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %726 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %727 = "ttir.logical_right_shift"(%715, %726) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %728 = "ttir.typecast"(%727) : (tensor<1xi32>) -> tensor<1xi8>
    %729 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %730 = "ttir.bitwise_and"(%728, %729) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %731 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %732 = "ttir.bitwise_xor"(%730, %731) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %733 = "ttir.bitwise_and"(%725, %732) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %734 = "ttir.bitwise_and"(%720, %732) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %735 = "ttir.bitwise_and"(%720, %725) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %736 = "ttir.bitwise_or"(%735, %734) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %737 = "ttir.bitwise_or"(%736, %733) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %738 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %739 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %740 = "ttir.slice_static"(%arg8) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %741 = "ttir.add"(%147, %740) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %742 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %743 = "ttir.logical_right_shift"(%147, %742) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %744 = "ttir.typecast"(%743) : (tensor<1xi32>) -> tensor<1xi8>
    %745 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %746 = "ttir.bitwise_and"(%744, %745) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %747 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %748 = "ttir.logical_right_shift"(%740, %747) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %749 = "ttir.typecast"(%748) : (tensor<1xi32>) -> tensor<1xi8>
    %750 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %751 = "ttir.bitwise_and"(%749, %750) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %752 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %753 = "ttir.logical_right_shift"(%741, %752) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %754 = "ttir.typecast"(%753) : (tensor<1xi32>) -> tensor<1xi8>
    %755 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %756 = "ttir.bitwise_and"(%754, %755) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %757 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %758 = "ttir.bitwise_xor"(%756, %757) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %759 = "ttir.bitwise_and"(%751, %758) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %760 = "ttir.bitwise_and"(%746, %758) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %761 = "ttir.bitwise_and"(%746, %751) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %762 = "ttir.bitwise_or"(%761, %760) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %763 = "ttir.bitwise_or"(%762, %759) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %764 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %765 = "ttir.typecast"(%737) : (tensor<1xi8>) -> tensor<1xi32>
    %766 = "ttir.typecast"(%764) : (tensor<1xi32>) -> tensor<1xi32>
    %767 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %768 = "ttir.logical_left_shift"(%766, %767) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %769 = "ttir.bitwise_or"(%765, %768) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %770 = "ttir.add"(%741, %769) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %771 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %772 = "ttir.logical_right_shift"(%770, %771) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %773 = "ttir.typecast"(%772) : (tensor<1xi32>) -> tensor<1xi8>
    %774 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %775 = "ttir.bitwise_and"(%773, %774) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %776 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %777 = "ttir.bitwise_xor"(%775, %776) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %778 = "ttir.bitwise_and"(%756, %777) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %779 = "ttir.bitwise_or"(%763, %778) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %780 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %781 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %782 = "ttir.slice_static"(%arg8) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %783 = "ttir.add"(%143, %782) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %784 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %785 = "ttir.logical_right_shift"(%143, %784) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %786 = "ttir.typecast"(%785) : (tensor<1xi32>) -> tensor<1xi8>
    %787 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %788 = "ttir.bitwise_and"(%786, %787) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %789 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %790 = "ttir.logical_right_shift"(%782, %789) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %791 = "ttir.typecast"(%790) : (tensor<1xi32>) -> tensor<1xi8>
    %792 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %793 = "ttir.bitwise_and"(%791, %792) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %794 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %795 = "ttir.logical_right_shift"(%783, %794) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %796 = "ttir.typecast"(%795) : (tensor<1xi32>) -> tensor<1xi8>
    %797 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %798 = "ttir.bitwise_and"(%796, %797) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %799 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %800 = "ttir.bitwise_xor"(%798, %799) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %801 = "ttir.bitwise_and"(%793, %800) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %802 = "ttir.bitwise_and"(%788, %800) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %803 = "ttir.bitwise_and"(%788, %793) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %804 = "ttir.bitwise_or"(%803, %802) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %805 = "ttir.bitwise_or"(%804, %801) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %806 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %807 = "ttir.typecast"(%779) : (tensor<1xi8>) -> tensor<1xi32>
    %808 = "ttir.typecast"(%806) : (tensor<1xi32>) -> tensor<1xi32>
    %809 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %810 = "ttir.logical_left_shift"(%808, %809) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %811 = "ttir.bitwise_or"(%807, %810) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %812 = "ttir.add"(%783, %811) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %813 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %814 = "ttir.logical_right_shift"(%812, %813) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %815 = "ttir.typecast"(%814) : (tensor<1xi32>) -> tensor<1xi8>
    %816 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %817 = "ttir.bitwise_and"(%815, %816) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %818 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %819 = "ttir.bitwise_xor"(%817, %818) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %820 = "ttir.bitwise_and"(%798, %819) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %821 = "ttir.bitwise_or"(%805, %820) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %822 = "ttir.concat"(%715, %770, %812) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %823 = "ttir.subtract"(%20, %25) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %824 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %825 = "ttir.slice_static"(%arg8) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %826 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %827 = "ttir.subtract"(%825, %145) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %828 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %829 = "ttir.logical_right_shift"(%825, %828) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %830 = "ttir.typecast"(%829) : (tensor<1xi32>) -> tensor<1xi8>
    %831 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %832 = "ttir.bitwise_and"(%830, %831) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %833 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %834 = "ttir.logical_right_shift"(%145, %833) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %835 = "ttir.typecast"(%834) : (tensor<1xi32>) -> tensor<1xi8>
    %836 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %837 = "ttir.bitwise_and"(%835, %836) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %838 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %839 = "ttir.logical_right_shift"(%827, %838) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %840 = "ttir.typecast"(%839) : (tensor<1xi32>) -> tensor<1xi8>
    %841 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %842 = "ttir.bitwise_and"(%840, %841) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %843 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %844 = "ttir.bitwise_xor"(%832, %843) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %845 = "ttir.bitwise_xor"(%832, %837) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %846 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %847 = "ttir.bitwise_xor"(%845, %846) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %848 = "ttir.bitwise_and"(%847, %842) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %849 = "ttir.bitwise_and"(%844, %837) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %850 = "ttir.bitwise_or"(%849, %848) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %851 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %852 = "ttir.slice_static"(%arg8) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %853 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %854 = "ttir.subtract"(%852, %147) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %855 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %856 = "ttir.logical_right_shift"(%852, %855) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %857 = "ttir.typecast"(%856) : (tensor<1xi32>) -> tensor<1xi8>
    %858 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %859 = "ttir.bitwise_and"(%857, %858) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %860 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %861 = "ttir.logical_right_shift"(%147, %860) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %862 = "ttir.typecast"(%861) : (tensor<1xi32>) -> tensor<1xi8>
    %863 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %864 = "ttir.bitwise_and"(%862, %863) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %865 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %866 = "ttir.logical_right_shift"(%854, %865) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %867 = "ttir.typecast"(%866) : (tensor<1xi32>) -> tensor<1xi8>
    %868 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %869 = "ttir.bitwise_and"(%867, %868) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %870 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %871 = "ttir.bitwise_xor"(%859, %870) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %872 = "ttir.bitwise_xor"(%859, %864) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %873 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %874 = "ttir.bitwise_xor"(%872, %873) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %875 = "ttir.bitwise_and"(%874, %869) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %876 = "ttir.bitwise_and"(%871, %864) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %877 = "ttir.bitwise_or"(%876, %875) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %878 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %879 = "ttir.typecast"(%850) : (tensor<1xi8>) -> tensor<1xi32>
    %880 = "ttir.typecast"(%878) : (tensor<1xi32>) -> tensor<1xi32>
    %881 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %882 = "ttir.logical_left_shift"(%880, %881) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %883 = "ttir.bitwise_or"(%879, %882) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %884 = "ttir.subtract"(%854, %883) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %885 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %886 = "ttir.logical_right_shift"(%884, %885) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %887 = "ttir.typecast"(%886) : (tensor<1xi32>) -> tensor<1xi8>
    %888 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %889 = "ttir.bitwise_and"(%887, %888) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %890 = "ttir.bitwise_and"(%871, %889) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %891 = "ttir.bitwise_or"(%877, %890) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %892 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %893 = "ttir.slice_static"(%arg8) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %894 = "ttir.constant"() <{value = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %895 = "ttir.subtract"(%893, %143) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %896 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %897 = "ttir.logical_right_shift"(%893, %896) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %898 = "ttir.typecast"(%897) : (tensor<1xi32>) -> tensor<1xi8>
    %899 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %900 = "ttir.bitwise_and"(%898, %899) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %901 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %902 = "ttir.logical_right_shift"(%143, %901) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %903 = "ttir.typecast"(%902) : (tensor<1xi32>) -> tensor<1xi8>
    %904 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %905 = "ttir.bitwise_and"(%903, %904) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %906 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %907 = "ttir.logical_right_shift"(%895, %906) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %908 = "ttir.typecast"(%907) : (tensor<1xi32>) -> tensor<1xi8>
    %909 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %910 = "ttir.bitwise_and"(%908, %909) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %911 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %912 = "ttir.bitwise_xor"(%900, %911) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %913 = "ttir.bitwise_xor"(%900, %905) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %914 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %915 = "ttir.bitwise_xor"(%913, %914) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %916 = "ttir.bitwise_and"(%915, %910) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %917 = "ttir.bitwise_and"(%912, %905) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %918 = "ttir.bitwise_or"(%917, %916) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %919 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %920 = "ttir.typecast"(%891) : (tensor<1xi8>) -> tensor<1xi32>
    %921 = "ttir.typecast"(%919) : (tensor<1xi32>) -> tensor<1xi32>
    %922 = "ttir.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %923 = "ttir.logical_left_shift"(%921, %922) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %924 = "ttir.bitwise_or"(%920, %923) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %925 = "ttir.subtract"(%895, %924) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %926 = "ttir.constant"() <{value = dense<31> : tensor<1xi32>}> : () -> tensor<1xi32>
    %927 = "ttir.logical_right_shift"(%925, %926) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %928 = "ttir.typecast"(%927) : (tensor<1xi32>) -> tensor<1xi8>
    %929 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %930 = "ttir.bitwise_and"(%928, %929) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %931 = "ttir.bitwise_and"(%912, %930) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %932 = "ttir.bitwise_or"(%918, %931) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %933 = "ttir.concat"(%827, %884, %925) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %934 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %935 = "ttir.slice_static"(%arg9) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %936 = "ttir.typecast"(%935) : (tensor<1xi32>) -> tensor<1xi8>
    %937 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %938 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %939 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %940 = "ttir.constant"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %941 = "ttir.slice_static"(%arg7) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %942 = "ttir.constant"() <{value = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %943 = "ttir.slice_static"(%arg7) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %944 = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %945 = "ttir.typecast"(%arg3) : (tensor<1xi32>) -> tensor<1xi32>
    %946 = "ttir.typecast"(%944) : (tensor<1xi32>) -> tensor<1xi32>
    %947 = "ttir.constant"() <{value = dense<16> : tensor<1xi32>}> : () -> tensor<1xi32>
    %948 = "ttir.logical_left_shift"(%946, %947) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %949 = "ttir.bitwise_or"(%945, %948) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %950 = "ttir.constant"() <{value = dense<65535> : tensor<1xi32>}> : () -> tensor<1xi32>
    %951 = "ttir.bitwise_and"(%949, %950) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %952 = "ttir.concat"(%941, %943, %951) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %953 = "ttir.typecast"(%15) : (tensor<1xi8>) -> tensor<1xi32>
    %954 = "ttir.typecast"(%arg4) : (tensor<1xi32>) -> tensor<1xi32>
    %955 = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %956 = "ttir.logical_left_shift"(%954, %955) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %957 = "ttir.bitwise_or"(%953, %956) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %958 = "ttir.typecast"(%15) : (tensor<1xi8>) -> tensor<1xi32>
    %959 = "ttir.constant"() <{value = dense<28> : tensor<1xi32>}> : () -> tensor<1xi32>
    %960 = "ttir.logical_left_shift"(%958, %959) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %961 = "ttir.bitwise_or"(%957, %960) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %962 = "ttir.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %963 = "ttir.logical_right_shift"(%arg5, %962) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %964 = "ttir.constant"() <{value = dense<16777215> : tensor<1xi32>}> : () -> tensor<1xi32>
    %965 = "ttir.bitwise_and"(%963, %964) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %966 = "ttir.typecast"(%965) : (tensor<1xi32>) -> tensor<1xi32>
    %967 = "ttir.typecast"(%13) : (tensor<1xi8>) -> tensor<1xi32>
    %968 = "ttir.constant"() <{value = dense<24> : tensor<1xi32>}> : () -> tensor<1xi32>
    %969 = "ttir.logical_left_shift"(%967, %968) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %970 = "ttir.bitwise_or"(%966, %969) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    "func.return"(%26, %32, %324, %667, %673, %683, %711, %822, %936, %952, %823, %933, %33, %50, %76, %134, %140, %221, %244, %323, %961, %970, %333, %354, %439, %481, %608, %658) : (tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<1xi32>, tensor<2xi32>, tensor<4xi32>, tensor<3xi32>, tensor<1xi8>, tensor<3xi32>, tensor<1xi8>, tensor<3xi32>, tensor<1xi32>, tensor<4xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>, tensor<3xi32>, tensor<1xi8>) -> ()
  }) : () -> ()
  func.func @wide_datapath(%a4: tensor<1xui8>, %a7: tensor<1xui8>, %b12: tensor<1xui32>, %b16: tensor<1xui32>, %c24: tensor<1xui32>, %c32: tensor<1xui32>, %d48__0: tensor<1xui32>, %d48__1: tensor<1xui32>, %d64__0: tensor<1xui32>, %d64__1: tensor<1xui32>, %e96__0: tensor<1xui32>, %e96__1: tensor<1xui32>, %e96__2: tensor<1xui32>, %e128__0: tensor<1xui32>, %e128__1: tensor<1xui32>, %e128__2: tensor<1xui32>, %e128__3: tensor<1xui32>, %f33__0: tensor<1xui32>, %f33__1: tensor<1xui32>, %f40__0: tensor<1xui32>, %f40__1: tensor<1xui32>, %g65__0: tensor<1xui32>, %g65__1: tensor<1xui32>, %g65__2: tensor<1xui32>, %g100__0: tensor<1xui32>, %g100__1: tensor<1xui32>, %g100__2: tensor<1xui32>, %g100__3: tensor<1xui32>) -> (tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>) {
    %_wi_a4 = "ttir.typecast"(%a4) : (tensor<1xui8>) -> tensor<1xi8>
    %_wi_a7 = "ttir.typecast"(%a7) : (tensor<1xui8>) -> tensor<1xi8>
    %_wi_b12 = "ttir.typecast"(%b12) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_b16 = "ttir.typecast"(%b16) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_c24 = "ttir.typecast"(%c24) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_c32 = "ttir.typecast"(%c32) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_d48__0 = "ttir.typecast"(%d48__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_d48__1 = "ttir.typecast"(%d48__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_d48 = "ttir.concat"(%_wi_d48__0, %_wi_d48__1) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %_wi_d64__0 = "ttir.typecast"(%d64__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_d64__1 = "ttir.typecast"(%d64__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_d64 = "ttir.concat"(%_wi_d64__0, %_wi_d64__1) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %_wi_e96__0 = "ttir.typecast"(%e96__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_e96__1 = "ttir.typecast"(%e96__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_e96__2 = "ttir.typecast"(%e96__2) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_e96 = "ttir.concat"(%_wi_e96__0, %_wi_e96__1, %_wi_e96__2) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %_wi_e128__0 = "ttir.typecast"(%e128__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_e128__1 = "ttir.typecast"(%e128__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_e128__2 = "ttir.typecast"(%e128__2) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_e128__3 = "ttir.typecast"(%e128__3) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_e128 = "ttir.concat"(%_wi_e128__0, %_wi_e128__1, %_wi_e128__2, %_wi_e128__3) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %_wi_f33__0 = "ttir.typecast"(%f33__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_f33__1 = "ttir.typecast"(%f33__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_f33 = "ttir.concat"(%_wi_f33__0, %_wi_f33__1) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %_wi_f40__0 = "ttir.typecast"(%f40__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_f40__1 = "ttir.typecast"(%f40__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_f40 = "ttir.concat"(%_wi_f40__0, %_wi_f40__1) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %_wi_g65__0 = "ttir.typecast"(%g65__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_g65__1 = "ttir.typecast"(%g65__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_g65__2 = "ttir.typecast"(%g65__2) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_g65 = "ttir.concat"(%_wi_g65__0, %_wi_g65__1, %_wi_g65__2) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %_wi_g100__0 = "ttir.typecast"(%g100__0) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_g100__1 = "ttir.typecast"(%g100__1) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_g100__2 = "ttir.typecast"(%g100__2) : (tensor<1xui32>) -> tensor<1xi32>
    %_wi_g100__3 = "ttir.typecast"(%g100__3) : (tensor<1xui32>) -> tensor<1xi32>
    %_grp_g100 = "ttir.concat"(%_wi_g100__0, %_wi_g100__1, %_wi_g100__2, %_wi_g100__3) <{dim = 0 : si32}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %_r0, %_r1, %_r2, %_r3, %_r4, %_r5, %_r6, %_r7, %_r8, %_r9, %_r10, %_r11, %_r12, %_r13, %_r14, %_r15, %_r16, %_r17, %_r18, %_r19, %_r20, %_r21, %_r22, %_r23, %_r24, %_r25, %_r26, %_r27 = call @wide_datapath_impl(%_wi_a4, %_wi_a7, %_wi_b12, %_wi_b16, %_wi_c24, %_wi_c32, %_grp_d48, %_grp_d64, %_grp_e96, %_grp_e128, %_grp_f33, %_grp_f40, %_grp_g65, %_grp_g100) : (tensor<1xi8>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<3xi32>, tensor<4xi32>, tensor<2xi32>, tensor<2xi32>, tensor<3xi32>, tensor<4xi32>) -> (tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<1xi32>, tensor<2xi32>, tensor<4xi32>, tensor<3xi32>, tensor<1xi8>, tensor<3xi32>, tensor<1xi8>, tensor<3xi32>, tensor<1xi32>, tensor<4xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<4xi32>, tensor<3xi32>, tensor<1xi8>)
    %_uo_add_i8 = "ttir.typecast"(%_r0) : (tensor<1xi8>) -> tensor<1xui8>
    %_uo_and_i16 = "ttir.typecast"(%_r1) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_or_i32 = "ttir.typecast"(%_r2) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_i64__0 = "ttir.slice_static"(%_r3) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_xor_i64__0 = "ttir.typecast"(%_sl_xor_i64__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_i64__1 = "ttir.slice_static"(%_r3) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_xor_i64__1 = "ttir.typecast"(%_sl_xor_i64__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_add_cross = "ttir.typecast"(%_r4) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_and_cross__0 = "ttir.slice_static"(%_r5) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_and_cross__0 = "ttir.typecast"(%_sl_and_cross__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_and_cross__1 = "ttir.slice_static"(%_r5) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_and_cross__1 = "ttir.typecast"(%_sl_and_cross__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_wide__0 = "ttir.slice_static"(%_r6) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_wide__0 = "ttir.typecast"(%_sl_xor_wide__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_wide__1 = "ttir.slice_static"(%_r6) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_wide__1 = "ttir.typecast"(%_sl_xor_wide__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_wide__2 = "ttir.slice_static"(%_r6) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_wide__2 = "ttir.typecast"(%_sl_xor_wide__2) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_wide__3 = "ttir.slice_static"(%_r6) <{begins = [3 : i32], ends = [4 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_wide__3 = "ttir.typecast"(%_sl_xor_wide__3) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_add_wide__0 = "ttir.slice_static"(%_r7) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_add_wide__0 = "ttir.typecast"(%_sl_add_wide__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_add_wide__1 = "ttir.slice_static"(%_r7) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_add_wide__1 = "ttir.typecast"(%_sl_add_wide__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_add_wide__2 = "ttir.slice_static"(%_r7) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_add_wide__2 = "ttir.typecast"(%_sl_add_wide__2) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_extract_lo = "ttir.typecast"(%_r8) : (tensor<1xi8>) -> tensor<1xui8>
    %_sl_concat_wide__0 = "ttir.slice_static"(%_r9) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_concat_wide__0 = "ttir.typecast"(%_sl_concat_wide__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_concat_wide__1 = "ttir.slice_static"(%_r9) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_concat_wide__1 = "ttir.typecast"(%_sl_concat_wide__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_concat_wide__2 = "ttir.slice_static"(%_r9) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_concat_wide__2 = "ttir.typecast"(%_sl_concat_wide__2) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_sub_i8 = "ttir.typecast"(%_r10) : (tensor<1xi8>) -> tensor<1xui8>
    %_sl_sub_wide__0 = "ttir.slice_static"(%_r11) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_sub_wide__0 = "ttir.typecast"(%_sl_sub_wide__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_sub_wide__1 = "ttir.slice_static"(%_r11) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_sub_wide__1 = "ttir.typecast"(%_sl_sub_wide__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_sub_wide__2 = "ttir.slice_static"(%_r11) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_sub_wide__2 = "ttir.typecast"(%_sl_sub_wide__2) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_not_i32 = "ttir.typecast"(%_r12) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_not_wide__0 = "ttir.slice_static"(%_r13) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_not_wide__0 = "ttir.typecast"(%_sl_not_wide__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_not_wide__1 = "ttir.slice_static"(%_r13) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_not_wide__1 = "ttir.typecast"(%_sl_not_wide__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_not_wide__2 = "ttir.slice_static"(%_r13) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_not_wide__2 = "ttir.typecast"(%_sl_not_wide__2) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_not_wide__3 = "ttir.slice_static"(%_r13) <{begins = [3 : i32], ends = [4 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_not_wide__3 = "ttir.typecast"(%_sl_not_wide__3) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_eq_i32 = "ttir.typecast"(%_r14) : (tensor<1xi8>) -> tensor<1xui8>
    %_uo_ne_wide = "ttir.typecast"(%_r15) : (tensor<1xi8>) -> tensor<1xui8>
    %_uo_lt_i16 = "ttir.typecast"(%_r16) : (tensor<1xi8>) -> tensor<1xui8>
    %_uo_ult_wide = "ttir.typecast"(%_r17) : (tensor<1xi8>) -> tensor<1xui8>
    %_uo_slt_i32 = "ttir.typecast"(%_r18) : (tensor<1xi8>) -> tensor<1xui8>
    %_uo_sge_wide = "ttir.typecast"(%_r19) : (tensor<1xi8>) -> tensor<1xui8>
    %_uo_shl_i32 = "ttir.typecast"(%_r20) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_shr_i32 = "ttir.typecast"(%_r21) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_mux_i32 = "ttir.typecast"(%_r22) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_mux_wide__0 = "ttir.slice_static"(%_r23) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_mux_wide__0 = "ttir.typecast"(%_sl_mux_wide__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_mux_wide__1 = "ttir.slice_static"(%_r23) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_mux_wide__1 = "ttir.typecast"(%_sl_mux_wide__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_add_odd__0 = "ttir.slice_static"(%_r24) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_add_odd__0 = "ttir.typecast"(%_sl_add_odd__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_add_odd__1 = "ttir.slice_static"(%_r24) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<2xi32>) -> tensor<1xi32>
    %_uo_add_odd__1 = "ttir.typecast"(%_sl_add_odd__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_odd__0 = "ttir.slice_static"(%_r25) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_odd__0 = "ttir.typecast"(%_sl_xor_odd__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_odd__1 = "ttir.slice_static"(%_r25) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_odd__1 = "ttir.typecast"(%_sl_xor_odd__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_odd__2 = "ttir.slice_static"(%_r25) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_odd__2 = "ttir.typecast"(%_sl_xor_odd__2) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_xor_odd__3 = "ttir.slice_static"(%_r25) <{begins = [3 : i32], ends = [4 : i32], step = [1 : i32]}> : (tensor<4xi32>) -> tensor<1xi32>
    %_uo_xor_odd__3 = "ttir.typecast"(%_sl_xor_odd__3) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_sub_odd__0 = "ttir.slice_static"(%_r26) <{begins = [0 : i32], ends = [1 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_sub_odd__0 = "ttir.typecast"(%_sl_sub_odd__0) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_sub_odd__1 = "ttir.slice_static"(%_r26) <{begins = [1 : i32], ends = [2 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_sub_odd__1 = "ttir.typecast"(%_sl_sub_odd__1) : (tensor<1xi32>) -> tensor<1xui32>
    %_sl_sub_odd__2 = "ttir.slice_static"(%_r26) <{begins = [2 : i32], ends = [3 : i32], step = [1 : i32]}> : (tensor<3xi32>) -> tensor<1xi32>
    %_uo_sub_odd__2 = "ttir.typecast"(%_sl_sub_odd__2) : (tensor<1xi32>) -> tensor<1xui32>
    %_uo_eq_odd = "ttir.typecast"(%_r27) : (tensor<1xi8>) -> tensor<1xui8>
    return %_uo_add_i8, %_uo_and_i16, %_uo_or_i32, %_uo_xor_i64__0, %_uo_xor_i64__1, %_uo_add_cross, %_uo_and_cross__0, %_uo_and_cross__1, %_uo_xor_wide__0, %_uo_xor_wide__1, %_uo_xor_wide__2, %_uo_xor_wide__3, %_uo_add_wide__0, %_uo_add_wide__1, %_uo_add_wide__2, %_uo_extract_lo, %_uo_concat_wide__0, %_uo_concat_wide__1, %_uo_concat_wide__2, %_uo_sub_i8, %_uo_sub_wide__0, %_uo_sub_wide__1, %_uo_sub_wide__2, %_uo_not_i32, %_uo_not_wide__0, %_uo_not_wide__1, %_uo_not_wide__2, %_uo_not_wide__3, %_uo_eq_i32, %_uo_ne_wide, %_uo_lt_i16, %_uo_ult_wide, %_uo_slt_i32, %_uo_sge_wide, %_uo_shl_i32, %_uo_shr_i32, %_uo_mux_i32, %_uo_mux_wide__0, %_uo_mux_wide__1, %_uo_add_odd__0, %_uo_add_odd__1, %_uo_xor_odd__0, %_uo_xor_odd__1, %_uo_xor_odd__2, %_uo_xor_odd__3, %_uo_sub_odd__0, %_uo_sub_odd__1, %_uo_sub_odd__2, %_uo_eq_odd : tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui8>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui8>
  }
}
