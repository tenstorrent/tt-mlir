module @IrToHlo.24214 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg1: tensor<2048x512xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg2: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg3: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg4: tensor<2048x4096xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg5: tensor<8192x2048xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg6: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg7: tensor<2048x512xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg8: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg9: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg10: tensor<2048x4096xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg11: tensor<4096x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg12: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg13: tensor<2048x512xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg14: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg15: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg16: tensor<2048x4096xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg17: tensor<4096x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg18: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg19: tensor<2048x512xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg20: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg21: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg22: tensor<2048x4096xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg23: tensor<4096x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg24: tensor<2048xbf16> {mhlo.sharding = "{replicated}"}, %arg25: tensor<1x494x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg26: tensor<128xbf16> {mhlo.sharding = "{replicated}"}, %arg27: tensor<32xbf16> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg28: tensor<32x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg29: tensor<32xbf16> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg30: tensor<32x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg31: tensor<8192x1x4xbf16> {mhlo.sharding = "{replicated}"}, %arg32: tensor<8192x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg33: tensor<i1> {mhlo.sharding = "{replicated}"}, %arg34: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg35: tensor<1x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg36: tensor<256x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg37: tensor<256x2048x512xbf16> {mhlo.sharding = "{replicated}"}, %arg38: tensor<256x1024x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg39: tensor<128xbf16> {mhlo.sharding = "{replicated}"}, %arg40: tensor<32xbf16> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg41: tensor<32x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg42: tensor<32xbf16> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg43: tensor<32x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg44: tensor<8192x1x4xbf16> {mhlo.sharding = "{replicated}"}, %arg45: tensor<8192x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg46: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg47: tensor<1x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg48: tensor<256x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg49: tensor<256x2048x512xbf16> {mhlo.sharding = "{replicated}"}, %arg50: tensor<256x1024x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg51: tensor<128xbf16> {mhlo.sharding = "{replicated}"}, %arg52: tensor<32xbf16> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg53: tensor<32x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg54: tensor<32xbf16> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg55: tensor<32x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg56: tensor<8192x1x4xbf16> {mhlo.sharding = "{replicated}"}, %arg57: tensor<8192x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg58: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg59: tensor<1x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg60: tensor<256x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg61: tensor<256x2048x512xbf16> {mhlo.sharding = "{replicated}"}, %arg62: tensor<256x1024x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg63: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg64: tensor<1x1x494x494xi1> {mhlo.sharding = "{replicated}"}, %arg65: tensor<256xbf16> {mhlo.sharding = "{replicated}"}, %arg66: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg67: tensor<3x1x494xi64> {mhlo.sharding = "{replicated}"}, %arg68: tensor<32xf32> {mhlo.sharding = "{replicated}"}, %arg69: tensor<256xbf16> {mhlo.sharding = "{replicated}"}, %arg70: tensor<512x2048xbf16> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg71: tensor<1x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg72: tensor<256x2048xbf16> {mhlo.sharding = "{replicated}"}, %arg73: tensor<256x2048x512xbf16> {mhlo.sharding = "{replicated}"}, %arg74: tensor<256x1024x2048xbf16> {mhlo.sharding = "{replicated}"}) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x16x494x494xf32>
    %cst_0 = stablehlo.constant dense<0xFFF0000000000000> : tensor<1x16x494x494xf64>
    %cst_1 = stablehlo.constant dense<0xFF80> : tensor<1x1x494x494xbf16>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x494x494xbf16>
    %cst_3 = stablehlo.constant dense<2.500000e-01> : tensor<1x16x256x494xf32>
    %cst_4 = stablehlo.constant dense<2.500000e-01> : tensor<1x16x494x256xf32>
    %c = stablehlo.constant dense<9> : tensor<32xi64>
    %cst_5 = stablehlo.constant dense<[-0.666666686, -0.333333343, 0.000000e+00, 0.333333343, 0.666666686, 1.000000e+00, 1.33333337, 1.66666663, 2.000000e+00, 2.33333325, 2.66666675, 3.000000e+00, 3.33333325, 3.66666675, 4.000000e+00, 4.33333349, 4.66666651, 5.000000e+00, 5.33333349, 5.66666651, 6.000000e+00, 6.33333349, 6.66666651, 7.000000e+00, 7.33333349, 7.66666651, 8.000000e+00, 8.33333301, 8.66666698, 9.000000e+00, 9.33333301, 9.66666698]> : tensor<32xf32>
    %c_6 = stablehlo.constant dense<11> : tensor<32xi64>
    %c_7 = stablehlo.constant dense<10> : tensor<32xi64>
    %cst_8 = stablehlo.constant dense<[-0.333333343, 0.000000e+00, 0.333333343, 0.666666686, 1.000000e+00, 1.33333337, 1.66666663, 2.000000e+00, 2.33333325, 2.66666675, 3.000000e+00, 3.33333325, 3.66666675, 4.000000e+00, 4.33333349, 4.66666651, 5.000000e+00, 5.33333349, 5.66666651, 6.000000e+00, 6.33333349, 6.66666651, 7.000000e+00, 7.33333349, 7.66666651, 8.000000e+00, 8.33333301, 8.66666698, 9.000000e+00, 9.33333301, 9.66666698, 1.000000e+01]> : tensor<32xf32>
    %c_9 = stablehlo.constant dense<0> : tensor<32xi64>
    %c_10 = stablehlo.constant dense<[false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true]> : tensor<32xi1>
    %c_11 = stablehlo.constant dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<32xi1>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<1x494x32xf32>
    %c_13 = stablehlo.constant dense<[false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false, false]> : tensor<32xi1>
    %c_14 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false]> : tensor<32xi1>
    %c_15 = stablehlo.constant dense<[false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<32xi1>
    %c_16 = stablehlo.constant dense<[[true], [false], [false]]> : tensor<3x1xi1>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<256xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<18x2048xbf16>
    %c_19 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xui16>
    %cst_20 = stablehlo.constant dense<"0x000000000000003D0000803D0000C03D0000003E0000203E0000403E0000603E0000803E0000903E0000A03E0000B03E0000C03E0000D03E0000E03E0000F03E0000003F0000083F0000103F0000183F0000203F0000283F0000303F0000383F0000403F0000483F0000503F0000583F0000603F0000683F0000703F0000783F0000803F0000843F0000883F00008C3F0000903F0000943F0000983F00009C3F0000A03F0000A43F0000A83F0000AC3F0000B03F0000B43F0000B83F0000BC3F0000C03F0000C43F0000C83F0000CC3F0000D03F0000D43F0000D83F0000DC3F0000E03F0000E43F0000E83F0000EC3F0000F03F0000F43F0000F83F0000FC3F000000400000024000000440000006400000084000000A4000000C4000000E40000010400000124000001440000016400000184000001A4000001C4000001E40000020400000224000002440000026400000284000002A4000002C4000002E40000030400000324000003440000036400000384000003A4000003C4000003E40000040400000424000004440000046400000484000004A4000004C4000004E40000050400000524000005440000056400000584000005A4000005C4000005E40000060400000624000006440000066400000684000006A4000006C4000006E40000070400000724000007440000076400000784000007A4000007C4000007E400000804000008140000082400000834000008440000085400000864000008740000088400000894000008A4000008B4000008C4000008D4000008E4000008F400000904000009140000092400000934000009440000095400000964000009740000098400000994000009A4000009B4000009C4000009D4000009E4000009F400000A0400000A1400000A2400000A3400000A4400000A5400000A6400000A7400000A8400000A9400000AA400000AB400000AC400000AD400000AE400000AF400000B0400000B1400000B2400000B3400000B4400000B5400000B6400000B7400000B8400000B9400000BA400000BB400000BC400000BD400000BE400000BF400000C0400000C1400000C2400000C3400000C4400000C5400000C6400000C7400000C8400000C9400000CA400000CB400000CC400000CD400000CE400000CF400000D0400000D1400000D2400000D3400000D4400000D5400000D6400000D7400000D8400000D9400000DA400000DB400000DC400000DD400000DE400000DF400000E0400000E1400000E2400000E3400000E4400000E5400000E6400000E7400000E8400000E9400000EA400000EB400000EC400000ED400000EE400000EF400000F0400000F1400000F2400000F3400000F4400000F5400000F6400000F7400000F8400000F9400000FA400000FB400000FC400000FD400000FE400000FF40"> : tensor<256xf32>
    %c_21 = stablehlo.constant dense<0> : tensor<18x8xi64>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<18x8xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x8x64x128xf32>
    %c_24 = stablehlo.constant dense<[[true, false, false, false, false, false, false, false]]> : tensor<1x8xi1>
    %c_25 = stablehlo.constant dense<[[false, true, false, false, false, false, false, false]]> : tensor<1x8xi1>
    %c_26 = stablehlo.constant dense<[[false, false, true, false, false, false, false, false]]> : tensor<1x8xi1>
    %c_27 = stablehlo.constant dense<[[false, false, false, true, false, false, false, false]]> : tensor<1x8xi1>
    %c_28 = stablehlo.constant dense<[[false, false, false, false, true, false, false, false]]> : tensor<1x8xi1>
    %c_29 = stablehlo.constant dense<[[false, false, false, false, false, true, false, false]]> : tensor<1x8xi1>
    %c_30 = stablehlo.constant dense<[[false, false, false, false, false, false, true, false]]> : tensor<1x8xi1>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x64x64xf32>
    %c_32 = stablehlo.constant dense<1> : tensor<64x64xi64>
    %c_33 = stablehlo.constant dense<63> : tensor<64xi64>
    %c_34 = stablehlo.constant dense<62> : tensor<64xi64>
    %c_35 = stablehlo.constant dense<61> : tensor<64xi64>
    %c_36 = stablehlo.constant dense<60> : tensor<64xi64>
    %c_37 = stablehlo.constant dense<59> : tensor<64xi64>
    %c_38 = stablehlo.constant dense<58> : tensor<64xi64>
    %c_39 = stablehlo.constant dense<57> : tensor<64xi64>
    %c_40 = stablehlo.constant dense<56> : tensor<64xi64>
    %c_41 = stablehlo.constant dense<55> : tensor<64xi64>
    %c_42 = stablehlo.constant dense<54> : tensor<64xi64>
    %c_43 = stablehlo.constant dense<53> : tensor<64xi64>
    %c_44 = stablehlo.constant dense<52> : tensor<64xi64>
    %c_45 = stablehlo.constant dense<51> : tensor<64xi64>
    %c_46 = stablehlo.constant dense<50> : tensor<64xi64>
    %c_47 = stablehlo.constant dense<49> : tensor<64xi64>
    %c_48 = stablehlo.constant dense<48> : tensor<64xi64>
    %c_49 = stablehlo.constant dense<47> : tensor<64xi64>
    %c_50 = stablehlo.constant dense<46> : tensor<64xi64>
    %c_51 = stablehlo.constant dense<45> : tensor<64xi64>
    %c_52 = stablehlo.constant dense<44> : tensor<64xi64>
    %c_53 = stablehlo.constant dense<43> : tensor<64xi64>
    %c_54 = stablehlo.constant dense<42> : tensor<64xi64>
    %c_55 = stablehlo.constant dense<41> : tensor<64xi64>
    %c_56 = stablehlo.constant dense<40> : tensor<64xi64>
    %c_57 = stablehlo.constant dense<39> : tensor<64xi64>
    %c_58 = stablehlo.constant dense<38> : tensor<64xi64>
    %c_59 = stablehlo.constant dense<37> : tensor<64xi64>
    %c_60 = stablehlo.constant dense<36> : tensor<64xi64>
    %c_61 = stablehlo.constant dense<35> : tensor<64xi64>
    %c_62 = stablehlo.constant dense<34> : tensor<64xi64>
    %c_63 = stablehlo.constant dense<33> : tensor<64xi64>
    %c_64 = stablehlo.constant dense<32> : tensor<64xi64>
    %c_65 = stablehlo.constant dense<31> : tensor<64xi64>
    %c_66 = stablehlo.constant dense<30> : tensor<64xi64>
    %c_67 = stablehlo.constant dense<29> : tensor<64xi64>
    %c_68 = stablehlo.constant dense<28> : tensor<64xi64>
    %c_69 = stablehlo.constant dense<27> : tensor<64xi64>
    %c_70 = stablehlo.constant dense<26> : tensor<64xi64>
    %c_71 = stablehlo.constant dense<25> : tensor<64xi64>
    %c_72 = stablehlo.constant dense<24> : tensor<64xi64>
    %c_73 = stablehlo.constant dense<23> : tensor<64xi64>
    %c_74 = stablehlo.constant dense<22> : tensor<64xi64>
    %c_75 = stablehlo.constant dense<21> : tensor<64xi64>
    %c_76 = stablehlo.constant dense<20> : tensor<64xi64>
    %c_77 = stablehlo.constant dense<19> : tensor<64xi64>
    %c_78 = stablehlo.constant dense<18> : tensor<64xi64>
    %c_79 = stablehlo.constant dense<17> : tensor<64xi64>
    %c_80 = stablehlo.constant dense<16> : tensor<64xi64>
    %c_81 = stablehlo.constant dense<15> : tensor<64xi64>
    %c_82 = stablehlo.constant dense<14> : tensor<64xi64>
    %c_83 = stablehlo.constant dense<13> : tensor<64xi64>
    %c_84 = stablehlo.constant dense<12> : tensor<64xi64>
    %c_85 = stablehlo.constant dense<11> : tensor<64xi64>
    %c_86 = stablehlo.constant dense<10> : tensor<64xi64>
    %c_87 = stablehlo.constant dense<9> : tensor<64xi64>
    %c_88 = stablehlo.constant dense<8> : tensor<64xi64>
    %c_89 = stablehlo.constant dense<7> : tensor<64xi64>
    %c_90 = stablehlo.constant dense<6> : tensor<64xi64>
    %c_91 = stablehlo.constant dense<5> : tensor<64xi64>
    %c_92 = stablehlo.constant dense<4> : tensor<64xi64>
    %c_93 = stablehlo.constant dense<3> : tensor<64xi64>
    %c_94 = stablehlo.constant dense<2> : tensor<64xi64>
    %c_95 = stablehlo.constant dense<1> : tensor<64xi64>
    %cst_96 = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01, 4.000000e+01, 4.100000e+01, 4.200000e+01, 4.300000e+01, 4.400000e+01, 4.500000e+01, 4.600000e+01, 4.700000e+01, 4.800000e+01, 4.900000e+01, 5.000000e+01, 5.100000e+01, 5.200000e+01, 5.300000e+01, 5.400000e+01, 5.500000e+01, 5.600000e+01, 5.700000e+01, 5.800000e+01, 5.900000e+01, 6.000000e+01, 6.100000e+01, 6.200000e+01, 6.300000e+01]> : tensor<64xf32>
    %c_97 = stablehlo.constant dense<0> : tensor<64xi64>
    %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x8x64x64xf32>
    %c_99 = stablehlo.constant dense<0> : tensor<64x64xi64>
    %c_100 = stablehlo.constant dense<[true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_101 = stablehlo.constant dense<[[false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_102 = stablehlo.constant dense<[true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_103 = stablehlo.constant dense<[[false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_104 = stablehlo.constant dense<[true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_105 = stablehlo.constant dense<[[false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_106 = stablehlo.constant dense<[true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_107 = stablehlo.constant dense<[[false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_108 = stablehlo.constant dense<[true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_109 = stablehlo.constant dense<[[false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_110 = stablehlo.constant dense<[true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_111 = stablehlo.constant dense<[[false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_112 = stablehlo.constant dense<[true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_113 = stablehlo.constant dense<[[false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_114 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_115 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_116 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_117 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_118 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_119 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_120 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_121 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_122 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_123 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_124 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_125 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_126 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_127 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_128 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_129 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_130 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_131 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_132 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_133 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_134 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_135 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_136 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_137 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_138 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_139 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_140 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_141 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_142 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_143 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_144 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_145 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_146 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_147 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_148 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_149 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_150 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_151 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_152 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_153 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_154 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_155 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_156 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_157 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_158 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_159 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_160 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_161 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_162 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_163 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_164 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_165 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_166 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_167 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_168 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_169 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_170 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_171 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_172 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_173 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_174 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_175 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_176 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_177 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_178 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_179 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_180 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_181 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_182 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_183 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_184 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_185 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_186 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_187 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_188 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_189 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_190 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_191 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_192 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_193 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_194 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_195 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_196 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_197 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_198 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_199 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_200 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_201 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_202 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_203 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_204 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_205 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_206 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_207 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_208 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_209 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_210 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_211 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_212 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : tensor<64xi1>
    %c_213 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_214 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false]> : tensor<64xi1>
    %c_215 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false]]> : tensor<1x64xi1>
    %c_216 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : tensor<64xi1>
    %c_217 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false]]> : tensor<1x64xi1>
    %c_218 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false]> : tensor<64xi1>
    %c_219 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false]]> : tensor<1x64xi1>
    %c_220 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false]> : tensor<64xi1>
    %c_221 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false]]> : tensor<1x64xi1>
    %c_222 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false]> : tensor<64xi1>
    %c_223 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false]]> : tensor<1x64xi1>
    %cst_224 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x8x64xf32>
    %c_225 = stablehlo.constant dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false]> : tensor<64xi1>
    %c_226 = stablehlo.constant dense<[[false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]]> : tensor<1x64xi1>
    %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<1x32x128x128xf32>
    %cst_228 = stablehlo.constant dense<2.000000e+01> : tensor<1x494x32xf32>
    %cst_229 = stablehlo.constant dense<0.0883883461> : tensor<1x32x512x128xf32>
    %cst_230 = stablehlo.constant dense<9.983770e-07> : tensor<1x494x32x1xbf16>
    %cst_231 = stablehlo.constant dense<1.000000e+00> : tensor<2048xf32>
    %c_232 = stablehlo.constant dense<[[false, false, false, false, false, false, false, true]]> : tensor<1x8xi1>
    %c_233 = stablehlo.constant dense<true> : tensor<i1>
    %c_234 = stablehlo.constant dense<false> : tensor<i1>
    %c_235 = stablehlo.constant dense<"0x00000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D000000000000001E000000000000001F0000000000000020000000000000002100000000000000220000000000000023000000000000002400000000000000250000000000000026000000000000002700000000000000280000000000000029000000000000002A000000000000002B000000000000002C000000000000002D000000000000002E000000000000002F0000000000000030000000000000003100000000000000320000000000000033000000000000003400000000000000350000000000000036000000000000003700000000000000380000000000000039000000000000003A000000000000003B000000000000003C000000000000003D000000000000003E000000000000003F0000000000000040000000000000004100000000000000420000000000000043000000000000004400000000000000450000000000000046000000000000004700000000000000480000000000000049000000000000004A000000000000004B000000000000004C000000000000004D000000000000004E000000000000004F0000000000000050000000000000005100000000000000520000000000000053000000000000005400000000000000550000000000000056000000000000005700000000000000580000000000000059000000000000005A000000000000005B000000000000005C000000000000005D000000000000005E000000000000005F0000000000000060000000000000006100000000000000620000000000000063000000000000006400000000000000650000000000000066000000000000006700000000000000680000000000000069000000000000006A000000000000006B000000000000006C000000000000006D000000000000006E000000000000006F0000000000000070000000000000007100000000000000720000000000000073000000000000007400000000000000750000000000000076000000000000007700000000000000780000000000000079000000000000007A000000000000007B000000000000007C000000000000007D000000000000007E000000000000007F0000000000000080000000000000008100000000000000820000000000000083000000000000008400000000000000850000000000000086000000000000008700000000000000880000000000000089000000000000008A000000000000008B000000000000008C000000000000008D000000000000008E000000000000008F0000000000000090000000000000009100000000000000920000000000000093000000000000009400000000000000950000000000000096000000000000009700000000000000980000000000000099000000000000009A000000000000009B000000000000009C000000000000009D000000000000009E000000000000009F00000000000000A000000000000000A100000000000000A200000000000000A300000000000000A400000000000000A500000000000000A600000000000000A700000000000000A800000000000000A900000000000000AA00000000000000AB00000000000000AC00000000000000AD00000000000000AE00000000000000AF00000000000000B000000000000000B100000000000000B200000000000000B300000000000000B400000000000000B500000000000000B600000000000000B700000000000000B800000000000000B900000000000000BA00000000000000BB00000000000000BC00000000000000BD00000000000000BE00000000000000BF00000000000000C000000000000000C100000000000000C200000000000000C300000000000000C400000000000000C500000000000000C600000000000000C700000000000000C800000000000000C900000000000000CA00000000000000CB00000000000000CC00000000000000CD00000000000000CE00000000000000CF00000000000000D000000000000000D100000000000000D200000000000000D300000000000000D400000000000000D500000000000000D600000000000000D700000000000000D800000000000000D900000000000000DA00000000000000DB00000000000000DC00000000000000DD00000000000000DE00000000000000DF00000000000000E000000000000000E100000000000000E200000000000000E300000000000000E400000000000000E500000000000000E600000000000000E700000000000000E800000000000000E900000000000000EA00000000000000EB00000000000000EC00000000000000ED00000000000000EE00000000000000EF00000000000000F000000000000000F100000000000000F200000000000000F300000000000000F400000000000000F500000000000000F600000000000000F700000000000000F800000000000000F900000000000000FA00000000000000FB00000000000000FC00000000000000FD00000000000000FE00000000000000FF00000000000000"> : tensor<256xi64>
    %cst_236 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_237 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xui8>
    %c_238 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
    %cst_239 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.custom_call @tt.mark_argument(%arg25) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "input", ttir.name = "arg72_1"}} : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %1 = stablehlo.broadcast_in_dim %c_232, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2 = stablehlo.reshape %arg24 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %3 = stablehlo.custom_call @tt.mark_argument(%2) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.input_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %4 = stablehlo.reshape %3 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %5 = stablehlo.convert %4 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %6 = stablehlo.add %5, %cst_231 : tensor<2048xf32>
    %7 = stablehlo.composite "tenstorrent.rms_norm" %0, %6 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_2} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %8 = stablehlo.reshape %7 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %9 = stablehlo.reshape %arg32 : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %10 = stablehlo.custom_call @tt.mark_argument(%9) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.in_proj_qkv.weight"}} : (tensor<1x8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %11 = stablehlo.reshape %10 : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %12 = stablehlo.transpose %11, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,8192]{0,1}"} : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %13 = stablehlo.dot_general %8, %12, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<494x8192xbf16>
    %14 = stablehlo.reshape %13 : (tensor<494x8192xbf16>) -> tensor<1x494x8192xbf16>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,8192,494]{1,2,0}"} : (tensor<1x494x8192xbf16>) -> tensor<1x8192x494xbf16>
    %16 = stablehlo.reshape %15 : (tensor<1x8192x494xbf16>) -> tensor<1x8192x1x494xbf16>
    %17 = stablehlo.custom_call @tt.mark_argument(%arg31) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.conv1d.weight"}} : (tensor<8192x1x4xbf16>) -> tensor<8192x1x4xbf16>
    %18 = stablehlo.reshape %17 : (tensor<8192x1x4xbf16>) -> tensor<8192x1x1x4xbf16>
    %19 = stablehlo.convolution(%16, %18) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 8192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x8192x1x494xbf16>, tensor<8192x1x1x4xbf16>) -> tensor<1x8192x1x497xbf16>
    %20 = stablehlo.reshape %19 : (tensor<1x8192x1x497xbf16>) -> tensor<1x8192x497xbf16>
    %21 = stablehlo.custom_call @tt.sharding_constraint(%20) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>"}} : (tensor<1x8192x497xbf16>) -> tensor<1x8192x497xbf16>
    %22 = stablehlo.slice %21 [0:1, 0:8192, 0:494] : (tensor<1x8192x497xbf16>) -> tensor<1x8192x494xbf16>
    %23 = stablehlo.convert %22 : (tensor<1x8192x494xbf16>) -> tensor<1x8192x494xf32>
    %24 = stablehlo.logistic %23 : tensor<1x8192x494xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<1x8192x494xf32>
    %26 = stablehlo.convert %25 : (tensor<1x8192x494xf32>) -> tensor<1x8192x494xbf16>
    %27 = stablehlo.transpose %26, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,494,8192]{1,2,0}"} : (tensor<1x8192x494xbf16>) -> tensor<1x494x8192xbf16>
    %28 = stablehlo.slice %27 [0:1, 0:494, 0:2048] : (tensor<1x494x8192xbf16>) -> tensor<1x494x2048xbf16>
    %29 = stablehlo.reshape %28 : (tensor<1x494x2048xbf16>) -> tensor<1x494x16x128xbf16>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 4] : (tensor<1x494x16x128xbf16>) -> tensor<1x494x16x2x128xbf16>
    %31 = stablehlo.reshape %30 : (tensor<1x494x16x2x128xbf16>) -> tensor<1x494x32x128xbf16>
    %32 = stablehlo.multiply %31, %31 : tensor<1x494x32x128xbf16>
    %33 = stablehlo.reduce(%32 init: %cst_239) applies stablehlo.add across dimensions = [3] : (tensor<1x494x32x128xbf16>, tensor<bf16>) -> tensor<1x494x32xbf16>
    %34 = stablehlo.reshape %33 : (tensor<1x494x32xbf16>) -> tensor<1x494x32x1xbf16>
    %35 = stablehlo.add %34, %cst_230 : tensor<1x494x32x1xbf16>
    %36 = stablehlo.rsqrt %35 : tensor<1x494x32x1xbf16>
    %37 = stablehlo.reshape %36 : (tensor<1x494x32x1xbf16>) -> tensor<1x494x32xbf16>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1, 2] : (tensor<1x494x32xbf16>) -> tensor<1x494x32x128xbf16>
    %39 = stablehlo.multiply %31, %38 : tensor<1x494x32x128xbf16>
    %40 = stablehlo.transpose %39, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %41 = stablehlo.convert %40 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %42 = stablehlo.pad %41, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %43 = stablehlo.multiply %42, %cst_229 : tensor<1x32x512x128xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %45 = stablehlo.slice %44 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %46 = stablehlo.reshape %45 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %47 = stablehlo.reshape %arg29 : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %48 = stablehlo.custom_call @tt.mark_argument(%47) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.A_log"}} : (tensor<1x1x32xbf16>) -> tensor<1x1x32xbf16>
    %49 = stablehlo.reshape %48 : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
    %50 = stablehlo.convert %49 : (tensor<32xbf16>) -> tensor<32xf32>
    %51 = stablehlo.exponential %50 : tensor<32xf32>
    %52 = stablehlo.negate %51 : tensor<32xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [2] : (tensor<32xf32>) -> tensor<1x494x32xf32>
    %54 = stablehlo.reshape %arg28 : (tensor<32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %55 = stablehlo.custom_call @tt.mark_argument(%54) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.in_proj_a.weight"}} : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %56 = stablehlo.reshape %55 : (tensor<1x32x2048xbf16>) -> tensor<32x2048xbf16>
    %57 = stablehlo.transpose %56, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,32]{0,1}"} : (tensor<32x2048xbf16>) -> tensor<2048x32xbf16>
    %58 = stablehlo.dot_general %8, %57, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x32xbf16>) -> tensor<494x32xbf16>
    %59 = stablehlo.reshape %58 : (tensor<494x32xbf16>) -> tensor<1x494x32xbf16>
    %60 = stablehlo.convert %59 : (tensor<1x494x32xbf16>) -> tensor<1x494x32xf32>
    %61 = stablehlo.reshape %arg27 : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %62 = stablehlo.custom_call @tt.mark_argument(%61) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.dt_bias"}} : (tensor<1x1x32xbf16>) -> tensor<1x1x32xbf16>
    %63 = stablehlo.reshape %62 : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
    %64 = stablehlo.convert %63 : (tensor<32xbf16>) -> tensor<32xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [2] : (tensor<32xf32>) -> tensor<1x494x32xf32>
    %66 = stablehlo.add %60, %65 : tensor<1x494x32xf32>
    %67 = stablehlo.compare  GT, %66, %cst_228 : (tensor<1x494x32xf32>, tensor<1x494x32xf32>) -> tensor<1x494x32xi1>
    %68 = stablehlo.exponential %66 : tensor<1x494x32xf32>
    %69 = stablehlo.log_plus_one %68 : tensor<1x494x32xf32>
    %70 = stablehlo.select %67, %66, %69 : tensor<1x494x32xi1>, tensor<1x494x32xf32>
    %71 = stablehlo.multiply %53, %70 : tensor<1x494x32xf32>
    %72 = stablehlo.transpose %71, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,494]{1,2,0}"} : (tensor<1x494x32xf32>) -> tensor<1x32x494xf32>
    %73 = stablehlo.pad %72, %cst_240, low = [0, 0, 0], high = [0, 0, 18], interior = [0, 0, 0] : (tensor<1x32x494xf32>, tensor<f32>) -> tensor<1x32x512xf32>
    %74 = stablehlo.reshape %73 : (tensor<1x32x512xf32>) -> tensor<1x32x8x64xf32>
    %75 = "stablehlo.reduce_window"(%74, %cst_240) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [63, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 64>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg75: tensor<f32>, %arg76: tensor<f32>):
      %5568 = stablehlo.add %arg75, %arg76 : tensor<f32>
      stablehlo.return %5568 : tensor<f32>
    }) : (tensor<1x32x8x64xf32>, tensor<f32>) -> tensor<1x32x8x64xf32>
    %76 = stablehlo.slice %75 [0:1, 0:32, 7:8, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %78 = stablehlo.exponential %77 : tensor<1x32x64x1xf32>
    %79 = stablehlo.reshape %78 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %81 = stablehlo.multiply %46, %80 : tensor<1x32x64x128xf32>
    %82 = stablehlo.slice %75 [0:1, 0:32, 0:1, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %83 = stablehlo.reshape %82 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %84 = stablehlo.slice %83 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %85 = stablehlo.reshape %84 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %86 = stablehlo.exponential %85 : tensor<1x32x1x1xf32>
    %87 = stablehlo.reshape %86 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %89 = stablehlo.multiply %88, %cst_227 : tensor<1x32x128x128xf32>
    %90 = stablehlo.slice %27 [0:1, 0:494, 2048:4096] : (tensor<1x494x8192xbf16>) -> tensor<1x494x2048xbf16>
    %91 = stablehlo.reshape %90 : (tensor<1x494x2048xbf16>) -> tensor<1x494x16x128xbf16>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2, 4] : (tensor<1x494x16x128xbf16>) -> tensor<1x494x16x2x128xbf16>
    %93 = stablehlo.reshape %92 : (tensor<1x494x16x2x128xbf16>) -> tensor<1x494x32x128xbf16>
    %94 = stablehlo.multiply %93, %93 : tensor<1x494x32x128xbf16>
    %95 = stablehlo.reduce(%94 init: %cst_239) applies stablehlo.add across dimensions = [3] : (tensor<1x494x32x128xbf16>, tensor<bf16>) -> tensor<1x494x32xbf16>
    %96 = stablehlo.reshape %95 : (tensor<1x494x32xbf16>) -> tensor<1x494x32x1xbf16>
    %97 = stablehlo.add %96, %cst_230 : tensor<1x494x32x1xbf16>
    %98 = stablehlo.rsqrt %97 : tensor<1x494x32x1xbf16>
    %99 = stablehlo.reshape %98 : (tensor<1x494x32x1xbf16>) -> tensor<1x494x32xbf16>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2] : (tensor<1x494x32xbf16>) -> tensor<1x494x32x128xbf16>
    %101 = stablehlo.multiply %93, %100 : tensor<1x494x32x128xbf16>
    %102 = stablehlo.transpose %101, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %103 = stablehlo.convert %102 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %104 = stablehlo.pad %103, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %105 = stablehlo.reshape %104 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %106 = stablehlo.slice %105 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %107 = stablehlo.reshape %106 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %108 = stablehlo.reshape %84 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %110 = stablehlo.subtract %109, %83 : tensor<1x32x64xf32>
    %111 = stablehlo.exponential %110 : tensor<1x32x64xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %113 = stablehlo.multiply %107, %112 : tensor<1x32x64x128xf32>
    %114 = stablehlo.transpose %113, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %115 = stablehlo.broadcast_in_dim %c_226, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %116 = stablehlo.broadcast_in_dim %arg33, dims = [] : (tensor<i1>) -> tensor<64xi1>
    %117 = stablehlo.and %116, %c_225 : tensor<64xi1>
    %118 = stablehlo.reshape %117 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %119 = stablehlo.reshape %117 : (tensor<64xi1>) -> tensor<1x64xi1>
    %120 = stablehlo.broadcast_in_dim %119, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %121 = stablehlo.not %118 : tensor<1x1x1x64xi1>
    %122 = stablehlo.reshape %121 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %124 = stablehlo.broadcast_in_dim %c_223, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %125 = stablehlo.and %116, %c_222 : tensor<64xi1>
    %126 = stablehlo.reshape %125 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %127 = stablehlo.reshape %125 : (tensor<64xi1>) -> tensor<1x64xi1>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %129 = stablehlo.not %126 : tensor<1x1x1x64xi1>
    %130 = stablehlo.reshape %129 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %132 = stablehlo.broadcast_in_dim %c_221, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %133 = stablehlo.and %116, %c_220 : tensor<64xi1>
    %134 = stablehlo.reshape %133 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %135 = stablehlo.reshape %133 : (tensor<64xi1>) -> tensor<1x64xi1>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %137 = stablehlo.not %134 : tensor<1x1x1x64xi1>
    %138 = stablehlo.reshape %137 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %140 = stablehlo.broadcast_in_dim %c_219, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %141 = stablehlo.and %116, %c_218 : tensor<64xi1>
    %142 = stablehlo.reshape %141 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %143 = stablehlo.reshape %141 : (tensor<64xi1>) -> tensor<1x64xi1>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %145 = stablehlo.not %142 : tensor<1x1x1x64xi1>
    %146 = stablehlo.reshape %145 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %148 = stablehlo.broadcast_in_dim %c_217, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %149 = stablehlo.and %116, %c_216 : tensor<64xi1>
    %150 = stablehlo.reshape %149 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %151 = stablehlo.reshape %149 : (tensor<64xi1>) -> tensor<1x64xi1>
    %152 = stablehlo.broadcast_in_dim %151, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %153 = stablehlo.not %150 : tensor<1x1x1x64xi1>
    %154 = stablehlo.reshape %153 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %156 = stablehlo.broadcast_in_dim %c_215, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %157 = stablehlo.and %116, %c_214 : tensor<64xi1>
    %158 = stablehlo.reshape %157 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %159 = stablehlo.reshape %157 : (tensor<64xi1>) -> tensor<1x64xi1>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %161 = stablehlo.not %158 : tensor<1x1x1x64xi1>
    %162 = stablehlo.reshape %161 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %163 = stablehlo.broadcast_in_dim %162, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %164 = stablehlo.broadcast_in_dim %c_213, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %165 = stablehlo.and %116, %c_212 : tensor<64xi1>
    %166 = stablehlo.reshape %165 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %167 = stablehlo.reshape %165 : (tensor<64xi1>) -> tensor<1x64xi1>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %169 = stablehlo.not %166 : tensor<1x1x1x64xi1>
    %170 = stablehlo.reshape %169 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %172 = stablehlo.broadcast_in_dim %c_211, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %173 = stablehlo.and %116, %c_210 : tensor<64xi1>
    %174 = stablehlo.reshape %173 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %175 = stablehlo.reshape %173 : (tensor<64xi1>) -> tensor<1x64xi1>
    %176 = stablehlo.broadcast_in_dim %175, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %177 = stablehlo.not %174 : tensor<1x1x1x64xi1>
    %178 = stablehlo.reshape %177 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %180 = stablehlo.broadcast_in_dim %c_209, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %181 = stablehlo.and %116, %c_208 : tensor<64xi1>
    %182 = stablehlo.reshape %181 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %183 = stablehlo.reshape %181 : (tensor<64xi1>) -> tensor<1x64xi1>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %185 = stablehlo.not %182 : tensor<1x1x1x64xi1>
    %186 = stablehlo.reshape %185 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %188 = stablehlo.broadcast_in_dim %c_207, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %189 = stablehlo.and %116, %c_206 : tensor<64xi1>
    %190 = stablehlo.reshape %189 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %191 = stablehlo.reshape %189 : (tensor<64xi1>) -> tensor<1x64xi1>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %193 = stablehlo.not %190 : tensor<1x1x1x64xi1>
    %194 = stablehlo.reshape %193 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %195 = stablehlo.broadcast_in_dim %194, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %196 = stablehlo.broadcast_in_dim %c_205, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %197 = stablehlo.and %116, %c_204 : tensor<64xi1>
    %198 = stablehlo.reshape %197 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %199 = stablehlo.reshape %197 : (tensor<64xi1>) -> tensor<1x64xi1>
    %200 = stablehlo.broadcast_in_dim %199, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %201 = stablehlo.not %198 : tensor<1x1x1x64xi1>
    %202 = stablehlo.reshape %201 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %204 = stablehlo.broadcast_in_dim %c_203, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %205 = stablehlo.and %116, %c_202 : tensor<64xi1>
    %206 = stablehlo.reshape %205 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %207 = stablehlo.reshape %205 : (tensor<64xi1>) -> tensor<1x64xi1>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %209 = stablehlo.not %206 : tensor<1x1x1x64xi1>
    %210 = stablehlo.reshape %209 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %211 = stablehlo.broadcast_in_dim %210, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %212 = stablehlo.broadcast_in_dim %c_201, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %213 = stablehlo.and %116, %c_200 : tensor<64xi1>
    %214 = stablehlo.reshape %213 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %215 = stablehlo.reshape %213 : (tensor<64xi1>) -> tensor<1x64xi1>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %217 = stablehlo.not %214 : tensor<1x1x1x64xi1>
    %218 = stablehlo.reshape %217 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %220 = stablehlo.broadcast_in_dim %c_199, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %221 = stablehlo.and %116, %c_198 : tensor<64xi1>
    %222 = stablehlo.reshape %221 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %223 = stablehlo.reshape %221 : (tensor<64xi1>) -> tensor<1x64xi1>
    %224 = stablehlo.broadcast_in_dim %223, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %225 = stablehlo.not %222 : tensor<1x1x1x64xi1>
    %226 = stablehlo.reshape %225 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %228 = stablehlo.broadcast_in_dim %c_197, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %229 = stablehlo.and %116, %c_196 : tensor<64xi1>
    %230 = stablehlo.reshape %229 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %231 = stablehlo.reshape %229 : (tensor<64xi1>) -> tensor<1x64xi1>
    %232 = stablehlo.broadcast_in_dim %231, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %233 = stablehlo.not %230 : tensor<1x1x1x64xi1>
    %234 = stablehlo.reshape %233 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %235 = stablehlo.broadcast_in_dim %234, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %236 = stablehlo.broadcast_in_dim %c_195, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %237 = stablehlo.and %116, %c_194 : tensor<64xi1>
    %238 = stablehlo.reshape %237 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %239 = stablehlo.reshape %237 : (tensor<64xi1>) -> tensor<1x64xi1>
    %240 = stablehlo.broadcast_in_dim %239, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %241 = stablehlo.not %238 : tensor<1x1x1x64xi1>
    %242 = stablehlo.reshape %241 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %243 = stablehlo.broadcast_in_dim %242, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %244 = stablehlo.broadcast_in_dim %c_193, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %245 = stablehlo.and %116, %c_192 : tensor<64xi1>
    %246 = stablehlo.reshape %245 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %247 = stablehlo.reshape %245 : (tensor<64xi1>) -> tensor<1x64xi1>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %249 = stablehlo.not %246 : tensor<1x1x1x64xi1>
    %250 = stablehlo.reshape %249 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %252 = stablehlo.broadcast_in_dim %c_191, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %253 = stablehlo.and %116, %c_190 : tensor<64xi1>
    %254 = stablehlo.reshape %253 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %255 = stablehlo.reshape %253 : (tensor<64xi1>) -> tensor<1x64xi1>
    %256 = stablehlo.broadcast_in_dim %255, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %257 = stablehlo.not %254 : tensor<1x1x1x64xi1>
    %258 = stablehlo.reshape %257 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %259 = stablehlo.broadcast_in_dim %258, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %260 = stablehlo.broadcast_in_dim %c_189, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %261 = stablehlo.and %116, %c_188 : tensor<64xi1>
    %262 = stablehlo.reshape %261 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %263 = stablehlo.reshape %261 : (tensor<64xi1>) -> tensor<1x64xi1>
    %264 = stablehlo.broadcast_in_dim %263, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %265 = stablehlo.not %262 : tensor<1x1x1x64xi1>
    %266 = stablehlo.reshape %265 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %268 = stablehlo.broadcast_in_dim %c_187, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %269 = stablehlo.and %116, %c_186 : tensor<64xi1>
    %270 = stablehlo.reshape %269 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %271 = stablehlo.reshape %269 : (tensor<64xi1>) -> tensor<1x64xi1>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %273 = stablehlo.not %270 : tensor<1x1x1x64xi1>
    %274 = stablehlo.reshape %273 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %276 = stablehlo.broadcast_in_dim %c_185, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %277 = stablehlo.and %116, %c_184 : tensor<64xi1>
    %278 = stablehlo.reshape %277 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %279 = stablehlo.reshape %277 : (tensor<64xi1>) -> tensor<1x64xi1>
    %280 = stablehlo.broadcast_in_dim %279, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %281 = stablehlo.not %278 : tensor<1x1x1x64xi1>
    %282 = stablehlo.reshape %281 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %284 = stablehlo.broadcast_in_dim %c_183, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %285 = stablehlo.and %116, %c_182 : tensor<64xi1>
    %286 = stablehlo.reshape %285 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %287 = stablehlo.reshape %285 : (tensor<64xi1>) -> tensor<1x64xi1>
    %288 = stablehlo.broadcast_in_dim %287, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %289 = stablehlo.not %286 : tensor<1x1x1x64xi1>
    %290 = stablehlo.reshape %289 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %292 = stablehlo.broadcast_in_dim %c_181, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %293 = stablehlo.and %116, %c_180 : tensor<64xi1>
    %294 = stablehlo.reshape %293 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %295 = stablehlo.reshape %293 : (tensor<64xi1>) -> tensor<1x64xi1>
    %296 = stablehlo.broadcast_in_dim %295, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %297 = stablehlo.not %294 : tensor<1x1x1x64xi1>
    %298 = stablehlo.reshape %297 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %299 = stablehlo.broadcast_in_dim %298, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %300 = stablehlo.broadcast_in_dim %c_179, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %301 = stablehlo.and %116, %c_178 : tensor<64xi1>
    %302 = stablehlo.reshape %301 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %303 = stablehlo.reshape %301 : (tensor<64xi1>) -> tensor<1x64xi1>
    %304 = stablehlo.broadcast_in_dim %303, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %305 = stablehlo.not %302 : tensor<1x1x1x64xi1>
    %306 = stablehlo.reshape %305 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %307 = stablehlo.broadcast_in_dim %306, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %308 = stablehlo.broadcast_in_dim %c_177, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %309 = stablehlo.and %116, %c_176 : tensor<64xi1>
    %310 = stablehlo.reshape %309 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %311 = stablehlo.reshape %309 : (tensor<64xi1>) -> tensor<1x64xi1>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %313 = stablehlo.not %310 : tensor<1x1x1x64xi1>
    %314 = stablehlo.reshape %313 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %315 = stablehlo.broadcast_in_dim %314, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %316 = stablehlo.broadcast_in_dim %c_175, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %317 = stablehlo.and %116, %c_174 : tensor<64xi1>
    %318 = stablehlo.reshape %317 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %319 = stablehlo.reshape %317 : (tensor<64xi1>) -> tensor<1x64xi1>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %321 = stablehlo.not %318 : tensor<1x1x1x64xi1>
    %322 = stablehlo.reshape %321 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %324 = stablehlo.broadcast_in_dim %c_173, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %325 = stablehlo.and %116, %c_172 : tensor<64xi1>
    %326 = stablehlo.reshape %325 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %327 = stablehlo.reshape %325 : (tensor<64xi1>) -> tensor<1x64xi1>
    %328 = stablehlo.broadcast_in_dim %327, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %329 = stablehlo.not %326 : tensor<1x1x1x64xi1>
    %330 = stablehlo.reshape %329 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %331 = stablehlo.broadcast_in_dim %330, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %332 = stablehlo.broadcast_in_dim %c_171, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %333 = stablehlo.and %116, %c_170 : tensor<64xi1>
    %334 = stablehlo.reshape %333 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %335 = stablehlo.reshape %333 : (tensor<64xi1>) -> tensor<1x64xi1>
    %336 = stablehlo.broadcast_in_dim %335, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %337 = stablehlo.not %334 : tensor<1x1x1x64xi1>
    %338 = stablehlo.reshape %337 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %339 = stablehlo.broadcast_in_dim %338, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %340 = stablehlo.broadcast_in_dim %c_169, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %341 = stablehlo.and %116, %c_168 : tensor<64xi1>
    %342 = stablehlo.reshape %341 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %343 = stablehlo.reshape %341 : (tensor<64xi1>) -> tensor<1x64xi1>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %345 = stablehlo.not %342 : tensor<1x1x1x64xi1>
    %346 = stablehlo.reshape %345 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %347 = stablehlo.broadcast_in_dim %346, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %348 = stablehlo.broadcast_in_dim %c_167, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %349 = stablehlo.and %116, %c_166 : tensor<64xi1>
    %350 = stablehlo.reshape %349 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %351 = stablehlo.reshape %349 : (tensor<64xi1>) -> tensor<1x64xi1>
    %352 = stablehlo.broadcast_in_dim %351, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %353 = stablehlo.not %350 : tensor<1x1x1x64xi1>
    %354 = stablehlo.reshape %353 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %355 = stablehlo.broadcast_in_dim %354, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %356 = stablehlo.broadcast_in_dim %c_165, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %357 = stablehlo.and %116, %c_164 : tensor<64xi1>
    %358 = stablehlo.reshape %357 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %359 = stablehlo.reshape %357 : (tensor<64xi1>) -> tensor<1x64xi1>
    %360 = stablehlo.broadcast_in_dim %359, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %361 = stablehlo.not %358 : tensor<1x1x1x64xi1>
    %362 = stablehlo.reshape %361 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %363 = stablehlo.broadcast_in_dim %362, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %364 = stablehlo.broadcast_in_dim %c_163, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %365 = stablehlo.and %116, %c_162 : tensor<64xi1>
    %366 = stablehlo.reshape %365 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %367 = stablehlo.reshape %365 : (tensor<64xi1>) -> tensor<1x64xi1>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %369 = stablehlo.not %366 : tensor<1x1x1x64xi1>
    %370 = stablehlo.reshape %369 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %371 = stablehlo.broadcast_in_dim %370, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %372 = stablehlo.broadcast_in_dim %c_161, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %373 = stablehlo.and %116, %c_160 : tensor<64xi1>
    %374 = stablehlo.reshape %373 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %375 = stablehlo.reshape %373 : (tensor<64xi1>) -> tensor<1x64xi1>
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %377 = stablehlo.not %374 : tensor<1x1x1x64xi1>
    %378 = stablehlo.reshape %377 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %379 = stablehlo.broadcast_in_dim %378, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %380 = stablehlo.broadcast_in_dim %c_159, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %381 = stablehlo.and %116, %c_158 : tensor<64xi1>
    %382 = stablehlo.reshape %381 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %383 = stablehlo.reshape %381 : (tensor<64xi1>) -> tensor<1x64xi1>
    %384 = stablehlo.broadcast_in_dim %383, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %385 = stablehlo.not %382 : tensor<1x1x1x64xi1>
    %386 = stablehlo.reshape %385 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %387 = stablehlo.broadcast_in_dim %386, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %388 = stablehlo.broadcast_in_dim %c_157, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %389 = stablehlo.and %116, %c_156 : tensor<64xi1>
    %390 = stablehlo.reshape %389 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %391 = stablehlo.reshape %389 : (tensor<64xi1>) -> tensor<1x64xi1>
    %392 = stablehlo.broadcast_in_dim %391, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %393 = stablehlo.not %390 : tensor<1x1x1x64xi1>
    %394 = stablehlo.reshape %393 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %395 = stablehlo.broadcast_in_dim %394, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %396 = stablehlo.broadcast_in_dim %c_155, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %397 = stablehlo.and %116, %c_154 : tensor<64xi1>
    %398 = stablehlo.reshape %397 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %399 = stablehlo.reshape %397 : (tensor<64xi1>) -> tensor<1x64xi1>
    %400 = stablehlo.broadcast_in_dim %399, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %401 = stablehlo.not %398 : tensor<1x1x1x64xi1>
    %402 = stablehlo.reshape %401 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %403 = stablehlo.broadcast_in_dim %402, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %404 = stablehlo.broadcast_in_dim %c_153, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %405 = stablehlo.and %116, %c_152 : tensor<64xi1>
    %406 = stablehlo.reshape %405 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %407 = stablehlo.reshape %405 : (tensor<64xi1>) -> tensor<1x64xi1>
    %408 = stablehlo.broadcast_in_dim %407, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %409 = stablehlo.not %406 : tensor<1x1x1x64xi1>
    %410 = stablehlo.reshape %409 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %411 = stablehlo.broadcast_in_dim %410, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %412 = stablehlo.broadcast_in_dim %c_151, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %413 = stablehlo.and %116, %c_150 : tensor<64xi1>
    %414 = stablehlo.reshape %413 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %415 = stablehlo.reshape %413 : (tensor<64xi1>) -> tensor<1x64xi1>
    %416 = stablehlo.broadcast_in_dim %415, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %417 = stablehlo.not %414 : tensor<1x1x1x64xi1>
    %418 = stablehlo.reshape %417 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %419 = stablehlo.broadcast_in_dim %418, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %420 = stablehlo.broadcast_in_dim %c_149, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %421 = stablehlo.and %116, %c_148 : tensor<64xi1>
    %422 = stablehlo.reshape %421 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %423 = stablehlo.reshape %421 : (tensor<64xi1>) -> tensor<1x64xi1>
    %424 = stablehlo.broadcast_in_dim %423, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %425 = stablehlo.not %422 : tensor<1x1x1x64xi1>
    %426 = stablehlo.reshape %425 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %427 = stablehlo.broadcast_in_dim %426, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %428 = stablehlo.broadcast_in_dim %c_147, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %429 = stablehlo.and %116, %c_146 : tensor<64xi1>
    %430 = stablehlo.reshape %429 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %431 = stablehlo.reshape %429 : (tensor<64xi1>) -> tensor<1x64xi1>
    %432 = stablehlo.broadcast_in_dim %431, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %433 = stablehlo.not %430 : tensor<1x1x1x64xi1>
    %434 = stablehlo.reshape %433 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %435 = stablehlo.broadcast_in_dim %434, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %436 = stablehlo.broadcast_in_dim %c_145, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %437 = stablehlo.and %116, %c_144 : tensor<64xi1>
    %438 = stablehlo.reshape %437 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %439 = stablehlo.reshape %437 : (tensor<64xi1>) -> tensor<1x64xi1>
    %440 = stablehlo.broadcast_in_dim %439, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %441 = stablehlo.not %438 : tensor<1x1x1x64xi1>
    %442 = stablehlo.reshape %441 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %444 = stablehlo.broadcast_in_dim %c_143, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %445 = stablehlo.and %116, %c_142 : tensor<64xi1>
    %446 = stablehlo.reshape %445 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %447 = stablehlo.reshape %445 : (tensor<64xi1>) -> tensor<1x64xi1>
    %448 = stablehlo.broadcast_in_dim %447, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %449 = stablehlo.not %446 : tensor<1x1x1x64xi1>
    %450 = stablehlo.reshape %449 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %451 = stablehlo.broadcast_in_dim %450, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %452 = stablehlo.broadcast_in_dim %c_141, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %453 = stablehlo.and %116, %c_140 : tensor<64xi1>
    %454 = stablehlo.reshape %453 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %455 = stablehlo.reshape %453 : (tensor<64xi1>) -> tensor<1x64xi1>
    %456 = stablehlo.broadcast_in_dim %455, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %457 = stablehlo.not %454 : tensor<1x1x1x64xi1>
    %458 = stablehlo.reshape %457 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %459 = stablehlo.broadcast_in_dim %458, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %460 = stablehlo.broadcast_in_dim %c_139, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %461 = stablehlo.and %116, %c_138 : tensor<64xi1>
    %462 = stablehlo.reshape %461 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %463 = stablehlo.reshape %461 : (tensor<64xi1>) -> tensor<1x64xi1>
    %464 = stablehlo.broadcast_in_dim %463, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %465 = stablehlo.not %462 : tensor<1x1x1x64xi1>
    %466 = stablehlo.reshape %465 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %467 = stablehlo.broadcast_in_dim %466, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %468 = stablehlo.broadcast_in_dim %c_137, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %469 = stablehlo.and %116, %c_136 : tensor<64xi1>
    %470 = stablehlo.reshape %469 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %471 = stablehlo.reshape %469 : (tensor<64xi1>) -> tensor<1x64xi1>
    %472 = stablehlo.broadcast_in_dim %471, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %473 = stablehlo.not %470 : tensor<1x1x1x64xi1>
    %474 = stablehlo.reshape %473 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %475 = stablehlo.broadcast_in_dim %474, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %476 = stablehlo.broadcast_in_dim %c_135, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %477 = stablehlo.and %116, %c_134 : tensor<64xi1>
    %478 = stablehlo.reshape %477 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %479 = stablehlo.reshape %477 : (tensor<64xi1>) -> tensor<1x64xi1>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %481 = stablehlo.not %478 : tensor<1x1x1x64xi1>
    %482 = stablehlo.reshape %481 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %483 = stablehlo.broadcast_in_dim %482, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %484 = stablehlo.broadcast_in_dim %c_133, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %485 = stablehlo.and %116, %c_132 : tensor<64xi1>
    %486 = stablehlo.reshape %485 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %487 = stablehlo.reshape %485 : (tensor<64xi1>) -> tensor<1x64xi1>
    %488 = stablehlo.broadcast_in_dim %487, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %489 = stablehlo.not %486 : tensor<1x1x1x64xi1>
    %490 = stablehlo.reshape %489 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %491 = stablehlo.broadcast_in_dim %490, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %492 = stablehlo.broadcast_in_dim %c_131, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %493 = stablehlo.and %116, %c_130 : tensor<64xi1>
    %494 = stablehlo.reshape %493 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %495 = stablehlo.reshape %493 : (tensor<64xi1>) -> tensor<1x64xi1>
    %496 = stablehlo.broadcast_in_dim %495, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %497 = stablehlo.not %494 : tensor<1x1x1x64xi1>
    %498 = stablehlo.reshape %497 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %499 = stablehlo.broadcast_in_dim %498, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %500 = stablehlo.broadcast_in_dim %c_129, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %501 = stablehlo.and %116, %c_128 : tensor<64xi1>
    %502 = stablehlo.reshape %501 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %503 = stablehlo.reshape %501 : (tensor<64xi1>) -> tensor<1x64xi1>
    %504 = stablehlo.broadcast_in_dim %503, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %505 = stablehlo.not %502 : tensor<1x1x1x64xi1>
    %506 = stablehlo.reshape %505 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %507 = stablehlo.broadcast_in_dim %506, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %508 = stablehlo.broadcast_in_dim %c_127, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %509 = stablehlo.and %116, %c_126 : tensor<64xi1>
    %510 = stablehlo.reshape %509 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %511 = stablehlo.reshape %509 : (tensor<64xi1>) -> tensor<1x64xi1>
    %512 = stablehlo.broadcast_in_dim %511, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %513 = stablehlo.not %510 : tensor<1x1x1x64xi1>
    %514 = stablehlo.reshape %513 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %516 = stablehlo.broadcast_in_dim %c_125, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %517 = stablehlo.and %116, %c_124 : tensor<64xi1>
    %518 = stablehlo.reshape %517 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %519 = stablehlo.reshape %517 : (tensor<64xi1>) -> tensor<1x64xi1>
    %520 = stablehlo.broadcast_in_dim %519, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %521 = stablehlo.not %518 : tensor<1x1x1x64xi1>
    %522 = stablehlo.reshape %521 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %523 = stablehlo.broadcast_in_dim %522, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %524 = stablehlo.broadcast_in_dim %c_123, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %525 = stablehlo.and %116, %c_122 : tensor<64xi1>
    %526 = stablehlo.reshape %525 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %527 = stablehlo.reshape %525 : (tensor<64xi1>) -> tensor<1x64xi1>
    %528 = stablehlo.broadcast_in_dim %527, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %529 = stablehlo.not %526 : tensor<1x1x1x64xi1>
    %530 = stablehlo.reshape %529 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %531 = stablehlo.broadcast_in_dim %530, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %532 = stablehlo.broadcast_in_dim %c_121, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %533 = stablehlo.and %116, %c_120 : tensor<64xi1>
    %534 = stablehlo.reshape %533 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %535 = stablehlo.reshape %533 : (tensor<64xi1>) -> tensor<1x64xi1>
    %536 = stablehlo.broadcast_in_dim %535, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %537 = stablehlo.not %534 : tensor<1x1x1x64xi1>
    %538 = stablehlo.reshape %537 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %539 = stablehlo.broadcast_in_dim %538, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %540 = stablehlo.broadcast_in_dim %c_119, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %541 = stablehlo.and %116, %c_118 : tensor<64xi1>
    %542 = stablehlo.reshape %541 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %543 = stablehlo.reshape %541 : (tensor<64xi1>) -> tensor<1x64xi1>
    %544 = stablehlo.broadcast_in_dim %543, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %545 = stablehlo.not %542 : tensor<1x1x1x64xi1>
    %546 = stablehlo.reshape %545 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %547 = stablehlo.broadcast_in_dim %546, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %548 = stablehlo.broadcast_in_dim %c_117, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %549 = stablehlo.and %116, %c_116 : tensor<64xi1>
    %550 = stablehlo.reshape %549 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %551 = stablehlo.reshape %549 : (tensor<64xi1>) -> tensor<1x64xi1>
    %552 = stablehlo.broadcast_in_dim %551, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %553 = stablehlo.not %550 : tensor<1x1x1x64xi1>
    %554 = stablehlo.reshape %553 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %555 = stablehlo.broadcast_in_dim %554, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %556 = stablehlo.broadcast_in_dim %c_115, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %557 = stablehlo.and %116, %c_114 : tensor<64xi1>
    %558 = stablehlo.reshape %557 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %559 = stablehlo.reshape %557 : (tensor<64xi1>) -> tensor<1x64xi1>
    %560 = stablehlo.broadcast_in_dim %559, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %561 = stablehlo.not %558 : tensor<1x1x1x64xi1>
    %562 = stablehlo.reshape %561 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %563 = stablehlo.broadcast_in_dim %562, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %564 = stablehlo.broadcast_in_dim %c_113, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %565 = stablehlo.and %116, %c_112 : tensor<64xi1>
    %566 = stablehlo.reshape %565 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %567 = stablehlo.reshape %565 : (tensor<64xi1>) -> tensor<1x64xi1>
    %568 = stablehlo.broadcast_in_dim %567, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %569 = stablehlo.not %566 : tensor<1x1x1x64xi1>
    %570 = stablehlo.reshape %569 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %572 = stablehlo.broadcast_in_dim %c_111, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %573 = stablehlo.and %116, %c_110 : tensor<64xi1>
    %574 = stablehlo.reshape %573 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %575 = stablehlo.reshape %573 : (tensor<64xi1>) -> tensor<1x64xi1>
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %577 = stablehlo.not %574 : tensor<1x1x1x64xi1>
    %578 = stablehlo.reshape %577 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %579 = stablehlo.broadcast_in_dim %578, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %580 = stablehlo.broadcast_in_dim %c_109, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %581 = stablehlo.and %116, %c_108 : tensor<64xi1>
    %582 = stablehlo.reshape %581 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %583 = stablehlo.reshape %581 : (tensor<64xi1>) -> tensor<1x64xi1>
    %584 = stablehlo.broadcast_in_dim %583, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %585 = stablehlo.not %582 : tensor<1x1x1x64xi1>
    %586 = stablehlo.reshape %585 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %587 = stablehlo.broadcast_in_dim %586, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %588 = stablehlo.broadcast_in_dim %c_107, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %589 = stablehlo.and %116, %c_106 : tensor<64xi1>
    %590 = stablehlo.reshape %589 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %591 = stablehlo.reshape %589 : (tensor<64xi1>) -> tensor<1x64xi1>
    %592 = stablehlo.broadcast_in_dim %591, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %593 = stablehlo.not %590 : tensor<1x1x1x64xi1>
    %594 = stablehlo.reshape %593 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %595 = stablehlo.broadcast_in_dim %594, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %596 = stablehlo.broadcast_in_dim %c_105, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %597 = stablehlo.and %116, %c_104 : tensor<64xi1>
    %598 = stablehlo.reshape %597 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %599 = stablehlo.reshape %597 : (tensor<64xi1>) -> tensor<1x64xi1>
    %600 = stablehlo.broadcast_in_dim %599, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %601 = stablehlo.not %598 : tensor<1x1x1x64xi1>
    %602 = stablehlo.reshape %601 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %603 = stablehlo.broadcast_in_dim %602, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %604 = stablehlo.broadcast_in_dim %c_103, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %605 = stablehlo.and %116, %c_102 : tensor<64xi1>
    %606 = stablehlo.reshape %605 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %607 = stablehlo.reshape %605 : (tensor<64xi1>) -> tensor<1x64xi1>
    %608 = stablehlo.broadcast_in_dim %607, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %609 = stablehlo.not %606 : tensor<1x1x1x64xi1>
    %610 = stablehlo.reshape %609 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %611 = stablehlo.broadcast_in_dim %610, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %612 = stablehlo.broadcast_in_dim %c_101, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64x64xi1>
    %613 = stablehlo.and %116, %c_100 : tensor<64xi1>
    %614 = stablehlo.reshape %613 : (tensor<64xi1>) -> tensor<1x1x1x64xi1>
    %615 = stablehlo.reshape %613 : (tensor<64xi1>) -> tensor<1x64xi1>
    %616 = stablehlo.broadcast_in_dim %615, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %617 = stablehlo.not %614 : tensor<1x1x1x64xi1>
    %618 = stablehlo.reshape %617 : (tensor<1x1x1x64xi1>) -> tensor<1x64xi1>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 3] : (tensor<1x64xi1>) -> tensor<1x32x8x64xi1>
    %620 = stablehlo.broadcast_in_dim %c_238, dims = [1] : (tensor<64xi64>) -> tensor<64x64xi64>
    %621 = stablehlo.broadcast_in_dim %c_238, dims = [0] : (tensor<64xi64>) -> tensor<64x64xi64>
    %622 = stablehlo.subtract %620, %621 : tensor<64x64xi64>
    %623 = stablehlo.compare  GE, %622, %c_99 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %624 = stablehlo.broadcast_in_dim %arg33, dims = [] : (tensor<i1>) -> tensor<64x64xi1>
    %625 = stablehlo.and %623, %624 : tensor<64x64xi1>
    %626 = stablehlo.reshape %625 : (tensor<64x64xi1>) -> tensor<1x64x64xi1>
    %627 = stablehlo.broadcast_in_dim %626, dims = [0, 3, 4] : (tensor<1x64x64xi1>) -> tensor<1x32x8x64x64xi1>
    %628 = stablehlo.reshape %arg30 : (tensor<32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %629 = stablehlo.custom_call @tt.mark_argument(%628) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.in_proj_b.weight"}} : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %630 = stablehlo.reshape %629 : (tensor<1x32x2048xbf16>) -> tensor<32x2048xbf16>
    %631 = stablehlo.transpose %630, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,32]{0,1}"} : (tensor<32x2048xbf16>) -> tensor<2048x32xbf16>
    %632 = stablehlo.dot_general %8, %631, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x32xbf16>) -> tensor<494x32xbf16>
    %633 = stablehlo.reshape %632 : (tensor<494x32xbf16>) -> tensor<1x494x32xbf16>
    %634 = stablehlo.logistic %633 : tensor<1x494x32xbf16>
    %635 = stablehlo.transpose %634, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,32,494]{1,2,0}"} : (tensor<1x494x32xbf16>) -> tensor<1x32x494xbf16>
    %636 = stablehlo.convert %635 {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,494]{1,2,0}"} : (tensor<1x32x494xbf16>) -> tensor<1x32x494xf32>
    %637 = stablehlo.pad %636, %cst_240, low = [0, 0, 0], high = [0, 0, 18], interior = [0, 0, 0] : (tensor<1x32x494xf32>, tensor<f32>) -> tensor<1x32x512xf32>
    %638 = stablehlo.broadcast_in_dim %637, dims = [0, 1, 2] : (tensor<1x32x512xf32>) -> tensor<1x32x512x128xf32>
    %639 = stablehlo.multiply %104, %638 : tensor<1x32x512x128xf32>
    %640 = stablehlo.reshape %639 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %641 = stablehlo.transpose %105, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[1,32,8,128,64]{3,4,2,1,0}"} : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x128x64xf32>
    %642 = stablehlo.dot_general %640, %641, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x128xf32>, tensor<1x32x8x128x64xf32>) -> tensor<1x32x8x64x64xf32>
    %643 = stablehlo.compare  LE, %622, %c_99 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %644 = stablehlo.reshape %643 : (tensor<64x64xi1>) -> tensor<1x64x64xi1>
    %645 = stablehlo.broadcast_in_dim %644, dims = [0, 3, 4] : (tensor<1x64x64xi1>) -> tensor<1x32x8x64x64xi1>
    %646 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %647 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %648 = stablehlo.subtract %646, %647 : tensor<1x32x8x64x64xf32>
    %649 = stablehlo.select %645, %648, %cst_98 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %650 = stablehlo.exponential %649 : tensor<1x32x8x64x64xf32>
    %651 = stablehlo.select %645, %650, %cst_98 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %652 = stablehlo.multiply %642, %651 : tensor<1x32x8x64x64xf32>
    %653 = stablehlo.select %627, %cst_98, %652 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %654 = stablehlo.negate %653 : tensor<1x32x8x64x64xf32>
    %655 = stablehlo.slice %654 [0:1, 0:32, 0:8, 1:2, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %656 = stablehlo.reshape %655 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %657 = stablehlo.slice %656 [0:1, 0:32, 0:8, 0:1] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x1xf32>
    %658 = stablehlo.reshape %657 : (tensor<1x32x8x1xf32>) -> tensor<1x32x8x1x1xf32>
    %659 = stablehlo.slice %654 [0:1, 0:32, 0:8, 0:1, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %660 = stablehlo.slice %659 [0:1, 0:32, 0:8, 0:1, 0:1] : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x1x1xf32>
    %661 = stablehlo.multiply %658, %660 : tensor<1x32x8x1x1xf32>
    %662 = stablehlo.reduce(%661 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x1x1xf32>, tensor<f32>) -> tensor<1x32x8x1xf32>
    %663 = stablehlo.add %657, %662 : tensor<1x32x8x1xf32>
    %664 = stablehlo.floor %cst_96 : tensor<64xf32>
    %665 = stablehlo.convert %664 : (tensor<64xf32>) -> tensor<64xi64>
    %666 = stablehlo.clamp %c_97, %665, %c_97 : tensor<64xi64>
    %667 = stablehlo.compare  LT, %666, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %668 = stablehlo.add %666, %c_95 : tensor<64xi64>
    %669 = stablehlo.select %667, %668, %666 : tensor<64xi1>, tensor<64xi64>
    %670 = stablehlo.reshape %669 : (tensor<64xi64>) -> tensor<64x1xi64>
    %671 = "stablehlo.gather"(%663, %670) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x1xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %672 = stablehlo.select %619, %cst_224, %671 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %673 = stablehlo.select %616, %672, %656 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %674 = stablehlo.broadcast_in_dim %673, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %675 = stablehlo.select %612, %674, %654 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %676 = stablehlo.slice %675 [0:1, 0:32, 0:8, 2:3, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %677 = stablehlo.reshape %676 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %678 = stablehlo.slice %677 [0:1, 0:32, 0:8, 0:2] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x2xf32>
    %679 = stablehlo.broadcast_in_dim %678, dims = [0, 1, 2, 3] : (tensor<1x32x8x2xf32>) -> tensor<1x32x8x2x2xf32>
    %680 = stablehlo.slice %675 [0:1, 0:32, 0:8, 0:2, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x2x64xf32>
    %681 = stablehlo.slice %680 [0:1, 0:32, 0:8, 0:2, 0:2] : (tensor<1x32x8x2x64xf32>) -> tensor<1x32x8x2x2xf32>
    %682 = stablehlo.multiply %679, %681 : tensor<1x32x8x2x2xf32>
    %683 = stablehlo.reduce(%682 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x2x2xf32>, tensor<f32>) -> tensor<1x32x8x2xf32>
    %684 = stablehlo.add %678, %683 : tensor<1x32x8x2xf32>
    %685 = stablehlo.clamp %c_97, %665, %c_95 : tensor<64xi64>
    %686 = stablehlo.compare  LT, %685, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %687 = stablehlo.add %685, %c_94 : tensor<64xi64>
    %688 = stablehlo.select %686, %687, %685 : tensor<64xi1>, tensor<64xi64>
    %689 = stablehlo.reshape %688 : (tensor<64xi64>) -> tensor<64x1xi64>
    %690 = "stablehlo.gather"(%684, %689) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x2xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %691 = stablehlo.select %611, %cst_224, %690 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %692 = stablehlo.select %608, %691, %677 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %693 = stablehlo.broadcast_in_dim %692, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %694 = stablehlo.select %604, %693, %675 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %695 = stablehlo.slice %694 [0:1, 0:32, 0:8, 3:4, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %696 = stablehlo.reshape %695 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %697 = stablehlo.slice %696 [0:1, 0:32, 0:8, 0:3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x3xf32>
    %698 = stablehlo.broadcast_in_dim %697, dims = [0, 1, 2, 3] : (tensor<1x32x8x3xf32>) -> tensor<1x32x8x3x3xf32>
    %699 = stablehlo.slice %694 [0:1, 0:32, 0:8, 0:3, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x3x64xf32>
    %700 = stablehlo.slice %699 [0:1, 0:32, 0:8, 0:3, 0:3] : (tensor<1x32x8x3x64xf32>) -> tensor<1x32x8x3x3xf32>
    %701 = stablehlo.multiply %698, %700 : tensor<1x32x8x3x3xf32>
    %702 = stablehlo.reduce(%701 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x3x3xf32>, tensor<f32>) -> tensor<1x32x8x3xf32>
    %703 = stablehlo.add %697, %702 : tensor<1x32x8x3xf32>
    %704 = stablehlo.clamp %c_97, %665, %c_94 : tensor<64xi64>
    %705 = stablehlo.compare  LT, %704, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %706 = stablehlo.add %704, %c_93 : tensor<64xi64>
    %707 = stablehlo.select %705, %706, %704 : tensor<64xi1>, tensor<64xi64>
    %708 = stablehlo.reshape %707 : (tensor<64xi64>) -> tensor<64x1xi64>
    %709 = "stablehlo.gather"(%703, %708) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x3xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %710 = stablehlo.select %603, %cst_224, %709 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %711 = stablehlo.select %600, %710, %696 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %712 = stablehlo.broadcast_in_dim %711, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %713 = stablehlo.select %596, %712, %694 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %714 = stablehlo.slice %713 [0:1, 0:32, 0:8, 4:5, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %715 = stablehlo.reshape %714 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %716 = stablehlo.slice %715 [0:1, 0:32, 0:8, 0:4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x4xf32>
    %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1, 2, 3] : (tensor<1x32x8x4xf32>) -> tensor<1x32x8x4x4xf32>
    %718 = stablehlo.slice %713 [0:1, 0:32, 0:8, 0:4, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x4x64xf32>
    %719 = stablehlo.slice %718 [0:1, 0:32, 0:8, 0:4, 0:4] : (tensor<1x32x8x4x64xf32>) -> tensor<1x32x8x4x4xf32>
    %720 = stablehlo.multiply %717, %719 : tensor<1x32x8x4x4xf32>
    %721 = stablehlo.reduce(%720 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x4x4xf32>, tensor<f32>) -> tensor<1x32x8x4xf32>
    %722 = stablehlo.add %716, %721 : tensor<1x32x8x4xf32>
    %723 = stablehlo.clamp %c_97, %665, %c_93 : tensor<64xi64>
    %724 = stablehlo.compare  LT, %723, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %725 = stablehlo.add %723, %c_92 : tensor<64xi64>
    %726 = stablehlo.select %724, %725, %723 : tensor<64xi1>, tensor<64xi64>
    %727 = stablehlo.reshape %726 : (tensor<64xi64>) -> tensor<64x1xi64>
    %728 = "stablehlo.gather"(%722, %727) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x4xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %729 = stablehlo.select %595, %cst_224, %728 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %730 = stablehlo.select %592, %729, %715 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %731 = stablehlo.broadcast_in_dim %730, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %732 = stablehlo.select %588, %731, %713 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %733 = stablehlo.slice %732 [0:1, 0:32, 0:8, 5:6, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %734 = stablehlo.reshape %733 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %735 = stablehlo.slice %734 [0:1, 0:32, 0:8, 0:5] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x5xf32>
    %736 = stablehlo.broadcast_in_dim %735, dims = [0, 1, 2, 3] : (tensor<1x32x8x5xf32>) -> tensor<1x32x8x5x5xf32>
    %737 = stablehlo.slice %732 [0:1, 0:32, 0:8, 0:5, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x5x64xf32>
    %738 = stablehlo.slice %737 [0:1, 0:32, 0:8, 0:5, 0:5] : (tensor<1x32x8x5x64xf32>) -> tensor<1x32x8x5x5xf32>
    %739 = stablehlo.multiply %736, %738 : tensor<1x32x8x5x5xf32>
    %740 = stablehlo.reduce(%739 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x5x5xf32>, tensor<f32>) -> tensor<1x32x8x5xf32>
    %741 = stablehlo.add %735, %740 : tensor<1x32x8x5xf32>
    %742 = stablehlo.clamp %c_97, %665, %c_92 : tensor<64xi64>
    %743 = stablehlo.compare  LT, %742, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %744 = stablehlo.add %742, %c_91 : tensor<64xi64>
    %745 = stablehlo.select %743, %744, %742 : tensor<64xi1>, tensor<64xi64>
    %746 = stablehlo.reshape %745 : (tensor<64xi64>) -> tensor<64x1xi64>
    %747 = "stablehlo.gather"(%741, %746) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x5xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %748 = stablehlo.select %587, %cst_224, %747 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %749 = stablehlo.select %584, %748, %734 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %750 = stablehlo.broadcast_in_dim %749, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %751 = stablehlo.select %580, %750, %732 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %752 = stablehlo.slice %751 [0:1, 0:32, 0:8, 6:7, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %753 = stablehlo.reshape %752 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %754 = stablehlo.slice %753 [0:1, 0:32, 0:8, 0:6] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x6xf32>
    %755 = stablehlo.broadcast_in_dim %754, dims = [0, 1, 2, 3] : (tensor<1x32x8x6xf32>) -> tensor<1x32x8x6x6xf32>
    %756 = stablehlo.slice %751 [0:1, 0:32, 0:8, 0:6, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x6x64xf32>
    %757 = stablehlo.slice %756 [0:1, 0:32, 0:8, 0:6, 0:6] : (tensor<1x32x8x6x64xf32>) -> tensor<1x32x8x6x6xf32>
    %758 = stablehlo.multiply %755, %757 : tensor<1x32x8x6x6xf32>
    %759 = stablehlo.reduce(%758 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x6x6xf32>, tensor<f32>) -> tensor<1x32x8x6xf32>
    %760 = stablehlo.add %754, %759 : tensor<1x32x8x6xf32>
    %761 = stablehlo.clamp %c_97, %665, %c_91 : tensor<64xi64>
    %762 = stablehlo.compare  LT, %761, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %763 = stablehlo.add %761, %c_90 : tensor<64xi64>
    %764 = stablehlo.select %762, %763, %761 : tensor<64xi1>, tensor<64xi64>
    %765 = stablehlo.reshape %764 : (tensor<64xi64>) -> tensor<64x1xi64>
    %766 = "stablehlo.gather"(%760, %765) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x6xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %767 = stablehlo.select %579, %cst_224, %766 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %768 = stablehlo.select %576, %767, %753 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %769 = stablehlo.broadcast_in_dim %768, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %770 = stablehlo.select %572, %769, %751 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %771 = stablehlo.slice %770 [0:1, 0:32, 0:8, 7:8, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %772 = stablehlo.reshape %771 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %773 = stablehlo.slice %772 [0:1, 0:32, 0:8, 0:7] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x7xf32>
    %774 = stablehlo.broadcast_in_dim %773, dims = [0, 1, 2, 3] : (tensor<1x32x8x7xf32>) -> tensor<1x32x8x7x7xf32>
    %775 = stablehlo.slice %770 [0:1, 0:32, 0:8, 0:7, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x7x64xf32>
    %776 = stablehlo.slice %775 [0:1, 0:32, 0:8, 0:7, 0:7] : (tensor<1x32x8x7x64xf32>) -> tensor<1x32x8x7x7xf32>
    %777 = stablehlo.multiply %774, %776 : tensor<1x32x8x7x7xf32>
    %778 = stablehlo.reduce(%777 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x7x7xf32>, tensor<f32>) -> tensor<1x32x8x7xf32>
    %779 = stablehlo.add %773, %778 : tensor<1x32x8x7xf32>
    %780 = stablehlo.clamp %c_97, %665, %c_90 : tensor<64xi64>
    %781 = stablehlo.compare  LT, %780, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %782 = stablehlo.add %780, %c_89 : tensor<64xi64>
    %783 = stablehlo.select %781, %782, %780 : tensor<64xi1>, tensor<64xi64>
    %784 = stablehlo.reshape %783 : (tensor<64xi64>) -> tensor<64x1xi64>
    %785 = "stablehlo.gather"(%779, %784) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x7xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %786 = stablehlo.select %571, %cst_224, %785 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %787 = stablehlo.select %568, %786, %772 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %789 = stablehlo.select %564, %788, %770 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %790 = stablehlo.slice %789 [0:1, 0:32, 0:8, 8:9, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %791 = stablehlo.reshape %790 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %792 = stablehlo.slice %791 [0:1, 0:32, 0:8, 0:8] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x8xf32>
    %793 = stablehlo.broadcast_in_dim %792, dims = [0, 1, 2, 3] : (tensor<1x32x8x8xf32>) -> tensor<1x32x8x8x8xf32>
    %794 = stablehlo.slice %789 [0:1, 0:32, 0:8, 0:8, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x8x64xf32>
    %795 = stablehlo.slice %794 [0:1, 0:32, 0:8, 0:8, 0:8] : (tensor<1x32x8x8x64xf32>) -> tensor<1x32x8x8x8xf32>
    %796 = stablehlo.multiply %793, %795 : tensor<1x32x8x8x8xf32>
    %797 = stablehlo.reduce(%796 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x8x8xf32>, tensor<f32>) -> tensor<1x32x8x8xf32>
    %798 = stablehlo.add %792, %797 : tensor<1x32x8x8xf32>
    %799 = stablehlo.clamp %c_97, %665, %c_89 : tensor<64xi64>
    %800 = stablehlo.compare  LT, %799, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %801 = stablehlo.add %799, %c_88 : tensor<64xi64>
    %802 = stablehlo.select %800, %801, %799 : tensor<64xi1>, tensor<64xi64>
    %803 = stablehlo.reshape %802 : (tensor<64xi64>) -> tensor<64x1xi64>
    %804 = "stablehlo.gather"(%798, %803) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x8xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %805 = stablehlo.select %563, %cst_224, %804 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %806 = stablehlo.select %560, %805, %791 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %808 = stablehlo.select %556, %807, %789 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %809 = stablehlo.slice %808 [0:1, 0:32, 0:8, 9:10, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %810 = stablehlo.reshape %809 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %811 = stablehlo.slice %810 [0:1, 0:32, 0:8, 0:9] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x9xf32>
    %812 = stablehlo.broadcast_in_dim %811, dims = [0, 1, 2, 3] : (tensor<1x32x8x9xf32>) -> tensor<1x32x8x9x9xf32>
    %813 = stablehlo.slice %808 [0:1, 0:32, 0:8, 0:9, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x9x64xf32>
    %814 = stablehlo.slice %813 [0:1, 0:32, 0:8, 0:9, 0:9] : (tensor<1x32x8x9x64xf32>) -> tensor<1x32x8x9x9xf32>
    %815 = stablehlo.multiply %812, %814 : tensor<1x32x8x9x9xf32>
    %816 = stablehlo.reduce(%815 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x9x9xf32>, tensor<f32>) -> tensor<1x32x8x9xf32>
    %817 = stablehlo.add %811, %816 : tensor<1x32x8x9xf32>
    %818 = stablehlo.clamp %c_97, %665, %c_88 : tensor<64xi64>
    %819 = stablehlo.compare  LT, %818, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %820 = stablehlo.add %818, %c_87 : tensor<64xi64>
    %821 = stablehlo.select %819, %820, %818 : tensor<64xi1>, tensor<64xi64>
    %822 = stablehlo.reshape %821 : (tensor<64xi64>) -> tensor<64x1xi64>
    %823 = "stablehlo.gather"(%817, %822) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x9xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %824 = stablehlo.select %555, %cst_224, %823 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %825 = stablehlo.select %552, %824, %810 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %826 = stablehlo.broadcast_in_dim %825, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %827 = stablehlo.select %548, %826, %808 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %828 = stablehlo.slice %827 [0:1, 0:32, 0:8, 10:11, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %829 = stablehlo.reshape %828 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %830 = stablehlo.slice %829 [0:1, 0:32, 0:8, 0:10] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x10xf32>
    %831 = stablehlo.broadcast_in_dim %830, dims = [0, 1, 2, 3] : (tensor<1x32x8x10xf32>) -> tensor<1x32x8x10x10xf32>
    %832 = stablehlo.slice %827 [0:1, 0:32, 0:8, 0:10, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x10x64xf32>
    %833 = stablehlo.slice %832 [0:1, 0:32, 0:8, 0:10, 0:10] : (tensor<1x32x8x10x64xf32>) -> tensor<1x32x8x10x10xf32>
    %834 = stablehlo.multiply %831, %833 : tensor<1x32x8x10x10xf32>
    %835 = stablehlo.reduce(%834 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x10x10xf32>, tensor<f32>) -> tensor<1x32x8x10xf32>
    %836 = stablehlo.add %830, %835 : tensor<1x32x8x10xf32>
    %837 = stablehlo.clamp %c_97, %665, %c_87 : tensor<64xi64>
    %838 = stablehlo.compare  LT, %837, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %839 = stablehlo.add %837, %c_86 : tensor<64xi64>
    %840 = stablehlo.select %838, %839, %837 : tensor<64xi1>, tensor<64xi64>
    %841 = stablehlo.reshape %840 : (tensor<64xi64>) -> tensor<64x1xi64>
    %842 = "stablehlo.gather"(%836, %841) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x10xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %843 = stablehlo.select %547, %cst_224, %842 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %844 = stablehlo.select %544, %843, %829 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %845 = stablehlo.broadcast_in_dim %844, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %846 = stablehlo.select %540, %845, %827 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %847 = stablehlo.slice %846 [0:1, 0:32, 0:8, 11:12, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %848 = stablehlo.reshape %847 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %849 = stablehlo.slice %848 [0:1, 0:32, 0:8, 0:11] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x11xf32>
    %850 = stablehlo.broadcast_in_dim %849, dims = [0, 1, 2, 3] : (tensor<1x32x8x11xf32>) -> tensor<1x32x8x11x11xf32>
    %851 = stablehlo.slice %846 [0:1, 0:32, 0:8, 0:11, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x11x64xf32>
    %852 = stablehlo.slice %851 [0:1, 0:32, 0:8, 0:11, 0:11] : (tensor<1x32x8x11x64xf32>) -> tensor<1x32x8x11x11xf32>
    %853 = stablehlo.multiply %850, %852 : tensor<1x32x8x11x11xf32>
    %854 = stablehlo.reduce(%853 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x11x11xf32>, tensor<f32>) -> tensor<1x32x8x11xf32>
    %855 = stablehlo.add %849, %854 : tensor<1x32x8x11xf32>
    %856 = stablehlo.clamp %c_97, %665, %c_86 : tensor<64xi64>
    %857 = stablehlo.compare  LT, %856, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %858 = stablehlo.add %856, %c_85 : tensor<64xi64>
    %859 = stablehlo.select %857, %858, %856 : tensor<64xi1>, tensor<64xi64>
    %860 = stablehlo.reshape %859 : (tensor<64xi64>) -> tensor<64x1xi64>
    %861 = "stablehlo.gather"(%855, %860) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x11xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %862 = stablehlo.select %539, %cst_224, %861 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %863 = stablehlo.select %536, %862, %848 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %864 = stablehlo.broadcast_in_dim %863, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %865 = stablehlo.select %532, %864, %846 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %866 = stablehlo.slice %865 [0:1, 0:32, 0:8, 12:13, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %867 = stablehlo.reshape %866 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %868 = stablehlo.slice %867 [0:1, 0:32, 0:8, 0:12] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x12xf32>
    %869 = stablehlo.broadcast_in_dim %868, dims = [0, 1, 2, 3] : (tensor<1x32x8x12xf32>) -> tensor<1x32x8x12x12xf32>
    %870 = stablehlo.slice %865 [0:1, 0:32, 0:8, 0:12, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x12x64xf32>
    %871 = stablehlo.slice %870 [0:1, 0:32, 0:8, 0:12, 0:12] : (tensor<1x32x8x12x64xf32>) -> tensor<1x32x8x12x12xf32>
    %872 = stablehlo.multiply %869, %871 : tensor<1x32x8x12x12xf32>
    %873 = stablehlo.reduce(%872 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x12x12xf32>, tensor<f32>) -> tensor<1x32x8x12xf32>
    %874 = stablehlo.add %868, %873 : tensor<1x32x8x12xf32>
    %875 = stablehlo.clamp %c_97, %665, %c_85 : tensor<64xi64>
    %876 = stablehlo.compare  LT, %875, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %877 = stablehlo.add %875, %c_84 : tensor<64xi64>
    %878 = stablehlo.select %876, %877, %875 : tensor<64xi1>, tensor<64xi64>
    %879 = stablehlo.reshape %878 : (tensor<64xi64>) -> tensor<64x1xi64>
    %880 = "stablehlo.gather"(%874, %879) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x12xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %881 = stablehlo.select %531, %cst_224, %880 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %882 = stablehlo.select %528, %881, %867 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %883 = stablehlo.broadcast_in_dim %882, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %884 = stablehlo.select %524, %883, %865 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %885 = stablehlo.slice %884 [0:1, 0:32, 0:8, 13:14, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %886 = stablehlo.reshape %885 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %887 = stablehlo.slice %886 [0:1, 0:32, 0:8, 0:13] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x13xf32>
    %888 = stablehlo.broadcast_in_dim %887, dims = [0, 1, 2, 3] : (tensor<1x32x8x13xf32>) -> tensor<1x32x8x13x13xf32>
    %889 = stablehlo.slice %884 [0:1, 0:32, 0:8, 0:13, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x13x64xf32>
    %890 = stablehlo.slice %889 [0:1, 0:32, 0:8, 0:13, 0:13] : (tensor<1x32x8x13x64xf32>) -> tensor<1x32x8x13x13xf32>
    %891 = stablehlo.multiply %888, %890 : tensor<1x32x8x13x13xf32>
    %892 = stablehlo.reduce(%891 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x13x13xf32>, tensor<f32>) -> tensor<1x32x8x13xf32>
    %893 = stablehlo.add %887, %892 : tensor<1x32x8x13xf32>
    %894 = stablehlo.clamp %c_97, %665, %c_84 : tensor<64xi64>
    %895 = stablehlo.compare  LT, %894, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %896 = stablehlo.add %894, %c_83 : tensor<64xi64>
    %897 = stablehlo.select %895, %896, %894 : tensor<64xi1>, tensor<64xi64>
    %898 = stablehlo.reshape %897 : (tensor<64xi64>) -> tensor<64x1xi64>
    %899 = "stablehlo.gather"(%893, %898) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x13xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %900 = stablehlo.select %523, %cst_224, %899 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %901 = stablehlo.select %520, %900, %886 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %902 = stablehlo.broadcast_in_dim %901, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %903 = stablehlo.select %516, %902, %884 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %904 = stablehlo.slice %903 [0:1, 0:32, 0:8, 14:15, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %905 = stablehlo.reshape %904 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %906 = stablehlo.slice %905 [0:1, 0:32, 0:8, 0:14] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x14xf32>
    %907 = stablehlo.broadcast_in_dim %906, dims = [0, 1, 2, 3] : (tensor<1x32x8x14xf32>) -> tensor<1x32x8x14x14xf32>
    %908 = stablehlo.slice %903 [0:1, 0:32, 0:8, 0:14, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x14x64xf32>
    %909 = stablehlo.slice %908 [0:1, 0:32, 0:8, 0:14, 0:14] : (tensor<1x32x8x14x64xf32>) -> tensor<1x32x8x14x14xf32>
    %910 = stablehlo.multiply %907, %909 : tensor<1x32x8x14x14xf32>
    %911 = stablehlo.reduce(%910 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x14x14xf32>, tensor<f32>) -> tensor<1x32x8x14xf32>
    %912 = stablehlo.add %906, %911 : tensor<1x32x8x14xf32>
    %913 = stablehlo.clamp %c_97, %665, %c_83 : tensor<64xi64>
    %914 = stablehlo.compare  LT, %913, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %915 = stablehlo.add %913, %c_82 : tensor<64xi64>
    %916 = stablehlo.select %914, %915, %913 : tensor<64xi1>, tensor<64xi64>
    %917 = stablehlo.reshape %916 : (tensor<64xi64>) -> tensor<64x1xi64>
    %918 = "stablehlo.gather"(%912, %917) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x14xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %919 = stablehlo.select %515, %cst_224, %918 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %920 = stablehlo.select %512, %919, %905 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %922 = stablehlo.select %508, %921, %903 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %923 = stablehlo.slice %922 [0:1, 0:32, 0:8, 15:16, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %924 = stablehlo.reshape %923 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %925 = stablehlo.slice %924 [0:1, 0:32, 0:8, 0:15] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x15xf32>
    %926 = stablehlo.broadcast_in_dim %925, dims = [0, 1, 2, 3] : (tensor<1x32x8x15xf32>) -> tensor<1x32x8x15x15xf32>
    %927 = stablehlo.slice %922 [0:1, 0:32, 0:8, 0:15, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x15x64xf32>
    %928 = stablehlo.slice %927 [0:1, 0:32, 0:8, 0:15, 0:15] : (tensor<1x32x8x15x64xf32>) -> tensor<1x32x8x15x15xf32>
    %929 = stablehlo.multiply %926, %928 : tensor<1x32x8x15x15xf32>
    %930 = stablehlo.reduce(%929 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x15x15xf32>, tensor<f32>) -> tensor<1x32x8x15xf32>
    %931 = stablehlo.add %925, %930 : tensor<1x32x8x15xf32>
    %932 = stablehlo.clamp %c_97, %665, %c_82 : tensor<64xi64>
    %933 = stablehlo.compare  LT, %932, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %934 = stablehlo.add %932, %c_81 : tensor<64xi64>
    %935 = stablehlo.select %933, %934, %932 : tensor<64xi1>, tensor<64xi64>
    %936 = stablehlo.reshape %935 : (tensor<64xi64>) -> tensor<64x1xi64>
    %937 = "stablehlo.gather"(%931, %936) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x15xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %938 = stablehlo.select %507, %cst_224, %937 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %939 = stablehlo.select %504, %938, %924 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %941 = stablehlo.select %500, %940, %922 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %942 = stablehlo.slice %941 [0:1, 0:32, 0:8, 16:17, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %943 = stablehlo.reshape %942 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %944 = stablehlo.slice %943 [0:1, 0:32, 0:8, 0:16] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x16xf32>
    %945 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2, 3] : (tensor<1x32x8x16xf32>) -> tensor<1x32x8x16x16xf32>
    %946 = stablehlo.slice %941 [0:1, 0:32, 0:8, 0:16, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x16x64xf32>
    %947 = stablehlo.slice %946 [0:1, 0:32, 0:8, 0:16, 0:16] : (tensor<1x32x8x16x64xf32>) -> tensor<1x32x8x16x16xf32>
    %948 = stablehlo.multiply %945, %947 : tensor<1x32x8x16x16xf32>
    %949 = stablehlo.reduce(%948 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x16x16xf32>, tensor<f32>) -> tensor<1x32x8x16xf32>
    %950 = stablehlo.add %944, %949 : tensor<1x32x8x16xf32>
    %951 = stablehlo.clamp %c_97, %665, %c_81 : tensor<64xi64>
    %952 = stablehlo.compare  LT, %951, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %953 = stablehlo.add %951, %c_80 : tensor<64xi64>
    %954 = stablehlo.select %952, %953, %951 : tensor<64xi1>, tensor<64xi64>
    %955 = stablehlo.reshape %954 : (tensor<64xi64>) -> tensor<64x1xi64>
    %956 = "stablehlo.gather"(%950, %955) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x16xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %957 = stablehlo.select %499, %cst_224, %956 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %958 = stablehlo.select %496, %957, %943 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %959 = stablehlo.broadcast_in_dim %958, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %960 = stablehlo.select %492, %959, %941 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %961 = stablehlo.slice %960 [0:1, 0:32, 0:8, 17:18, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %962 = stablehlo.reshape %961 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %963 = stablehlo.slice %962 [0:1, 0:32, 0:8, 0:17] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x17xf32>
    %964 = stablehlo.broadcast_in_dim %963, dims = [0, 1, 2, 3] : (tensor<1x32x8x17xf32>) -> tensor<1x32x8x17x17xf32>
    %965 = stablehlo.slice %960 [0:1, 0:32, 0:8, 0:17, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x17x64xf32>
    %966 = stablehlo.slice %965 [0:1, 0:32, 0:8, 0:17, 0:17] : (tensor<1x32x8x17x64xf32>) -> tensor<1x32x8x17x17xf32>
    %967 = stablehlo.multiply %964, %966 : tensor<1x32x8x17x17xf32>
    %968 = stablehlo.reduce(%967 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x17x17xf32>, tensor<f32>) -> tensor<1x32x8x17xf32>
    %969 = stablehlo.add %963, %968 : tensor<1x32x8x17xf32>
    %970 = stablehlo.clamp %c_97, %665, %c_80 : tensor<64xi64>
    %971 = stablehlo.compare  LT, %970, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %972 = stablehlo.add %970, %c_79 : tensor<64xi64>
    %973 = stablehlo.select %971, %972, %970 : tensor<64xi1>, tensor<64xi64>
    %974 = stablehlo.reshape %973 : (tensor<64xi64>) -> tensor<64x1xi64>
    %975 = "stablehlo.gather"(%969, %974) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x17xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %976 = stablehlo.select %491, %cst_224, %975 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %977 = stablehlo.select %488, %976, %962 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %978 = stablehlo.broadcast_in_dim %977, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %979 = stablehlo.select %484, %978, %960 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %980 = stablehlo.slice %979 [0:1, 0:32, 0:8, 18:19, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %981 = stablehlo.reshape %980 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %982 = stablehlo.slice %981 [0:1, 0:32, 0:8, 0:18] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x18xf32>
    %983 = stablehlo.broadcast_in_dim %982, dims = [0, 1, 2, 3] : (tensor<1x32x8x18xf32>) -> tensor<1x32x8x18x18xf32>
    %984 = stablehlo.slice %979 [0:1, 0:32, 0:8, 0:18, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x18x64xf32>
    %985 = stablehlo.slice %984 [0:1, 0:32, 0:8, 0:18, 0:18] : (tensor<1x32x8x18x64xf32>) -> tensor<1x32x8x18x18xf32>
    %986 = stablehlo.multiply %983, %985 : tensor<1x32x8x18x18xf32>
    %987 = stablehlo.reduce(%986 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x18x18xf32>, tensor<f32>) -> tensor<1x32x8x18xf32>
    %988 = stablehlo.add %982, %987 : tensor<1x32x8x18xf32>
    %989 = stablehlo.clamp %c_97, %665, %c_79 : tensor<64xi64>
    %990 = stablehlo.compare  LT, %989, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %991 = stablehlo.add %989, %c_78 : tensor<64xi64>
    %992 = stablehlo.select %990, %991, %989 : tensor<64xi1>, tensor<64xi64>
    %993 = stablehlo.reshape %992 : (tensor<64xi64>) -> tensor<64x1xi64>
    %994 = "stablehlo.gather"(%988, %993) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x18xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %995 = stablehlo.select %483, %cst_224, %994 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %996 = stablehlo.select %480, %995, %981 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %997 = stablehlo.broadcast_in_dim %996, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %998 = stablehlo.select %476, %997, %979 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %999 = stablehlo.slice %998 [0:1, 0:32, 0:8, 19:20, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1000 = stablehlo.reshape %999 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1001 = stablehlo.slice %1000 [0:1, 0:32, 0:8, 0:19] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x19xf32>
    %1002 = stablehlo.broadcast_in_dim %1001, dims = [0, 1, 2, 3] : (tensor<1x32x8x19xf32>) -> tensor<1x32x8x19x19xf32>
    %1003 = stablehlo.slice %998 [0:1, 0:32, 0:8, 0:19, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x19x64xf32>
    %1004 = stablehlo.slice %1003 [0:1, 0:32, 0:8, 0:19, 0:19] : (tensor<1x32x8x19x64xf32>) -> tensor<1x32x8x19x19xf32>
    %1005 = stablehlo.multiply %1002, %1004 : tensor<1x32x8x19x19xf32>
    %1006 = stablehlo.reduce(%1005 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x19x19xf32>, tensor<f32>) -> tensor<1x32x8x19xf32>
    %1007 = stablehlo.add %1001, %1006 : tensor<1x32x8x19xf32>
    %1008 = stablehlo.clamp %c_97, %665, %c_78 : tensor<64xi64>
    %1009 = stablehlo.compare  LT, %1008, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1010 = stablehlo.add %1008, %c_77 : tensor<64xi64>
    %1011 = stablehlo.select %1009, %1010, %1008 : tensor<64xi1>, tensor<64xi64>
    %1012 = stablehlo.reshape %1011 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1013 = "stablehlo.gather"(%1007, %1012) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x19xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1014 = stablehlo.select %475, %cst_224, %1013 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1015 = stablehlo.select %472, %1014, %1000 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1016 = stablehlo.broadcast_in_dim %1015, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1017 = stablehlo.select %468, %1016, %998 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1018 = stablehlo.slice %1017 [0:1, 0:32, 0:8, 20:21, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1019 = stablehlo.reshape %1018 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1020 = stablehlo.slice %1019 [0:1, 0:32, 0:8, 0:20] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x20xf32>
    %1021 = stablehlo.broadcast_in_dim %1020, dims = [0, 1, 2, 3] : (tensor<1x32x8x20xf32>) -> tensor<1x32x8x20x20xf32>
    %1022 = stablehlo.slice %1017 [0:1, 0:32, 0:8, 0:20, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x20x64xf32>
    %1023 = stablehlo.slice %1022 [0:1, 0:32, 0:8, 0:20, 0:20] : (tensor<1x32x8x20x64xf32>) -> tensor<1x32x8x20x20xf32>
    %1024 = stablehlo.multiply %1021, %1023 : tensor<1x32x8x20x20xf32>
    %1025 = stablehlo.reduce(%1024 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x20x20xf32>, tensor<f32>) -> tensor<1x32x8x20xf32>
    %1026 = stablehlo.add %1020, %1025 : tensor<1x32x8x20xf32>
    %1027 = stablehlo.clamp %c_97, %665, %c_77 : tensor<64xi64>
    %1028 = stablehlo.compare  LT, %1027, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1029 = stablehlo.add %1027, %c_76 : tensor<64xi64>
    %1030 = stablehlo.select %1028, %1029, %1027 : tensor<64xi1>, tensor<64xi64>
    %1031 = stablehlo.reshape %1030 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1032 = "stablehlo.gather"(%1026, %1031) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x20xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1033 = stablehlo.select %467, %cst_224, %1032 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1034 = stablehlo.select %464, %1033, %1019 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1035 = stablehlo.broadcast_in_dim %1034, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1036 = stablehlo.select %460, %1035, %1017 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1037 = stablehlo.slice %1036 [0:1, 0:32, 0:8, 21:22, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1038 = stablehlo.reshape %1037 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1039 = stablehlo.slice %1038 [0:1, 0:32, 0:8, 0:21] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x21xf32>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2, 3] : (tensor<1x32x8x21xf32>) -> tensor<1x32x8x21x21xf32>
    %1041 = stablehlo.slice %1036 [0:1, 0:32, 0:8, 0:21, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x21x64xf32>
    %1042 = stablehlo.slice %1041 [0:1, 0:32, 0:8, 0:21, 0:21] : (tensor<1x32x8x21x64xf32>) -> tensor<1x32x8x21x21xf32>
    %1043 = stablehlo.multiply %1040, %1042 : tensor<1x32x8x21x21xf32>
    %1044 = stablehlo.reduce(%1043 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x21x21xf32>, tensor<f32>) -> tensor<1x32x8x21xf32>
    %1045 = stablehlo.add %1039, %1044 : tensor<1x32x8x21xf32>
    %1046 = stablehlo.clamp %c_97, %665, %c_76 : tensor<64xi64>
    %1047 = stablehlo.compare  LT, %1046, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1048 = stablehlo.add %1046, %c_75 : tensor<64xi64>
    %1049 = stablehlo.select %1047, %1048, %1046 : tensor<64xi1>, tensor<64xi64>
    %1050 = stablehlo.reshape %1049 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1051 = "stablehlo.gather"(%1045, %1050) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x21xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1052 = stablehlo.select %459, %cst_224, %1051 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1053 = stablehlo.select %456, %1052, %1038 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1054 = stablehlo.broadcast_in_dim %1053, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1055 = stablehlo.select %452, %1054, %1036 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1056 = stablehlo.slice %1055 [0:1, 0:32, 0:8, 22:23, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1057 = stablehlo.reshape %1056 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1058 = stablehlo.slice %1057 [0:1, 0:32, 0:8, 0:22] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x22xf32>
    %1059 = stablehlo.broadcast_in_dim %1058, dims = [0, 1, 2, 3] : (tensor<1x32x8x22xf32>) -> tensor<1x32x8x22x22xf32>
    %1060 = stablehlo.slice %1055 [0:1, 0:32, 0:8, 0:22, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x22x64xf32>
    %1061 = stablehlo.slice %1060 [0:1, 0:32, 0:8, 0:22, 0:22] : (tensor<1x32x8x22x64xf32>) -> tensor<1x32x8x22x22xf32>
    %1062 = stablehlo.multiply %1059, %1061 : tensor<1x32x8x22x22xf32>
    %1063 = stablehlo.reduce(%1062 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x22x22xf32>, tensor<f32>) -> tensor<1x32x8x22xf32>
    %1064 = stablehlo.add %1058, %1063 : tensor<1x32x8x22xf32>
    %1065 = stablehlo.clamp %c_97, %665, %c_75 : tensor<64xi64>
    %1066 = stablehlo.compare  LT, %1065, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1067 = stablehlo.add %1065, %c_74 : tensor<64xi64>
    %1068 = stablehlo.select %1066, %1067, %1065 : tensor<64xi1>, tensor<64xi64>
    %1069 = stablehlo.reshape %1068 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1070 = "stablehlo.gather"(%1064, %1069) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x22xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1071 = stablehlo.select %451, %cst_224, %1070 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1072 = stablehlo.select %448, %1071, %1057 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1073 = stablehlo.broadcast_in_dim %1072, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1074 = stablehlo.select %444, %1073, %1055 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1075 = stablehlo.slice %1074 [0:1, 0:32, 0:8, 23:24, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1076 = stablehlo.reshape %1075 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1077 = stablehlo.slice %1076 [0:1, 0:32, 0:8, 0:23] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x23xf32>
    %1078 = stablehlo.broadcast_in_dim %1077, dims = [0, 1, 2, 3] : (tensor<1x32x8x23xf32>) -> tensor<1x32x8x23x23xf32>
    %1079 = stablehlo.slice %1074 [0:1, 0:32, 0:8, 0:23, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x23x64xf32>
    %1080 = stablehlo.slice %1079 [0:1, 0:32, 0:8, 0:23, 0:23] : (tensor<1x32x8x23x64xf32>) -> tensor<1x32x8x23x23xf32>
    %1081 = stablehlo.multiply %1078, %1080 : tensor<1x32x8x23x23xf32>
    %1082 = stablehlo.reduce(%1081 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x23x23xf32>, tensor<f32>) -> tensor<1x32x8x23xf32>
    %1083 = stablehlo.add %1077, %1082 : tensor<1x32x8x23xf32>
    %1084 = stablehlo.clamp %c_97, %665, %c_74 : tensor<64xi64>
    %1085 = stablehlo.compare  LT, %1084, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1086 = stablehlo.add %1084, %c_73 : tensor<64xi64>
    %1087 = stablehlo.select %1085, %1086, %1084 : tensor<64xi1>, tensor<64xi64>
    %1088 = stablehlo.reshape %1087 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1089 = "stablehlo.gather"(%1083, %1088) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x23xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1090 = stablehlo.select %443, %cst_224, %1089 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1091 = stablehlo.select %440, %1090, %1076 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1092 = stablehlo.broadcast_in_dim %1091, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1093 = stablehlo.select %436, %1092, %1074 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1094 = stablehlo.slice %1093 [0:1, 0:32, 0:8, 24:25, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1095 = stablehlo.reshape %1094 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1096 = stablehlo.slice %1095 [0:1, 0:32, 0:8, 0:24] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x24xf32>
    %1097 = stablehlo.broadcast_in_dim %1096, dims = [0, 1, 2, 3] : (tensor<1x32x8x24xf32>) -> tensor<1x32x8x24x24xf32>
    %1098 = stablehlo.slice %1093 [0:1, 0:32, 0:8, 0:24, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x24x64xf32>
    %1099 = stablehlo.slice %1098 [0:1, 0:32, 0:8, 0:24, 0:24] : (tensor<1x32x8x24x64xf32>) -> tensor<1x32x8x24x24xf32>
    %1100 = stablehlo.multiply %1097, %1099 : tensor<1x32x8x24x24xf32>
    %1101 = stablehlo.reduce(%1100 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x24x24xf32>, tensor<f32>) -> tensor<1x32x8x24xf32>
    %1102 = stablehlo.add %1096, %1101 : tensor<1x32x8x24xf32>
    %1103 = stablehlo.clamp %c_97, %665, %c_73 : tensor<64xi64>
    %1104 = stablehlo.compare  LT, %1103, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1105 = stablehlo.add %1103, %c_72 : tensor<64xi64>
    %1106 = stablehlo.select %1104, %1105, %1103 : tensor<64xi1>, tensor<64xi64>
    %1107 = stablehlo.reshape %1106 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1108 = "stablehlo.gather"(%1102, %1107) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x24xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1109 = stablehlo.select %435, %cst_224, %1108 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1110 = stablehlo.select %432, %1109, %1095 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1111 = stablehlo.broadcast_in_dim %1110, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1112 = stablehlo.select %428, %1111, %1093 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1113 = stablehlo.slice %1112 [0:1, 0:32, 0:8, 25:26, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1114 = stablehlo.reshape %1113 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1115 = stablehlo.slice %1114 [0:1, 0:32, 0:8, 0:25] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x25xf32>
    %1116 = stablehlo.broadcast_in_dim %1115, dims = [0, 1, 2, 3] : (tensor<1x32x8x25xf32>) -> tensor<1x32x8x25x25xf32>
    %1117 = stablehlo.slice %1112 [0:1, 0:32, 0:8, 0:25, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x25x64xf32>
    %1118 = stablehlo.slice %1117 [0:1, 0:32, 0:8, 0:25, 0:25] : (tensor<1x32x8x25x64xf32>) -> tensor<1x32x8x25x25xf32>
    %1119 = stablehlo.multiply %1116, %1118 : tensor<1x32x8x25x25xf32>
    %1120 = stablehlo.reduce(%1119 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x25x25xf32>, tensor<f32>) -> tensor<1x32x8x25xf32>
    %1121 = stablehlo.add %1115, %1120 : tensor<1x32x8x25xf32>
    %1122 = stablehlo.clamp %c_97, %665, %c_72 : tensor<64xi64>
    %1123 = stablehlo.compare  LT, %1122, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1124 = stablehlo.add %1122, %c_71 : tensor<64xi64>
    %1125 = stablehlo.select %1123, %1124, %1122 : tensor<64xi1>, tensor<64xi64>
    %1126 = stablehlo.reshape %1125 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1127 = "stablehlo.gather"(%1121, %1126) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x25xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1128 = stablehlo.select %427, %cst_224, %1127 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1129 = stablehlo.select %424, %1128, %1114 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1130 = stablehlo.broadcast_in_dim %1129, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1131 = stablehlo.select %420, %1130, %1112 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1132 = stablehlo.slice %1131 [0:1, 0:32, 0:8, 26:27, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1133 = stablehlo.reshape %1132 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1134 = stablehlo.slice %1133 [0:1, 0:32, 0:8, 0:26] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x26xf32>
    %1135 = stablehlo.broadcast_in_dim %1134, dims = [0, 1, 2, 3] : (tensor<1x32x8x26xf32>) -> tensor<1x32x8x26x26xf32>
    %1136 = stablehlo.slice %1131 [0:1, 0:32, 0:8, 0:26, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x26x64xf32>
    %1137 = stablehlo.slice %1136 [0:1, 0:32, 0:8, 0:26, 0:26] : (tensor<1x32x8x26x64xf32>) -> tensor<1x32x8x26x26xf32>
    %1138 = stablehlo.multiply %1135, %1137 : tensor<1x32x8x26x26xf32>
    %1139 = stablehlo.reduce(%1138 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x26x26xf32>, tensor<f32>) -> tensor<1x32x8x26xf32>
    %1140 = stablehlo.add %1134, %1139 : tensor<1x32x8x26xf32>
    %1141 = stablehlo.clamp %c_97, %665, %c_71 : tensor<64xi64>
    %1142 = stablehlo.compare  LT, %1141, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1143 = stablehlo.add %1141, %c_70 : tensor<64xi64>
    %1144 = stablehlo.select %1142, %1143, %1141 : tensor<64xi1>, tensor<64xi64>
    %1145 = stablehlo.reshape %1144 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1146 = "stablehlo.gather"(%1140, %1145) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x26xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1147 = stablehlo.select %419, %cst_224, %1146 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1148 = stablehlo.select %416, %1147, %1133 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1149 = stablehlo.broadcast_in_dim %1148, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1150 = stablehlo.select %412, %1149, %1131 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1151 = stablehlo.slice %1150 [0:1, 0:32, 0:8, 27:28, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1152 = stablehlo.reshape %1151 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1153 = stablehlo.slice %1152 [0:1, 0:32, 0:8, 0:27] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x27xf32>
    %1154 = stablehlo.broadcast_in_dim %1153, dims = [0, 1, 2, 3] : (tensor<1x32x8x27xf32>) -> tensor<1x32x8x27x27xf32>
    %1155 = stablehlo.slice %1150 [0:1, 0:32, 0:8, 0:27, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x27x64xf32>
    %1156 = stablehlo.slice %1155 [0:1, 0:32, 0:8, 0:27, 0:27] : (tensor<1x32x8x27x64xf32>) -> tensor<1x32x8x27x27xf32>
    %1157 = stablehlo.multiply %1154, %1156 : tensor<1x32x8x27x27xf32>
    %1158 = stablehlo.reduce(%1157 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x27x27xf32>, tensor<f32>) -> tensor<1x32x8x27xf32>
    %1159 = stablehlo.add %1153, %1158 : tensor<1x32x8x27xf32>
    %1160 = stablehlo.clamp %c_97, %665, %c_70 : tensor<64xi64>
    %1161 = stablehlo.compare  LT, %1160, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1162 = stablehlo.add %1160, %c_69 : tensor<64xi64>
    %1163 = stablehlo.select %1161, %1162, %1160 : tensor<64xi1>, tensor<64xi64>
    %1164 = stablehlo.reshape %1163 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1165 = "stablehlo.gather"(%1159, %1164) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x27xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1166 = stablehlo.select %411, %cst_224, %1165 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1167 = stablehlo.select %408, %1166, %1152 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1168 = stablehlo.broadcast_in_dim %1167, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1169 = stablehlo.select %404, %1168, %1150 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1170 = stablehlo.slice %1169 [0:1, 0:32, 0:8, 28:29, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1171 = stablehlo.reshape %1170 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1172 = stablehlo.slice %1171 [0:1, 0:32, 0:8, 0:28] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x28xf32>
    %1173 = stablehlo.broadcast_in_dim %1172, dims = [0, 1, 2, 3] : (tensor<1x32x8x28xf32>) -> tensor<1x32x8x28x28xf32>
    %1174 = stablehlo.slice %1169 [0:1, 0:32, 0:8, 0:28, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x28x64xf32>
    %1175 = stablehlo.slice %1174 [0:1, 0:32, 0:8, 0:28, 0:28] : (tensor<1x32x8x28x64xf32>) -> tensor<1x32x8x28x28xf32>
    %1176 = stablehlo.multiply %1173, %1175 : tensor<1x32x8x28x28xf32>
    %1177 = stablehlo.reduce(%1176 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x28x28xf32>, tensor<f32>) -> tensor<1x32x8x28xf32>
    %1178 = stablehlo.add %1172, %1177 : tensor<1x32x8x28xf32>
    %1179 = stablehlo.clamp %c_97, %665, %c_69 : tensor<64xi64>
    %1180 = stablehlo.compare  LT, %1179, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1181 = stablehlo.add %1179, %c_68 : tensor<64xi64>
    %1182 = stablehlo.select %1180, %1181, %1179 : tensor<64xi1>, tensor<64xi64>
    %1183 = stablehlo.reshape %1182 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1184 = "stablehlo.gather"(%1178, %1183) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x28xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1185 = stablehlo.select %403, %cst_224, %1184 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1186 = stablehlo.select %400, %1185, %1171 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1187 = stablehlo.broadcast_in_dim %1186, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1188 = stablehlo.select %396, %1187, %1169 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1189 = stablehlo.slice %1188 [0:1, 0:32, 0:8, 29:30, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1190 = stablehlo.reshape %1189 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1191 = stablehlo.slice %1190 [0:1, 0:32, 0:8, 0:29] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x29xf32>
    %1192 = stablehlo.broadcast_in_dim %1191, dims = [0, 1, 2, 3] : (tensor<1x32x8x29xf32>) -> tensor<1x32x8x29x29xf32>
    %1193 = stablehlo.slice %1188 [0:1, 0:32, 0:8, 0:29, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x29x64xf32>
    %1194 = stablehlo.slice %1193 [0:1, 0:32, 0:8, 0:29, 0:29] : (tensor<1x32x8x29x64xf32>) -> tensor<1x32x8x29x29xf32>
    %1195 = stablehlo.multiply %1192, %1194 : tensor<1x32x8x29x29xf32>
    %1196 = stablehlo.reduce(%1195 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x29x29xf32>, tensor<f32>) -> tensor<1x32x8x29xf32>
    %1197 = stablehlo.add %1191, %1196 : tensor<1x32x8x29xf32>
    %1198 = stablehlo.clamp %c_97, %665, %c_68 : tensor<64xi64>
    %1199 = stablehlo.compare  LT, %1198, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1200 = stablehlo.add %1198, %c_67 : tensor<64xi64>
    %1201 = stablehlo.select %1199, %1200, %1198 : tensor<64xi1>, tensor<64xi64>
    %1202 = stablehlo.reshape %1201 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1203 = "stablehlo.gather"(%1197, %1202) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x29xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1204 = stablehlo.select %395, %cst_224, %1203 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1205 = stablehlo.select %392, %1204, %1190 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1206 = stablehlo.broadcast_in_dim %1205, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1207 = stablehlo.select %388, %1206, %1188 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1208 = stablehlo.slice %1207 [0:1, 0:32, 0:8, 30:31, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1209 = stablehlo.reshape %1208 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1210 = stablehlo.slice %1209 [0:1, 0:32, 0:8, 0:30] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x30xf32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<1x32x8x30xf32>) -> tensor<1x32x8x30x30xf32>
    %1212 = stablehlo.slice %1207 [0:1, 0:32, 0:8, 0:30, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x30x64xf32>
    %1213 = stablehlo.slice %1212 [0:1, 0:32, 0:8, 0:30, 0:30] : (tensor<1x32x8x30x64xf32>) -> tensor<1x32x8x30x30xf32>
    %1214 = stablehlo.multiply %1211, %1213 : tensor<1x32x8x30x30xf32>
    %1215 = stablehlo.reduce(%1214 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x30x30xf32>, tensor<f32>) -> tensor<1x32x8x30xf32>
    %1216 = stablehlo.add %1210, %1215 : tensor<1x32x8x30xf32>
    %1217 = stablehlo.clamp %c_97, %665, %c_67 : tensor<64xi64>
    %1218 = stablehlo.compare  LT, %1217, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1219 = stablehlo.add %1217, %c_66 : tensor<64xi64>
    %1220 = stablehlo.select %1218, %1219, %1217 : tensor<64xi1>, tensor<64xi64>
    %1221 = stablehlo.reshape %1220 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1222 = "stablehlo.gather"(%1216, %1221) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x30xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1223 = stablehlo.select %387, %cst_224, %1222 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1224 = stablehlo.select %384, %1223, %1209 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1225 = stablehlo.broadcast_in_dim %1224, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1226 = stablehlo.select %380, %1225, %1207 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1227 = stablehlo.slice %1226 [0:1, 0:32, 0:8, 31:32, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1228 = stablehlo.reshape %1227 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1229 = stablehlo.slice %1228 [0:1, 0:32, 0:8, 0:31] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x31xf32>
    %1230 = stablehlo.broadcast_in_dim %1229, dims = [0, 1, 2, 3] : (tensor<1x32x8x31xf32>) -> tensor<1x32x8x31x31xf32>
    %1231 = stablehlo.slice %1226 [0:1, 0:32, 0:8, 0:31, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x31x64xf32>
    %1232 = stablehlo.slice %1231 [0:1, 0:32, 0:8, 0:31, 0:31] : (tensor<1x32x8x31x64xf32>) -> tensor<1x32x8x31x31xf32>
    %1233 = stablehlo.multiply %1230, %1232 : tensor<1x32x8x31x31xf32>
    %1234 = stablehlo.reduce(%1233 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x31x31xf32>, tensor<f32>) -> tensor<1x32x8x31xf32>
    %1235 = stablehlo.add %1229, %1234 : tensor<1x32x8x31xf32>
    %1236 = stablehlo.clamp %c_97, %665, %c_66 : tensor<64xi64>
    %1237 = stablehlo.compare  LT, %1236, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1238 = stablehlo.add %1236, %c_65 : tensor<64xi64>
    %1239 = stablehlo.select %1237, %1238, %1236 : tensor<64xi1>, tensor<64xi64>
    %1240 = stablehlo.reshape %1239 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1241 = "stablehlo.gather"(%1235, %1240) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x31xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1242 = stablehlo.select %379, %cst_224, %1241 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1243 = stablehlo.select %376, %1242, %1228 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1244 = stablehlo.broadcast_in_dim %1243, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1245 = stablehlo.select %372, %1244, %1226 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1246 = stablehlo.slice %1245 [0:1, 0:32, 0:8, 32:33, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1247 = stablehlo.reshape %1246 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1248 = stablehlo.slice %1247 [0:1, 0:32, 0:8, 0:32] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x32xf32>
    %1249 = stablehlo.broadcast_in_dim %1248, dims = [0, 1, 2, 3] : (tensor<1x32x8x32xf32>) -> tensor<1x32x8x32x32xf32>
    %1250 = stablehlo.slice %1245 [0:1, 0:32, 0:8, 0:32, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x32x64xf32>
    %1251 = stablehlo.slice %1250 [0:1, 0:32, 0:8, 0:32, 0:32] : (tensor<1x32x8x32x64xf32>) -> tensor<1x32x8x32x32xf32>
    %1252 = stablehlo.multiply %1249, %1251 : tensor<1x32x8x32x32xf32>
    %1253 = stablehlo.reduce(%1252 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x32x32xf32>, tensor<f32>) -> tensor<1x32x8x32xf32>
    %1254 = stablehlo.add %1248, %1253 : tensor<1x32x8x32xf32>
    %1255 = stablehlo.clamp %c_97, %665, %c_65 : tensor<64xi64>
    %1256 = stablehlo.compare  LT, %1255, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1257 = stablehlo.add %1255, %c_64 : tensor<64xi64>
    %1258 = stablehlo.select %1256, %1257, %1255 : tensor<64xi1>, tensor<64xi64>
    %1259 = stablehlo.reshape %1258 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1260 = "stablehlo.gather"(%1254, %1259) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x32xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1261 = stablehlo.select %371, %cst_224, %1260 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1262 = stablehlo.select %368, %1261, %1247 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1263 = stablehlo.broadcast_in_dim %1262, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1264 = stablehlo.select %364, %1263, %1245 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1265 = stablehlo.slice %1264 [0:1, 0:32, 0:8, 33:34, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1266 = stablehlo.reshape %1265 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1267 = stablehlo.slice %1266 [0:1, 0:32, 0:8, 0:33] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x33xf32>
    %1268 = stablehlo.broadcast_in_dim %1267, dims = [0, 1, 2, 3] : (tensor<1x32x8x33xf32>) -> tensor<1x32x8x33x33xf32>
    %1269 = stablehlo.slice %1264 [0:1, 0:32, 0:8, 0:33, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x33x64xf32>
    %1270 = stablehlo.slice %1269 [0:1, 0:32, 0:8, 0:33, 0:33] : (tensor<1x32x8x33x64xf32>) -> tensor<1x32x8x33x33xf32>
    %1271 = stablehlo.multiply %1268, %1270 : tensor<1x32x8x33x33xf32>
    %1272 = stablehlo.reduce(%1271 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x33x33xf32>, tensor<f32>) -> tensor<1x32x8x33xf32>
    %1273 = stablehlo.add %1267, %1272 : tensor<1x32x8x33xf32>
    %1274 = stablehlo.clamp %c_97, %665, %c_64 : tensor<64xi64>
    %1275 = stablehlo.compare  LT, %1274, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1276 = stablehlo.add %1274, %c_63 : tensor<64xi64>
    %1277 = stablehlo.select %1275, %1276, %1274 : tensor<64xi1>, tensor<64xi64>
    %1278 = stablehlo.reshape %1277 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1279 = "stablehlo.gather"(%1273, %1278) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x33xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1280 = stablehlo.select %363, %cst_224, %1279 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1281 = stablehlo.select %360, %1280, %1266 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1282 = stablehlo.broadcast_in_dim %1281, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1283 = stablehlo.select %356, %1282, %1264 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1284 = stablehlo.slice %1283 [0:1, 0:32, 0:8, 34:35, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1285 = stablehlo.reshape %1284 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1286 = stablehlo.slice %1285 [0:1, 0:32, 0:8, 0:34] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x34xf32>
    %1287 = stablehlo.broadcast_in_dim %1286, dims = [0, 1, 2, 3] : (tensor<1x32x8x34xf32>) -> tensor<1x32x8x34x34xf32>
    %1288 = stablehlo.slice %1283 [0:1, 0:32, 0:8, 0:34, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x34x64xf32>
    %1289 = stablehlo.slice %1288 [0:1, 0:32, 0:8, 0:34, 0:34] : (tensor<1x32x8x34x64xf32>) -> tensor<1x32x8x34x34xf32>
    %1290 = stablehlo.multiply %1287, %1289 : tensor<1x32x8x34x34xf32>
    %1291 = stablehlo.reduce(%1290 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x34x34xf32>, tensor<f32>) -> tensor<1x32x8x34xf32>
    %1292 = stablehlo.add %1286, %1291 : tensor<1x32x8x34xf32>
    %1293 = stablehlo.clamp %c_97, %665, %c_63 : tensor<64xi64>
    %1294 = stablehlo.compare  LT, %1293, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1295 = stablehlo.add %1293, %c_62 : tensor<64xi64>
    %1296 = stablehlo.select %1294, %1295, %1293 : tensor<64xi1>, tensor<64xi64>
    %1297 = stablehlo.reshape %1296 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1298 = "stablehlo.gather"(%1292, %1297) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x34xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1299 = stablehlo.select %355, %cst_224, %1298 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1300 = stablehlo.select %352, %1299, %1285 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1301 = stablehlo.broadcast_in_dim %1300, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1302 = stablehlo.select %348, %1301, %1283 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1303 = stablehlo.slice %1302 [0:1, 0:32, 0:8, 35:36, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1304 = stablehlo.reshape %1303 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1305 = stablehlo.slice %1304 [0:1, 0:32, 0:8, 0:35] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x35xf32>
    %1306 = stablehlo.broadcast_in_dim %1305, dims = [0, 1, 2, 3] : (tensor<1x32x8x35xf32>) -> tensor<1x32x8x35x35xf32>
    %1307 = stablehlo.slice %1302 [0:1, 0:32, 0:8, 0:35, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x35x64xf32>
    %1308 = stablehlo.slice %1307 [0:1, 0:32, 0:8, 0:35, 0:35] : (tensor<1x32x8x35x64xf32>) -> tensor<1x32x8x35x35xf32>
    %1309 = stablehlo.multiply %1306, %1308 : tensor<1x32x8x35x35xf32>
    %1310 = stablehlo.reduce(%1309 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x35x35xf32>, tensor<f32>) -> tensor<1x32x8x35xf32>
    %1311 = stablehlo.add %1305, %1310 : tensor<1x32x8x35xf32>
    %1312 = stablehlo.clamp %c_97, %665, %c_62 : tensor<64xi64>
    %1313 = stablehlo.compare  LT, %1312, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1314 = stablehlo.add %1312, %c_61 : tensor<64xi64>
    %1315 = stablehlo.select %1313, %1314, %1312 : tensor<64xi1>, tensor<64xi64>
    %1316 = stablehlo.reshape %1315 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1317 = "stablehlo.gather"(%1311, %1316) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x35xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1318 = stablehlo.select %347, %cst_224, %1317 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1319 = stablehlo.select %344, %1318, %1304 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1320 = stablehlo.broadcast_in_dim %1319, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1321 = stablehlo.select %340, %1320, %1302 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1322 = stablehlo.slice %1321 [0:1, 0:32, 0:8, 36:37, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1323 = stablehlo.reshape %1322 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1324 = stablehlo.slice %1323 [0:1, 0:32, 0:8, 0:36] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x36xf32>
    %1325 = stablehlo.broadcast_in_dim %1324, dims = [0, 1, 2, 3] : (tensor<1x32x8x36xf32>) -> tensor<1x32x8x36x36xf32>
    %1326 = stablehlo.slice %1321 [0:1, 0:32, 0:8, 0:36, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x36x64xf32>
    %1327 = stablehlo.slice %1326 [0:1, 0:32, 0:8, 0:36, 0:36] : (tensor<1x32x8x36x64xf32>) -> tensor<1x32x8x36x36xf32>
    %1328 = stablehlo.multiply %1325, %1327 : tensor<1x32x8x36x36xf32>
    %1329 = stablehlo.reduce(%1328 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x36x36xf32>, tensor<f32>) -> tensor<1x32x8x36xf32>
    %1330 = stablehlo.add %1324, %1329 : tensor<1x32x8x36xf32>
    %1331 = stablehlo.clamp %c_97, %665, %c_61 : tensor<64xi64>
    %1332 = stablehlo.compare  LT, %1331, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1333 = stablehlo.add %1331, %c_60 : tensor<64xi64>
    %1334 = stablehlo.select %1332, %1333, %1331 : tensor<64xi1>, tensor<64xi64>
    %1335 = stablehlo.reshape %1334 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1336 = "stablehlo.gather"(%1330, %1335) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x36xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1337 = stablehlo.select %339, %cst_224, %1336 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1338 = stablehlo.select %336, %1337, %1323 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1339 = stablehlo.broadcast_in_dim %1338, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1340 = stablehlo.select %332, %1339, %1321 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1341 = stablehlo.slice %1340 [0:1, 0:32, 0:8, 37:38, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1342 = stablehlo.reshape %1341 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1343 = stablehlo.slice %1342 [0:1, 0:32, 0:8, 0:37] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x37xf32>
    %1344 = stablehlo.broadcast_in_dim %1343, dims = [0, 1, 2, 3] : (tensor<1x32x8x37xf32>) -> tensor<1x32x8x37x37xf32>
    %1345 = stablehlo.slice %1340 [0:1, 0:32, 0:8, 0:37, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x37x64xf32>
    %1346 = stablehlo.slice %1345 [0:1, 0:32, 0:8, 0:37, 0:37] : (tensor<1x32x8x37x64xf32>) -> tensor<1x32x8x37x37xf32>
    %1347 = stablehlo.multiply %1344, %1346 : tensor<1x32x8x37x37xf32>
    %1348 = stablehlo.reduce(%1347 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x37x37xf32>, tensor<f32>) -> tensor<1x32x8x37xf32>
    %1349 = stablehlo.add %1343, %1348 : tensor<1x32x8x37xf32>
    %1350 = stablehlo.clamp %c_97, %665, %c_60 : tensor<64xi64>
    %1351 = stablehlo.compare  LT, %1350, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1352 = stablehlo.add %1350, %c_59 : tensor<64xi64>
    %1353 = stablehlo.select %1351, %1352, %1350 : tensor<64xi1>, tensor<64xi64>
    %1354 = stablehlo.reshape %1353 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1355 = "stablehlo.gather"(%1349, %1354) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x37xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1356 = stablehlo.select %331, %cst_224, %1355 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1357 = stablehlo.select %328, %1356, %1342 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1358 = stablehlo.broadcast_in_dim %1357, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1359 = stablehlo.select %324, %1358, %1340 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1360 = stablehlo.slice %1359 [0:1, 0:32, 0:8, 38:39, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1361 = stablehlo.reshape %1360 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1362 = stablehlo.slice %1361 [0:1, 0:32, 0:8, 0:38] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x38xf32>
    %1363 = stablehlo.broadcast_in_dim %1362, dims = [0, 1, 2, 3] : (tensor<1x32x8x38xf32>) -> tensor<1x32x8x38x38xf32>
    %1364 = stablehlo.slice %1359 [0:1, 0:32, 0:8, 0:38, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x38x64xf32>
    %1365 = stablehlo.slice %1364 [0:1, 0:32, 0:8, 0:38, 0:38] : (tensor<1x32x8x38x64xf32>) -> tensor<1x32x8x38x38xf32>
    %1366 = stablehlo.multiply %1363, %1365 : tensor<1x32x8x38x38xf32>
    %1367 = stablehlo.reduce(%1366 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x38x38xf32>, tensor<f32>) -> tensor<1x32x8x38xf32>
    %1368 = stablehlo.add %1362, %1367 : tensor<1x32x8x38xf32>
    %1369 = stablehlo.clamp %c_97, %665, %c_59 : tensor<64xi64>
    %1370 = stablehlo.compare  LT, %1369, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1371 = stablehlo.add %1369, %c_58 : tensor<64xi64>
    %1372 = stablehlo.select %1370, %1371, %1369 : tensor<64xi1>, tensor<64xi64>
    %1373 = stablehlo.reshape %1372 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1374 = "stablehlo.gather"(%1368, %1373) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x38xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1375 = stablehlo.select %323, %cst_224, %1374 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1376 = stablehlo.select %320, %1375, %1361 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1377 = stablehlo.broadcast_in_dim %1376, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1378 = stablehlo.select %316, %1377, %1359 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1379 = stablehlo.slice %1378 [0:1, 0:32, 0:8, 39:40, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1380 = stablehlo.reshape %1379 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1381 = stablehlo.slice %1380 [0:1, 0:32, 0:8, 0:39] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x39xf32>
    %1382 = stablehlo.broadcast_in_dim %1381, dims = [0, 1, 2, 3] : (tensor<1x32x8x39xf32>) -> tensor<1x32x8x39x39xf32>
    %1383 = stablehlo.slice %1378 [0:1, 0:32, 0:8, 0:39, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x39x64xf32>
    %1384 = stablehlo.slice %1383 [0:1, 0:32, 0:8, 0:39, 0:39] : (tensor<1x32x8x39x64xf32>) -> tensor<1x32x8x39x39xf32>
    %1385 = stablehlo.multiply %1382, %1384 : tensor<1x32x8x39x39xf32>
    %1386 = stablehlo.reduce(%1385 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x39x39xf32>, tensor<f32>) -> tensor<1x32x8x39xf32>
    %1387 = stablehlo.add %1381, %1386 : tensor<1x32x8x39xf32>
    %1388 = stablehlo.clamp %c_97, %665, %c_58 : tensor<64xi64>
    %1389 = stablehlo.compare  LT, %1388, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1390 = stablehlo.add %1388, %c_57 : tensor<64xi64>
    %1391 = stablehlo.select %1389, %1390, %1388 : tensor<64xi1>, tensor<64xi64>
    %1392 = stablehlo.reshape %1391 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1393 = "stablehlo.gather"(%1387, %1392) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x39xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1394 = stablehlo.select %315, %cst_224, %1393 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1395 = stablehlo.select %312, %1394, %1380 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1396 = stablehlo.broadcast_in_dim %1395, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1397 = stablehlo.select %308, %1396, %1378 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1398 = stablehlo.slice %1397 [0:1, 0:32, 0:8, 40:41, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1399 = stablehlo.reshape %1398 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1400 = stablehlo.slice %1399 [0:1, 0:32, 0:8, 0:40] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x40xf32>
    %1401 = stablehlo.broadcast_in_dim %1400, dims = [0, 1, 2, 3] : (tensor<1x32x8x40xf32>) -> tensor<1x32x8x40x40xf32>
    %1402 = stablehlo.slice %1397 [0:1, 0:32, 0:8, 0:40, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x40x64xf32>
    %1403 = stablehlo.slice %1402 [0:1, 0:32, 0:8, 0:40, 0:40] : (tensor<1x32x8x40x64xf32>) -> tensor<1x32x8x40x40xf32>
    %1404 = stablehlo.multiply %1401, %1403 : tensor<1x32x8x40x40xf32>
    %1405 = stablehlo.reduce(%1404 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x40x40xf32>, tensor<f32>) -> tensor<1x32x8x40xf32>
    %1406 = stablehlo.add %1400, %1405 : tensor<1x32x8x40xf32>
    %1407 = stablehlo.clamp %c_97, %665, %c_57 : tensor<64xi64>
    %1408 = stablehlo.compare  LT, %1407, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1409 = stablehlo.add %1407, %c_56 : tensor<64xi64>
    %1410 = stablehlo.select %1408, %1409, %1407 : tensor<64xi1>, tensor<64xi64>
    %1411 = stablehlo.reshape %1410 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1412 = "stablehlo.gather"(%1406, %1411) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x40xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1413 = stablehlo.select %307, %cst_224, %1412 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1414 = stablehlo.select %304, %1413, %1399 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1415 = stablehlo.broadcast_in_dim %1414, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1416 = stablehlo.select %300, %1415, %1397 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1417 = stablehlo.slice %1416 [0:1, 0:32, 0:8, 41:42, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1418 = stablehlo.reshape %1417 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1419 = stablehlo.slice %1418 [0:1, 0:32, 0:8, 0:41] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x41xf32>
    %1420 = stablehlo.broadcast_in_dim %1419, dims = [0, 1, 2, 3] : (tensor<1x32x8x41xf32>) -> tensor<1x32x8x41x41xf32>
    %1421 = stablehlo.slice %1416 [0:1, 0:32, 0:8, 0:41, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x41x64xf32>
    %1422 = stablehlo.slice %1421 [0:1, 0:32, 0:8, 0:41, 0:41] : (tensor<1x32x8x41x64xf32>) -> tensor<1x32x8x41x41xf32>
    %1423 = stablehlo.multiply %1420, %1422 : tensor<1x32x8x41x41xf32>
    %1424 = stablehlo.reduce(%1423 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x41x41xf32>, tensor<f32>) -> tensor<1x32x8x41xf32>
    %1425 = stablehlo.add %1419, %1424 : tensor<1x32x8x41xf32>
    %1426 = stablehlo.clamp %c_97, %665, %c_56 : tensor<64xi64>
    %1427 = stablehlo.compare  LT, %1426, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1428 = stablehlo.add %1426, %c_55 : tensor<64xi64>
    %1429 = stablehlo.select %1427, %1428, %1426 : tensor<64xi1>, tensor<64xi64>
    %1430 = stablehlo.reshape %1429 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1431 = "stablehlo.gather"(%1425, %1430) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x41xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1432 = stablehlo.select %299, %cst_224, %1431 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1433 = stablehlo.select %296, %1432, %1418 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1434 = stablehlo.broadcast_in_dim %1433, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1435 = stablehlo.select %292, %1434, %1416 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1436 = stablehlo.slice %1435 [0:1, 0:32, 0:8, 42:43, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1437 = stablehlo.reshape %1436 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1438 = stablehlo.slice %1437 [0:1, 0:32, 0:8, 0:42] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x42xf32>
    %1439 = stablehlo.broadcast_in_dim %1438, dims = [0, 1, 2, 3] : (tensor<1x32x8x42xf32>) -> tensor<1x32x8x42x42xf32>
    %1440 = stablehlo.slice %1435 [0:1, 0:32, 0:8, 0:42, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x42x64xf32>
    %1441 = stablehlo.slice %1440 [0:1, 0:32, 0:8, 0:42, 0:42] : (tensor<1x32x8x42x64xf32>) -> tensor<1x32x8x42x42xf32>
    %1442 = stablehlo.multiply %1439, %1441 : tensor<1x32x8x42x42xf32>
    %1443 = stablehlo.reduce(%1442 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x42x42xf32>, tensor<f32>) -> tensor<1x32x8x42xf32>
    %1444 = stablehlo.add %1438, %1443 : tensor<1x32x8x42xf32>
    %1445 = stablehlo.clamp %c_97, %665, %c_55 : tensor<64xi64>
    %1446 = stablehlo.compare  LT, %1445, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1447 = stablehlo.add %1445, %c_54 : tensor<64xi64>
    %1448 = stablehlo.select %1446, %1447, %1445 : tensor<64xi1>, tensor<64xi64>
    %1449 = stablehlo.reshape %1448 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1450 = "stablehlo.gather"(%1444, %1449) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x42xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1451 = stablehlo.select %291, %cst_224, %1450 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1452 = stablehlo.select %288, %1451, %1437 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1453 = stablehlo.broadcast_in_dim %1452, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1454 = stablehlo.select %284, %1453, %1435 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1455 = stablehlo.slice %1454 [0:1, 0:32, 0:8, 43:44, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1456 = stablehlo.reshape %1455 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1457 = stablehlo.slice %1456 [0:1, 0:32, 0:8, 0:43] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x43xf32>
    %1458 = stablehlo.broadcast_in_dim %1457, dims = [0, 1, 2, 3] : (tensor<1x32x8x43xf32>) -> tensor<1x32x8x43x43xf32>
    %1459 = stablehlo.slice %1454 [0:1, 0:32, 0:8, 0:43, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x43x64xf32>
    %1460 = stablehlo.slice %1459 [0:1, 0:32, 0:8, 0:43, 0:43] : (tensor<1x32x8x43x64xf32>) -> tensor<1x32x8x43x43xf32>
    %1461 = stablehlo.multiply %1458, %1460 : tensor<1x32x8x43x43xf32>
    %1462 = stablehlo.reduce(%1461 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x43x43xf32>, tensor<f32>) -> tensor<1x32x8x43xf32>
    %1463 = stablehlo.add %1457, %1462 : tensor<1x32x8x43xf32>
    %1464 = stablehlo.clamp %c_97, %665, %c_54 : tensor<64xi64>
    %1465 = stablehlo.compare  LT, %1464, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1466 = stablehlo.add %1464, %c_53 : tensor<64xi64>
    %1467 = stablehlo.select %1465, %1466, %1464 : tensor<64xi1>, tensor<64xi64>
    %1468 = stablehlo.reshape %1467 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1469 = "stablehlo.gather"(%1463, %1468) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x43xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1470 = stablehlo.select %283, %cst_224, %1469 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1471 = stablehlo.select %280, %1470, %1456 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1472 = stablehlo.broadcast_in_dim %1471, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1473 = stablehlo.select %276, %1472, %1454 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1474 = stablehlo.slice %1473 [0:1, 0:32, 0:8, 44:45, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1475 = stablehlo.reshape %1474 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1476 = stablehlo.slice %1475 [0:1, 0:32, 0:8, 0:44] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x44xf32>
    %1477 = stablehlo.broadcast_in_dim %1476, dims = [0, 1, 2, 3] : (tensor<1x32x8x44xf32>) -> tensor<1x32x8x44x44xf32>
    %1478 = stablehlo.slice %1473 [0:1, 0:32, 0:8, 0:44, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x44x64xf32>
    %1479 = stablehlo.slice %1478 [0:1, 0:32, 0:8, 0:44, 0:44] : (tensor<1x32x8x44x64xf32>) -> tensor<1x32x8x44x44xf32>
    %1480 = stablehlo.multiply %1477, %1479 : tensor<1x32x8x44x44xf32>
    %1481 = stablehlo.reduce(%1480 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x44x44xf32>, tensor<f32>) -> tensor<1x32x8x44xf32>
    %1482 = stablehlo.add %1476, %1481 : tensor<1x32x8x44xf32>
    %1483 = stablehlo.clamp %c_97, %665, %c_53 : tensor<64xi64>
    %1484 = stablehlo.compare  LT, %1483, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1485 = stablehlo.add %1483, %c_52 : tensor<64xi64>
    %1486 = stablehlo.select %1484, %1485, %1483 : tensor<64xi1>, tensor<64xi64>
    %1487 = stablehlo.reshape %1486 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1488 = "stablehlo.gather"(%1482, %1487) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x44xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1489 = stablehlo.select %275, %cst_224, %1488 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1490 = stablehlo.select %272, %1489, %1475 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1491 = stablehlo.broadcast_in_dim %1490, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1492 = stablehlo.select %268, %1491, %1473 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1493 = stablehlo.slice %1492 [0:1, 0:32, 0:8, 45:46, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1494 = stablehlo.reshape %1493 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1495 = stablehlo.slice %1494 [0:1, 0:32, 0:8, 0:45] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x45xf32>
    %1496 = stablehlo.broadcast_in_dim %1495, dims = [0, 1, 2, 3] : (tensor<1x32x8x45xf32>) -> tensor<1x32x8x45x45xf32>
    %1497 = stablehlo.slice %1492 [0:1, 0:32, 0:8, 0:45, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x45x64xf32>
    %1498 = stablehlo.slice %1497 [0:1, 0:32, 0:8, 0:45, 0:45] : (tensor<1x32x8x45x64xf32>) -> tensor<1x32x8x45x45xf32>
    %1499 = stablehlo.multiply %1496, %1498 : tensor<1x32x8x45x45xf32>
    %1500 = stablehlo.reduce(%1499 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x45x45xf32>, tensor<f32>) -> tensor<1x32x8x45xf32>
    %1501 = stablehlo.add %1495, %1500 : tensor<1x32x8x45xf32>
    %1502 = stablehlo.clamp %c_97, %665, %c_52 : tensor<64xi64>
    %1503 = stablehlo.compare  LT, %1502, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1504 = stablehlo.add %1502, %c_51 : tensor<64xi64>
    %1505 = stablehlo.select %1503, %1504, %1502 : tensor<64xi1>, tensor<64xi64>
    %1506 = stablehlo.reshape %1505 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1507 = "stablehlo.gather"(%1501, %1506) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x45xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1508 = stablehlo.select %267, %cst_224, %1507 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1509 = stablehlo.select %264, %1508, %1494 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1510 = stablehlo.broadcast_in_dim %1509, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1511 = stablehlo.select %260, %1510, %1492 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1512 = stablehlo.slice %1511 [0:1, 0:32, 0:8, 46:47, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1513 = stablehlo.reshape %1512 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1514 = stablehlo.slice %1513 [0:1, 0:32, 0:8, 0:46] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x46xf32>
    %1515 = stablehlo.broadcast_in_dim %1514, dims = [0, 1, 2, 3] : (tensor<1x32x8x46xf32>) -> tensor<1x32x8x46x46xf32>
    %1516 = stablehlo.slice %1511 [0:1, 0:32, 0:8, 0:46, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x46x64xf32>
    %1517 = stablehlo.slice %1516 [0:1, 0:32, 0:8, 0:46, 0:46] : (tensor<1x32x8x46x64xf32>) -> tensor<1x32x8x46x46xf32>
    %1518 = stablehlo.multiply %1515, %1517 : tensor<1x32x8x46x46xf32>
    %1519 = stablehlo.reduce(%1518 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x46x46xf32>, tensor<f32>) -> tensor<1x32x8x46xf32>
    %1520 = stablehlo.add %1514, %1519 : tensor<1x32x8x46xf32>
    %1521 = stablehlo.clamp %c_97, %665, %c_51 : tensor<64xi64>
    %1522 = stablehlo.compare  LT, %1521, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1523 = stablehlo.add %1521, %c_50 : tensor<64xi64>
    %1524 = stablehlo.select %1522, %1523, %1521 : tensor<64xi1>, tensor<64xi64>
    %1525 = stablehlo.reshape %1524 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1526 = "stablehlo.gather"(%1520, %1525) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x46xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1527 = stablehlo.select %259, %cst_224, %1526 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1528 = stablehlo.select %256, %1527, %1513 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1529 = stablehlo.broadcast_in_dim %1528, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1530 = stablehlo.select %252, %1529, %1511 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1531 = stablehlo.slice %1530 [0:1, 0:32, 0:8, 47:48, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1532 = stablehlo.reshape %1531 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1533 = stablehlo.slice %1532 [0:1, 0:32, 0:8, 0:47] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x47xf32>
    %1534 = stablehlo.broadcast_in_dim %1533, dims = [0, 1, 2, 3] : (tensor<1x32x8x47xf32>) -> tensor<1x32x8x47x47xf32>
    %1535 = stablehlo.slice %1530 [0:1, 0:32, 0:8, 0:47, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x47x64xf32>
    %1536 = stablehlo.slice %1535 [0:1, 0:32, 0:8, 0:47, 0:47] : (tensor<1x32x8x47x64xf32>) -> tensor<1x32x8x47x47xf32>
    %1537 = stablehlo.multiply %1534, %1536 : tensor<1x32x8x47x47xf32>
    %1538 = stablehlo.reduce(%1537 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x47x47xf32>, tensor<f32>) -> tensor<1x32x8x47xf32>
    %1539 = stablehlo.add %1533, %1538 : tensor<1x32x8x47xf32>
    %1540 = stablehlo.clamp %c_97, %665, %c_50 : tensor<64xi64>
    %1541 = stablehlo.compare  LT, %1540, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1542 = stablehlo.add %1540, %c_49 : tensor<64xi64>
    %1543 = stablehlo.select %1541, %1542, %1540 : tensor<64xi1>, tensor<64xi64>
    %1544 = stablehlo.reshape %1543 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1545 = "stablehlo.gather"(%1539, %1544) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x47xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1546 = stablehlo.select %251, %cst_224, %1545 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1547 = stablehlo.select %248, %1546, %1532 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1548 = stablehlo.broadcast_in_dim %1547, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1549 = stablehlo.select %244, %1548, %1530 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1550 = stablehlo.slice %1549 [0:1, 0:32, 0:8, 48:49, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1551 = stablehlo.reshape %1550 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1552 = stablehlo.slice %1551 [0:1, 0:32, 0:8, 0:48] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x48xf32>
    %1553 = stablehlo.broadcast_in_dim %1552, dims = [0, 1, 2, 3] : (tensor<1x32x8x48xf32>) -> tensor<1x32x8x48x48xf32>
    %1554 = stablehlo.slice %1549 [0:1, 0:32, 0:8, 0:48, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x48x64xf32>
    %1555 = stablehlo.slice %1554 [0:1, 0:32, 0:8, 0:48, 0:48] : (tensor<1x32x8x48x64xf32>) -> tensor<1x32x8x48x48xf32>
    %1556 = stablehlo.multiply %1553, %1555 : tensor<1x32x8x48x48xf32>
    %1557 = stablehlo.reduce(%1556 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x48x48xf32>, tensor<f32>) -> tensor<1x32x8x48xf32>
    %1558 = stablehlo.add %1552, %1557 : tensor<1x32x8x48xf32>
    %1559 = stablehlo.clamp %c_97, %665, %c_49 : tensor<64xi64>
    %1560 = stablehlo.compare  LT, %1559, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1561 = stablehlo.add %1559, %c_48 : tensor<64xi64>
    %1562 = stablehlo.select %1560, %1561, %1559 : tensor<64xi1>, tensor<64xi64>
    %1563 = stablehlo.reshape %1562 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1564 = "stablehlo.gather"(%1558, %1563) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x48xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1565 = stablehlo.select %243, %cst_224, %1564 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1566 = stablehlo.select %240, %1565, %1551 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1567 = stablehlo.broadcast_in_dim %1566, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1568 = stablehlo.select %236, %1567, %1549 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1569 = stablehlo.slice %1568 [0:1, 0:32, 0:8, 49:50, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1570 = stablehlo.reshape %1569 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1571 = stablehlo.slice %1570 [0:1, 0:32, 0:8, 0:49] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x49xf32>
    %1572 = stablehlo.broadcast_in_dim %1571, dims = [0, 1, 2, 3] : (tensor<1x32x8x49xf32>) -> tensor<1x32x8x49x49xf32>
    %1573 = stablehlo.slice %1568 [0:1, 0:32, 0:8, 0:49, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x49x64xf32>
    %1574 = stablehlo.slice %1573 [0:1, 0:32, 0:8, 0:49, 0:49] : (tensor<1x32x8x49x64xf32>) -> tensor<1x32x8x49x49xf32>
    %1575 = stablehlo.multiply %1572, %1574 : tensor<1x32x8x49x49xf32>
    %1576 = stablehlo.reduce(%1575 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x49x49xf32>, tensor<f32>) -> tensor<1x32x8x49xf32>
    %1577 = stablehlo.add %1571, %1576 : tensor<1x32x8x49xf32>
    %1578 = stablehlo.clamp %c_97, %665, %c_48 : tensor<64xi64>
    %1579 = stablehlo.compare  LT, %1578, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1580 = stablehlo.add %1578, %c_47 : tensor<64xi64>
    %1581 = stablehlo.select %1579, %1580, %1578 : tensor<64xi1>, tensor<64xi64>
    %1582 = stablehlo.reshape %1581 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1583 = "stablehlo.gather"(%1577, %1582) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x49xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1584 = stablehlo.select %235, %cst_224, %1583 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1585 = stablehlo.select %232, %1584, %1570 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1586 = stablehlo.broadcast_in_dim %1585, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1587 = stablehlo.select %228, %1586, %1568 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1588 = stablehlo.slice %1587 [0:1, 0:32, 0:8, 50:51, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1589 = stablehlo.reshape %1588 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1590 = stablehlo.slice %1589 [0:1, 0:32, 0:8, 0:50] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x50xf32>
    %1591 = stablehlo.broadcast_in_dim %1590, dims = [0, 1, 2, 3] : (tensor<1x32x8x50xf32>) -> tensor<1x32x8x50x50xf32>
    %1592 = stablehlo.slice %1587 [0:1, 0:32, 0:8, 0:50, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x50x64xf32>
    %1593 = stablehlo.slice %1592 [0:1, 0:32, 0:8, 0:50, 0:50] : (tensor<1x32x8x50x64xf32>) -> tensor<1x32x8x50x50xf32>
    %1594 = stablehlo.multiply %1591, %1593 : tensor<1x32x8x50x50xf32>
    %1595 = stablehlo.reduce(%1594 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x50x50xf32>, tensor<f32>) -> tensor<1x32x8x50xf32>
    %1596 = stablehlo.add %1590, %1595 : tensor<1x32x8x50xf32>
    %1597 = stablehlo.clamp %c_97, %665, %c_47 : tensor<64xi64>
    %1598 = stablehlo.compare  LT, %1597, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1599 = stablehlo.add %1597, %c_46 : tensor<64xi64>
    %1600 = stablehlo.select %1598, %1599, %1597 : tensor<64xi1>, tensor<64xi64>
    %1601 = stablehlo.reshape %1600 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1602 = "stablehlo.gather"(%1596, %1601) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x50xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1603 = stablehlo.select %227, %cst_224, %1602 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1604 = stablehlo.select %224, %1603, %1589 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1605 = stablehlo.broadcast_in_dim %1604, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1606 = stablehlo.select %220, %1605, %1587 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1607 = stablehlo.slice %1606 [0:1, 0:32, 0:8, 51:52, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1608 = stablehlo.reshape %1607 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1609 = stablehlo.slice %1608 [0:1, 0:32, 0:8, 0:51] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x51xf32>
    %1610 = stablehlo.broadcast_in_dim %1609, dims = [0, 1, 2, 3] : (tensor<1x32x8x51xf32>) -> tensor<1x32x8x51x51xf32>
    %1611 = stablehlo.slice %1606 [0:1, 0:32, 0:8, 0:51, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x51x64xf32>
    %1612 = stablehlo.slice %1611 [0:1, 0:32, 0:8, 0:51, 0:51] : (tensor<1x32x8x51x64xf32>) -> tensor<1x32x8x51x51xf32>
    %1613 = stablehlo.multiply %1610, %1612 : tensor<1x32x8x51x51xf32>
    %1614 = stablehlo.reduce(%1613 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x51x51xf32>, tensor<f32>) -> tensor<1x32x8x51xf32>
    %1615 = stablehlo.add %1609, %1614 : tensor<1x32x8x51xf32>
    %1616 = stablehlo.clamp %c_97, %665, %c_46 : tensor<64xi64>
    %1617 = stablehlo.compare  LT, %1616, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1618 = stablehlo.add %1616, %c_45 : tensor<64xi64>
    %1619 = stablehlo.select %1617, %1618, %1616 : tensor<64xi1>, tensor<64xi64>
    %1620 = stablehlo.reshape %1619 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1621 = "stablehlo.gather"(%1615, %1620) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x51xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1622 = stablehlo.select %219, %cst_224, %1621 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1623 = stablehlo.select %216, %1622, %1608 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1624 = stablehlo.broadcast_in_dim %1623, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1625 = stablehlo.select %212, %1624, %1606 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1626 = stablehlo.slice %1625 [0:1, 0:32, 0:8, 52:53, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1627 = stablehlo.reshape %1626 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1628 = stablehlo.slice %1627 [0:1, 0:32, 0:8, 0:52] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x52xf32>
    %1629 = stablehlo.broadcast_in_dim %1628, dims = [0, 1, 2, 3] : (tensor<1x32x8x52xf32>) -> tensor<1x32x8x52x52xf32>
    %1630 = stablehlo.slice %1625 [0:1, 0:32, 0:8, 0:52, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x52x64xf32>
    %1631 = stablehlo.slice %1630 [0:1, 0:32, 0:8, 0:52, 0:52] : (tensor<1x32x8x52x64xf32>) -> tensor<1x32x8x52x52xf32>
    %1632 = stablehlo.multiply %1629, %1631 : tensor<1x32x8x52x52xf32>
    %1633 = stablehlo.reduce(%1632 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x52x52xf32>, tensor<f32>) -> tensor<1x32x8x52xf32>
    %1634 = stablehlo.add %1628, %1633 : tensor<1x32x8x52xf32>
    %1635 = stablehlo.clamp %c_97, %665, %c_45 : tensor<64xi64>
    %1636 = stablehlo.compare  LT, %1635, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1637 = stablehlo.add %1635, %c_44 : tensor<64xi64>
    %1638 = stablehlo.select %1636, %1637, %1635 : tensor<64xi1>, tensor<64xi64>
    %1639 = stablehlo.reshape %1638 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1640 = "stablehlo.gather"(%1634, %1639) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x52xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1641 = stablehlo.select %211, %cst_224, %1640 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1642 = stablehlo.select %208, %1641, %1627 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1643 = stablehlo.broadcast_in_dim %1642, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1644 = stablehlo.select %204, %1643, %1625 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1645 = stablehlo.slice %1644 [0:1, 0:32, 0:8, 53:54, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1646 = stablehlo.reshape %1645 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1647 = stablehlo.slice %1646 [0:1, 0:32, 0:8, 0:53] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x53xf32>
    %1648 = stablehlo.broadcast_in_dim %1647, dims = [0, 1, 2, 3] : (tensor<1x32x8x53xf32>) -> tensor<1x32x8x53x53xf32>
    %1649 = stablehlo.slice %1644 [0:1, 0:32, 0:8, 0:53, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x53x64xf32>
    %1650 = stablehlo.slice %1649 [0:1, 0:32, 0:8, 0:53, 0:53] : (tensor<1x32x8x53x64xf32>) -> tensor<1x32x8x53x53xf32>
    %1651 = stablehlo.multiply %1648, %1650 : tensor<1x32x8x53x53xf32>
    %1652 = stablehlo.reduce(%1651 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x53x53xf32>, tensor<f32>) -> tensor<1x32x8x53xf32>
    %1653 = stablehlo.add %1647, %1652 : tensor<1x32x8x53xf32>
    %1654 = stablehlo.clamp %c_97, %665, %c_44 : tensor<64xi64>
    %1655 = stablehlo.compare  LT, %1654, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1656 = stablehlo.add %1654, %c_43 : tensor<64xi64>
    %1657 = stablehlo.select %1655, %1656, %1654 : tensor<64xi1>, tensor<64xi64>
    %1658 = stablehlo.reshape %1657 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1659 = "stablehlo.gather"(%1653, %1658) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x53xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1660 = stablehlo.select %203, %cst_224, %1659 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1661 = stablehlo.select %200, %1660, %1646 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1662 = stablehlo.broadcast_in_dim %1661, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1663 = stablehlo.select %196, %1662, %1644 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1664 = stablehlo.slice %1663 [0:1, 0:32, 0:8, 54:55, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1665 = stablehlo.reshape %1664 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1666 = stablehlo.slice %1665 [0:1, 0:32, 0:8, 0:54] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x54xf32>
    %1667 = stablehlo.broadcast_in_dim %1666, dims = [0, 1, 2, 3] : (tensor<1x32x8x54xf32>) -> tensor<1x32x8x54x54xf32>
    %1668 = stablehlo.slice %1663 [0:1, 0:32, 0:8, 0:54, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x54x64xf32>
    %1669 = stablehlo.slice %1668 [0:1, 0:32, 0:8, 0:54, 0:54] : (tensor<1x32x8x54x64xf32>) -> tensor<1x32x8x54x54xf32>
    %1670 = stablehlo.multiply %1667, %1669 : tensor<1x32x8x54x54xf32>
    %1671 = stablehlo.reduce(%1670 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x54x54xf32>, tensor<f32>) -> tensor<1x32x8x54xf32>
    %1672 = stablehlo.add %1666, %1671 : tensor<1x32x8x54xf32>
    %1673 = stablehlo.clamp %c_97, %665, %c_43 : tensor<64xi64>
    %1674 = stablehlo.compare  LT, %1673, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1675 = stablehlo.add %1673, %c_42 : tensor<64xi64>
    %1676 = stablehlo.select %1674, %1675, %1673 : tensor<64xi1>, tensor<64xi64>
    %1677 = stablehlo.reshape %1676 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1678 = "stablehlo.gather"(%1672, %1677) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x54xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1679 = stablehlo.select %195, %cst_224, %1678 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1680 = stablehlo.select %192, %1679, %1665 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1681 = stablehlo.broadcast_in_dim %1680, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1682 = stablehlo.select %188, %1681, %1663 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1683 = stablehlo.slice %1682 [0:1, 0:32, 0:8, 55:56, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1684 = stablehlo.reshape %1683 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1685 = stablehlo.slice %1684 [0:1, 0:32, 0:8, 0:55] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x55xf32>
    %1686 = stablehlo.broadcast_in_dim %1685, dims = [0, 1, 2, 3] : (tensor<1x32x8x55xf32>) -> tensor<1x32x8x55x55xf32>
    %1687 = stablehlo.slice %1682 [0:1, 0:32, 0:8, 0:55, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x55x64xf32>
    %1688 = stablehlo.slice %1687 [0:1, 0:32, 0:8, 0:55, 0:55] : (tensor<1x32x8x55x64xf32>) -> tensor<1x32x8x55x55xf32>
    %1689 = stablehlo.multiply %1686, %1688 : tensor<1x32x8x55x55xf32>
    %1690 = stablehlo.reduce(%1689 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x55x55xf32>, tensor<f32>) -> tensor<1x32x8x55xf32>
    %1691 = stablehlo.add %1685, %1690 : tensor<1x32x8x55xf32>
    %1692 = stablehlo.clamp %c_97, %665, %c_42 : tensor<64xi64>
    %1693 = stablehlo.compare  LT, %1692, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1694 = stablehlo.add %1692, %c_41 : tensor<64xi64>
    %1695 = stablehlo.select %1693, %1694, %1692 : tensor<64xi1>, tensor<64xi64>
    %1696 = stablehlo.reshape %1695 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1697 = "stablehlo.gather"(%1691, %1696) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x55xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1698 = stablehlo.select %187, %cst_224, %1697 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1699 = stablehlo.select %184, %1698, %1684 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1700 = stablehlo.broadcast_in_dim %1699, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1701 = stablehlo.select %180, %1700, %1682 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1702 = stablehlo.slice %1701 [0:1, 0:32, 0:8, 56:57, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1703 = stablehlo.reshape %1702 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1704 = stablehlo.slice %1703 [0:1, 0:32, 0:8, 0:56] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x56xf32>
    %1705 = stablehlo.broadcast_in_dim %1704, dims = [0, 1, 2, 3] : (tensor<1x32x8x56xf32>) -> tensor<1x32x8x56x56xf32>
    %1706 = stablehlo.slice %1701 [0:1, 0:32, 0:8, 0:56, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x56x64xf32>
    %1707 = stablehlo.slice %1706 [0:1, 0:32, 0:8, 0:56, 0:56] : (tensor<1x32x8x56x64xf32>) -> tensor<1x32x8x56x56xf32>
    %1708 = stablehlo.multiply %1705, %1707 : tensor<1x32x8x56x56xf32>
    %1709 = stablehlo.reduce(%1708 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x56x56xf32>, tensor<f32>) -> tensor<1x32x8x56xf32>
    %1710 = stablehlo.add %1704, %1709 : tensor<1x32x8x56xf32>
    %1711 = stablehlo.clamp %c_97, %665, %c_41 : tensor<64xi64>
    %1712 = stablehlo.compare  LT, %1711, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1713 = stablehlo.add %1711, %c_40 : tensor<64xi64>
    %1714 = stablehlo.select %1712, %1713, %1711 : tensor<64xi1>, tensor<64xi64>
    %1715 = stablehlo.reshape %1714 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1716 = "stablehlo.gather"(%1710, %1715) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x56xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1717 = stablehlo.select %179, %cst_224, %1716 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1718 = stablehlo.select %176, %1717, %1703 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1719 = stablehlo.broadcast_in_dim %1718, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1720 = stablehlo.select %172, %1719, %1701 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1721 = stablehlo.slice %1720 [0:1, 0:32, 0:8, 57:58, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1722 = stablehlo.reshape %1721 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1723 = stablehlo.slice %1722 [0:1, 0:32, 0:8, 0:57] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x57xf32>
    %1724 = stablehlo.broadcast_in_dim %1723, dims = [0, 1, 2, 3] : (tensor<1x32x8x57xf32>) -> tensor<1x32x8x57x57xf32>
    %1725 = stablehlo.slice %1720 [0:1, 0:32, 0:8, 0:57, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x57x64xf32>
    %1726 = stablehlo.slice %1725 [0:1, 0:32, 0:8, 0:57, 0:57] : (tensor<1x32x8x57x64xf32>) -> tensor<1x32x8x57x57xf32>
    %1727 = stablehlo.multiply %1724, %1726 : tensor<1x32x8x57x57xf32>
    %1728 = stablehlo.reduce(%1727 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x57x57xf32>, tensor<f32>) -> tensor<1x32x8x57xf32>
    %1729 = stablehlo.add %1723, %1728 : tensor<1x32x8x57xf32>
    %1730 = stablehlo.clamp %c_97, %665, %c_40 : tensor<64xi64>
    %1731 = stablehlo.compare  LT, %1730, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1732 = stablehlo.add %1730, %c_39 : tensor<64xi64>
    %1733 = stablehlo.select %1731, %1732, %1730 : tensor<64xi1>, tensor<64xi64>
    %1734 = stablehlo.reshape %1733 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1735 = "stablehlo.gather"(%1729, %1734) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x57xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1736 = stablehlo.select %171, %cst_224, %1735 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1737 = stablehlo.select %168, %1736, %1722 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1738 = stablehlo.broadcast_in_dim %1737, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1739 = stablehlo.select %164, %1738, %1720 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1740 = stablehlo.slice %1739 [0:1, 0:32, 0:8, 58:59, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1741 = stablehlo.reshape %1740 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1742 = stablehlo.slice %1741 [0:1, 0:32, 0:8, 0:58] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x58xf32>
    %1743 = stablehlo.broadcast_in_dim %1742, dims = [0, 1, 2, 3] : (tensor<1x32x8x58xf32>) -> tensor<1x32x8x58x58xf32>
    %1744 = stablehlo.slice %1739 [0:1, 0:32, 0:8, 0:58, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x58x64xf32>
    %1745 = stablehlo.slice %1744 [0:1, 0:32, 0:8, 0:58, 0:58] : (tensor<1x32x8x58x64xf32>) -> tensor<1x32x8x58x58xf32>
    %1746 = stablehlo.multiply %1743, %1745 : tensor<1x32x8x58x58xf32>
    %1747 = stablehlo.reduce(%1746 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x58x58xf32>, tensor<f32>) -> tensor<1x32x8x58xf32>
    %1748 = stablehlo.add %1742, %1747 : tensor<1x32x8x58xf32>
    %1749 = stablehlo.clamp %c_97, %665, %c_39 : tensor<64xi64>
    %1750 = stablehlo.compare  LT, %1749, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1751 = stablehlo.add %1749, %c_38 : tensor<64xi64>
    %1752 = stablehlo.select %1750, %1751, %1749 : tensor<64xi1>, tensor<64xi64>
    %1753 = stablehlo.reshape %1752 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1754 = "stablehlo.gather"(%1748, %1753) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x58xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1755 = stablehlo.select %163, %cst_224, %1754 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1756 = stablehlo.select %160, %1755, %1741 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1757 = stablehlo.broadcast_in_dim %1756, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1758 = stablehlo.select %156, %1757, %1739 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1759 = stablehlo.slice %1758 [0:1, 0:32, 0:8, 59:60, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1760 = stablehlo.reshape %1759 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1761 = stablehlo.slice %1760 [0:1, 0:32, 0:8, 0:59] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x59xf32>
    %1762 = stablehlo.broadcast_in_dim %1761, dims = [0, 1, 2, 3] : (tensor<1x32x8x59xf32>) -> tensor<1x32x8x59x59xf32>
    %1763 = stablehlo.slice %1758 [0:1, 0:32, 0:8, 0:59, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x59x64xf32>
    %1764 = stablehlo.slice %1763 [0:1, 0:32, 0:8, 0:59, 0:59] : (tensor<1x32x8x59x64xf32>) -> tensor<1x32x8x59x59xf32>
    %1765 = stablehlo.multiply %1762, %1764 : tensor<1x32x8x59x59xf32>
    %1766 = stablehlo.reduce(%1765 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x59x59xf32>, tensor<f32>) -> tensor<1x32x8x59xf32>
    %1767 = stablehlo.add %1761, %1766 : tensor<1x32x8x59xf32>
    %1768 = stablehlo.clamp %c_97, %665, %c_38 : tensor<64xi64>
    %1769 = stablehlo.compare  LT, %1768, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1770 = stablehlo.add %1768, %c_37 : tensor<64xi64>
    %1771 = stablehlo.select %1769, %1770, %1768 : tensor<64xi1>, tensor<64xi64>
    %1772 = stablehlo.reshape %1771 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1773 = "stablehlo.gather"(%1767, %1772) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x59xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1774 = stablehlo.select %155, %cst_224, %1773 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1775 = stablehlo.select %152, %1774, %1760 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1776 = stablehlo.broadcast_in_dim %1775, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1777 = stablehlo.select %148, %1776, %1758 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1778 = stablehlo.slice %1777 [0:1, 0:32, 0:8, 60:61, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1779 = stablehlo.reshape %1778 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1780 = stablehlo.slice %1779 [0:1, 0:32, 0:8, 0:60] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x60xf32>
    %1781 = stablehlo.broadcast_in_dim %1780, dims = [0, 1, 2, 3] : (tensor<1x32x8x60xf32>) -> tensor<1x32x8x60x60xf32>
    %1782 = stablehlo.slice %1777 [0:1, 0:32, 0:8, 0:60, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x60x64xf32>
    %1783 = stablehlo.slice %1782 [0:1, 0:32, 0:8, 0:60, 0:60] : (tensor<1x32x8x60x64xf32>) -> tensor<1x32x8x60x60xf32>
    %1784 = stablehlo.multiply %1781, %1783 : tensor<1x32x8x60x60xf32>
    %1785 = stablehlo.reduce(%1784 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x60x60xf32>, tensor<f32>) -> tensor<1x32x8x60xf32>
    %1786 = stablehlo.add %1780, %1785 : tensor<1x32x8x60xf32>
    %1787 = stablehlo.clamp %c_97, %665, %c_37 : tensor<64xi64>
    %1788 = stablehlo.compare  LT, %1787, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1789 = stablehlo.add %1787, %c_36 : tensor<64xi64>
    %1790 = stablehlo.select %1788, %1789, %1787 : tensor<64xi1>, tensor<64xi64>
    %1791 = stablehlo.reshape %1790 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1792 = "stablehlo.gather"(%1786, %1791) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x60xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1793 = stablehlo.select %147, %cst_224, %1792 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1794 = stablehlo.select %144, %1793, %1779 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1795 = stablehlo.broadcast_in_dim %1794, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1796 = stablehlo.select %140, %1795, %1777 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1797 = stablehlo.slice %1796 [0:1, 0:32, 0:8, 61:62, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1798 = stablehlo.reshape %1797 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1799 = stablehlo.slice %1798 [0:1, 0:32, 0:8, 0:61] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x61xf32>
    %1800 = stablehlo.broadcast_in_dim %1799, dims = [0, 1, 2, 3] : (tensor<1x32x8x61xf32>) -> tensor<1x32x8x61x61xf32>
    %1801 = stablehlo.slice %1796 [0:1, 0:32, 0:8, 0:61, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x61x64xf32>
    %1802 = stablehlo.slice %1801 [0:1, 0:32, 0:8, 0:61, 0:61] : (tensor<1x32x8x61x64xf32>) -> tensor<1x32x8x61x61xf32>
    %1803 = stablehlo.multiply %1800, %1802 : tensor<1x32x8x61x61xf32>
    %1804 = stablehlo.reduce(%1803 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x61x61xf32>, tensor<f32>) -> tensor<1x32x8x61xf32>
    %1805 = stablehlo.add %1799, %1804 : tensor<1x32x8x61xf32>
    %1806 = stablehlo.clamp %c_97, %665, %c_36 : tensor<64xi64>
    %1807 = stablehlo.compare  LT, %1806, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1808 = stablehlo.add %1806, %c_35 : tensor<64xi64>
    %1809 = stablehlo.select %1807, %1808, %1806 : tensor<64xi1>, tensor<64xi64>
    %1810 = stablehlo.reshape %1809 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1811 = "stablehlo.gather"(%1805, %1810) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x61xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1812 = stablehlo.select %139, %cst_224, %1811 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1813 = stablehlo.select %136, %1812, %1798 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1814 = stablehlo.broadcast_in_dim %1813, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1815 = stablehlo.select %132, %1814, %1796 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1816 = stablehlo.slice %1815 [0:1, 0:32, 0:8, 62:63, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1817 = stablehlo.reshape %1816 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1818 = stablehlo.slice %1817 [0:1, 0:32, 0:8, 0:62] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x62xf32>
    %1819 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2, 3] : (tensor<1x32x8x62xf32>) -> tensor<1x32x8x62x62xf32>
    %1820 = stablehlo.slice %1815 [0:1, 0:32, 0:8, 0:62, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x62x64xf32>
    %1821 = stablehlo.slice %1820 [0:1, 0:32, 0:8, 0:62, 0:62] : (tensor<1x32x8x62x64xf32>) -> tensor<1x32x8x62x62xf32>
    %1822 = stablehlo.multiply %1819, %1821 : tensor<1x32x8x62x62xf32>
    %1823 = stablehlo.reduce(%1822 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x62x62xf32>, tensor<f32>) -> tensor<1x32x8x62xf32>
    %1824 = stablehlo.add %1818, %1823 : tensor<1x32x8x62xf32>
    %1825 = stablehlo.clamp %c_97, %665, %c_35 : tensor<64xi64>
    %1826 = stablehlo.compare  LT, %1825, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1827 = stablehlo.add %1825, %c_34 : tensor<64xi64>
    %1828 = stablehlo.select %1826, %1827, %1825 : tensor<64xi1>, tensor<64xi64>
    %1829 = stablehlo.reshape %1828 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1830 = "stablehlo.gather"(%1824, %1829) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x62xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1831 = stablehlo.select %131, %cst_224, %1830 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1832 = stablehlo.select %128, %1831, %1817 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1833 = stablehlo.broadcast_in_dim %1832, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1834 = stablehlo.select %124, %1833, %1815 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1835 = stablehlo.slice %1834 [0:1, 0:32, 0:8, 63:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %1836 = stablehlo.reshape %1835 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %1837 = stablehlo.slice %1836 [0:1, 0:32, 0:8, 0:63] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x63xf32>
    %1838 = stablehlo.broadcast_in_dim %1837, dims = [0, 1, 2, 3] : (tensor<1x32x8x63xf32>) -> tensor<1x32x8x63x63xf32>
    %1839 = stablehlo.slice %1834 [0:1, 0:32, 0:8, 0:63, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x63x64xf32>
    %1840 = stablehlo.slice %1839 [0:1, 0:32, 0:8, 0:63, 0:63] : (tensor<1x32x8x63x64xf32>) -> tensor<1x32x8x63x63xf32>
    %1841 = stablehlo.multiply %1838, %1840 : tensor<1x32x8x63x63xf32>
    %1842 = stablehlo.reduce(%1841 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x63x63xf32>, tensor<f32>) -> tensor<1x32x8x63xf32>
    %1843 = stablehlo.add %1837, %1842 : tensor<1x32x8x63xf32>
    %1844 = stablehlo.clamp %c_97, %665, %c_34 : tensor<64xi64>
    %1845 = stablehlo.compare  LT, %1844, %c_97 : (tensor<64xi64>, tensor<64xi64>) -> tensor<64xi1>
    %1846 = stablehlo.add %1844, %c_33 : tensor<64xi64>
    %1847 = stablehlo.select %1845, %1846, %1844 : tensor<64xi1>, tensor<64xi64>
    %1848 = stablehlo.reshape %1847 : (tensor<64xi64>) -> tensor<64x1xi64>
    %1849 = "stablehlo.gather"(%1843, %1848) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x63xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %1850 = stablehlo.select %123, %cst_224, %1849 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1851 = stablehlo.select %120, %1850, %1836 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %1852 = stablehlo.broadcast_in_dim %1851, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1853 = stablehlo.select %115, %1852, %1834 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %1854 = stablehlo.broadcast_in_dim %c_237, dims = [0] : (tensor<64xui8>) -> tensor<64x64xui8>
    %1855 = stablehlo.broadcast_in_dim %c_237, dims = [1] : (tensor<64xui8>) -> tensor<64x64xui8>
    %1856 = stablehlo.compare  EQ, %1854, %1855 : (tensor<64x64xui8>, tensor<64x64xui8>) -> tensor<64x64xi1>
    %1857 = stablehlo.convert %1856 : (tensor<64x64xi1>) -> tensor<64x64xf32>
    %1858 = stablehlo.broadcast_in_dim %1857, dims = [3, 4] : (tensor<64x64xf32>) -> tensor<1x32x8x64x64xf32>
    %1859 = stablehlo.add %1853, %1858 : tensor<1x32x8x64x64xf32>
    %1860 = stablehlo.slice %27 [0:1, 0:494, 4096:8192] : (tensor<1x494x8192xbf16>) -> tensor<1x494x4096xbf16>
    %1861 = stablehlo.reshape %1860 : (tensor<1x494x4096xbf16>) -> tensor<1x494x32x128xbf16>
    %1862 = stablehlo.transpose %1861, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %1863 = stablehlo.convert %1862 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %1864 = stablehlo.pad %1863, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %1865 = stablehlo.multiply %1864, %638 : tensor<1x32x512x128xf32>
    %1866 = stablehlo.reshape %1865 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %1867 = stablehlo.dot_general %1859, %1866, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x64xf32>, tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %1868 = stablehlo.slice %1867 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1869 = stablehlo.reshape %1868 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1870 = stablehlo.exponential %75 : tensor<1x32x8x64xf32>
    %1871 = stablehlo.broadcast_in_dim %1870, dims = [0, 1, 2, 3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x128xf32>
    %1872 = stablehlo.multiply %640, %1871 : tensor<1x32x8x64x128xf32>
    %1873 = stablehlo.dot_general %1859, %1872, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x64xf32>, tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %1874 = stablehlo.slice %1873 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1875 = stablehlo.reshape %1874 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1876 = stablehlo.dot_general %1875, %cst_227, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %1877 = stablehlo.subtract %1869, %1876 : tensor<1x32x64x128xf32>
    %1878 = stablehlo.dot_general %114, %1877, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %1879 = stablehlo.add %89, %1878 : tensor<1x32x128x128xf32>
    %1880 = stablehlo.slice %75 [0:1, 0:32, 1:2, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %1881 = stablehlo.reshape %1880 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %1882 = stablehlo.slice %1881 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %1883 = stablehlo.reshape %1882 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %1884 = stablehlo.exponential %1883 : tensor<1x32x1x1xf32>
    %1885 = stablehlo.reshape %1884 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %1886 = stablehlo.broadcast_in_dim %1885, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %1887 = stablehlo.multiply %1879, %1886 : tensor<1x32x128x128xf32>
    %1888 = stablehlo.slice %105 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1889 = stablehlo.reshape %1888 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1890 = stablehlo.reshape %1882 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %1891 = stablehlo.broadcast_in_dim %1890, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %1892 = stablehlo.subtract %1891, %1881 : tensor<1x32x64xf32>
    %1893 = stablehlo.exponential %1892 : tensor<1x32x64xf32>
    %1894 = stablehlo.broadcast_in_dim %1893, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %1895 = stablehlo.multiply %1889, %1894 : tensor<1x32x64x128xf32>
    %1896 = stablehlo.transpose %1895, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %1897 = stablehlo.slice %1867 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1898 = stablehlo.reshape %1897 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1899 = stablehlo.slice %1873 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1900 = stablehlo.reshape %1899 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1901 = stablehlo.dot_general %1900, %1879, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %1902 = stablehlo.subtract %1898, %1901 : tensor<1x32x64x128xf32>
    %1903 = stablehlo.dot_general %1896, %1902, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %1904 = stablehlo.add %1887, %1903 : tensor<1x32x128x128xf32>
    %1905 = stablehlo.slice %75 [0:1, 0:32, 2:3, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %1906 = stablehlo.reshape %1905 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %1907 = stablehlo.slice %1906 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %1908 = stablehlo.reshape %1907 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %1909 = stablehlo.exponential %1908 : tensor<1x32x1x1xf32>
    %1910 = stablehlo.reshape %1909 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %1911 = stablehlo.broadcast_in_dim %1910, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %1912 = stablehlo.multiply %1904, %1911 : tensor<1x32x128x128xf32>
    %1913 = stablehlo.slice %105 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1914 = stablehlo.reshape %1913 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1915 = stablehlo.reshape %1907 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %1916 = stablehlo.broadcast_in_dim %1915, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %1917 = stablehlo.subtract %1916, %1906 : tensor<1x32x64xf32>
    %1918 = stablehlo.exponential %1917 : tensor<1x32x64xf32>
    %1919 = stablehlo.broadcast_in_dim %1918, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %1920 = stablehlo.multiply %1914, %1919 : tensor<1x32x64x128xf32>
    %1921 = stablehlo.transpose %1920, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %1922 = stablehlo.slice %1867 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1923 = stablehlo.reshape %1922 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1924 = stablehlo.slice %1873 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1925 = stablehlo.reshape %1924 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1926 = stablehlo.dot_general %1925, %1904, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %1927 = stablehlo.subtract %1923, %1926 : tensor<1x32x64x128xf32>
    %1928 = stablehlo.dot_general %1921, %1927, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %1929 = stablehlo.add %1912, %1928 : tensor<1x32x128x128xf32>
    %1930 = stablehlo.slice %75 [0:1, 0:32, 3:4, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %1931 = stablehlo.reshape %1930 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %1932 = stablehlo.slice %1931 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %1933 = stablehlo.reshape %1932 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %1934 = stablehlo.exponential %1933 : tensor<1x32x1x1xf32>
    %1935 = stablehlo.reshape %1934 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %1936 = stablehlo.broadcast_in_dim %1935, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %1937 = stablehlo.multiply %1929, %1936 : tensor<1x32x128x128xf32>
    %1938 = stablehlo.slice %105 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1939 = stablehlo.reshape %1938 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1940 = stablehlo.reshape %1932 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %1941 = stablehlo.broadcast_in_dim %1940, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %1942 = stablehlo.subtract %1941, %1931 : tensor<1x32x64xf32>
    %1943 = stablehlo.exponential %1942 : tensor<1x32x64xf32>
    %1944 = stablehlo.broadcast_in_dim %1943, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %1945 = stablehlo.multiply %1939, %1944 : tensor<1x32x64x128xf32>
    %1946 = stablehlo.transpose %1945, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %1947 = stablehlo.slice %1867 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1948 = stablehlo.reshape %1947 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1949 = stablehlo.slice %1873 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1950 = stablehlo.reshape %1949 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1951 = stablehlo.dot_general %1950, %1929, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %1952 = stablehlo.subtract %1948, %1951 : tensor<1x32x64x128xf32>
    %1953 = stablehlo.dot_general %1946, %1952, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %1954 = stablehlo.add %1937, %1953 : tensor<1x32x128x128xf32>
    %1955 = stablehlo.slice %75 [0:1, 0:32, 4:5, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %1956 = stablehlo.reshape %1955 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %1957 = stablehlo.slice %1956 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %1958 = stablehlo.reshape %1957 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %1959 = stablehlo.exponential %1958 : tensor<1x32x1x1xf32>
    %1960 = stablehlo.reshape %1959 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %1961 = stablehlo.broadcast_in_dim %1960, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %1962 = stablehlo.multiply %1954, %1961 : tensor<1x32x128x128xf32>
    %1963 = stablehlo.slice %105 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1964 = stablehlo.reshape %1963 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1965 = stablehlo.reshape %1957 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %1966 = stablehlo.broadcast_in_dim %1965, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %1967 = stablehlo.subtract %1966, %1956 : tensor<1x32x64xf32>
    %1968 = stablehlo.exponential %1967 : tensor<1x32x64xf32>
    %1969 = stablehlo.broadcast_in_dim %1968, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %1970 = stablehlo.multiply %1964, %1969 : tensor<1x32x64x128xf32>
    %1971 = stablehlo.transpose %1970, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %1972 = stablehlo.slice %1867 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1973 = stablehlo.reshape %1972 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1974 = stablehlo.slice %1873 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1975 = stablehlo.reshape %1974 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1976 = stablehlo.dot_general %1975, %1954, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %1977 = stablehlo.subtract %1973, %1976 : tensor<1x32x64x128xf32>
    %1978 = stablehlo.dot_general %1971, %1977, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %1979 = stablehlo.add %1962, %1978 : tensor<1x32x128x128xf32>
    %1980 = stablehlo.slice %75 [0:1, 0:32, 5:6, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %1981 = stablehlo.reshape %1980 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %1982 = stablehlo.slice %1981 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %1983 = stablehlo.reshape %1982 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %1984 = stablehlo.exponential %1983 : tensor<1x32x1x1xf32>
    %1985 = stablehlo.reshape %1984 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %1986 = stablehlo.broadcast_in_dim %1985, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %1987 = stablehlo.multiply %1979, %1986 : tensor<1x32x128x128xf32>
    %1988 = stablehlo.slice %105 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1989 = stablehlo.reshape %1988 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1990 = stablehlo.reshape %1982 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %1991 = stablehlo.broadcast_in_dim %1990, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %1992 = stablehlo.subtract %1991, %1981 : tensor<1x32x64xf32>
    %1993 = stablehlo.exponential %1992 : tensor<1x32x64xf32>
    %1994 = stablehlo.broadcast_in_dim %1993, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %1995 = stablehlo.multiply %1989, %1994 : tensor<1x32x64x128xf32>
    %1996 = stablehlo.transpose %1995, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %1997 = stablehlo.slice %1867 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %1998 = stablehlo.reshape %1997 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %1999 = stablehlo.slice %1873 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2000 = stablehlo.reshape %1999 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2001 = stablehlo.dot_general %2000, %1979, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2002 = stablehlo.subtract %1998, %2001 : tensor<1x32x64x128xf32>
    %2003 = stablehlo.dot_general %1996, %2002, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %2004 = stablehlo.add %1987, %2003 : tensor<1x32x128x128xf32>
    %2005 = stablehlo.slice %75 [0:1, 0:32, 6:7, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %2006 = stablehlo.reshape %2005 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %2007 = stablehlo.slice %2006 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %2008 = stablehlo.reshape %2007 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %2009 = stablehlo.exponential %2008 : tensor<1x32x1x1xf32>
    %2010 = stablehlo.reshape %2009 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %2011 = stablehlo.broadcast_in_dim %2010, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %2012 = stablehlo.multiply %2004, %2011 : tensor<1x32x128x128xf32>
    %2013 = stablehlo.slice %105 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2014 = stablehlo.reshape %2013 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2015 = stablehlo.reshape %2007 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %2016 = stablehlo.broadcast_in_dim %2015, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %2017 = stablehlo.subtract %2016, %2006 : tensor<1x32x64xf32>
    %2018 = stablehlo.exponential %2017 : tensor<1x32x64xf32>
    %2019 = stablehlo.broadcast_in_dim %2018, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2020 = stablehlo.multiply %2014, %2019 : tensor<1x32x64x128xf32>
    %2021 = stablehlo.transpose %2020, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2022 = stablehlo.slice %1867 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2023 = stablehlo.reshape %2022 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2024 = stablehlo.slice %1873 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2025 = stablehlo.reshape %2024 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2026 = stablehlo.dot_general %2025, %2004, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2027 = stablehlo.subtract %2023, %2026 : tensor<1x32x64x128xf32>
    %2028 = stablehlo.dot_general %2021, %2027, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %2029 = stablehlo.add %2012, %2028 : tensor<1x32x128x128xf32>
    %2030 = stablehlo.dot_general %81, %2029, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2031 = stablehlo.compare  GE, %622, %c_32 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %2032 = stablehlo.and %2031, %624 : tensor<64x64xi1>
    %2033 = stablehlo.reshape %2032 : (tensor<64x64xi1>) -> tensor<1x64x64xi1>
    %2034 = stablehlo.broadcast_in_dim %2033, dims = [0, 2, 3] : (tensor<1x64x64xi1>) -> tensor<1x32x64x64xi1>
    %2035 = stablehlo.slice %105 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2036 = stablehlo.reshape %2035 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2037 = stablehlo.transpose %2036, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2038 = stablehlo.dot_general %46, %2037, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2039 = stablehlo.slice %651 [0:1, 0:32, 7:8, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2040 = stablehlo.reshape %2039 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2041 = stablehlo.multiply %2038, %2040 : tensor<1x32x64x64xf32>
    %2042 = stablehlo.select %2034, %cst_31, %2041 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2043 = stablehlo.slice %1867 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2044 = stablehlo.reshape %2043 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2045 = stablehlo.slice %1873 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2046 = stablehlo.reshape %2045 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2047 = stablehlo.dot_general %2046, %2029, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2048 = stablehlo.subtract %2044, %2047 : tensor<1x32x64x128xf32>
    %2049 = stablehlo.dot_general %2042, %2048, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2050 = stablehlo.add %2030, %2049 : tensor<1x32x64x128xf32>
    %2051 = stablehlo.broadcast_in_dim %2050, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2052 = stablehlo.broadcast_in_dim %c_30, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2053 = stablehlo.slice %44 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2054 = stablehlo.reshape %2053 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2055 = stablehlo.reshape %2005 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2056 = stablehlo.exponential %2055 : tensor<1x32x64x1xf32>
    %2057 = stablehlo.reshape %2056 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2058 = stablehlo.broadcast_in_dim %2057, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2059 = stablehlo.multiply %2054, %2058 : tensor<1x32x64x128xf32>
    %2060 = stablehlo.dot_general %2059, %2004, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2061 = stablehlo.transpose %2014, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2062 = stablehlo.dot_general %2054, %2061, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2063 = stablehlo.slice %651 [0:1, 0:32, 6:7, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2064 = stablehlo.reshape %2063 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2065 = stablehlo.multiply %2062, %2064 : tensor<1x32x64x64xf32>
    %2066 = stablehlo.select %2034, %cst_31, %2065 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2067 = stablehlo.dot_general %2066, %2027, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2068 = stablehlo.add %2060, %2067 : tensor<1x32x64x128xf32>
    %2069 = stablehlo.broadcast_in_dim %2068, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2070 = stablehlo.broadcast_in_dim %c_29, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2071 = stablehlo.slice %44 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2072 = stablehlo.reshape %2071 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2073 = stablehlo.reshape %1980 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2074 = stablehlo.exponential %2073 : tensor<1x32x64x1xf32>
    %2075 = stablehlo.reshape %2074 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2076 = stablehlo.broadcast_in_dim %2075, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2077 = stablehlo.multiply %2072, %2076 : tensor<1x32x64x128xf32>
    %2078 = stablehlo.dot_general %2077, %1979, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2079 = stablehlo.transpose %1989, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2080 = stablehlo.dot_general %2072, %2079, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2081 = stablehlo.slice %651 [0:1, 0:32, 5:6, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2082 = stablehlo.reshape %2081 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2083 = stablehlo.multiply %2080, %2082 : tensor<1x32x64x64xf32>
    %2084 = stablehlo.select %2034, %cst_31, %2083 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2085 = stablehlo.dot_general %2084, %2002, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2086 = stablehlo.add %2078, %2085 : tensor<1x32x64x128xf32>
    %2087 = stablehlo.broadcast_in_dim %2086, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2088 = stablehlo.broadcast_in_dim %c_28, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2089 = stablehlo.slice %44 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2090 = stablehlo.reshape %2089 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2091 = stablehlo.reshape %1955 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2092 = stablehlo.exponential %2091 : tensor<1x32x64x1xf32>
    %2093 = stablehlo.reshape %2092 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2094 = stablehlo.broadcast_in_dim %2093, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2095 = stablehlo.multiply %2090, %2094 : tensor<1x32x64x128xf32>
    %2096 = stablehlo.dot_general %2095, %1954, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2097 = stablehlo.transpose %1964, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2098 = stablehlo.dot_general %2090, %2097, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2099 = stablehlo.slice %651 [0:1, 0:32, 4:5, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2100 = stablehlo.reshape %2099 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2101 = stablehlo.multiply %2098, %2100 : tensor<1x32x64x64xf32>
    %2102 = stablehlo.select %2034, %cst_31, %2101 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2103 = stablehlo.dot_general %2102, %1977, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2104 = stablehlo.add %2096, %2103 : tensor<1x32x64x128xf32>
    %2105 = stablehlo.broadcast_in_dim %2104, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2106 = stablehlo.broadcast_in_dim %c_27, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2107 = stablehlo.slice %44 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2108 = stablehlo.reshape %2107 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2109 = stablehlo.reshape %1930 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2110 = stablehlo.exponential %2109 : tensor<1x32x64x1xf32>
    %2111 = stablehlo.reshape %2110 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2112 = stablehlo.broadcast_in_dim %2111, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2113 = stablehlo.multiply %2108, %2112 : tensor<1x32x64x128xf32>
    %2114 = stablehlo.dot_general %2113, %1929, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2115 = stablehlo.transpose %1939, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2116 = stablehlo.dot_general %2108, %2115, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2117 = stablehlo.slice %651 [0:1, 0:32, 3:4, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2118 = stablehlo.reshape %2117 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2119 = stablehlo.multiply %2116, %2118 : tensor<1x32x64x64xf32>
    %2120 = stablehlo.select %2034, %cst_31, %2119 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2121 = stablehlo.dot_general %2120, %1952, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2122 = stablehlo.add %2114, %2121 : tensor<1x32x64x128xf32>
    %2123 = stablehlo.broadcast_in_dim %2122, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2124 = stablehlo.broadcast_in_dim %c_26, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2125 = stablehlo.slice %44 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2126 = stablehlo.reshape %2125 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2127 = stablehlo.reshape %1905 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2128 = stablehlo.exponential %2127 : tensor<1x32x64x1xf32>
    %2129 = stablehlo.reshape %2128 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2130 = stablehlo.broadcast_in_dim %2129, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2131 = stablehlo.multiply %2126, %2130 : tensor<1x32x64x128xf32>
    %2132 = stablehlo.dot_general %2131, %1904, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2133 = stablehlo.transpose %1914, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2134 = stablehlo.dot_general %2126, %2133, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2135 = stablehlo.slice %651 [0:1, 0:32, 2:3, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2136 = stablehlo.reshape %2135 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2137 = stablehlo.multiply %2134, %2136 : tensor<1x32x64x64xf32>
    %2138 = stablehlo.select %2034, %cst_31, %2137 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2139 = stablehlo.dot_general %2138, %1927, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2140 = stablehlo.add %2132, %2139 : tensor<1x32x64x128xf32>
    %2141 = stablehlo.broadcast_in_dim %2140, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2142 = stablehlo.broadcast_in_dim %c_25, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2143 = stablehlo.slice %44 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2144 = stablehlo.reshape %2143 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2145 = stablehlo.reshape %1880 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2146 = stablehlo.exponential %2145 : tensor<1x32x64x1xf32>
    %2147 = stablehlo.reshape %2146 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2148 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2149 = stablehlo.multiply %2144, %2148 : tensor<1x32x64x128xf32>
    %2150 = stablehlo.dot_general %2149, %1879, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2151 = stablehlo.transpose %1889, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2152 = stablehlo.dot_general %2144, %2151, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2153 = stablehlo.slice %651 [0:1, 0:32, 1:2, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2154 = stablehlo.reshape %2153 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2155 = stablehlo.multiply %2152, %2154 : tensor<1x32x64x64xf32>
    %2156 = stablehlo.select %2034, %cst_31, %2155 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2157 = stablehlo.dot_general %2156, %1902, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2158 = stablehlo.add %2150, %2157 : tensor<1x32x64x128xf32>
    %2159 = stablehlo.broadcast_in_dim %2158, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2160 = stablehlo.broadcast_in_dim %c_24, dims = [0, 2] : (tensor<1x8xi1>) -> tensor<1x32x8x64x128xi1>
    %2161 = stablehlo.slice %44 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2162 = stablehlo.reshape %2161 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2163 = stablehlo.reshape %82 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2164 = stablehlo.exponential %2163 : tensor<1x32x64x1xf32>
    %2165 = stablehlo.reshape %2164 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2166 = stablehlo.broadcast_in_dim %2165, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2167 = stablehlo.multiply %2162, %2166 : tensor<1x32x64x128xf32>
    %2168 = stablehlo.dot_general %2167, %cst_227, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %2169 = stablehlo.transpose %107, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2170 = stablehlo.dot_general %2162, %2169, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %2171 = stablehlo.slice %651 [0:1, 0:32, 0:1, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %2172 = stablehlo.reshape %2171 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %2173 = stablehlo.multiply %2170, %2172 : tensor<1x32x64x64xf32>
    %2174 = stablehlo.select %2034, %cst_31, %2173 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %2175 = stablehlo.dot_general %2174, %1877, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2176 = stablehlo.add %2168, %2175 : tensor<1x32x64x128xf32>
    %2177 = stablehlo.broadcast_in_dim %2176, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2178 = stablehlo.select %2160, %2177, %cst_23 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2179 = stablehlo.select %2142, %2159, %2178 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2180 = stablehlo.select %2124, %2141, %2179 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2181 = stablehlo.select %2106, %2123, %2180 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2182 = stablehlo.select %2088, %2105, %2181 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2183 = stablehlo.select %2070, %2087, %2182 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2184 = stablehlo.select %2052, %2069, %2183 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2185 = stablehlo.select %1, %2051, %2184 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %2186 = stablehlo.reshape %2185 : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x512x128xf32>
    %2187 = stablehlo.slice %2186 [0:1, 0:32, 0:494, 0:128] : (tensor<1x32x512x128xf32>) -> tensor<1x32x494x128xf32>
    %2188 = stablehlo.transpose %2187, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,494,32,128]{3,1,2,0}"} : (tensor<1x32x494x128xf32>) -> tensor<1x494x32x128xf32>
    %2189 = stablehlo.convert %2188 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,494,32,128]{3,1,2,0}"} : (tensor<1x494x32x128xf32>) -> tensor<1x494x32x128xbf16>
    %2190 = stablehlo.reshape %2189 : (tensor<1x494x32x128xbf16>) -> tensor<15808x128xbf16>
    %2191 = stablehlo.reshape %arg26 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
    %2192 = stablehlo.custom_call @tt.mark_argument(%2191) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.norm.weight"}} : (tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16>
    %2193 = stablehlo.reshape %2192 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
    %2194 = stablehlo.composite "tenstorrent.rms_norm" %2190, %2193 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_3} : (tensor<15808x128xbf16>, tensor<128xbf16>) -> tensor<15808x128xbf16>
    %2195 = stablehlo.convert %2194 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %2196 = stablehlo.reshape %arg23 : (tensor<4096x2048xbf16>) -> tensor<1x4096x2048xbf16>
    %2197 = stablehlo.custom_call @tt.mark_argument(%2196) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.in_proj_z.weight"}} : (tensor<1x4096x2048xbf16>) -> tensor<1x4096x2048xbf16>
    %2198 = stablehlo.reshape %2197 : (tensor<1x4096x2048xbf16>) -> tensor<4096x2048xbf16>
    %2199 = stablehlo.transpose %2198, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,4096]{0,1}"} : (tensor<4096x2048xbf16>) -> tensor<2048x4096xbf16>
    %2200 = stablehlo.dot_general %8, %2199, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x4096xbf16>) -> tensor<494x4096xbf16>
    %2201 = stablehlo.reshape %2200 : (tensor<494x4096xbf16>) -> tensor<15808x128xbf16>
    %2202 = stablehlo.convert %2201 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %2203 = stablehlo.logistic %2202 : tensor<15808x128xf32>
    %2204 = stablehlo.multiply %2202, %2203 : tensor<15808x128xf32>
    %2205 = stablehlo.multiply %2195, %2204 : tensor<15808x128xf32>
    %2206 = stablehlo.convert %2205 : (tensor<15808x128xf32>) -> tensor<15808x128xbf16>
    %2207 = stablehlo.reshape %2206 : (tensor<15808x128xbf16>) -> tensor<494x4096xbf16>
    %2208 = stablehlo.reshape %arg22 : (tensor<2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %2209 = stablehlo.custom_call @tt.mark_argument(%2208) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.linear_attn.out_proj.weight"}} : (tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %2210 = stablehlo.reshape %2209 : (tensor<1x2048x4096xbf16>) -> tensor<2048x4096xbf16>
    %2211 = stablehlo.transpose %2210, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[4096,2048]{0,1}"} : (tensor<2048x4096xbf16>) -> tensor<4096x2048xbf16>
    %2212 = stablehlo.dot_general %2207, %2211, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<494x2048xbf16>
    %2213 = stablehlo.reshape %2212 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %2214 = stablehlo.add %0, %2213 : tensor<1x494x2048xbf16>
    %2215 = stablehlo.reshape %arg21 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %2216 = stablehlo.custom_call @tt.mark_argument(%2215) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.post_attention_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %2217 = stablehlo.reshape %2216 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %2218 = stablehlo.convert %2217 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %2219 = stablehlo.add %2218, %cst_231 : tensor<2048xf32>
    %2220 = stablehlo.composite "tenstorrent.rms_norm" %2214, %2219 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_5} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %2221 = stablehlo.reshape %2220 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %2222 = stablehlo.reshape %arg36 : (tensor<256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %2223 = stablehlo.custom_call @tt.mark_argument(%2222) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.mlp.gate.weight"}} : (tensor<1x256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %2224 = stablehlo.reshape %2223 : (tensor<1x256x2048xbf16>) -> tensor<256x2048xbf16>
    %2225 = stablehlo.transpose %2224, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,256]{0,1}"} : (tensor<256x2048xbf16>) -> tensor<2048x256xbf16>
    %2226 = stablehlo.dot_general %2221, %2225, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x256xbf16>) -> tensor<494x256xbf16>
    %2227 = stablehlo.convert %2226 : (tensor<494x256xbf16>) -> tensor<494x256xf32>
    %2228 = stablehlo.reduce(%2227 init: %cst_236) applies stablehlo.maximum across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %2229 = stablehlo.broadcast_in_dim %2228, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %2230 = stablehlo.subtract %2227, %2229 : tensor<494x256xf32>
    %2231 = stablehlo.exponential %2230 : tensor<494x256xf32>
    %2232 = stablehlo.reduce(%2231 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %2233 = stablehlo.broadcast_in_dim %2232, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %2234 = stablehlo.divide %2231, %2233 : tensor<494x256xf32>
    %2235:2 = stablehlo.composite "tenstorrent.topk" %2234 {composite_attributes = {dim = -1 : i64, k = 8 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk.impl_0} : (tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>)
    %2236 = stablehlo.reduce(%2235#0 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x8xf32>, tensor<f32>) -> tensor<494xf32>
    %2237 = stablehlo.broadcast_in_dim %2236, dims = [0] : (tensor<494xf32>) -> tensor<494x8xf32>
    %2238 = stablehlo.divide %2235#0, %2237 : tensor<494x8xf32>
    %2239 = stablehlo.concatenate %2238, %cst_22, dim = 0 : (tensor<494x8xf32>, tensor<18x8xf32>) -> tensor<512x8xf32>
    %2240 = stablehlo.convert %2239 : (tensor<512x8xf32>) -> tensor<512x8xbf16>
    %2241 = stablehlo.reshape %2240 : (tensor<512x8xbf16>) -> tensor<512x1x8xbf16>
    %2242 = stablehlo.concatenate %2235#1, %c_21, dim = 0 : (tensor<494x8xi64>, tensor<18x8xi64>) -> tensor<512x8xi64>
    %2243 = stablehlo.broadcast_in_dim %2242, dims = [0, 1] : (tensor<512x8xi64>) -> tensor<512x8x256xi64>
    %2244 = stablehlo.broadcast_in_dim %c_235, dims = [2] : (tensor<256xi64>) -> tensor<512x8x256xi64>
    %2245 = stablehlo.compare  EQ, %2243, %2244 : (tensor<512x8x256xi64>, tensor<512x8x256xi64>) -> tensor<512x8x256xi1>
    %2246 = stablehlo.convert %2245 : (tensor<512x8x256xi1>) -> tensor<512x8x256xbf16>
    %2247 = stablehlo.dot_general %2241, %2246, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x1x8xbf16>, tensor<512x8x256xbf16>) -> tensor<512x1x256xbf16>
    %2248 = stablehlo.reshape %2247 : (tensor<512x1x256xbf16>) -> tensor<1x512x256xbf16>
    %2249 = stablehlo.concatenate %2248, %2248, %2248, %2248, %2248, %2248, %2248, %2248, dim = 1 : (tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>) -> tensor<1x4096x256xbf16>
    %2250 = stablehlo.reshape %2249 : (tensor<1x4096x256xbf16>) -> tensor<1x1x4096x256xbf16>
    %2251 = stablehlo.floor %cst_20 : tensor<256xf32>
    %2252 = stablehlo.convert %2251 : (tensor<256xf32>) -> tensor<256xi64>
    %2253 = stablehlo.convert %2252 : (tensor<256xi64>) -> tensor<256xui16>
    %2254 = stablehlo.broadcast_in_dim %2253, dims = [0] : (tensor<256xui16>) -> tensor<256x8xui16>
    %2255 = stablehlo.broadcast_in_dim %c_19, dims = [1] : (tensor<8xui16>) -> tensor<256x8xui16>
    %2256 = stablehlo.compare  EQ, %2254, %2255 : (tensor<256x8xui16>, tensor<256x8xui16>) -> tensor<256x8xi1>
    %2257 = stablehlo.convert %2256 : (tensor<256x8xi1>) -> tensor<256x8xui16>
    %2258 = stablehlo.reshape %2257 : (tensor<256x8xui16>) -> tensor<1x1x256x8xui16>
    %2259 = stablehlo.concatenate %2221, %cst_18, dim = 0 : (tensor<494x2048xbf16>, tensor<18x2048xbf16>) -> tensor<512x2048xbf16>
    %2260 = stablehlo.reshape %2259 : (tensor<512x2048xbf16>) -> tensor<1x1x512x2048xbf16>
    %2261 = stablehlo.reshape %2242 : (tensor<512x8xi64>) -> tensor<1x1x512x8xi64>
    %2262 = stablehlo.custom_call @tt.all_to_all_dispatch(%2260, %2261, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8"}, xla_shape = "(bf16[1,8,512,2048]{3,2,1,0}, s64[1,8,512,8]{3,2,1,0})"} : (tensor<1x1x512x2048xbf16>, tensor<1x1x512x8xi64>, tensor<1x1x256x8xui16>) -> tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>
    %2263 = stablehlo.get_tuple_element %2262[1] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x8xi64>
    %2264 = stablehlo.reshape %2263 : (tensor<1x8x512x8xi64>) -> tensor<1x1x4096x8xi64>
    %2265 = stablehlo.custom_call @tt.moe_expert_token_remap(%2250, %2258, %2264) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {num_devices = "8", reduction_size = "32"}, xla_shape = "(bf16[1,1,4096,256]{3,2,1,0}, bf16[1,1,128,256]{3,2,1,0})"} : (tensor<1x1x4096x256xbf16>, tensor<1x1x256x8xui16>, tensor<1x1x4096x8xi64>) -> tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>
    %2266 = stablehlo.get_tuple_element %2262[0] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x2048xbf16>
    %2267 = stablehlo.reshape %2266 : (tensor<1x8x512x2048xbf16>) -> tensor<8x16x32x2048xbf16>
    %2268 = stablehlo.custom_call @tt.mark_argument(%arg38) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.mlp.experts.gate_up_proj"}} : (tensor<256x1024x2048xbf16>) -> tensor<256x1024x2048xbf16>
    %2269 = stablehlo.transpose %2268, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,2048,1024]{1,2,0}"} : (tensor<256x1024x2048xbf16>) -> tensor<256x2048x1024xbf16>
    %2270 = stablehlo.reshape %2269 : (tensor<256x2048x1024xbf16>) -> tensor<1x256x2048x1024xbf16>
    %2271 = stablehlo.get_tuple_element %2265[1] : (tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>) -> tensor<1x1x128x256xbf16>
    %2272 = stablehlo.reshape %2271 : (tensor<1x1x128x256xbf16>) -> tensor<8x16x1x256xbf16>
    %2273 = stablehlo.custom_call @tt.sparse_matmul(%2267, %2270, %2272) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True", nnz = "0"}} : (tensor<8x16x32x2048xbf16>, tensor<1x256x2048x1024xbf16>, tensor<8x16x1x256xbf16>) -> tensor<8x16x1x256x32x1024xbf16>
    %2274 = stablehlo.reshape %2273 : (tensor<8x16x1x256x32x1024xbf16>) -> tensor<8x16x256x32x1024xbf16>
    %2275 = stablehlo.slice %2274 [0:8, 0:16, 0:256, 0:32, 0:512] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %2276 = stablehlo.convert %2275 : (tensor<8x16x256x32x512xbf16>) -> tensor<8x16x256x32x512xf32>
    %2277 = stablehlo.logistic %2276 : tensor<8x16x256x32x512xf32>
    %2278 = stablehlo.multiply %2276, %2277 : tensor<8x16x256x32x512xf32>
    %2279 = stablehlo.convert %2278 : (tensor<8x16x256x32x512xf32>) -> tensor<8x16x256x32x512xbf16>
    %2280 = stablehlo.slice %2274 [0:8, 0:16, 0:256, 0:32, 512:1024] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %2281 = stablehlo.multiply %2279, %2280 : tensor<8x16x256x32x512xbf16>
    %2282 = stablehlo.reshape %2281 : (tensor<8x16x256x32x512xbf16>) -> tensor<128x256x32x512xbf16>
    %2283 = stablehlo.custom_call @tt.mark_argument(%arg37) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.mlp.experts.down_proj"}} : (tensor<256x2048x512xbf16>) -> tensor<256x2048x512xbf16>
    %2284 = stablehlo.transpose %2283, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,512,2048]{1,2,0}"} : (tensor<256x2048x512xbf16>) -> tensor<256x512x2048xbf16>
    %2285 = stablehlo.reshape %2284 : (tensor<256x512x2048xbf16>) -> tensor<1x256x512x2048xbf16>
    %2286 = stablehlo.custom_call @tt.sparse_matmul(%2282, %2285, %2271) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "True", is_input_b_sparse = "False", nnz = "0"}} : (tensor<128x256x32x512xbf16>, tensor<1x256x512x2048xbf16>, tensor<1x1x128x256xbf16>) -> tensor<128x256x32x2048xbf16>
    %2287 = stablehlo.transpose %2286, dims = [1, 0, 2, 3] {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, xla_shape = "bf16[256,128,32,2048]{3,2,0,1}"} : (tensor<128x256x32x2048xbf16>) -> tensor<256x128x32x2048xbf16>
    %2288 = stablehlo.reshape %2287 : (tensor<256x128x32x2048xbf16>) -> tensor<256x1x4096x2048xbf16>
    %2289 = stablehlo.custom_call @tt.all_to_all_combine(%2288, %2264, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8", num_experts_per_tok = "8", output_shard_dim = "2"}} : (tensor<256x1x4096x2048xbf16>, tensor<1x1x4096x8xi64>, tensor<1x1x256x8xui16>) -> tensor<8x1x512x2048xbf16>
    %2290 = stablehlo.transpose %2239, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[8,512]{0,1}"} : (tensor<512x8xf32>) -> tensor<8x512xf32>
    %2291 = stablehlo.reshape %2290 : (tensor<8x512xf32>) -> tensor<8x1x512x1xf32>
    %2292 = stablehlo.convert %2291 : (tensor<8x1x512x1xf32>) -> tensor<8x1x512x1xbf16>
    %2293 = stablehlo.reshape %2292 : (tensor<8x1x512x1xbf16>) -> tensor<8x1x512xbf16>
    %2294 = stablehlo.broadcast_in_dim %2293, dims = [0, 1, 2] : (tensor<8x1x512xbf16>) -> tensor<8x1x512x2048xbf16>
    %2295 = stablehlo.multiply %2289, %2294 : tensor<8x1x512x2048xbf16>
    %2296 = stablehlo.reduce(%2295 init: %cst_239) applies stablehlo.add across dimensions = [0] : (tensor<8x1x512x2048xbf16>, tensor<bf16>) -> tensor<1x512x2048xbf16>
    %2297 = stablehlo.reshape %2296 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %2298 = stablehlo.slice %2297 [0:494, 0:2048] : (tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %2299 = stablehlo.reshape %arg35 : (tensor<1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %2300 = stablehlo.custom_call @tt.mark_argument(%2299) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.mlp.shared_expert_gate.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %2301 = stablehlo.reshape %2300 : (tensor<1x1x2048xbf16>) -> tensor<2048x1xbf16>
    %2302 = stablehlo.dot_general %2221, %2301, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x1xbf16>) -> tensor<494x1xbf16>
    %2303 = stablehlo.logistic %2302 : tensor<494x1xbf16>
    %2304 = stablehlo.reshape %2303 : (tensor<494x1xbf16>) -> tensor<494xbf16>
    %2305 = stablehlo.broadcast_in_dim %2304, dims = [0] : (tensor<494xbf16>) -> tensor<494x2048xbf16>
    %2306 = stablehlo.reshape %arg34 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %2307 = stablehlo.custom_call @tt.mark_argument(%2306) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.mlp.shared_expert.gate_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %2308 = stablehlo.reshape %2307 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %2309 = stablehlo.transpose %2308, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %2310 = stablehlo.dot_general %2221, %2309, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %2311 = stablehlo.convert %2310 : (tensor<494x512xbf16>) -> tensor<494x512xf32>
    %2312 = stablehlo.logistic %2311 : tensor<494x512xf32>
    %2313 = stablehlo.multiply %2311, %2312 : tensor<494x512xf32>
    %2314 = stablehlo.convert %2313 : (tensor<494x512xf32>) -> tensor<494x512xbf16>
    %2315 = stablehlo.reshape %arg20 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %2316 = stablehlo.custom_call @tt.mark_argument(%2315) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.mlp.shared_expert.up_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %2317 = stablehlo.reshape %2316 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %2318 = stablehlo.transpose %2317, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %2319 = stablehlo.dot_general %2221, %2318, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %2320 = stablehlo.multiply %2314, %2319 : tensor<494x512xbf16>
    %2321 = stablehlo.reshape %arg19 : (tensor<2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %2322 = stablehlo.custom_call @tt.mark_argument(%2321) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.0.mlp.shared_expert.down_proj.weight"}} : (tensor<1x2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %2323 = stablehlo.reshape %2322 : (tensor<1x2048x512xbf16>) -> tensor<2048x512xbf16>
    %2324 = stablehlo.transpose %2323, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[512,2048]{0,1}"} : (tensor<2048x512xbf16>) -> tensor<512x2048xbf16>
    %2325 = stablehlo.dot_general %2320, %2324, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x512xbf16>, tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %2326 = stablehlo.multiply %2305, %2325 : tensor<494x2048xbf16>
    %2327 = stablehlo.add %2298, %2326 : tensor<494x2048xbf16>
    %2328 = stablehlo.reshape %2327 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %2329 = stablehlo.add %2214, %2328 : tensor<1x494x2048xbf16>
    %2330 = stablehlo.reshape %arg18 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %2331 = stablehlo.custom_call @tt.mark_argument(%2330) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.input_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %2332 = stablehlo.reshape %2331 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %2333 = stablehlo.convert %2332 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %2334 = stablehlo.add %2333, %cst_231 : tensor<2048xf32>
    %2335 = stablehlo.composite "tenstorrent.rms_norm" %2329, %2334 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_6} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %2336 = stablehlo.reshape %2335 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %2337 = stablehlo.reshape %arg45 : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %2338 = stablehlo.custom_call @tt.mark_argument(%2337) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.in_proj_qkv.weight"}} : (tensor<1x8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %2339 = stablehlo.reshape %2338 : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %2340 = stablehlo.transpose %2339, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,8192]{0,1}"} : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %2341 = stablehlo.dot_general %2336, %2340, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<494x8192xbf16>
    %2342 = stablehlo.reshape %2341 : (tensor<494x8192xbf16>) -> tensor<1x494x8192xbf16>
    %2343 = stablehlo.transpose %2342, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,8192,494]{1,2,0}"} : (tensor<1x494x8192xbf16>) -> tensor<1x8192x494xbf16>
    %2344 = stablehlo.reshape %2343 : (tensor<1x8192x494xbf16>) -> tensor<1x8192x1x494xbf16>
    %2345 = stablehlo.custom_call @tt.mark_argument(%arg44) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.conv1d.weight"}} : (tensor<8192x1x4xbf16>) -> tensor<8192x1x4xbf16>
    %2346 = stablehlo.reshape %2345 : (tensor<8192x1x4xbf16>) -> tensor<8192x1x1x4xbf16>
    %2347 = stablehlo.convolution(%2344, %2346) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 8192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x8192x1x494xbf16>, tensor<8192x1x1x4xbf16>) -> tensor<1x8192x1x497xbf16>
    %2348 = stablehlo.reshape %2347 : (tensor<1x8192x1x497xbf16>) -> tensor<1x8192x497xbf16>
    %2349 = stablehlo.custom_call @tt.sharding_constraint(%2348) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>"}} : (tensor<1x8192x497xbf16>) -> tensor<1x8192x497xbf16>
    %2350 = stablehlo.slice %2349 [0:1, 0:8192, 0:494] : (tensor<1x8192x497xbf16>) -> tensor<1x8192x494xbf16>
    %2351 = stablehlo.convert %2350 : (tensor<1x8192x494xbf16>) -> tensor<1x8192x494xf32>
    %2352 = stablehlo.logistic %2351 : tensor<1x8192x494xf32>
    %2353 = stablehlo.multiply %2351, %2352 : tensor<1x8192x494xf32>
    %2354 = stablehlo.convert %2353 : (tensor<1x8192x494xf32>) -> tensor<1x8192x494xbf16>
    %2355 = stablehlo.transpose %2354, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,494,8192]{1,2,0}"} : (tensor<1x8192x494xbf16>) -> tensor<1x494x8192xbf16>
    %2356 = stablehlo.slice %2355 [0:1, 0:494, 0:2048] : (tensor<1x494x8192xbf16>) -> tensor<1x494x2048xbf16>
    %2357 = stablehlo.reshape %2356 : (tensor<1x494x2048xbf16>) -> tensor<1x494x16x128xbf16>
    %2358 = stablehlo.broadcast_in_dim %2357, dims = [0, 1, 2, 4] : (tensor<1x494x16x128xbf16>) -> tensor<1x494x16x2x128xbf16>
    %2359 = stablehlo.reshape %2358 : (tensor<1x494x16x2x128xbf16>) -> tensor<1x494x32x128xbf16>
    %2360 = stablehlo.multiply %2359, %2359 : tensor<1x494x32x128xbf16>
    %2361 = stablehlo.reduce(%2360 init: %cst_239) applies stablehlo.add across dimensions = [3] : (tensor<1x494x32x128xbf16>, tensor<bf16>) -> tensor<1x494x32xbf16>
    %2362 = stablehlo.reshape %2361 : (tensor<1x494x32xbf16>) -> tensor<1x494x32x1xbf16>
    %2363 = stablehlo.add %2362, %cst_230 : tensor<1x494x32x1xbf16>
    %2364 = stablehlo.rsqrt %2363 : tensor<1x494x32x1xbf16>
    %2365 = stablehlo.reshape %2364 : (tensor<1x494x32x1xbf16>) -> tensor<1x494x32xbf16>
    %2366 = stablehlo.broadcast_in_dim %2365, dims = [0, 1, 2] : (tensor<1x494x32xbf16>) -> tensor<1x494x32x128xbf16>
    %2367 = stablehlo.multiply %2359, %2366 : tensor<1x494x32x128xbf16>
    %2368 = stablehlo.transpose %2367, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %2369 = stablehlo.convert %2368 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %2370 = stablehlo.pad %2369, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %2371 = stablehlo.multiply %2370, %cst_229 : tensor<1x32x512x128xf32>
    %2372 = stablehlo.reshape %2371 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2373 = stablehlo.slice %2372 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2374 = stablehlo.reshape %2373 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2375 = stablehlo.reshape %arg42 : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %2376 = stablehlo.custom_call @tt.mark_argument(%2375) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.A_log"}} : (tensor<1x1x32xbf16>) -> tensor<1x1x32xbf16>
    %2377 = stablehlo.reshape %2376 : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
    %2378 = stablehlo.convert %2377 : (tensor<32xbf16>) -> tensor<32xf32>
    %2379 = stablehlo.exponential %2378 : tensor<32xf32>
    %2380 = stablehlo.negate %2379 : tensor<32xf32>
    %2381 = stablehlo.broadcast_in_dim %2380, dims = [2] : (tensor<32xf32>) -> tensor<1x494x32xf32>
    %2382 = stablehlo.reshape %arg41 : (tensor<32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %2383 = stablehlo.custom_call @tt.mark_argument(%2382) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.in_proj_a.weight"}} : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %2384 = stablehlo.reshape %2383 : (tensor<1x32x2048xbf16>) -> tensor<32x2048xbf16>
    %2385 = stablehlo.transpose %2384, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,32]{0,1}"} : (tensor<32x2048xbf16>) -> tensor<2048x32xbf16>
    %2386 = stablehlo.dot_general %2336, %2385, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x32xbf16>) -> tensor<494x32xbf16>
    %2387 = stablehlo.reshape %2386 : (tensor<494x32xbf16>) -> tensor<1x494x32xbf16>
    %2388 = stablehlo.convert %2387 : (tensor<1x494x32xbf16>) -> tensor<1x494x32xf32>
    %2389 = stablehlo.reshape %arg40 : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %2390 = stablehlo.custom_call @tt.mark_argument(%2389) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.dt_bias"}} : (tensor<1x1x32xbf16>) -> tensor<1x1x32xbf16>
    %2391 = stablehlo.reshape %2390 : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
    %2392 = stablehlo.convert %2391 : (tensor<32xbf16>) -> tensor<32xf32>
    %2393 = stablehlo.broadcast_in_dim %2392, dims = [2] : (tensor<32xf32>) -> tensor<1x494x32xf32>
    %2394 = stablehlo.add %2388, %2393 : tensor<1x494x32xf32>
    %2395 = stablehlo.compare  GT, %2394, %cst_228 : (tensor<1x494x32xf32>, tensor<1x494x32xf32>) -> tensor<1x494x32xi1>
    %2396 = stablehlo.exponential %2394 : tensor<1x494x32xf32>
    %2397 = stablehlo.log_plus_one %2396 : tensor<1x494x32xf32>
    %2398 = stablehlo.select %2395, %2394, %2397 : tensor<1x494x32xi1>, tensor<1x494x32xf32>
    %2399 = stablehlo.multiply %2381, %2398 : tensor<1x494x32xf32>
    %2400 = stablehlo.transpose %2399, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,494]{1,2,0}"} : (tensor<1x494x32xf32>) -> tensor<1x32x494xf32>
    %2401 = stablehlo.pad %2400, %cst_240, low = [0, 0, 0], high = [0, 0, 18], interior = [0, 0, 0] : (tensor<1x32x494xf32>, tensor<f32>) -> tensor<1x32x512xf32>
    %2402 = stablehlo.reshape %2401 : (tensor<1x32x512xf32>) -> tensor<1x32x8x64xf32>
    %2403 = "stablehlo.reduce_window"(%2402, %cst_240) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [63, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 64>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg75: tensor<f32>, %arg76: tensor<f32>):
      %5568 = stablehlo.add %arg75, %arg76 : tensor<f32>
      stablehlo.return %5568 : tensor<f32>
    }) : (tensor<1x32x8x64xf32>, tensor<f32>) -> tensor<1x32x8x64xf32>
    %2404 = stablehlo.slice %2403 [0:1, 0:32, 7:8, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %2405 = stablehlo.reshape %2404 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %2406 = stablehlo.exponential %2405 : tensor<1x32x64x1xf32>
    %2407 = stablehlo.reshape %2406 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2408 = stablehlo.broadcast_in_dim %2407, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2409 = stablehlo.multiply %2374, %2408 : tensor<1x32x64x128xf32>
    %2410 = stablehlo.slice %2403 [0:1, 0:32, 0:1, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %2411 = stablehlo.reshape %2410 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %2412 = stablehlo.slice %2411 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %2413 = stablehlo.reshape %2412 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %2414 = stablehlo.exponential %2413 : tensor<1x32x1x1xf32>
    %2415 = stablehlo.reshape %2414 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %2416 = stablehlo.broadcast_in_dim %2415, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %2417 = stablehlo.multiply %2416, %cst_227 : tensor<1x32x128x128xf32>
    %2418 = stablehlo.slice %2355 [0:1, 0:494, 2048:4096] : (tensor<1x494x8192xbf16>) -> tensor<1x494x2048xbf16>
    %2419 = stablehlo.reshape %2418 : (tensor<1x494x2048xbf16>) -> tensor<1x494x16x128xbf16>
    %2420 = stablehlo.broadcast_in_dim %2419, dims = [0, 1, 2, 4] : (tensor<1x494x16x128xbf16>) -> tensor<1x494x16x2x128xbf16>
    %2421 = stablehlo.reshape %2420 : (tensor<1x494x16x2x128xbf16>) -> tensor<1x494x32x128xbf16>
    %2422 = stablehlo.multiply %2421, %2421 : tensor<1x494x32x128xbf16>
    %2423 = stablehlo.reduce(%2422 init: %cst_239) applies stablehlo.add across dimensions = [3] : (tensor<1x494x32x128xbf16>, tensor<bf16>) -> tensor<1x494x32xbf16>
    %2424 = stablehlo.reshape %2423 : (tensor<1x494x32xbf16>) -> tensor<1x494x32x1xbf16>
    %2425 = stablehlo.add %2424, %cst_230 : tensor<1x494x32x1xbf16>
    %2426 = stablehlo.rsqrt %2425 : tensor<1x494x32x1xbf16>
    %2427 = stablehlo.reshape %2426 : (tensor<1x494x32x1xbf16>) -> tensor<1x494x32xbf16>
    %2428 = stablehlo.broadcast_in_dim %2427, dims = [0, 1, 2] : (tensor<1x494x32xbf16>) -> tensor<1x494x32x128xbf16>
    %2429 = stablehlo.multiply %2421, %2428 : tensor<1x494x32x128xbf16>
    %2430 = stablehlo.transpose %2429, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %2431 = stablehlo.convert %2430 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %2432 = stablehlo.pad %2431, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %2433 = stablehlo.reshape %2432 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2434 = stablehlo.slice %2433 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %2435 = stablehlo.reshape %2434 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %2436 = stablehlo.reshape %2412 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %2437 = stablehlo.broadcast_in_dim %2436, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %2438 = stablehlo.subtract %2437, %2411 : tensor<1x32x64xf32>
    %2439 = stablehlo.exponential %2438 : tensor<1x32x64xf32>
    %2440 = stablehlo.broadcast_in_dim %2439, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %2441 = stablehlo.multiply %2435, %2440 : tensor<1x32x64x128xf32>
    %2442 = stablehlo.transpose %2441, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %2443 = stablehlo.reshape %arg43 : (tensor<32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %2444 = stablehlo.custom_call @tt.mark_argument(%2443) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.in_proj_b.weight"}} : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %2445 = stablehlo.reshape %2444 : (tensor<1x32x2048xbf16>) -> tensor<32x2048xbf16>
    %2446 = stablehlo.transpose %2445, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,32]{0,1}"} : (tensor<32x2048xbf16>) -> tensor<2048x32xbf16>
    %2447 = stablehlo.dot_general %2336, %2446, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x32xbf16>) -> tensor<494x32xbf16>
    %2448 = stablehlo.reshape %2447 : (tensor<494x32xbf16>) -> tensor<1x494x32xbf16>
    %2449 = stablehlo.logistic %2448 : tensor<1x494x32xbf16>
    %2450 = stablehlo.transpose %2449, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,32,494]{1,2,0}"} : (tensor<1x494x32xbf16>) -> tensor<1x32x494xbf16>
    %2451 = stablehlo.convert %2450 {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,494]{1,2,0}"} : (tensor<1x32x494xbf16>) -> tensor<1x32x494xf32>
    %2452 = stablehlo.pad %2451, %cst_240, low = [0, 0, 0], high = [0, 0, 18], interior = [0, 0, 0] : (tensor<1x32x494xf32>, tensor<f32>) -> tensor<1x32x512xf32>
    %2453 = stablehlo.broadcast_in_dim %2452, dims = [0, 1, 2] : (tensor<1x32x512xf32>) -> tensor<1x32x512x128xf32>
    %2454 = stablehlo.multiply %2432, %2453 : tensor<1x32x512x128xf32>
    %2455 = stablehlo.reshape %2454 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %2456 = stablehlo.transpose %2433, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[1,32,8,128,64]{3,4,2,1,0}"} : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x128x64xf32>
    %2457 = stablehlo.dot_general %2455, %2456, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x128xf32>, tensor<1x32x8x128x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2458 = stablehlo.broadcast_in_dim %2403, dims = [0, 1, 2, 3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2459 = stablehlo.broadcast_in_dim %2403, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2460 = stablehlo.subtract %2458, %2459 : tensor<1x32x8x64x64xf32>
    %2461 = stablehlo.select %645, %2460, %cst_98 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2462 = stablehlo.exponential %2461 : tensor<1x32x8x64x64xf32>
    %2463 = stablehlo.select %645, %2462, %cst_98 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2464 = stablehlo.multiply %2457, %2463 : tensor<1x32x8x64x64xf32>
    %2465 = stablehlo.select %627, %cst_98, %2464 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2466 = stablehlo.negate %2465 : tensor<1x32x8x64x64xf32>
    %2467 = stablehlo.slice %2466 [0:1, 0:32, 0:8, 1:2, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2468 = stablehlo.reshape %2467 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2469 = stablehlo.slice %2468 [0:1, 0:32, 0:8, 0:1] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x1xf32>
    %2470 = stablehlo.reshape %2469 : (tensor<1x32x8x1xf32>) -> tensor<1x32x8x1x1xf32>
    %2471 = stablehlo.slice %2466 [0:1, 0:32, 0:8, 0:1, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2472 = stablehlo.slice %2471 [0:1, 0:32, 0:8, 0:1, 0:1] : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x1x1xf32>
    %2473 = stablehlo.multiply %2470, %2472 : tensor<1x32x8x1x1xf32>
    %2474 = stablehlo.reduce(%2473 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x1x1xf32>, tensor<f32>) -> tensor<1x32x8x1xf32>
    %2475 = stablehlo.add %2469, %2474 : tensor<1x32x8x1xf32>
    %2476 = "stablehlo.gather"(%2475, %670) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x1xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2477 = stablehlo.select %619, %cst_224, %2476 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2478 = stablehlo.select %616, %2477, %2468 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2479 = stablehlo.broadcast_in_dim %2478, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2480 = stablehlo.select %612, %2479, %2466 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2481 = stablehlo.slice %2480 [0:1, 0:32, 0:8, 2:3, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2482 = stablehlo.reshape %2481 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2483 = stablehlo.slice %2482 [0:1, 0:32, 0:8, 0:2] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x2xf32>
    %2484 = stablehlo.broadcast_in_dim %2483, dims = [0, 1, 2, 3] : (tensor<1x32x8x2xf32>) -> tensor<1x32x8x2x2xf32>
    %2485 = stablehlo.slice %2480 [0:1, 0:32, 0:8, 0:2, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x2x64xf32>
    %2486 = stablehlo.slice %2485 [0:1, 0:32, 0:8, 0:2, 0:2] : (tensor<1x32x8x2x64xf32>) -> tensor<1x32x8x2x2xf32>
    %2487 = stablehlo.multiply %2484, %2486 : tensor<1x32x8x2x2xf32>
    %2488 = stablehlo.reduce(%2487 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x2x2xf32>, tensor<f32>) -> tensor<1x32x8x2xf32>
    %2489 = stablehlo.add %2483, %2488 : tensor<1x32x8x2xf32>
    %2490 = "stablehlo.gather"(%2489, %689) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x2xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2491 = stablehlo.select %611, %cst_224, %2490 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2492 = stablehlo.select %608, %2491, %2482 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2493 = stablehlo.broadcast_in_dim %2492, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2494 = stablehlo.select %604, %2493, %2480 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2495 = stablehlo.slice %2494 [0:1, 0:32, 0:8, 3:4, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2496 = stablehlo.reshape %2495 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2497 = stablehlo.slice %2496 [0:1, 0:32, 0:8, 0:3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x3xf32>
    %2498 = stablehlo.broadcast_in_dim %2497, dims = [0, 1, 2, 3] : (tensor<1x32x8x3xf32>) -> tensor<1x32x8x3x3xf32>
    %2499 = stablehlo.slice %2494 [0:1, 0:32, 0:8, 0:3, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x3x64xf32>
    %2500 = stablehlo.slice %2499 [0:1, 0:32, 0:8, 0:3, 0:3] : (tensor<1x32x8x3x64xf32>) -> tensor<1x32x8x3x3xf32>
    %2501 = stablehlo.multiply %2498, %2500 : tensor<1x32x8x3x3xf32>
    %2502 = stablehlo.reduce(%2501 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x3x3xf32>, tensor<f32>) -> tensor<1x32x8x3xf32>
    %2503 = stablehlo.add %2497, %2502 : tensor<1x32x8x3xf32>
    %2504 = "stablehlo.gather"(%2503, %708) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x3xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2505 = stablehlo.select %603, %cst_224, %2504 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2506 = stablehlo.select %600, %2505, %2496 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2507 = stablehlo.broadcast_in_dim %2506, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2508 = stablehlo.select %596, %2507, %2494 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2509 = stablehlo.slice %2508 [0:1, 0:32, 0:8, 4:5, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2510 = stablehlo.reshape %2509 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2511 = stablehlo.slice %2510 [0:1, 0:32, 0:8, 0:4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x4xf32>
    %2512 = stablehlo.broadcast_in_dim %2511, dims = [0, 1, 2, 3] : (tensor<1x32x8x4xf32>) -> tensor<1x32x8x4x4xf32>
    %2513 = stablehlo.slice %2508 [0:1, 0:32, 0:8, 0:4, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x4x64xf32>
    %2514 = stablehlo.slice %2513 [0:1, 0:32, 0:8, 0:4, 0:4] : (tensor<1x32x8x4x64xf32>) -> tensor<1x32x8x4x4xf32>
    %2515 = stablehlo.multiply %2512, %2514 : tensor<1x32x8x4x4xf32>
    %2516 = stablehlo.reduce(%2515 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x4x4xf32>, tensor<f32>) -> tensor<1x32x8x4xf32>
    %2517 = stablehlo.add %2511, %2516 : tensor<1x32x8x4xf32>
    %2518 = "stablehlo.gather"(%2517, %727) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x4xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2519 = stablehlo.select %595, %cst_224, %2518 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2520 = stablehlo.select %592, %2519, %2510 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2521 = stablehlo.broadcast_in_dim %2520, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2522 = stablehlo.select %588, %2521, %2508 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2523 = stablehlo.slice %2522 [0:1, 0:32, 0:8, 5:6, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2524 = stablehlo.reshape %2523 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2525 = stablehlo.slice %2524 [0:1, 0:32, 0:8, 0:5] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x5xf32>
    %2526 = stablehlo.broadcast_in_dim %2525, dims = [0, 1, 2, 3] : (tensor<1x32x8x5xf32>) -> tensor<1x32x8x5x5xf32>
    %2527 = stablehlo.slice %2522 [0:1, 0:32, 0:8, 0:5, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x5x64xf32>
    %2528 = stablehlo.slice %2527 [0:1, 0:32, 0:8, 0:5, 0:5] : (tensor<1x32x8x5x64xf32>) -> tensor<1x32x8x5x5xf32>
    %2529 = stablehlo.multiply %2526, %2528 : tensor<1x32x8x5x5xf32>
    %2530 = stablehlo.reduce(%2529 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x5x5xf32>, tensor<f32>) -> tensor<1x32x8x5xf32>
    %2531 = stablehlo.add %2525, %2530 : tensor<1x32x8x5xf32>
    %2532 = "stablehlo.gather"(%2531, %746) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x5xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2533 = stablehlo.select %587, %cst_224, %2532 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2534 = stablehlo.select %584, %2533, %2524 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2535 = stablehlo.broadcast_in_dim %2534, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2536 = stablehlo.select %580, %2535, %2522 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2537 = stablehlo.slice %2536 [0:1, 0:32, 0:8, 6:7, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2538 = stablehlo.reshape %2537 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2539 = stablehlo.slice %2538 [0:1, 0:32, 0:8, 0:6] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x6xf32>
    %2540 = stablehlo.broadcast_in_dim %2539, dims = [0, 1, 2, 3] : (tensor<1x32x8x6xf32>) -> tensor<1x32x8x6x6xf32>
    %2541 = stablehlo.slice %2536 [0:1, 0:32, 0:8, 0:6, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x6x64xf32>
    %2542 = stablehlo.slice %2541 [0:1, 0:32, 0:8, 0:6, 0:6] : (tensor<1x32x8x6x64xf32>) -> tensor<1x32x8x6x6xf32>
    %2543 = stablehlo.multiply %2540, %2542 : tensor<1x32x8x6x6xf32>
    %2544 = stablehlo.reduce(%2543 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x6x6xf32>, tensor<f32>) -> tensor<1x32x8x6xf32>
    %2545 = stablehlo.add %2539, %2544 : tensor<1x32x8x6xf32>
    %2546 = "stablehlo.gather"(%2545, %765) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x6xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2547 = stablehlo.select %579, %cst_224, %2546 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2548 = stablehlo.select %576, %2547, %2538 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2549 = stablehlo.broadcast_in_dim %2548, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2550 = stablehlo.select %572, %2549, %2536 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2551 = stablehlo.slice %2550 [0:1, 0:32, 0:8, 7:8, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2552 = stablehlo.reshape %2551 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2553 = stablehlo.slice %2552 [0:1, 0:32, 0:8, 0:7] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x7xf32>
    %2554 = stablehlo.broadcast_in_dim %2553, dims = [0, 1, 2, 3] : (tensor<1x32x8x7xf32>) -> tensor<1x32x8x7x7xf32>
    %2555 = stablehlo.slice %2550 [0:1, 0:32, 0:8, 0:7, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x7x64xf32>
    %2556 = stablehlo.slice %2555 [0:1, 0:32, 0:8, 0:7, 0:7] : (tensor<1x32x8x7x64xf32>) -> tensor<1x32x8x7x7xf32>
    %2557 = stablehlo.multiply %2554, %2556 : tensor<1x32x8x7x7xf32>
    %2558 = stablehlo.reduce(%2557 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x7x7xf32>, tensor<f32>) -> tensor<1x32x8x7xf32>
    %2559 = stablehlo.add %2553, %2558 : tensor<1x32x8x7xf32>
    %2560 = "stablehlo.gather"(%2559, %784) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x7xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2561 = stablehlo.select %571, %cst_224, %2560 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2562 = stablehlo.select %568, %2561, %2552 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2563 = stablehlo.broadcast_in_dim %2562, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2564 = stablehlo.select %564, %2563, %2550 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2565 = stablehlo.slice %2564 [0:1, 0:32, 0:8, 8:9, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2566 = stablehlo.reshape %2565 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2567 = stablehlo.slice %2566 [0:1, 0:32, 0:8, 0:8] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x8xf32>
    %2568 = stablehlo.broadcast_in_dim %2567, dims = [0, 1, 2, 3] : (tensor<1x32x8x8xf32>) -> tensor<1x32x8x8x8xf32>
    %2569 = stablehlo.slice %2564 [0:1, 0:32, 0:8, 0:8, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x8x64xf32>
    %2570 = stablehlo.slice %2569 [0:1, 0:32, 0:8, 0:8, 0:8] : (tensor<1x32x8x8x64xf32>) -> tensor<1x32x8x8x8xf32>
    %2571 = stablehlo.multiply %2568, %2570 : tensor<1x32x8x8x8xf32>
    %2572 = stablehlo.reduce(%2571 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x8x8xf32>, tensor<f32>) -> tensor<1x32x8x8xf32>
    %2573 = stablehlo.add %2567, %2572 : tensor<1x32x8x8xf32>
    %2574 = "stablehlo.gather"(%2573, %803) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x8xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2575 = stablehlo.select %563, %cst_224, %2574 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2576 = stablehlo.select %560, %2575, %2566 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2577 = stablehlo.broadcast_in_dim %2576, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2578 = stablehlo.select %556, %2577, %2564 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2579 = stablehlo.slice %2578 [0:1, 0:32, 0:8, 9:10, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2580 = stablehlo.reshape %2579 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2581 = stablehlo.slice %2580 [0:1, 0:32, 0:8, 0:9] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x9xf32>
    %2582 = stablehlo.broadcast_in_dim %2581, dims = [0, 1, 2, 3] : (tensor<1x32x8x9xf32>) -> tensor<1x32x8x9x9xf32>
    %2583 = stablehlo.slice %2578 [0:1, 0:32, 0:8, 0:9, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x9x64xf32>
    %2584 = stablehlo.slice %2583 [0:1, 0:32, 0:8, 0:9, 0:9] : (tensor<1x32x8x9x64xf32>) -> tensor<1x32x8x9x9xf32>
    %2585 = stablehlo.multiply %2582, %2584 : tensor<1x32x8x9x9xf32>
    %2586 = stablehlo.reduce(%2585 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x9x9xf32>, tensor<f32>) -> tensor<1x32x8x9xf32>
    %2587 = stablehlo.add %2581, %2586 : tensor<1x32x8x9xf32>
    %2588 = "stablehlo.gather"(%2587, %822) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x9xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2589 = stablehlo.select %555, %cst_224, %2588 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2590 = stablehlo.select %552, %2589, %2580 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2591 = stablehlo.broadcast_in_dim %2590, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2592 = stablehlo.select %548, %2591, %2578 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2593 = stablehlo.slice %2592 [0:1, 0:32, 0:8, 10:11, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2594 = stablehlo.reshape %2593 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2595 = stablehlo.slice %2594 [0:1, 0:32, 0:8, 0:10] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x10xf32>
    %2596 = stablehlo.broadcast_in_dim %2595, dims = [0, 1, 2, 3] : (tensor<1x32x8x10xf32>) -> tensor<1x32x8x10x10xf32>
    %2597 = stablehlo.slice %2592 [0:1, 0:32, 0:8, 0:10, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x10x64xf32>
    %2598 = stablehlo.slice %2597 [0:1, 0:32, 0:8, 0:10, 0:10] : (tensor<1x32x8x10x64xf32>) -> tensor<1x32x8x10x10xf32>
    %2599 = stablehlo.multiply %2596, %2598 : tensor<1x32x8x10x10xf32>
    %2600 = stablehlo.reduce(%2599 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x10x10xf32>, tensor<f32>) -> tensor<1x32x8x10xf32>
    %2601 = stablehlo.add %2595, %2600 : tensor<1x32x8x10xf32>
    %2602 = "stablehlo.gather"(%2601, %841) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x10xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2603 = stablehlo.select %547, %cst_224, %2602 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2604 = stablehlo.select %544, %2603, %2594 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2605 = stablehlo.broadcast_in_dim %2604, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2606 = stablehlo.select %540, %2605, %2592 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2607 = stablehlo.slice %2606 [0:1, 0:32, 0:8, 11:12, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2608 = stablehlo.reshape %2607 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2609 = stablehlo.slice %2608 [0:1, 0:32, 0:8, 0:11] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x11xf32>
    %2610 = stablehlo.broadcast_in_dim %2609, dims = [0, 1, 2, 3] : (tensor<1x32x8x11xf32>) -> tensor<1x32x8x11x11xf32>
    %2611 = stablehlo.slice %2606 [0:1, 0:32, 0:8, 0:11, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x11x64xf32>
    %2612 = stablehlo.slice %2611 [0:1, 0:32, 0:8, 0:11, 0:11] : (tensor<1x32x8x11x64xf32>) -> tensor<1x32x8x11x11xf32>
    %2613 = stablehlo.multiply %2610, %2612 : tensor<1x32x8x11x11xf32>
    %2614 = stablehlo.reduce(%2613 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x11x11xf32>, tensor<f32>) -> tensor<1x32x8x11xf32>
    %2615 = stablehlo.add %2609, %2614 : tensor<1x32x8x11xf32>
    %2616 = "stablehlo.gather"(%2615, %860) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x11xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2617 = stablehlo.select %539, %cst_224, %2616 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2618 = stablehlo.select %536, %2617, %2608 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2619 = stablehlo.broadcast_in_dim %2618, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2620 = stablehlo.select %532, %2619, %2606 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2621 = stablehlo.slice %2620 [0:1, 0:32, 0:8, 12:13, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2622 = stablehlo.reshape %2621 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2623 = stablehlo.slice %2622 [0:1, 0:32, 0:8, 0:12] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x12xf32>
    %2624 = stablehlo.broadcast_in_dim %2623, dims = [0, 1, 2, 3] : (tensor<1x32x8x12xf32>) -> tensor<1x32x8x12x12xf32>
    %2625 = stablehlo.slice %2620 [0:1, 0:32, 0:8, 0:12, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x12x64xf32>
    %2626 = stablehlo.slice %2625 [0:1, 0:32, 0:8, 0:12, 0:12] : (tensor<1x32x8x12x64xf32>) -> tensor<1x32x8x12x12xf32>
    %2627 = stablehlo.multiply %2624, %2626 : tensor<1x32x8x12x12xf32>
    %2628 = stablehlo.reduce(%2627 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x12x12xf32>, tensor<f32>) -> tensor<1x32x8x12xf32>
    %2629 = stablehlo.add %2623, %2628 : tensor<1x32x8x12xf32>
    %2630 = "stablehlo.gather"(%2629, %879) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x12xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2631 = stablehlo.select %531, %cst_224, %2630 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2632 = stablehlo.select %528, %2631, %2622 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2633 = stablehlo.broadcast_in_dim %2632, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2634 = stablehlo.select %524, %2633, %2620 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2635 = stablehlo.slice %2634 [0:1, 0:32, 0:8, 13:14, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2636 = stablehlo.reshape %2635 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2637 = stablehlo.slice %2636 [0:1, 0:32, 0:8, 0:13] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x13xf32>
    %2638 = stablehlo.broadcast_in_dim %2637, dims = [0, 1, 2, 3] : (tensor<1x32x8x13xf32>) -> tensor<1x32x8x13x13xf32>
    %2639 = stablehlo.slice %2634 [0:1, 0:32, 0:8, 0:13, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x13x64xf32>
    %2640 = stablehlo.slice %2639 [0:1, 0:32, 0:8, 0:13, 0:13] : (tensor<1x32x8x13x64xf32>) -> tensor<1x32x8x13x13xf32>
    %2641 = stablehlo.multiply %2638, %2640 : tensor<1x32x8x13x13xf32>
    %2642 = stablehlo.reduce(%2641 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x13x13xf32>, tensor<f32>) -> tensor<1x32x8x13xf32>
    %2643 = stablehlo.add %2637, %2642 : tensor<1x32x8x13xf32>
    %2644 = "stablehlo.gather"(%2643, %898) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x13xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2645 = stablehlo.select %523, %cst_224, %2644 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2646 = stablehlo.select %520, %2645, %2636 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2647 = stablehlo.broadcast_in_dim %2646, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2648 = stablehlo.select %516, %2647, %2634 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2649 = stablehlo.slice %2648 [0:1, 0:32, 0:8, 14:15, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2650 = stablehlo.reshape %2649 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2651 = stablehlo.slice %2650 [0:1, 0:32, 0:8, 0:14] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x14xf32>
    %2652 = stablehlo.broadcast_in_dim %2651, dims = [0, 1, 2, 3] : (tensor<1x32x8x14xf32>) -> tensor<1x32x8x14x14xf32>
    %2653 = stablehlo.slice %2648 [0:1, 0:32, 0:8, 0:14, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x14x64xf32>
    %2654 = stablehlo.slice %2653 [0:1, 0:32, 0:8, 0:14, 0:14] : (tensor<1x32x8x14x64xf32>) -> tensor<1x32x8x14x14xf32>
    %2655 = stablehlo.multiply %2652, %2654 : tensor<1x32x8x14x14xf32>
    %2656 = stablehlo.reduce(%2655 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x14x14xf32>, tensor<f32>) -> tensor<1x32x8x14xf32>
    %2657 = stablehlo.add %2651, %2656 : tensor<1x32x8x14xf32>
    %2658 = "stablehlo.gather"(%2657, %917) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x14xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2659 = stablehlo.select %515, %cst_224, %2658 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2660 = stablehlo.select %512, %2659, %2650 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2661 = stablehlo.broadcast_in_dim %2660, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2662 = stablehlo.select %508, %2661, %2648 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2663 = stablehlo.slice %2662 [0:1, 0:32, 0:8, 15:16, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2664 = stablehlo.reshape %2663 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2665 = stablehlo.slice %2664 [0:1, 0:32, 0:8, 0:15] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x15xf32>
    %2666 = stablehlo.broadcast_in_dim %2665, dims = [0, 1, 2, 3] : (tensor<1x32x8x15xf32>) -> tensor<1x32x8x15x15xf32>
    %2667 = stablehlo.slice %2662 [0:1, 0:32, 0:8, 0:15, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x15x64xf32>
    %2668 = stablehlo.slice %2667 [0:1, 0:32, 0:8, 0:15, 0:15] : (tensor<1x32x8x15x64xf32>) -> tensor<1x32x8x15x15xf32>
    %2669 = stablehlo.multiply %2666, %2668 : tensor<1x32x8x15x15xf32>
    %2670 = stablehlo.reduce(%2669 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x15x15xf32>, tensor<f32>) -> tensor<1x32x8x15xf32>
    %2671 = stablehlo.add %2665, %2670 : tensor<1x32x8x15xf32>
    %2672 = "stablehlo.gather"(%2671, %936) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x15xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2673 = stablehlo.select %507, %cst_224, %2672 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2674 = stablehlo.select %504, %2673, %2664 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2675 = stablehlo.broadcast_in_dim %2674, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2676 = stablehlo.select %500, %2675, %2662 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2677 = stablehlo.slice %2676 [0:1, 0:32, 0:8, 16:17, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2678 = stablehlo.reshape %2677 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2679 = stablehlo.slice %2678 [0:1, 0:32, 0:8, 0:16] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x16xf32>
    %2680 = stablehlo.broadcast_in_dim %2679, dims = [0, 1, 2, 3] : (tensor<1x32x8x16xf32>) -> tensor<1x32x8x16x16xf32>
    %2681 = stablehlo.slice %2676 [0:1, 0:32, 0:8, 0:16, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x16x64xf32>
    %2682 = stablehlo.slice %2681 [0:1, 0:32, 0:8, 0:16, 0:16] : (tensor<1x32x8x16x64xf32>) -> tensor<1x32x8x16x16xf32>
    %2683 = stablehlo.multiply %2680, %2682 : tensor<1x32x8x16x16xf32>
    %2684 = stablehlo.reduce(%2683 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x16x16xf32>, tensor<f32>) -> tensor<1x32x8x16xf32>
    %2685 = stablehlo.add %2679, %2684 : tensor<1x32x8x16xf32>
    %2686 = "stablehlo.gather"(%2685, %955) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x16xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2687 = stablehlo.select %499, %cst_224, %2686 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2688 = stablehlo.select %496, %2687, %2678 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2689 = stablehlo.broadcast_in_dim %2688, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2690 = stablehlo.select %492, %2689, %2676 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2691 = stablehlo.slice %2690 [0:1, 0:32, 0:8, 17:18, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2692 = stablehlo.reshape %2691 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2693 = stablehlo.slice %2692 [0:1, 0:32, 0:8, 0:17] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x17xf32>
    %2694 = stablehlo.broadcast_in_dim %2693, dims = [0, 1, 2, 3] : (tensor<1x32x8x17xf32>) -> tensor<1x32x8x17x17xf32>
    %2695 = stablehlo.slice %2690 [0:1, 0:32, 0:8, 0:17, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x17x64xf32>
    %2696 = stablehlo.slice %2695 [0:1, 0:32, 0:8, 0:17, 0:17] : (tensor<1x32x8x17x64xf32>) -> tensor<1x32x8x17x17xf32>
    %2697 = stablehlo.multiply %2694, %2696 : tensor<1x32x8x17x17xf32>
    %2698 = stablehlo.reduce(%2697 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x17x17xf32>, tensor<f32>) -> tensor<1x32x8x17xf32>
    %2699 = stablehlo.add %2693, %2698 : tensor<1x32x8x17xf32>
    %2700 = "stablehlo.gather"(%2699, %974) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x17xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2701 = stablehlo.select %491, %cst_224, %2700 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2702 = stablehlo.select %488, %2701, %2692 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2703 = stablehlo.broadcast_in_dim %2702, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2704 = stablehlo.select %484, %2703, %2690 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2705 = stablehlo.slice %2704 [0:1, 0:32, 0:8, 18:19, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2706 = stablehlo.reshape %2705 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2707 = stablehlo.slice %2706 [0:1, 0:32, 0:8, 0:18] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x18xf32>
    %2708 = stablehlo.broadcast_in_dim %2707, dims = [0, 1, 2, 3] : (tensor<1x32x8x18xf32>) -> tensor<1x32x8x18x18xf32>
    %2709 = stablehlo.slice %2704 [0:1, 0:32, 0:8, 0:18, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x18x64xf32>
    %2710 = stablehlo.slice %2709 [0:1, 0:32, 0:8, 0:18, 0:18] : (tensor<1x32x8x18x64xf32>) -> tensor<1x32x8x18x18xf32>
    %2711 = stablehlo.multiply %2708, %2710 : tensor<1x32x8x18x18xf32>
    %2712 = stablehlo.reduce(%2711 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x18x18xf32>, tensor<f32>) -> tensor<1x32x8x18xf32>
    %2713 = stablehlo.add %2707, %2712 : tensor<1x32x8x18xf32>
    %2714 = "stablehlo.gather"(%2713, %993) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x18xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2715 = stablehlo.select %483, %cst_224, %2714 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2716 = stablehlo.select %480, %2715, %2706 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2717 = stablehlo.broadcast_in_dim %2716, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2718 = stablehlo.select %476, %2717, %2704 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2719 = stablehlo.slice %2718 [0:1, 0:32, 0:8, 19:20, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2720 = stablehlo.reshape %2719 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2721 = stablehlo.slice %2720 [0:1, 0:32, 0:8, 0:19] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x19xf32>
    %2722 = stablehlo.broadcast_in_dim %2721, dims = [0, 1, 2, 3] : (tensor<1x32x8x19xf32>) -> tensor<1x32x8x19x19xf32>
    %2723 = stablehlo.slice %2718 [0:1, 0:32, 0:8, 0:19, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x19x64xf32>
    %2724 = stablehlo.slice %2723 [0:1, 0:32, 0:8, 0:19, 0:19] : (tensor<1x32x8x19x64xf32>) -> tensor<1x32x8x19x19xf32>
    %2725 = stablehlo.multiply %2722, %2724 : tensor<1x32x8x19x19xf32>
    %2726 = stablehlo.reduce(%2725 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x19x19xf32>, tensor<f32>) -> tensor<1x32x8x19xf32>
    %2727 = stablehlo.add %2721, %2726 : tensor<1x32x8x19xf32>
    %2728 = "stablehlo.gather"(%2727, %1012) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x19xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2729 = stablehlo.select %475, %cst_224, %2728 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2730 = stablehlo.select %472, %2729, %2720 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2731 = stablehlo.broadcast_in_dim %2730, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2732 = stablehlo.select %468, %2731, %2718 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2733 = stablehlo.slice %2732 [0:1, 0:32, 0:8, 20:21, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2734 = stablehlo.reshape %2733 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2735 = stablehlo.slice %2734 [0:1, 0:32, 0:8, 0:20] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x20xf32>
    %2736 = stablehlo.broadcast_in_dim %2735, dims = [0, 1, 2, 3] : (tensor<1x32x8x20xf32>) -> tensor<1x32x8x20x20xf32>
    %2737 = stablehlo.slice %2732 [0:1, 0:32, 0:8, 0:20, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x20x64xf32>
    %2738 = stablehlo.slice %2737 [0:1, 0:32, 0:8, 0:20, 0:20] : (tensor<1x32x8x20x64xf32>) -> tensor<1x32x8x20x20xf32>
    %2739 = stablehlo.multiply %2736, %2738 : tensor<1x32x8x20x20xf32>
    %2740 = stablehlo.reduce(%2739 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x20x20xf32>, tensor<f32>) -> tensor<1x32x8x20xf32>
    %2741 = stablehlo.add %2735, %2740 : tensor<1x32x8x20xf32>
    %2742 = "stablehlo.gather"(%2741, %1031) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x20xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2743 = stablehlo.select %467, %cst_224, %2742 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2744 = stablehlo.select %464, %2743, %2734 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2745 = stablehlo.broadcast_in_dim %2744, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2746 = stablehlo.select %460, %2745, %2732 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2747 = stablehlo.slice %2746 [0:1, 0:32, 0:8, 21:22, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2748 = stablehlo.reshape %2747 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2749 = stablehlo.slice %2748 [0:1, 0:32, 0:8, 0:21] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x21xf32>
    %2750 = stablehlo.broadcast_in_dim %2749, dims = [0, 1, 2, 3] : (tensor<1x32x8x21xf32>) -> tensor<1x32x8x21x21xf32>
    %2751 = stablehlo.slice %2746 [0:1, 0:32, 0:8, 0:21, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x21x64xf32>
    %2752 = stablehlo.slice %2751 [0:1, 0:32, 0:8, 0:21, 0:21] : (tensor<1x32x8x21x64xf32>) -> tensor<1x32x8x21x21xf32>
    %2753 = stablehlo.multiply %2750, %2752 : tensor<1x32x8x21x21xf32>
    %2754 = stablehlo.reduce(%2753 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x21x21xf32>, tensor<f32>) -> tensor<1x32x8x21xf32>
    %2755 = stablehlo.add %2749, %2754 : tensor<1x32x8x21xf32>
    %2756 = "stablehlo.gather"(%2755, %1050) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x21xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2757 = stablehlo.select %459, %cst_224, %2756 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2758 = stablehlo.select %456, %2757, %2748 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2759 = stablehlo.broadcast_in_dim %2758, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2760 = stablehlo.select %452, %2759, %2746 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2761 = stablehlo.slice %2760 [0:1, 0:32, 0:8, 22:23, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2762 = stablehlo.reshape %2761 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2763 = stablehlo.slice %2762 [0:1, 0:32, 0:8, 0:22] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x22xf32>
    %2764 = stablehlo.broadcast_in_dim %2763, dims = [0, 1, 2, 3] : (tensor<1x32x8x22xf32>) -> tensor<1x32x8x22x22xf32>
    %2765 = stablehlo.slice %2760 [0:1, 0:32, 0:8, 0:22, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x22x64xf32>
    %2766 = stablehlo.slice %2765 [0:1, 0:32, 0:8, 0:22, 0:22] : (tensor<1x32x8x22x64xf32>) -> tensor<1x32x8x22x22xf32>
    %2767 = stablehlo.multiply %2764, %2766 : tensor<1x32x8x22x22xf32>
    %2768 = stablehlo.reduce(%2767 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x22x22xf32>, tensor<f32>) -> tensor<1x32x8x22xf32>
    %2769 = stablehlo.add %2763, %2768 : tensor<1x32x8x22xf32>
    %2770 = "stablehlo.gather"(%2769, %1069) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x22xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2771 = stablehlo.select %451, %cst_224, %2770 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2772 = stablehlo.select %448, %2771, %2762 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2773 = stablehlo.broadcast_in_dim %2772, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2774 = stablehlo.select %444, %2773, %2760 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2775 = stablehlo.slice %2774 [0:1, 0:32, 0:8, 23:24, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2776 = stablehlo.reshape %2775 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2777 = stablehlo.slice %2776 [0:1, 0:32, 0:8, 0:23] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x23xf32>
    %2778 = stablehlo.broadcast_in_dim %2777, dims = [0, 1, 2, 3] : (tensor<1x32x8x23xf32>) -> tensor<1x32x8x23x23xf32>
    %2779 = stablehlo.slice %2774 [0:1, 0:32, 0:8, 0:23, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x23x64xf32>
    %2780 = stablehlo.slice %2779 [0:1, 0:32, 0:8, 0:23, 0:23] : (tensor<1x32x8x23x64xf32>) -> tensor<1x32x8x23x23xf32>
    %2781 = stablehlo.multiply %2778, %2780 : tensor<1x32x8x23x23xf32>
    %2782 = stablehlo.reduce(%2781 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x23x23xf32>, tensor<f32>) -> tensor<1x32x8x23xf32>
    %2783 = stablehlo.add %2777, %2782 : tensor<1x32x8x23xf32>
    %2784 = "stablehlo.gather"(%2783, %1088) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x23xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2785 = stablehlo.select %443, %cst_224, %2784 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2786 = stablehlo.select %440, %2785, %2776 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2787 = stablehlo.broadcast_in_dim %2786, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2788 = stablehlo.select %436, %2787, %2774 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2789 = stablehlo.slice %2788 [0:1, 0:32, 0:8, 24:25, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2790 = stablehlo.reshape %2789 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2791 = stablehlo.slice %2790 [0:1, 0:32, 0:8, 0:24] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x24xf32>
    %2792 = stablehlo.broadcast_in_dim %2791, dims = [0, 1, 2, 3] : (tensor<1x32x8x24xf32>) -> tensor<1x32x8x24x24xf32>
    %2793 = stablehlo.slice %2788 [0:1, 0:32, 0:8, 0:24, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x24x64xf32>
    %2794 = stablehlo.slice %2793 [0:1, 0:32, 0:8, 0:24, 0:24] : (tensor<1x32x8x24x64xf32>) -> tensor<1x32x8x24x24xf32>
    %2795 = stablehlo.multiply %2792, %2794 : tensor<1x32x8x24x24xf32>
    %2796 = stablehlo.reduce(%2795 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x24x24xf32>, tensor<f32>) -> tensor<1x32x8x24xf32>
    %2797 = stablehlo.add %2791, %2796 : tensor<1x32x8x24xf32>
    %2798 = "stablehlo.gather"(%2797, %1107) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x24xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2799 = stablehlo.select %435, %cst_224, %2798 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2800 = stablehlo.select %432, %2799, %2790 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2801 = stablehlo.broadcast_in_dim %2800, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2802 = stablehlo.select %428, %2801, %2788 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2803 = stablehlo.slice %2802 [0:1, 0:32, 0:8, 25:26, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2804 = stablehlo.reshape %2803 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2805 = stablehlo.slice %2804 [0:1, 0:32, 0:8, 0:25] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x25xf32>
    %2806 = stablehlo.broadcast_in_dim %2805, dims = [0, 1, 2, 3] : (tensor<1x32x8x25xf32>) -> tensor<1x32x8x25x25xf32>
    %2807 = stablehlo.slice %2802 [0:1, 0:32, 0:8, 0:25, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x25x64xf32>
    %2808 = stablehlo.slice %2807 [0:1, 0:32, 0:8, 0:25, 0:25] : (tensor<1x32x8x25x64xf32>) -> tensor<1x32x8x25x25xf32>
    %2809 = stablehlo.multiply %2806, %2808 : tensor<1x32x8x25x25xf32>
    %2810 = stablehlo.reduce(%2809 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x25x25xf32>, tensor<f32>) -> tensor<1x32x8x25xf32>
    %2811 = stablehlo.add %2805, %2810 : tensor<1x32x8x25xf32>
    %2812 = "stablehlo.gather"(%2811, %1126) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x25xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2813 = stablehlo.select %427, %cst_224, %2812 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2814 = stablehlo.select %424, %2813, %2804 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2815 = stablehlo.broadcast_in_dim %2814, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2816 = stablehlo.select %420, %2815, %2802 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2817 = stablehlo.slice %2816 [0:1, 0:32, 0:8, 26:27, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2818 = stablehlo.reshape %2817 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2819 = stablehlo.slice %2818 [0:1, 0:32, 0:8, 0:26] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x26xf32>
    %2820 = stablehlo.broadcast_in_dim %2819, dims = [0, 1, 2, 3] : (tensor<1x32x8x26xf32>) -> tensor<1x32x8x26x26xf32>
    %2821 = stablehlo.slice %2816 [0:1, 0:32, 0:8, 0:26, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x26x64xf32>
    %2822 = stablehlo.slice %2821 [0:1, 0:32, 0:8, 0:26, 0:26] : (tensor<1x32x8x26x64xf32>) -> tensor<1x32x8x26x26xf32>
    %2823 = stablehlo.multiply %2820, %2822 : tensor<1x32x8x26x26xf32>
    %2824 = stablehlo.reduce(%2823 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x26x26xf32>, tensor<f32>) -> tensor<1x32x8x26xf32>
    %2825 = stablehlo.add %2819, %2824 : tensor<1x32x8x26xf32>
    %2826 = "stablehlo.gather"(%2825, %1145) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x26xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2827 = stablehlo.select %419, %cst_224, %2826 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2828 = stablehlo.select %416, %2827, %2818 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2829 = stablehlo.broadcast_in_dim %2828, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2830 = stablehlo.select %412, %2829, %2816 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2831 = stablehlo.slice %2830 [0:1, 0:32, 0:8, 27:28, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2832 = stablehlo.reshape %2831 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2833 = stablehlo.slice %2832 [0:1, 0:32, 0:8, 0:27] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x27xf32>
    %2834 = stablehlo.broadcast_in_dim %2833, dims = [0, 1, 2, 3] : (tensor<1x32x8x27xf32>) -> tensor<1x32x8x27x27xf32>
    %2835 = stablehlo.slice %2830 [0:1, 0:32, 0:8, 0:27, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x27x64xf32>
    %2836 = stablehlo.slice %2835 [0:1, 0:32, 0:8, 0:27, 0:27] : (tensor<1x32x8x27x64xf32>) -> tensor<1x32x8x27x27xf32>
    %2837 = stablehlo.multiply %2834, %2836 : tensor<1x32x8x27x27xf32>
    %2838 = stablehlo.reduce(%2837 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x27x27xf32>, tensor<f32>) -> tensor<1x32x8x27xf32>
    %2839 = stablehlo.add %2833, %2838 : tensor<1x32x8x27xf32>
    %2840 = "stablehlo.gather"(%2839, %1164) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x27xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2841 = stablehlo.select %411, %cst_224, %2840 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2842 = stablehlo.select %408, %2841, %2832 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2843 = stablehlo.broadcast_in_dim %2842, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2844 = stablehlo.select %404, %2843, %2830 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2845 = stablehlo.slice %2844 [0:1, 0:32, 0:8, 28:29, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2846 = stablehlo.reshape %2845 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2847 = stablehlo.slice %2846 [0:1, 0:32, 0:8, 0:28] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x28xf32>
    %2848 = stablehlo.broadcast_in_dim %2847, dims = [0, 1, 2, 3] : (tensor<1x32x8x28xf32>) -> tensor<1x32x8x28x28xf32>
    %2849 = stablehlo.slice %2844 [0:1, 0:32, 0:8, 0:28, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x28x64xf32>
    %2850 = stablehlo.slice %2849 [0:1, 0:32, 0:8, 0:28, 0:28] : (tensor<1x32x8x28x64xf32>) -> tensor<1x32x8x28x28xf32>
    %2851 = stablehlo.multiply %2848, %2850 : tensor<1x32x8x28x28xf32>
    %2852 = stablehlo.reduce(%2851 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x28x28xf32>, tensor<f32>) -> tensor<1x32x8x28xf32>
    %2853 = stablehlo.add %2847, %2852 : tensor<1x32x8x28xf32>
    %2854 = "stablehlo.gather"(%2853, %1183) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x28xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2855 = stablehlo.select %403, %cst_224, %2854 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2856 = stablehlo.select %400, %2855, %2846 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2857 = stablehlo.broadcast_in_dim %2856, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2858 = stablehlo.select %396, %2857, %2844 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2859 = stablehlo.slice %2858 [0:1, 0:32, 0:8, 29:30, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2860 = stablehlo.reshape %2859 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2861 = stablehlo.slice %2860 [0:1, 0:32, 0:8, 0:29] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x29xf32>
    %2862 = stablehlo.broadcast_in_dim %2861, dims = [0, 1, 2, 3] : (tensor<1x32x8x29xf32>) -> tensor<1x32x8x29x29xf32>
    %2863 = stablehlo.slice %2858 [0:1, 0:32, 0:8, 0:29, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x29x64xf32>
    %2864 = stablehlo.slice %2863 [0:1, 0:32, 0:8, 0:29, 0:29] : (tensor<1x32x8x29x64xf32>) -> tensor<1x32x8x29x29xf32>
    %2865 = stablehlo.multiply %2862, %2864 : tensor<1x32x8x29x29xf32>
    %2866 = stablehlo.reduce(%2865 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x29x29xf32>, tensor<f32>) -> tensor<1x32x8x29xf32>
    %2867 = stablehlo.add %2861, %2866 : tensor<1x32x8x29xf32>
    %2868 = "stablehlo.gather"(%2867, %1202) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x29xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2869 = stablehlo.select %395, %cst_224, %2868 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2870 = stablehlo.select %392, %2869, %2860 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2871 = stablehlo.broadcast_in_dim %2870, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2872 = stablehlo.select %388, %2871, %2858 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2873 = stablehlo.slice %2872 [0:1, 0:32, 0:8, 30:31, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2874 = stablehlo.reshape %2873 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2875 = stablehlo.slice %2874 [0:1, 0:32, 0:8, 0:30] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x30xf32>
    %2876 = stablehlo.broadcast_in_dim %2875, dims = [0, 1, 2, 3] : (tensor<1x32x8x30xf32>) -> tensor<1x32x8x30x30xf32>
    %2877 = stablehlo.slice %2872 [0:1, 0:32, 0:8, 0:30, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x30x64xf32>
    %2878 = stablehlo.slice %2877 [0:1, 0:32, 0:8, 0:30, 0:30] : (tensor<1x32x8x30x64xf32>) -> tensor<1x32x8x30x30xf32>
    %2879 = stablehlo.multiply %2876, %2878 : tensor<1x32x8x30x30xf32>
    %2880 = stablehlo.reduce(%2879 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x30x30xf32>, tensor<f32>) -> tensor<1x32x8x30xf32>
    %2881 = stablehlo.add %2875, %2880 : tensor<1x32x8x30xf32>
    %2882 = "stablehlo.gather"(%2881, %1221) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x30xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2883 = stablehlo.select %387, %cst_224, %2882 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2884 = stablehlo.select %384, %2883, %2874 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2885 = stablehlo.broadcast_in_dim %2884, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2886 = stablehlo.select %380, %2885, %2872 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2887 = stablehlo.slice %2886 [0:1, 0:32, 0:8, 31:32, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2888 = stablehlo.reshape %2887 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2889 = stablehlo.slice %2888 [0:1, 0:32, 0:8, 0:31] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x31xf32>
    %2890 = stablehlo.broadcast_in_dim %2889, dims = [0, 1, 2, 3] : (tensor<1x32x8x31xf32>) -> tensor<1x32x8x31x31xf32>
    %2891 = stablehlo.slice %2886 [0:1, 0:32, 0:8, 0:31, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x31x64xf32>
    %2892 = stablehlo.slice %2891 [0:1, 0:32, 0:8, 0:31, 0:31] : (tensor<1x32x8x31x64xf32>) -> tensor<1x32x8x31x31xf32>
    %2893 = stablehlo.multiply %2890, %2892 : tensor<1x32x8x31x31xf32>
    %2894 = stablehlo.reduce(%2893 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x31x31xf32>, tensor<f32>) -> tensor<1x32x8x31xf32>
    %2895 = stablehlo.add %2889, %2894 : tensor<1x32x8x31xf32>
    %2896 = "stablehlo.gather"(%2895, %1240) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x31xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2897 = stablehlo.select %379, %cst_224, %2896 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2898 = stablehlo.select %376, %2897, %2888 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2899 = stablehlo.broadcast_in_dim %2898, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2900 = stablehlo.select %372, %2899, %2886 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2901 = stablehlo.slice %2900 [0:1, 0:32, 0:8, 32:33, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2902 = stablehlo.reshape %2901 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2903 = stablehlo.slice %2902 [0:1, 0:32, 0:8, 0:32] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x32xf32>
    %2904 = stablehlo.broadcast_in_dim %2903, dims = [0, 1, 2, 3] : (tensor<1x32x8x32xf32>) -> tensor<1x32x8x32x32xf32>
    %2905 = stablehlo.slice %2900 [0:1, 0:32, 0:8, 0:32, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x32x64xf32>
    %2906 = stablehlo.slice %2905 [0:1, 0:32, 0:8, 0:32, 0:32] : (tensor<1x32x8x32x64xf32>) -> tensor<1x32x8x32x32xf32>
    %2907 = stablehlo.multiply %2904, %2906 : tensor<1x32x8x32x32xf32>
    %2908 = stablehlo.reduce(%2907 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x32x32xf32>, tensor<f32>) -> tensor<1x32x8x32xf32>
    %2909 = stablehlo.add %2903, %2908 : tensor<1x32x8x32xf32>
    %2910 = "stablehlo.gather"(%2909, %1259) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x32xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2911 = stablehlo.select %371, %cst_224, %2910 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2912 = stablehlo.select %368, %2911, %2902 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2913 = stablehlo.broadcast_in_dim %2912, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2914 = stablehlo.select %364, %2913, %2900 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2915 = stablehlo.slice %2914 [0:1, 0:32, 0:8, 33:34, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2916 = stablehlo.reshape %2915 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2917 = stablehlo.slice %2916 [0:1, 0:32, 0:8, 0:33] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x33xf32>
    %2918 = stablehlo.broadcast_in_dim %2917, dims = [0, 1, 2, 3] : (tensor<1x32x8x33xf32>) -> tensor<1x32x8x33x33xf32>
    %2919 = stablehlo.slice %2914 [0:1, 0:32, 0:8, 0:33, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x33x64xf32>
    %2920 = stablehlo.slice %2919 [0:1, 0:32, 0:8, 0:33, 0:33] : (tensor<1x32x8x33x64xf32>) -> tensor<1x32x8x33x33xf32>
    %2921 = stablehlo.multiply %2918, %2920 : tensor<1x32x8x33x33xf32>
    %2922 = stablehlo.reduce(%2921 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x33x33xf32>, tensor<f32>) -> tensor<1x32x8x33xf32>
    %2923 = stablehlo.add %2917, %2922 : tensor<1x32x8x33xf32>
    %2924 = "stablehlo.gather"(%2923, %1278) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x33xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2925 = stablehlo.select %363, %cst_224, %2924 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2926 = stablehlo.select %360, %2925, %2916 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2927 = stablehlo.broadcast_in_dim %2926, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2928 = stablehlo.select %356, %2927, %2914 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2929 = stablehlo.slice %2928 [0:1, 0:32, 0:8, 34:35, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2930 = stablehlo.reshape %2929 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2931 = stablehlo.slice %2930 [0:1, 0:32, 0:8, 0:34] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x34xf32>
    %2932 = stablehlo.broadcast_in_dim %2931, dims = [0, 1, 2, 3] : (tensor<1x32x8x34xf32>) -> tensor<1x32x8x34x34xf32>
    %2933 = stablehlo.slice %2928 [0:1, 0:32, 0:8, 0:34, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x34x64xf32>
    %2934 = stablehlo.slice %2933 [0:1, 0:32, 0:8, 0:34, 0:34] : (tensor<1x32x8x34x64xf32>) -> tensor<1x32x8x34x34xf32>
    %2935 = stablehlo.multiply %2932, %2934 : tensor<1x32x8x34x34xf32>
    %2936 = stablehlo.reduce(%2935 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x34x34xf32>, tensor<f32>) -> tensor<1x32x8x34xf32>
    %2937 = stablehlo.add %2931, %2936 : tensor<1x32x8x34xf32>
    %2938 = "stablehlo.gather"(%2937, %1297) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x34xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2939 = stablehlo.select %355, %cst_224, %2938 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2940 = stablehlo.select %352, %2939, %2930 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2941 = stablehlo.broadcast_in_dim %2940, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2942 = stablehlo.select %348, %2941, %2928 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2943 = stablehlo.slice %2942 [0:1, 0:32, 0:8, 35:36, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2944 = stablehlo.reshape %2943 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2945 = stablehlo.slice %2944 [0:1, 0:32, 0:8, 0:35] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x35xf32>
    %2946 = stablehlo.broadcast_in_dim %2945, dims = [0, 1, 2, 3] : (tensor<1x32x8x35xf32>) -> tensor<1x32x8x35x35xf32>
    %2947 = stablehlo.slice %2942 [0:1, 0:32, 0:8, 0:35, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x35x64xf32>
    %2948 = stablehlo.slice %2947 [0:1, 0:32, 0:8, 0:35, 0:35] : (tensor<1x32x8x35x64xf32>) -> tensor<1x32x8x35x35xf32>
    %2949 = stablehlo.multiply %2946, %2948 : tensor<1x32x8x35x35xf32>
    %2950 = stablehlo.reduce(%2949 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x35x35xf32>, tensor<f32>) -> tensor<1x32x8x35xf32>
    %2951 = stablehlo.add %2945, %2950 : tensor<1x32x8x35xf32>
    %2952 = "stablehlo.gather"(%2951, %1316) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x35xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2953 = stablehlo.select %347, %cst_224, %2952 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2954 = stablehlo.select %344, %2953, %2944 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2955 = stablehlo.broadcast_in_dim %2954, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2956 = stablehlo.select %340, %2955, %2942 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2957 = stablehlo.slice %2956 [0:1, 0:32, 0:8, 36:37, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2958 = stablehlo.reshape %2957 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2959 = stablehlo.slice %2958 [0:1, 0:32, 0:8, 0:36] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x36xf32>
    %2960 = stablehlo.broadcast_in_dim %2959, dims = [0, 1, 2, 3] : (tensor<1x32x8x36xf32>) -> tensor<1x32x8x36x36xf32>
    %2961 = stablehlo.slice %2956 [0:1, 0:32, 0:8, 0:36, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x36x64xf32>
    %2962 = stablehlo.slice %2961 [0:1, 0:32, 0:8, 0:36, 0:36] : (tensor<1x32x8x36x64xf32>) -> tensor<1x32x8x36x36xf32>
    %2963 = stablehlo.multiply %2960, %2962 : tensor<1x32x8x36x36xf32>
    %2964 = stablehlo.reduce(%2963 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x36x36xf32>, tensor<f32>) -> tensor<1x32x8x36xf32>
    %2965 = stablehlo.add %2959, %2964 : tensor<1x32x8x36xf32>
    %2966 = "stablehlo.gather"(%2965, %1335) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x36xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2967 = stablehlo.select %339, %cst_224, %2966 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2968 = stablehlo.select %336, %2967, %2958 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2969 = stablehlo.broadcast_in_dim %2968, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2970 = stablehlo.select %332, %2969, %2956 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2971 = stablehlo.slice %2970 [0:1, 0:32, 0:8, 37:38, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2972 = stablehlo.reshape %2971 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2973 = stablehlo.slice %2972 [0:1, 0:32, 0:8, 0:37] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x37xf32>
    %2974 = stablehlo.broadcast_in_dim %2973, dims = [0, 1, 2, 3] : (tensor<1x32x8x37xf32>) -> tensor<1x32x8x37x37xf32>
    %2975 = stablehlo.slice %2970 [0:1, 0:32, 0:8, 0:37, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x37x64xf32>
    %2976 = stablehlo.slice %2975 [0:1, 0:32, 0:8, 0:37, 0:37] : (tensor<1x32x8x37x64xf32>) -> tensor<1x32x8x37x37xf32>
    %2977 = stablehlo.multiply %2974, %2976 : tensor<1x32x8x37x37xf32>
    %2978 = stablehlo.reduce(%2977 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x37x37xf32>, tensor<f32>) -> tensor<1x32x8x37xf32>
    %2979 = stablehlo.add %2973, %2978 : tensor<1x32x8x37xf32>
    %2980 = "stablehlo.gather"(%2979, %1354) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x37xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2981 = stablehlo.select %331, %cst_224, %2980 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2982 = stablehlo.select %328, %2981, %2972 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2983 = stablehlo.broadcast_in_dim %2982, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2984 = stablehlo.select %324, %2983, %2970 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2985 = stablehlo.slice %2984 [0:1, 0:32, 0:8, 38:39, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %2986 = stablehlo.reshape %2985 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %2987 = stablehlo.slice %2986 [0:1, 0:32, 0:8, 0:38] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x38xf32>
    %2988 = stablehlo.broadcast_in_dim %2987, dims = [0, 1, 2, 3] : (tensor<1x32x8x38xf32>) -> tensor<1x32x8x38x38xf32>
    %2989 = stablehlo.slice %2984 [0:1, 0:32, 0:8, 0:38, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x38x64xf32>
    %2990 = stablehlo.slice %2989 [0:1, 0:32, 0:8, 0:38, 0:38] : (tensor<1x32x8x38x64xf32>) -> tensor<1x32x8x38x38xf32>
    %2991 = stablehlo.multiply %2988, %2990 : tensor<1x32x8x38x38xf32>
    %2992 = stablehlo.reduce(%2991 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x38x38xf32>, tensor<f32>) -> tensor<1x32x8x38xf32>
    %2993 = stablehlo.add %2987, %2992 : tensor<1x32x8x38xf32>
    %2994 = "stablehlo.gather"(%2993, %1373) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x38xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %2995 = stablehlo.select %323, %cst_224, %2994 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2996 = stablehlo.select %320, %2995, %2986 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %2997 = stablehlo.broadcast_in_dim %2996, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %2998 = stablehlo.select %316, %2997, %2984 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %2999 = stablehlo.slice %2998 [0:1, 0:32, 0:8, 39:40, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3000 = stablehlo.reshape %2999 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3001 = stablehlo.slice %3000 [0:1, 0:32, 0:8, 0:39] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x39xf32>
    %3002 = stablehlo.broadcast_in_dim %3001, dims = [0, 1, 2, 3] : (tensor<1x32x8x39xf32>) -> tensor<1x32x8x39x39xf32>
    %3003 = stablehlo.slice %2998 [0:1, 0:32, 0:8, 0:39, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x39x64xf32>
    %3004 = stablehlo.slice %3003 [0:1, 0:32, 0:8, 0:39, 0:39] : (tensor<1x32x8x39x64xf32>) -> tensor<1x32x8x39x39xf32>
    %3005 = stablehlo.multiply %3002, %3004 : tensor<1x32x8x39x39xf32>
    %3006 = stablehlo.reduce(%3005 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x39x39xf32>, tensor<f32>) -> tensor<1x32x8x39xf32>
    %3007 = stablehlo.add %3001, %3006 : tensor<1x32x8x39xf32>
    %3008 = "stablehlo.gather"(%3007, %1392) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x39xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3009 = stablehlo.select %315, %cst_224, %3008 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3010 = stablehlo.select %312, %3009, %3000 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3011 = stablehlo.broadcast_in_dim %3010, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3012 = stablehlo.select %308, %3011, %2998 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3013 = stablehlo.slice %3012 [0:1, 0:32, 0:8, 40:41, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3014 = stablehlo.reshape %3013 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3015 = stablehlo.slice %3014 [0:1, 0:32, 0:8, 0:40] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x40xf32>
    %3016 = stablehlo.broadcast_in_dim %3015, dims = [0, 1, 2, 3] : (tensor<1x32x8x40xf32>) -> tensor<1x32x8x40x40xf32>
    %3017 = stablehlo.slice %3012 [0:1, 0:32, 0:8, 0:40, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x40x64xf32>
    %3018 = stablehlo.slice %3017 [0:1, 0:32, 0:8, 0:40, 0:40] : (tensor<1x32x8x40x64xf32>) -> tensor<1x32x8x40x40xf32>
    %3019 = stablehlo.multiply %3016, %3018 : tensor<1x32x8x40x40xf32>
    %3020 = stablehlo.reduce(%3019 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x40x40xf32>, tensor<f32>) -> tensor<1x32x8x40xf32>
    %3021 = stablehlo.add %3015, %3020 : tensor<1x32x8x40xf32>
    %3022 = "stablehlo.gather"(%3021, %1411) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x40xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3023 = stablehlo.select %307, %cst_224, %3022 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3024 = stablehlo.select %304, %3023, %3014 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3025 = stablehlo.broadcast_in_dim %3024, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3026 = stablehlo.select %300, %3025, %3012 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3027 = stablehlo.slice %3026 [0:1, 0:32, 0:8, 41:42, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3028 = stablehlo.reshape %3027 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3029 = stablehlo.slice %3028 [0:1, 0:32, 0:8, 0:41] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x41xf32>
    %3030 = stablehlo.broadcast_in_dim %3029, dims = [0, 1, 2, 3] : (tensor<1x32x8x41xf32>) -> tensor<1x32x8x41x41xf32>
    %3031 = stablehlo.slice %3026 [0:1, 0:32, 0:8, 0:41, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x41x64xf32>
    %3032 = stablehlo.slice %3031 [0:1, 0:32, 0:8, 0:41, 0:41] : (tensor<1x32x8x41x64xf32>) -> tensor<1x32x8x41x41xf32>
    %3033 = stablehlo.multiply %3030, %3032 : tensor<1x32x8x41x41xf32>
    %3034 = stablehlo.reduce(%3033 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x41x41xf32>, tensor<f32>) -> tensor<1x32x8x41xf32>
    %3035 = stablehlo.add %3029, %3034 : tensor<1x32x8x41xf32>
    %3036 = "stablehlo.gather"(%3035, %1430) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x41xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3037 = stablehlo.select %299, %cst_224, %3036 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3038 = stablehlo.select %296, %3037, %3028 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3039 = stablehlo.broadcast_in_dim %3038, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3040 = stablehlo.select %292, %3039, %3026 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3041 = stablehlo.slice %3040 [0:1, 0:32, 0:8, 42:43, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3042 = stablehlo.reshape %3041 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3043 = stablehlo.slice %3042 [0:1, 0:32, 0:8, 0:42] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x42xf32>
    %3044 = stablehlo.broadcast_in_dim %3043, dims = [0, 1, 2, 3] : (tensor<1x32x8x42xf32>) -> tensor<1x32x8x42x42xf32>
    %3045 = stablehlo.slice %3040 [0:1, 0:32, 0:8, 0:42, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x42x64xf32>
    %3046 = stablehlo.slice %3045 [0:1, 0:32, 0:8, 0:42, 0:42] : (tensor<1x32x8x42x64xf32>) -> tensor<1x32x8x42x42xf32>
    %3047 = stablehlo.multiply %3044, %3046 : tensor<1x32x8x42x42xf32>
    %3048 = stablehlo.reduce(%3047 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x42x42xf32>, tensor<f32>) -> tensor<1x32x8x42xf32>
    %3049 = stablehlo.add %3043, %3048 : tensor<1x32x8x42xf32>
    %3050 = "stablehlo.gather"(%3049, %1449) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x42xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3051 = stablehlo.select %291, %cst_224, %3050 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3052 = stablehlo.select %288, %3051, %3042 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3053 = stablehlo.broadcast_in_dim %3052, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3054 = stablehlo.select %284, %3053, %3040 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3055 = stablehlo.slice %3054 [0:1, 0:32, 0:8, 43:44, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3056 = stablehlo.reshape %3055 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3057 = stablehlo.slice %3056 [0:1, 0:32, 0:8, 0:43] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x43xf32>
    %3058 = stablehlo.broadcast_in_dim %3057, dims = [0, 1, 2, 3] : (tensor<1x32x8x43xf32>) -> tensor<1x32x8x43x43xf32>
    %3059 = stablehlo.slice %3054 [0:1, 0:32, 0:8, 0:43, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x43x64xf32>
    %3060 = stablehlo.slice %3059 [0:1, 0:32, 0:8, 0:43, 0:43] : (tensor<1x32x8x43x64xf32>) -> tensor<1x32x8x43x43xf32>
    %3061 = stablehlo.multiply %3058, %3060 : tensor<1x32x8x43x43xf32>
    %3062 = stablehlo.reduce(%3061 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x43x43xf32>, tensor<f32>) -> tensor<1x32x8x43xf32>
    %3063 = stablehlo.add %3057, %3062 : tensor<1x32x8x43xf32>
    %3064 = "stablehlo.gather"(%3063, %1468) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x43xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3065 = stablehlo.select %283, %cst_224, %3064 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3066 = stablehlo.select %280, %3065, %3056 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3067 = stablehlo.broadcast_in_dim %3066, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3068 = stablehlo.select %276, %3067, %3054 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3069 = stablehlo.slice %3068 [0:1, 0:32, 0:8, 44:45, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3070 = stablehlo.reshape %3069 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3071 = stablehlo.slice %3070 [0:1, 0:32, 0:8, 0:44] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x44xf32>
    %3072 = stablehlo.broadcast_in_dim %3071, dims = [0, 1, 2, 3] : (tensor<1x32x8x44xf32>) -> tensor<1x32x8x44x44xf32>
    %3073 = stablehlo.slice %3068 [0:1, 0:32, 0:8, 0:44, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x44x64xf32>
    %3074 = stablehlo.slice %3073 [0:1, 0:32, 0:8, 0:44, 0:44] : (tensor<1x32x8x44x64xf32>) -> tensor<1x32x8x44x44xf32>
    %3075 = stablehlo.multiply %3072, %3074 : tensor<1x32x8x44x44xf32>
    %3076 = stablehlo.reduce(%3075 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x44x44xf32>, tensor<f32>) -> tensor<1x32x8x44xf32>
    %3077 = stablehlo.add %3071, %3076 : tensor<1x32x8x44xf32>
    %3078 = "stablehlo.gather"(%3077, %1487) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x44xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3079 = stablehlo.select %275, %cst_224, %3078 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3080 = stablehlo.select %272, %3079, %3070 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3081 = stablehlo.broadcast_in_dim %3080, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3082 = stablehlo.select %268, %3081, %3068 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3083 = stablehlo.slice %3082 [0:1, 0:32, 0:8, 45:46, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3084 = stablehlo.reshape %3083 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3085 = stablehlo.slice %3084 [0:1, 0:32, 0:8, 0:45] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x45xf32>
    %3086 = stablehlo.broadcast_in_dim %3085, dims = [0, 1, 2, 3] : (tensor<1x32x8x45xf32>) -> tensor<1x32x8x45x45xf32>
    %3087 = stablehlo.slice %3082 [0:1, 0:32, 0:8, 0:45, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x45x64xf32>
    %3088 = stablehlo.slice %3087 [0:1, 0:32, 0:8, 0:45, 0:45] : (tensor<1x32x8x45x64xf32>) -> tensor<1x32x8x45x45xf32>
    %3089 = stablehlo.multiply %3086, %3088 : tensor<1x32x8x45x45xf32>
    %3090 = stablehlo.reduce(%3089 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x45x45xf32>, tensor<f32>) -> tensor<1x32x8x45xf32>
    %3091 = stablehlo.add %3085, %3090 : tensor<1x32x8x45xf32>
    %3092 = "stablehlo.gather"(%3091, %1506) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x45xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3093 = stablehlo.select %267, %cst_224, %3092 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3094 = stablehlo.select %264, %3093, %3084 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3095 = stablehlo.broadcast_in_dim %3094, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3096 = stablehlo.select %260, %3095, %3082 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3097 = stablehlo.slice %3096 [0:1, 0:32, 0:8, 46:47, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3098 = stablehlo.reshape %3097 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3099 = stablehlo.slice %3098 [0:1, 0:32, 0:8, 0:46] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x46xf32>
    %3100 = stablehlo.broadcast_in_dim %3099, dims = [0, 1, 2, 3] : (tensor<1x32x8x46xf32>) -> tensor<1x32x8x46x46xf32>
    %3101 = stablehlo.slice %3096 [0:1, 0:32, 0:8, 0:46, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x46x64xf32>
    %3102 = stablehlo.slice %3101 [0:1, 0:32, 0:8, 0:46, 0:46] : (tensor<1x32x8x46x64xf32>) -> tensor<1x32x8x46x46xf32>
    %3103 = stablehlo.multiply %3100, %3102 : tensor<1x32x8x46x46xf32>
    %3104 = stablehlo.reduce(%3103 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x46x46xf32>, tensor<f32>) -> tensor<1x32x8x46xf32>
    %3105 = stablehlo.add %3099, %3104 : tensor<1x32x8x46xf32>
    %3106 = "stablehlo.gather"(%3105, %1525) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x46xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3107 = stablehlo.select %259, %cst_224, %3106 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3108 = stablehlo.select %256, %3107, %3098 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3109 = stablehlo.broadcast_in_dim %3108, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3110 = stablehlo.select %252, %3109, %3096 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3111 = stablehlo.slice %3110 [0:1, 0:32, 0:8, 47:48, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3112 = stablehlo.reshape %3111 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3113 = stablehlo.slice %3112 [0:1, 0:32, 0:8, 0:47] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x47xf32>
    %3114 = stablehlo.broadcast_in_dim %3113, dims = [0, 1, 2, 3] : (tensor<1x32x8x47xf32>) -> tensor<1x32x8x47x47xf32>
    %3115 = stablehlo.slice %3110 [0:1, 0:32, 0:8, 0:47, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x47x64xf32>
    %3116 = stablehlo.slice %3115 [0:1, 0:32, 0:8, 0:47, 0:47] : (tensor<1x32x8x47x64xf32>) -> tensor<1x32x8x47x47xf32>
    %3117 = stablehlo.multiply %3114, %3116 : tensor<1x32x8x47x47xf32>
    %3118 = stablehlo.reduce(%3117 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x47x47xf32>, tensor<f32>) -> tensor<1x32x8x47xf32>
    %3119 = stablehlo.add %3113, %3118 : tensor<1x32x8x47xf32>
    %3120 = "stablehlo.gather"(%3119, %1544) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x47xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3121 = stablehlo.select %251, %cst_224, %3120 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3122 = stablehlo.select %248, %3121, %3112 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3123 = stablehlo.broadcast_in_dim %3122, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3124 = stablehlo.select %244, %3123, %3110 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3125 = stablehlo.slice %3124 [0:1, 0:32, 0:8, 48:49, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3126 = stablehlo.reshape %3125 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3127 = stablehlo.slice %3126 [0:1, 0:32, 0:8, 0:48] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x48xf32>
    %3128 = stablehlo.broadcast_in_dim %3127, dims = [0, 1, 2, 3] : (tensor<1x32x8x48xf32>) -> tensor<1x32x8x48x48xf32>
    %3129 = stablehlo.slice %3124 [0:1, 0:32, 0:8, 0:48, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x48x64xf32>
    %3130 = stablehlo.slice %3129 [0:1, 0:32, 0:8, 0:48, 0:48] : (tensor<1x32x8x48x64xf32>) -> tensor<1x32x8x48x48xf32>
    %3131 = stablehlo.multiply %3128, %3130 : tensor<1x32x8x48x48xf32>
    %3132 = stablehlo.reduce(%3131 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x48x48xf32>, tensor<f32>) -> tensor<1x32x8x48xf32>
    %3133 = stablehlo.add %3127, %3132 : tensor<1x32x8x48xf32>
    %3134 = "stablehlo.gather"(%3133, %1563) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x48xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3135 = stablehlo.select %243, %cst_224, %3134 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3136 = stablehlo.select %240, %3135, %3126 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3137 = stablehlo.broadcast_in_dim %3136, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3138 = stablehlo.select %236, %3137, %3124 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3139 = stablehlo.slice %3138 [0:1, 0:32, 0:8, 49:50, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3140 = stablehlo.reshape %3139 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3141 = stablehlo.slice %3140 [0:1, 0:32, 0:8, 0:49] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x49xf32>
    %3142 = stablehlo.broadcast_in_dim %3141, dims = [0, 1, 2, 3] : (tensor<1x32x8x49xf32>) -> tensor<1x32x8x49x49xf32>
    %3143 = stablehlo.slice %3138 [0:1, 0:32, 0:8, 0:49, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x49x64xf32>
    %3144 = stablehlo.slice %3143 [0:1, 0:32, 0:8, 0:49, 0:49] : (tensor<1x32x8x49x64xf32>) -> tensor<1x32x8x49x49xf32>
    %3145 = stablehlo.multiply %3142, %3144 : tensor<1x32x8x49x49xf32>
    %3146 = stablehlo.reduce(%3145 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x49x49xf32>, tensor<f32>) -> tensor<1x32x8x49xf32>
    %3147 = stablehlo.add %3141, %3146 : tensor<1x32x8x49xf32>
    %3148 = "stablehlo.gather"(%3147, %1582) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x49xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3149 = stablehlo.select %235, %cst_224, %3148 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3150 = stablehlo.select %232, %3149, %3140 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3151 = stablehlo.broadcast_in_dim %3150, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3152 = stablehlo.select %228, %3151, %3138 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3153 = stablehlo.slice %3152 [0:1, 0:32, 0:8, 50:51, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3154 = stablehlo.reshape %3153 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3155 = stablehlo.slice %3154 [0:1, 0:32, 0:8, 0:50] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x50xf32>
    %3156 = stablehlo.broadcast_in_dim %3155, dims = [0, 1, 2, 3] : (tensor<1x32x8x50xf32>) -> tensor<1x32x8x50x50xf32>
    %3157 = stablehlo.slice %3152 [0:1, 0:32, 0:8, 0:50, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x50x64xf32>
    %3158 = stablehlo.slice %3157 [0:1, 0:32, 0:8, 0:50, 0:50] : (tensor<1x32x8x50x64xf32>) -> tensor<1x32x8x50x50xf32>
    %3159 = stablehlo.multiply %3156, %3158 : tensor<1x32x8x50x50xf32>
    %3160 = stablehlo.reduce(%3159 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x50x50xf32>, tensor<f32>) -> tensor<1x32x8x50xf32>
    %3161 = stablehlo.add %3155, %3160 : tensor<1x32x8x50xf32>
    %3162 = "stablehlo.gather"(%3161, %1601) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x50xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3163 = stablehlo.select %227, %cst_224, %3162 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3164 = stablehlo.select %224, %3163, %3154 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3165 = stablehlo.broadcast_in_dim %3164, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3166 = stablehlo.select %220, %3165, %3152 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3167 = stablehlo.slice %3166 [0:1, 0:32, 0:8, 51:52, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3168 = stablehlo.reshape %3167 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3169 = stablehlo.slice %3168 [0:1, 0:32, 0:8, 0:51] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x51xf32>
    %3170 = stablehlo.broadcast_in_dim %3169, dims = [0, 1, 2, 3] : (tensor<1x32x8x51xf32>) -> tensor<1x32x8x51x51xf32>
    %3171 = stablehlo.slice %3166 [0:1, 0:32, 0:8, 0:51, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x51x64xf32>
    %3172 = stablehlo.slice %3171 [0:1, 0:32, 0:8, 0:51, 0:51] : (tensor<1x32x8x51x64xf32>) -> tensor<1x32x8x51x51xf32>
    %3173 = stablehlo.multiply %3170, %3172 : tensor<1x32x8x51x51xf32>
    %3174 = stablehlo.reduce(%3173 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x51x51xf32>, tensor<f32>) -> tensor<1x32x8x51xf32>
    %3175 = stablehlo.add %3169, %3174 : tensor<1x32x8x51xf32>
    %3176 = "stablehlo.gather"(%3175, %1620) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x51xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3177 = stablehlo.select %219, %cst_224, %3176 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3178 = stablehlo.select %216, %3177, %3168 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3179 = stablehlo.broadcast_in_dim %3178, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3180 = stablehlo.select %212, %3179, %3166 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3181 = stablehlo.slice %3180 [0:1, 0:32, 0:8, 52:53, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3182 = stablehlo.reshape %3181 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3183 = stablehlo.slice %3182 [0:1, 0:32, 0:8, 0:52] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x52xf32>
    %3184 = stablehlo.broadcast_in_dim %3183, dims = [0, 1, 2, 3] : (tensor<1x32x8x52xf32>) -> tensor<1x32x8x52x52xf32>
    %3185 = stablehlo.slice %3180 [0:1, 0:32, 0:8, 0:52, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x52x64xf32>
    %3186 = stablehlo.slice %3185 [0:1, 0:32, 0:8, 0:52, 0:52] : (tensor<1x32x8x52x64xf32>) -> tensor<1x32x8x52x52xf32>
    %3187 = stablehlo.multiply %3184, %3186 : tensor<1x32x8x52x52xf32>
    %3188 = stablehlo.reduce(%3187 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x52x52xf32>, tensor<f32>) -> tensor<1x32x8x52xf32>
    %3189 = stablehlo.add %3183, %3188 : tensor<1x32x8x52xf32>
    %3190 = "stablehlo.gather"(%3189, %1639) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x52xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3191 = stablehlo.select %211, %cst_224, %3190 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3192 = stablehlo.select %208, %3191, %3182 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3193 = stablehlo.broadcast_in_dim %3192, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3194 = stablehlo.select %204, %3193, %3180 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3195 = stablehlo.slice %3194 [0:1, 0:32, 0:8, 53:54, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3196 = stablehlo.reshape %3195 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3197 = stablehlo.slice %3196 [0:1, 0:32, 0:8, 0:53] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x53xf32>
    %3198 = stablehlo.broadcast_in_dim %3197, dims = [0, 1, 2, 3] : (tensor<1x32x8x53xf32>) -> tensor<1x32x8x53x53xf32>
    %3199 = stablehlo.slice %3194 [0:1, 0:32, 0:8, 0:53, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x53x64xf32>
    %3200 = stablehlo.slice %3199 [0:1, 0:32, 0:8, 0:53, 0:53] : (tensor<1x32x8x53x64xf32>) -> tensor<1x32x8x53x53xf32>
    %3201 = stablehlo.multiply %3198, %3200 : tensor<1x32x8x53x53xf32>
    %3202 = stablehlo.reduce(%3201 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x53x53xf32>, tensor<f32>) -> tensor<1x32x8x53xf32>
    %3203 = stablehlo.add %3197, %3202 : tensor<1x32x8x53xf32>
    %3204 = "stablehlo.gather"(%3203, %1658) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x53xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3205 = stablehlo.select %203, %cst_224, %3204 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3206 = stablehlo.select %200, %3205, %3196 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3207 = stablehlo.broadcast_in_dim %3206, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3208 = stablehlo.select %196, %3207, %3194 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3209 = stablehlo.slice %3208 [0:1, 0:32, 0:8, 54:55, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3210 = stablehlo.reshape %3209 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3211 = stablehlo.slice %3210 [0:1, 0:32, 0:8, 0:54] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x54xf32>
    %3212 = stablehlo.broadcast_in_dim %3211, dims = [0, 1, 2, 3] : (tensor<1x32x8x54xf32>) -> tensor<1x32x8x54x54xf32>
    %3213 = stablehlo.slice %3208 [0:1, 0:32, 0:8, 0:54, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x54x64xf32>
    %3214 = stablehlo.slice %3213 [0:1, 0:32, 0:8, 0:54, 0:54] : (tensor<1x32x8x54x64xf32>) -> tensor<1x32x8x54x54xf32>
    %3215 = stablehlo.multiply %3212, %3214 : tensor<1x32x8x54x54xf32>
    %3216 = stablehlo.reduce(%3215 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x54x54xf32>, tensor<f32>) -> tensor<1x32x8x54xf32>
    %3217 = stablehlo.add %3211, %3216 : tensor<1x32x8x54xf32>
    %3218 = "stablehlo.gather"(%3217, %1677) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x54xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3219 = stablehlo.select %195, %cst_224, %3218 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3220 = stablehlo.select %192, %3219, %3210 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3221 = stablehlo.broadcast_in_dim %3220, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3222 = stablehlo.select %188, %3221, %3208 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3223 = stablehlo.slice %3222 [0:1, 0:32, 0:8, 55:56, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3224 = stablehlo.reshape %3223 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3225 = stablehlo.slice %3224 [0:1, 0:32, 0:8, 0:55] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x55xf32>
    %3226 = stablehlo.broadcast_in_dim %3225, dims = [0, 1, 2, 3] : (tensor<1x32x8x55xf32>) -> tensor<1x32x8x55x55xf32>
    %3227 = stablehlo.slice %3222 [0:1, 0:32, 0:8, 0:55, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x55x64xf32>
    %3228 = stablehlo.slice %3227 [0:1, 0:32, 0:8, 0:55, 0:55] : (tensor<1x32x8x55x64xf32>) -> tensor<1x32x8x55x55xf32>
    %3229 = stablehlo.multiply %3226, %3228 : tensor<1x32x8x55x55xf32>
    %3230 = stablehlo.reduce(%3229 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x55x55xf32>, tensor<f32>) -> tensor<1x32x8x55xf32>
    %3231 = stablehlo.add %3225, %3230 : tensor<1x32x8x55xf32>
    %3232 = "stablehlo.gather"(%3231, %1696) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x55xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3233 = stablehlo.select %187, %cst_224, %3232 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3234 = stablehlo.select %184, %3233, %3224 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3235 = stablehlo.broadcast_in_dim %3234, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3236 = stablehlo.select %180, %3235, %3222 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3237 = stablehlo.slice %3236 [0:1, 0:32, 0:8, 56:57, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3238 = stablehlo.reshape %3237 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3239 = stablehlo.slice %3238 [0:1, 0:32, 0:8, 0:56] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x56xf32>
    %3240 = stablehlo.broadcast_in_dim %3239, dims = [0, 1, 2, 3] : (tensor<1x32x8x56xf32>) -> tensor<1x32x8x56x56xf32>
    %3241 = stablehlo.slice %3236 [0:1, 0:32, 0:8, 0:56, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x56x64xf32>
    %3242 = stablehlo.slice %3241 [0:1, 0:32, 0:8, 0:56, 0:56] : (tensor<1x32x8x56x64xf32>) -> tensor<1x32x8x56x56xf32>
    %3243 = stablehlo.multiply %3240, %3242 : tensor<1x32x8x56x56xf32>
    %3244 = stablehlo.reduce(%3243 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x56x56xf32>, tensor<f32>) -> tensor<1x32x8x56xf32>
    %3245 = stablehlo.add %3239, %3244 : tensor<1x32x8x56xf32>
    %3246 = "stablehlo.gather"(%3245, %1715) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x56xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3247 = stablehlo.select %179, %cst_224, %3246 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3248 = stablehlo.select %176, %3247, %3238 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3249 = stablehlo.broadcast_in_dim %3248, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3250 = stablehlo.select %172, %3249, %3236 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3251 = stablehlo.slice %3250 [0:1, 0:32, 0:8, 57:58, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3252 = stablehlo.reshape %3251 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3253 = stablehlo.slice %3252 [0:1, 0:32, 0:8, 0:57] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x57xf32>
    %3254 = stablehlo.broadcast_in_dim %3253, dims = [0, 1, 2, 3] : (tensor<1x32x8x57xf32>) -> tensor<1x32x8x57x57xf32>
    %3255 = stablehlo.slice %3250 [0:1, 0:32, 0:8, 0:57, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x57x64xf32>
    %3256 = stablehlo.slice %3255 [0:1, 0:32, 0:8, 0:57, 0:57] : (tensor<1x32x8x57x64xf32>) -> tensor<1x32x8x57x57xf32>
    %3257 = stablehlo.multiply %3254, %3256 : tensor<1x32x8x57x57xf32>
    %3258 = stablehlo.reduce(%3257 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x57x57xf32>, tensor<f32>) -> tensor<1x32x8x57xf32>
    %3259 = stablehlo.add %3253, %3258 : tensor<1x32x8x57xf32>
    %3260 = "stablehlo.gather"(%3259, %1734) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x57xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3261 = stablehlo.select %171, %cst_224, %3260 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3262 = stablehlo.select %168, %3261, %3252 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3263 = stablehlo.broadcast_in_dim %3262, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3264 = stablehlo.select %164, %3263, %3250 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3265 = stablehlo.slice %3264 [0:1, 0:32, 0:8, 58:59, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3266 = stablehlo.reshape %3265 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3267 = stablehlo.slice %3266 [0:1, 0:32, 0:8, 0:58] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x58xf32>
    %3268 = stablehlo.broadcast_in_dim %3267, dims = [0, 1, 2, 3] : (tensor<1x32x8x58xf32>) -> tensor<1x32x8x58x58xf32>
    %3269 = stablehlo.slice %3264 [0:1, 0:32, 0:8, 0:58, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x58x64xf32>
    %3270 = stablehlo.slice %3269 [0:1, 0:32, 0:8, 0:58, 0:58] : (tensor<1x32x8x58x64xf32>) -> tensor<1x32x8x58x58xf32>
    %3271 = stablehlo.multiply %3268, %3270 : tensor<1x32x8x58x58xf32>
    %3272 = stablehlo.reduce(%3271 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x58x58xf32>, tensor<f32>) -> tensor<1x32x8x58xf32>
    %3273 = stablehlo.add %3267, %3272 : tensor<1x32x8x58xf32>
    %3274 = "stablehlo.gather"(%3273, %1753) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x58xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3275 = stablehlo.select %163, %cst_224, %3274 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3276 = stablehlo.select %160, %3275, %3266 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3277 = stablehlo.broadcast_in_dim %3276, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3278 = stablehlo.select %156, %3277, %3264 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3279 = stablehlo.slice %3278 [0:1, 0:32, 0:8, 59:60, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3280 = stablehlo.reshape %3279 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3281 = stablehlo.slice %3280 [0:1, 0:32, 0:8, 0:59] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x59xf32>
    %3282 = stablehlo.broadcast_in_dim %3281, dims = [0, 1, 2, 3] : (tensor<1x32x8x59xf32>) -> tensor<1x32x8x59x59xf32>
    %3283 = stablehlo.slice %3278 [0:1, 0:32, 0:8, 0:59, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x59x64xf32>
    %3284 = stablehlo.slice %3283 [0:1, 0:32, 0:8, 0:59, 0:59] : (tensor<1x32x8x59x64xf32>) -> tensor<1x32x8x59x59xf32>
    %3285 = stablehlo.multiply %3282, %3284 : tensor<1x32x8x59x59xf32>
    %3286 = stablehlo.reduce(%3285 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x59x59xf32>, tensor<f32>) -> tensor<1x32x8x59xf32>
    %3287 = stablehlo.add %3281, %3286 : tensor<1x32x8x59xf32>
    %3288 = "stablehlo.gather"(%3287, %1772) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x59xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3289 = stablehlo.select %155, %cst_224, %3288 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3290 = stablehlo.select %152, %3289, %3280 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3291 = stablehlo.broadcast_in_dim %3290, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3292 = stablehlo.select %148, %3291, %3278 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3293 = stablehlo.slice %3292 [0:1, 0:32, 0:8, 60:61, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3294 = stablehlo.reshape %3293 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3295 = stablehlo.slice %3294 [0:1, 0:32, 0:8, 0:60] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x60xf32>
    %3296 = stablehlo.broadcast_in_dim %3295, dims = [0, 1, 2, 3] : (tensor<1x32x8x60xf32>) -> tensor<1x32x8x60x60xf32>
    %3297 = stablehlo.slice %3292 [0:1, 0:32, 0:8, 0:60, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x60x64xf32>
    %3298 = stablehlo.slice %3297 [0:1, 0:32, 0:8, 0:60, 0:60] : (tensor<1x32x8x60x64xf32>) -> tensor<1x32x8x60x60xf32>
    %3299 = stablehlo.multiply %3296, %3298 : tensor<1x32x8x60x60xf32>
    %3300 = stablehlo.reduce(%3299 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x60x60xf32>, tensor<f32>) -> tensor<1x32x8x60xf32>
    %3301 = stablehlo.add %3295, %3300 : tensor<1x32x8x60xf32>
    %3302 = "stablehlo.gather"(%3301, %1791) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x60xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3303 = stablehlo.select %147, %cst_224, %3302 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3304 = stablehlo.select %144, %3303, %3294 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3305 = stablehlo.broadcast_in_dim %3304, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3306 = stablehlo.select %140, %3305, %3292 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3307 = stablehlo.slice %3306 [0:1, 0:32, 0:8, 61:62, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3308 = stablehlo.reshape %3307 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3309 = stablehlo.slice %3308 [0:1, 0:32, 0:8, 0:61] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x61xf32>
    %3310 = stablehlo.broadcast_in_dim %3309, dims = [0, 1, 2, 3] : (tensor<1x32x8x61xf32>) -> tensor<1x32x8x61x61xf32>
    %3311 = stablehlo.slice %3306 [0:1, 0:32, 0:8, 0:61, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x61x64xf32>
    %3312 = stablehlo.slice %3311 [0:1, 0:32, 0:8, 0:61, 0:61] : (tensor<1x32x8x61x64xf32>) -> tensor<1x32x8x61x61xf32>
    %3313 = stablehlo.multiply %3310, %3312 : tensor<1x32x8x61x61xf32>
    %3314 = stablehlo.reduce(%3313 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x61x61xf32>, tensor<f32>) -> tensor<1x32x8x61xf32>
    %3315 = stablehlo.add %3309, %3314 : tensor<1x32x8x61xf32>
    %3316 = "stablehlo.gather"(%3315, %1810) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x61xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3317 = stablehlo.select %139, %cst_224, %3316 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3318 = stablehlo.select %136, %3317, %3308 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3319 = stablehlo.broadcast_in_dim %3318, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3320 = stablehlo.select %132, %3319, %3306 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3321 = stablehlo.slice %3320 [0:1, 0:32, 0:8, 62:63, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3322 = stablehlo.reshape %3321 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3323 = stablehlo.slice %3322 [0:1, 0:32, 0:8, 0:62] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x62xf32>
    %3324 = stablehlo.broadcast_in_dim %3323, dims = [0, 1, 2, 3] : (tensor<1x32x8x62xf32>) -> tensor<1x32x8x62x62xf32>
    %3325 = stablehlo.slice %3320 [0:1, 0:32, 0:8, 0:62, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x62x64xf32>
    %3326 = stablehlo.slice %3325 [0:1, 0:32, 0:8, 0:62, 0:62] : (tensor<1x32x8x62x64xf32>) -> tensor<1x32x8x62x62xf32>
    %3327 = stablehlo.multiply %3324, %3326 : tensor<1x32x8x62x62xf32>
    %3328 = stablehlo.reduce(%3327 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x62x62xf32>, tensor<f32>) -> tensor<1x32x8x62xf32>
    %3329 = stablehlo.add %3323, %3328 : tensor<1x32x8x62xf32>
    %3330 = "stablehlo.gather"(%3329, %1829) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x62xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3331 = stablehlo.select %131, %cst_224, %3330 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3332 = stablehlo.select %128, %3331, %3322 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3333 = stablehlo.broadcast_in_dim %3332, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3334 = stablehlo.select %124, %3333, %3320 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3335 = stablehlo.slice %3334 [0:1, 0:32, 0:8, 63:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3336 = stablehlo.reshape %3335 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3337 = stablehlo.slice %3336 [0:1, 0:32, 0:8, 0:63] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x63xf32>
    %3338 = stablehlo.broadcast_in_dim %3337, dims = [0, 1, 2, 3] : (tensor<1x32x8x63xf32>) -> tensor<1x32x8x63x63xf32>
    %3339 = stablehlo.slice %3334 [0:1, 0:32, 0:8, 0:63, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x63x64xf32>
    %3340 = stablehlo.slice %3339 [0:1, 0:32, 0:8, 0:63, 0:63] : (tensor<1x32x8x63x64xf32>) -> tensor<1x32x8x63x63xf32>
    %3341 = stablehlo.multiply %3338, %3340 : tensor<1x32x8x63x63xf32>
    %3342 = stablehlo.reduce(%3341 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x63x63xf32>, tensor<f32>) -> tensor<1x32x8x63xf32>
    %3343 = stablehlo.add %3337, %3342 : tensor<1x32x8x63xf32>
    %3344 = "stablehlo.gather"(%3343, %1848) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x63xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3345 = stablehlo.select %123, %cst_224, %3344 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3346 = stablehlo.select %120, %3345, %3336 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3347 = stablehlo.broadcast_in_dim %3346, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3348 = stablehlo.select %115, %3347, %3334 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3349 = stablehlo.add %3348, %1858 : tensor<1x32x8x64x64xf32>
    %3350 = stablehlo.slice %2355 [0:1, 0:494, 4096:8192] : (tensor<1x494x8192xbf16>) -> tensor<1x494x4096xbf16>
    %3351 = stablehlo.reshape %3350 : (tensor<1x494x4096xbf16>) -> tensor<1x494x32x128xbf16>
    %3352 = stablehlo.transpose %3351, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %3353 = stablehlo.convert %3352 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %3354 = stablehlo.pad %3353, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %3355 = stablehlo.multiply %3354, %2453 : tensor<1x32x512x128xf32>
    %3356 = stablehlo.reshape %3355 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3357 = stablehlo.dot_general %3349, %3356, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x64xf32>, tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3358 = stablehlo.slice %3357 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3359 = stablehlo.reshape %3358 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3360 = stablehlo.exponential %2403 : tensor<1x32x8x64xf32>
    %3361 = stablehlo.broadcast_in_dim %3360, dims = [0, 1, 2, 3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x128xf32>
    %3362 = stablehlo.multiply %2455, %3361 : tensor<1x32x8x64x128xf32>
    %3363 = stablehlo.dot_general %3349, %3362, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x64xf32>, tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3364 = stablehlo.slice %3363 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3365 = stablehlo.reshape %3364 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3366 = stablehlo.dot_general %3365, %cst_227, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3367 = stablehlo.subtract %3359, %3366 : tensor<1x32x64x128xf32>
    %3368 = stablehlo.dot_general %2442, %3367, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %3369 = stablehlo.add %2417, %3368 : tensor<1x32x128x128xf32>
    %3370 = stablehlo.slice %2403 [0:1, 0:32, 1:2, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3371 = stablehlo.reshape %3370 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %3372 = stablehlo.slice %3371 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %3373 = stablehlo.reshape %3372 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %3374 = stablehlo.exponential %3373 : tensor<1x32x1x1xf32>
    %3375 = stablehlo.reshape %3374 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %3376 = stablehlo.broadcast_in_dim %3375, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %3377 = stablehlo.multiply %3369, %3376 : tensor<1x32x128x128xf32>
    %3378 = stablehlo.slice %2433 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3379 = stablehlo.reshape %3378 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3380 = stablehlo.reshape %3372 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %3381 = stablehlo.broadcast_in_dim %3380, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %3382 = stablehlo.subtract %3381, %3371 : tensor<1x32x64xf32>
    %3383 = stablehlo.exponential %3382 : tensor<1x32x64xf32>
    %3384 = stablehlo.broadcast_in_dim %3383, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3385 = stablehlo.multiply %3379, %3384 : tensor<1x32x64x128xf32>
    %3386 = stablehlo.transpose %3385, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3387 = stablehlo.slice %3357 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3388 = stablehlo.reshape %3387 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3389 = stablehlo.slice %3363 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3390 = stablehlo.reshape %3389 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3391 = stablehlo.dot_general %3390, %3369, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3392 = stablehlo.subtract %3388, %3391 : tensor<1x32x64x128xf32>
    %3393 = stablehlo.dot_general %3386, %3392, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %3394 = stablehlo.add %3377, %3393 : tensor<1x32x128x128xf32>
    %3395 = stablehlo.slice %2403 [0:1, 0:32, 2:3, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3396 = stablehlo.reshape %3395 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %3397 = stablehlo.slice %3396 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %3398 = stablehlo.reshape %3397 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %3399 = stablehlo.exponential %3398 : tensor<1x32x1x1xf32>
    %3400 = stablehlo.reshape %3399 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %3401 = stablehlo.broadcast_in_dim %3400, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %3402 = stablehlo.multiply %3394, %3401 : tensor<1x32x128x128xf32>
    %3403 = stablehlo.slice %2433 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3404 = stablehlo.reshape %3403 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3405 = stablehlo.reshape %3397 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %3406 = stablehlo.broadcast_in_dim %3405, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %3407 = stablehlo.subtract %3406, %3396 : tensor<1x32x64xf32>
    %3408 = stablehlo.exponential %3407 : tensor<1x32x64xf32>
    %3409 = stablehlo.broadcast_in_dim %3408, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3410 = stablehlo.multiply %3404, %3409 : tensor<1x32x64x128xf32>
    %3411 = stablehlo.transpose %3410, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3412 = stablehlo.slice %3357 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3413 = stablehlo.reshape %3412 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3414 = stablehlo.slice %3363 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3415 = stablehlo.reshape %3414 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3416 = stablehlo.dot_general %3415, %3394, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3417 = stablehlo.subtract %3413, %3416 : tensor<1x32x64x128xf32>
    %3418 = stablehlo.dot_general %3411, %3417, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %3419 = stablehlo.add %3402, %3418 : tensor<1x32x128x128xf32>
    %3420 = stablehlo.slice %2403 [0:1, 0:32, 3:4, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3421 = stablehlo.reshape %3420 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %3422 = stablehlo.slice %3421 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %3423 = stablehlo.reshape %3422 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %3424 = stablehlo.exponential %3423 : tensor<1x32x1x1xf32>
    %3425 = stablehlo.reshape %3424 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %3426 = stablehlo.broadcast_in_dim %3425, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %3427 = stablehlo.multiply %3419, %3426 : tensor<1x32x128x128xf32>
    %3428 = stablehlo.slice %2433 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3429 = stablehlo.reshape %3428 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3430 = stablehlo.reshape %3422 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %3431 = stablehlo.broadcast_in_dim %3430, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %3432 = stablehlo.subtract %3431, %3421 : tensor<1x32x64xf32>
    %3433 = stablehlo.exponential %3432 : tensor<1x32x64xf32>
    %3434 = stablehlo.broadcast_in_dim %3433, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3435 = stablehlo.multiply %3429, %3434 : tensor<1x32x64x128xf32>
    %3436 = stablehlo.transpose %3435, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3437 = stablehlo.slice %3357 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3438 = stablehlo.reshape %3437 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3439 = stablehlo.slice %3363 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3440 = stablehlo.reshape %3439 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3441 = stablehlo.dot_general %3440, %3419, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3442 = stablehlo.subtract %3438, %3441 : tensor<1x32x64x128xf32>
    %3443 = stablehlo.dot_general %3436, %3442, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %3444 = stablehlo.add %3427, %3443 : tensor<1x32x128x128xf32>
    %3445 = stablehlo.slice %2403 [0:1, 0:32, 4:5, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3446 = stablehlo.reshape %3445 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %3447 = stablehlo.slice %3446 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %3448 = stablehlo.reshape %3447 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %3449 = stablehlo.exponential %3448 : tensor<1x32x1x1xf32>
    %3450 = stablehlo.reshape %3449 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %3451 = stablehlo.broadcast_in_dim %3450, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %3452 = stablehlo.multiply %3444, %3451 : tensor<1x32x128x128xf32>
    %3453 = stablehlo.slice %2433 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3454 = stablehlo.reshape %3453 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3455 = stablehlo.reshape %3447 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %3456 = stablehlo.broadcast_in_dim %3455, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %3457 = stablehlo.subtract %3456, %3446 : tensor<1x32x64xf32>
    %3458 = stablehlo.exponential %3457 : tensor<1x32x64xf32>
    %3459 = stablehlo.broadcast_in_dim %3458, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3460 = stablehlo.multiply %3454, %3459 : tensor<1x32x64x128xf32>
    %3461 = stablehlo.transpose %3460, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3462 = stablehlo.slice %3357 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3463 = stablehlo.reshape %3462 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3464 = stablehlo.slice %3363 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3465 = stablehlo.reshape %3464 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3466 = stablehlo.dot_general %3465, %3444, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3467 = stablehlo.subtract %3463, %3466 : tensor<1x32x64x128xf32>
    %3468 = stablehlo.dot_general %3461, %3467, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %3469 = stablehlo.add %3452, %3468 : tensor<1x32x128x128xf32>
    %3470 = stablehlo.slice %2403 [0:1, 0:32, 5:6, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3471 = stablehlo.reshape %3470 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %3472 = stablehlo.slice %3471 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %3473 = stablehlo.reshape %3472 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %3474 = stablehlo.exponential %3473 : tensor<1x32x1x1xf32>
    %3475 = stablehlo.reshape %3474 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %3476 = stablehlo.broadcast_in_dim %3475, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %3477 = stablehlo.multiply %3469, %3476 : tensor<1x32x128x128xf32>
    %3478 = stablehlo.slice %2433 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3479 = stablehlo.reshape %3478 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3480 = stablehlo.reshape %3472 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %3481 = stablehlo.broadcast_in_dim %3480, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %3482 = stablehlo.subtract %3481, %3471 : tensor<1x32x64xf32>
    %3483 = stablehlo.exponential %3482 : tensor<1x32x64xf32>
    %3484 = stablehlo.broadcast_in_dim %3483, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3485 = stablehlo.multiply %3479, %3484 : tensor<1x32x64x128xf32>
    %3486 = stablehlo.transpose %3485, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3487 = stablehlo.slice %3357 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3488 = stablehlo.reshape %3487 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3489 = stablehlo.slice %3363 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3490 = stablehlo.reshape %3489 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3491 = stablehlo.dot_general %3490, %3469, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3492 = stablehlo.subtract %3488, %3491 : tensor<1x32x64x128xf32>
    %3493 = stablehlo.dot_general %3486, %3492, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %3494 = stablehlo.add %3477, %3493 : tensor<1x32x128x128xf32>
    %3495 = stablehlo.slice %2403 [0:1, 0:32, 6:7, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3496 = stablehlo.reshape %3495 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %3497 = stablehlo.slice %3496 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %3498 = stablehlo.reshape %3497 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %3499 = stablehlo.exponential %3498 : tensor<1x32x1x1xf32>
    %3500 = stablehlo.reshape %3499 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %3501 = stablehlo.broadcast_in_dim %3500, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %3502 = stablehlo.multiply %3494, %3501 : tensor<1x32x128x128xf32>
    %3503 = stablehlo.slice %2433 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3504 = stablehlo.reshape %3503 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3505 = stablehlo.reshape %3497 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %3506 = stablehlo.broadcast_in_dim %3505, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %3507 = stablehlo.subtract %3506, %3496 : tensor<1x32x64xf32>
    %3508 = stablehlo.exponential %3507 : tensor<1x32x64xf32>
    %3509 = stablehlo.broadcast_in_dim %3508, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3510 = stablehlo.multiply %3504, %3509 : tensor<1x32x64x128xf32>
    %3511 = stablehlo.transpose %3510, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3512 = stablehlo.slice %3357 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3513 = stablehlo.reshape %3512 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3514 = stablehlo.slice %3363 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3515 = stablehlo.reshape %3514 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3516 = stablehlo.dot_general %3515, %3494, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3517 = stablehlo.subtract %3513, %3516 : tensor<1x32x64x128xf32>
    %3518 = stablehlo.dot_general %3511, %3517, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %3519 = stablehlo.add %3502, %3518 : tensor<1x32x128x128xf32>
    %3520 = stablehlo.dot_general %2409, %3519, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3521 = stablehlo.slice %2433 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3522 = stablehlo.reshape %3521 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3523 = stablehlo.transpose %3522, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3524 = stablehlo.dot_general %2374, %3523, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3525 = stablehlo.slice %2463 [0:1, 0:32, 7:8, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3526 = stablehlo.reshape %3525 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3527 = stablehlo.multiply %3524, %3526 : tensor<1x32x64x64xf32>
    %3528 = stablehlo.select %2034, %cst_31, %3527 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3529 = stablehlo.slice %3357 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3530 = stablehlo.reshape %3529 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3531 = stablehlo.slice %3363 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3532 = stablehlo.reshape %3531 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3533 = stablehlo.dot_general %3532, %3519, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3534 = stablehlo.subtract %3530, %3533 : tensor<1x32x64x128xf32>
    %3535 = stablehlo.dot_general %3528, %3534, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3536 = stablehlo.add %3520, %3535 : tensor<1x32x64x128xf32>
    %3537 = stablehlo.broadcast_in_dim %3536, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3538 = stablehlo.slice %2372 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3539 = stablehlo.reshape %3538 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3540 = stablehlo.reshape %3495 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3541 = stablehlo.exponential %3540 : tensor<1x32x64x1xf32>
    %3542 = stablehlo.reshape %3541 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3543 = stablehlo.broadcast_in_dim %3542, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3544 = stablehlo.multiply %3539, %3543 : tensor<1x32x64x128xf32>
    %3545 = stablehlo.dot_general %3544, %3494, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3546 = stablehlo.transpose %3504, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3547 = stablehlo.dot_general %3539, %3546, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3548 = stablehlo.slice %2463 [0:1, 0:32, 6:7, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3549 = stablehlo.reshape %3548 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3550 = stablehlo.multiply %3547, %3549 : tensor<1x32x64x64xf32>
    %3551 = stablehlo.select %2034, %cst_31, %3550 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3552 = stablehlo.dot_general %3551, %3517, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3553 = stablehlo.add %3545, %3552 : tensor<1x32x64x128xf32>
    %3554 = stablehlo.broadcast_in_dim %3553, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3555 = stablehlo.slice %2372 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3556 = stablehlo.reshape %3555 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3557 = stablehlo.reshape %3470 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3558 = stablehlo.exponential %3557 : tensor<1x32x64x1xf32>
    %3559 = stablehlo.reshape %3558 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3560 = stablehlo.broadcast_in_dim %3559, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3561 = stablehlo.multiply %3556, %3560 : tensor<1x32x64x128xf32>
    %3562 = stablehlo.dot_general %3561, %3469, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3563 = stablehlo.transpose %3479, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3564 = stablehlo.dot_general %3556, %3563, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3565 = stablehlo.slice %2463 [0:1, 0:32, 5:6, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3566 = stablehlo.reshape %3565 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3567 = stablehlo.multiply %3564, %3566 : tensor<1x32x64x64xf32>
    %3568 = stablehlo.select %2034, %cst_31, %3567 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3569 = stablehlo.dot_general %3568, %3492, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3570 = stablehlo.add %3562, %3569 : tensor<1x32x64x128xf32>
    %3571 = stablehlo.broadcast_in_dim %3570, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3572 = stablehlo.slice %2372 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3573 = stablehlo.reshape %3572 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3574 = stablehlo.reshape %3445 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3575 = stablehlo.exponential %3574 : tensor<1x32x64x1xf32>
    %3576 = stablehlo.reshape %3575 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3577 = stablehlo.broadcast_in_dim %3576, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3578 = stablehlo.multiply %3573, %3577 : tensor<1x32x64x128xf32>
    %3579 = stablehlo.dot_general %3578, %3444, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3580 = stablehlo.transpose %3454, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3581 = stablehlo.dot_general %3573, %3580, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3582 = stablehlo.slice %2463 [0:1, 0:32, 4:5, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3583 = stablehlo.reshape %3582 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3584 = stablehlo.multiply %3581, %3583 : tensor<1x32x64x64xf32>
    %3585 = stablehlo.select %2034, %cst_31, %3584 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3586 = stablehlo.dot_general %3585, %3467, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3587 = stablehlo.add %3579, %3586 : tensor<1x32x64x128xf32>
    %3588 = stablehlo.broadcast_in_dim %3587, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3589 = stablehlo.slice %2372 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3590 = stablehlo.reshape %3589 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3591 = stablehlo.reshape %3420 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3592 = stablehlo.exponential %3591 : tensor<1x32x64x1xf32>
    %3593 = stablehlo.reshape %3592 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3594 = stablehlo.broadcast_in_dim %3593, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3595 = stablehlo.multiply %3590, %3594 : tensor<1x32x64x128xf32>
    %3596 = stablehlo.dot_general %3595, %3419, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3597 = stablehlo.transpose %3429, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3598 = stablehlo.dot_general %3590, %3597, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3599 = stablehlo.slice %2463 [0:1, 0:32, 3:4, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3600 = stablehlo.reshape %3599 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3601 = stablehlo.multiply %3598, %3600 : tensor<1x32x64x64xf32>
    %3602 = stablehlo.select %2034, %cst_31, %3601 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3603 = stablehlo.dot_general %3602, %3442, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3604 = stablehlo.add %3596, %3603 : tensor<1x32x64x128xf32>
    %3605 = stablehlo.broadcast_in_dim %3604, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3606 = stablehlo.slice %2372 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3607 = stablehlo.reshape %3606 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3608 = stablehlo.reshape %3395 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3609 = stablehlo.exponential %3608 : tensor<1x32x64x1xf32>
    %3610 = stablehlo.reshape %3609 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3611 = stablehlo.broadcast_in_dim %3610, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3612 = stablehlo.multiply %3607, %3611 : tensor<1x32x64x128xf32>
    %3613 = stablehlo.dot_general %3612, %3394, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3614 = stablehlo.transpose %3404, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3615 = stablehlo.dot_general %3607, %3614, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3616 = stablehlo.slice %2463 [0:1, 0:32, 2:3, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3617 = stablehlo.reshape %3616 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3618 = stablehlo.multiply %3615, %3617 : tensor<1x32x64x64xf32>
    %3619 = stablehlo.select %2034, %cst_31, %3618 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3620 = stablehlo.dot_general %3619, %3417, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3621 = stablehlo.add %3613, %3620 : tensor<1x32x64x128xf32>
    %3622 = stablehlo.broadcast_in_dim %3621, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3623 = stablehlo.slice %2372 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3624 = stablehlo.reshape %3623 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3625 = stablehlo.reshape %3370 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3626 = stablehlo.exponential %3625 : tensor<1x32x64x1xf32>
    %3627 = stablehlo.reshape %3626 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3628 = stablehlo.broadcast_in_dim %3627, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3629 = stablehlo.multiply %3624, %3628 : tensor<1x32x64x128xf32>
    %3630 = stablehlo.dot_general %3629, %3369, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3631 = stablehlo.transpose %3379, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3632 = stablehlo.dot_general %3624, %3631, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3633 = stablehlo.slice %2463 [0:1, 0:32, 1:2, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3634 = stablehlo.reshape %3633 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3635 = stablehlo.multiply %3632, %3634 : tensor<1x32x64x64xf32>
    %3636 = stablehlo.select %2034, %cst_31, %3635 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3637 = stablehlo.dot_general %3636, %3392, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3638 = stablehlo.add %3630, %3637 : tensor<1x32x64x128xf32>
    %3639 = stablehlo.broadcast_in_dim %3638, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3640 = stablehlo.slice %2372 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3641 = stablehlo.reshape %3640 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3642 = stablehlo.reshape %2410 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3643 = stablehlo.exponential %3642 : tensor<1x32x64x1xf32>
    %3644 = stablehlo.reshape %3643 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3645 = stablehlo.broadcast_in_dim %3644, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3646 = stablehlo.multiply %3641, %3645 : tensor<1x32x64x128xf32>
    %3647 = stablehlo.dot_general %3646, %cst_227, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %3648 = stablehlo.transpose %2435, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3649 = stablehlo.dot_general %3641, %3648, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %3650 = stablehlo.slice %2463 [0:1, 0:32, 0:1, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %3651 = stablehlo.reshape %3650 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %3652 = stablehlo.multiply %3649, %3651 : tensor<1x32x64x64xf32>
    %3653 = stablehlo.select %2034, %cst_31, %3652 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %3654 = stablehlo.dot_general %3653, %3367, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3655 = stablehlo.add %3647, %3654 : tensor<1x32x64x128xf32>
    %3656 = stablehlo.broadcast_in_dim %3655, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3657 = stablehlo.select %2160, %3656, %cst_23 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3658 = stablehlo.select %2142, %3639, %3657 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3659 = stablehlo.select %2124, %3622, %3658 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3660 = stablehlo.select %2106, %3605, %3659 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3661 = stablehlo.select %2088, %3588, %3660 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3662 = stablehlo.select %2070, %3571, %3661 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3663 = stablehlo.select %2052, %3554, %3662 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3664 = stablehlo.select %1, %3537, %3663 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %3665 = stablehlo.reshape %3664 : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x512x128xf32>
    %3666 = stablehlo.slice %3665 [0:1, 0:32, 0:494, 0:128] : (tensor<1x32x512x128xf32>) -> tensor<1x32x494x128xf32>
    %3667 = stablehlo.transpose %3666, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,494,32,128]{3,1,2,0}"} : (tensor<1x32x494x128xf32>) -> tensor<1x494x32x128xf32>
    %3668 = stablehlo.convert %3667 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,494,32,128]{3,1,2,0}"} : (tensor<1x494x32x128xf32>) -> tensor<1x494x32x128xbf16>
    %3669 = stablehlo.reshape %3668 : (tensor<1x494x32x128xbf16>) -> tensor<15808x128xbf16>
    %3670 = stablehlo.reshape %arg39 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
    %3671 = stablehlo.custom_call @tt.mark_argument(%3670) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.norm.weight"}} : (tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16>
    %3672 = stablehlo.reshape %3671 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
    %3673 = stablehlo.composite "tenstorrent.rms_norm" %3669, %3672 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_4} : (tensor<15808x128xbf16>, tensor<128xbf16>) -> tensor<15808x128xbf16>
    %3674 = stablehlo.convert %3673 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %3675 = stablehlo.reshape %arg17 : (tensor<4096x2048xbf16>) -> tensor<1x4096x2048xbf16>
    %3676 = stablehlo.custom_call @tt.mark_argument(%3675) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.in_proj_z.weight"}} : (tensor<1x4096x2048xbf16>) -> tensor<1x4096x2048xbf16>
    %3677 = stablehlo.reshape %3676 : (tensor<1x4096x2048xbf16>) -> tensor<4096x2048xbf16>
    %3678 = stablehlo.transpose %3677, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,4096]{0,1}"} : (tensor<4096x2048xbf16>) -> tensor<2048x4096xbf16>
    %3679 = stablehlo.dot_general %2336, %3678, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x4096xbf16>) -> tensor<494x4096xbf16>
    %3680 = stablehlo.reshape %3679 : (tensor<494x4096xbf16>) -> tensor<15808x128xbf16>
    %3681 = stablehlo.convert %3680 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %3682 = stablehlo.logistic %3681 : tensor<15808x128xf32>
    %3683 = stablehlo.multiply %3681, %3682 : tensor<15808x128xf32>
    %3684 = stablehlo.multiply %3674, %3683 : tensor<15808x128xf32>
    %3685 = stablehlo.convert %3684 : (tensor<15808x128xf32>) -> tensor<15808x128xbf16>
    %3686 = stablehlo.reshape %3685 : (tensor<15808x128xbf16>) -> tensor<494x4096xbf16>
    %3687 = stablehlo.reshape %arg16 : (tensor<2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %3688 = stablehlo.custom_call @tt.mark_argument(%3687) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.linear_attn.out_proj.weight"}} : (tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %3689 = stablehlo.reshape %3688 : (tensor<1x2048x4096xbf16>) -> tensor<2048x4096xbf16>
    %3690 = stablehlo.transpose %3689, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[4096,2048]{0,1}"} : (tensor<2048x4096xbf16>) -> tensor<4096x2048xbf16>
    %3691 = stablehlo.dot_general %3686, %3690, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<494x2048xbf16>
    %3692 = stablehlo.reshape %3691 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %3693 = stablehlo.add %2329, %3692 : tensor<1x494x2048xbf16>
    %3694 = stablehlo.reshape %arg15 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %3695 = stablehlo.custom_call @tt.mark_argument(%3694) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.post_attention_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %3696 = stablehlo.reshape %3695 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %3697 = stablehlo.convert %3696 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %3698 = stablehlo.add %3697, %cst_231 : tensor<2048xf32>
    %3699 = stablehlo.composite "tenstorrent.rms_norm" %3693, %3698 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_7} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %3700 = stablehlo.reshape %3699 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %3701 = stablehlo.reshape %arg48 : (tensor<256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %3702 = stablehlo.custom_call @tt.mark_argument(%3701) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.mlp.gate.weight"}} : (tensor<1x256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %3703 = stablehlo.reshape %3702 : (tensor<1x256x2048xbf16>) -> tensor<256x2048xbf16>
    %3704 = stablehlo.transpose %3703, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,256]{0,1}"} : (tensor<256x2048xbf16>) -> tensor<2048x256xbf16>
    %3705 = stablehlo.dot_general %3700, %3704, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x256xbf16>) -> tensor<494x256xbf16>
    %3706 = stablehlo.convert %3705 : (tensor<494x256xbf16>) -> tensor<494x256xf32>
    %3707 = stablehlo.reduce(%3706 init: %cst_236) applies stablehlo.maximum across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %3708 = stablehlo.broadcast_in_dim %3707, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %3709 = stablehlo.subtract %3706, %3708 : tensor<494x256xf32>
    %3710 = stablehlo.exponential %3709 : tensor<494x256xf32>
    %3711 = stablehlo.reduce(%3710 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %3712 = stablehlo.broadcast_in_dim %3711, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %3713 = stablehlo.divide %3710, %3712 : tensor<494x256xf32>
    %3714:2 = stablehlo.composite "tenstorrent.topk" %3713 {composite_attributes = {dim = -1 : i64, k = 8 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk.impl_2} : (tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>)
    %3715 = stablehlo.reduce(%3714#0 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x8xf32>, tensor<f32>) -> tensor<494xf32>
    %3716 = stablehlo.broadcast_in_dim %3715, dims = [0] : (tensor<494xf32>) -> tensor<494x8xf32>
    %3717 = stablehlo.divide %3714#0, %3716 : tensor<494x8xf32>
    %3718 = stablehlo.concatenate %3717, %cst_22, dim = 0 : (tensor<494x8xf32>, tensor<18x8xf32>) -> tensor<512x8xf32>
    %3719 = stablehlo.convert %3718 : (tensor<512x8xf32>) -> tensor<512x8xbf16>
    %3720 = stablehlo.reshape %3719 : (tensor<512x8xbf16>) -> tensor<512x1x8xbf16>
    %3721 = stablehlo.concatenate %3714#1, %c_21, dim = 0 : (tensor<494x8xi64>, tensor<18x8xi64>) -> tensor<512x8xi64>
    %3722 = stablehlo.broadcast_in_dim %3721, dims = [0, 1] : (tensor<512x8xi64>) -> tensor<512x8x256xi64>
    %3723 = stablehlo.compare  EQ, %3722, %2244 : (tensor<512x8x256xi64>, tensor<512x8x256xi64>) -> tensor<512x8x256xi1>
    %3724 = stablehlo.convert %3723 : (tensor<512x8x256xi1>) -> tensor<512x8x256xbf16>
    %3725 = stablehlo.dot_general %3720, %3724, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x1x8xbf16>, tensor<512x8x256xbf16>) -> tensor<512x1x256xbf16>
    %3726 = stablehlo.reshape %3725 : (tensor<512x1x256xbf16>) -> tensor<1x512x256xbf16>
    %3727 = stablehlo.concatenate %3726, %3726, %3726, %3726, %3726, %3726, %3726, %3726, dim = 1 : (tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>) -> tensor<1x4096x256xbf16>
    %3728 = stablehlo.reshape %3727 : (tensor<1x4096x256xbf16>) -> tensor<1x1x4096x256xbf16>
    %3729 = stablehlo.concatenate %3700, %cst_18, dim = 0 : (tensor<494x2048xbf16>, tensor<18x2048xbf16>) -> tensor<512x2048xbf16>
    %3730 = stablehlo.reshape %3729 : (tensor<512x2048xbf16>) -> tensor<1x1x512x2048xbf16>
    %3731 = stablehlo.reshape %3721 : (tensor<512x8xi64>) -> tensor<1x1x512x8xi64>
    %3732 = stablehlo.custom_call @tt.all_to_all_dispatch(%3730, %3731, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8"}, xla_shape = "(bf16[1,8,512,2048]{3,2,1,0}, s64[1,8,512,8]{3,2,1,0})"} : (tensor<1x1x512x2048xbf16>, tensor<1x1x512x8xi64>, tensor<1x1x256x8xui16>) -> tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>
    %3733 = stablehlo.get_tuple_element %3732[1] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x8xi64>
    %3734 = stablehlo.reshape %3733 : (tensor<1x8x512x8xi64>) -> tensor<1x1x4096x8xi64>
    %3735 = stablehlo.custom_call @tt.moe_expert_token_remap(%3728, %2258, %3734) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {num_devices = "8", reduction_size = "32"}, xla_shape = "(bf16[1,1,4096,256]{3,2,1,0}, bf16[1,1,128,256]{3,2,1,0})"} : (tensor<1x1x4096x256xbf16>, tensor<1x1x256x8xui16>, tensor<1x1x4096x8xi64>) -> tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>
    %3736 = stablehlo.get_tuple_element %3732[0] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x2048xbf16>
    %3737 = stablehlo.reshape %3736 : (tensor<1x8x512x2048xbf16>) -> tensor<8x16x32x2048xbf16>
    %3738 = stablehlo.custom_call @tt.mark_argument(%arg50) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.mlp.experts.gate_up_proj"}} : (tensor<256x1024x2048xbf16>) -> tensor<256x1024x2048xbf16>
    %3739 = stablehlo.transpose %3738, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,2048,1024]{1,2,0}"} : (tensor<256x1024x2048xbf16>) -> tensor<256x2048x1024xbf16>
    %3740 = stablehlo.reshape %3739 : (tensor<256x2048x1024xbf16>) -> tensor<1x256x2048x1024xbf16>
    %3741 = stablehlo.get_tuple_element %3735[1] : (tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>) -> tensor<1x1x128x256xbf16>
    %3742 = stablehlo.reshape %3741 : (tensor<1x1x128x256xbf16>) -> tensor<8x16x1x256xbf16>
    %3743 = stablehlo.custom_call @tt.sparse_matmul(%3737, %3740, %3742) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True", nnz = "0"}} : (tensor<8x16x32x2048xbf16>, tensor<1x256x2048x1024xbf16>, tensor<8x16x1x256xbf16>) -> tensor<8x16x1x256x32x1024xbf16>
    %3744 = stablehlo.reshape %3743 : (tensor<8x16x1x256x32x1024xbf16>) -> tensor<8x16x256x32x1024xbf16>
    %3745 = stablehlo.slice %3744 [0:8, 0:16, 0:256, 0:32, 0:512] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %3746 = stablehlo.convert %3745 : (tensor<8x16x256x32x512xbf16>) -> tensor<8x16x256x32x512xf32>
    %3747 = stablehlo.logistic %3746 : tensor<8x16x256x32x512xf32>
    %3748 = stablehlo.multiply %3746, %3747 : tensor<8x16x256x32x512xf32>
    %3749 = stablehlo.convert %3748 : (tensor<8x16x256x32x512xf32>) -> tensor<8x16x256x32x512xbf16>
    %3750 = stablehlo.slice %3744 [0:8, 0:16, 0:256, 0:32, 512:1024] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %3751 = stablehlo.multiply %3749, %3750 : tensor<8x16x256x32x512xbf16>
    %3752 = stablehlo.reshape %3751 : (tensor<8x16x256x32x512xbf16>) -> tensor<128x256x32x512xbf16>
    %3753 = stablehlo.custom_call @tt.mark_argument(%arg49) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.mlp.experts.down_proj"}} : (tensor<256x2048x512xbf16>) -> tensor<256x2048x512xbf16>
    %3754 = stablehlo.transpose %3753, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,512,2048]{1,2,0}"} : (tensor<256x2048x512xbf16>) -> tensor<256x512x2048xbf16>
    %3755 = stablehlo.reshape %3754 : (tensor<256x512x2048xbf16>) -> tensor<1x256x512x2048xbf16>
    %3756 = stablehlo.custom_call @tt.sparse_matmul(%3752, %3755, %3741) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "True", is_input_b_sparse = "False", nnz = "0"}} : (tensor<128x256x32x512xbf16>, tensor<1x256x512x2048xbf16>, tensor<1x1x128x256xbf16>) -> tensor<128x256x32x2048xbf16>
    %3757 = stablehlo.transpose %3756, dims = [1, 0, 2, 3] {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, xla_shape = "bf16[256,128,32,2048]{3,2,0,1}"} : (tensor<128x256x32x2048xbf16>) -> tensor<256x128x32x2048xbf16>
    %3758 = stablehlo.reshape %3757 : (tensor<256x128x32x2048xbf16>) -> tensor<256x1x4096x2048xbf16>
    %3759 = stablehlo.custom_call @tt.all_to_all_combine(%3758, %3734, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8", num_experts_per_tok = "8", output_shard_dim = "2"}} : (tensor<256x1x4096x2048xbf16>, tensor<1x1x4096x8xi64>, tensor<1x1x256x8xui16>) -> tensor<8x1x512x2048xbf16>
    %3760 = stablehlo.transpose %3718, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[8,512]{0,1}"} : (tensor<512x8xf32>) -> tensor<8x512xf32>
    %3761 = stablehlo.reshape %3760 : (tensor<8x512xf32>) -> tensor<8x1x512x1xf32>
    %3762 = stablehlo.convert %3761 : (tensor<8x1x512x1xf32>) -> tensor<8x1x512x1xbf16>
    %3763 = stablehlo.reshape %3762 : (tensor<8x1x512x1xbf16>) -> tensor<8x1x512xbf16>
    %3764 = stablehlo.broadcast_in_dim %3763, dims = [0, 1, 2] : (tensor<8x1x512xbf16>) -> tensor<8x1x512x2048xbf16>
    %3765 = stablehlo.multiply %3759, %3764 : tensor<8x1x512x2048xbf16>
    %3766 = stablehlo.reduce(%3765 init: %cst_239) applies stablehlo.add across dimensions = [0] : (tensor<8x1x512x2048xbf16>, tensor<bf16>) -> tensor<1x512x2048xbf16>
    %3767 = stablehlo.reshape %3766 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %3768 = stablehlo.slice %3767 [0:494, 0:2048] : (tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %3769 = stablehlo.reshape %arg47 : (tensor<1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %3770 = stablehlo.custom_call @tt.mark_argument(%3769) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.mlp.shared_expert_gate.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %3771 = stablehlo.reshape %3770 : (tensor<1x1x2048xbf16>) -> tensor<2048x1xbf16>
    %3772 = stablehlo.dot_general %3700, %3771, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x1xbf16>) -> tensor<494x1xbf16>
    %3773 = stablehlo.logistic %3772 : tensor<494x1xbf16>
    %3774 = stablehlo.reshape %3773 : (tensor<494x1xbf16>) -> tensor<494xbf16>
    %3775 = stablehlo.broadcast_in_dim %3774, dims = [0] : (tensor<494xbf16>) -> tensor<494x2048xbf16>
    %3776 = stablehlo.reshape %arg46 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %3777 = stablehlo.custom_call @tt.mark_argument(%3776) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.mlp.shared_expert.gate_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %3778 = stablehlo.reshape %3777 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %3779 = stablehlo.transpose %3778, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %3780 = stablehlo.dot_general %3700, %3779, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %3781 = stablehlo.convert %3780 : (tensor<494x512xbf16>) -> tensor<494x512xf32>
    %3782 = stablehlo.logistic %3781 : tensor<494x512xf32>
    %3783 = stablehlo.multiply %3781, %3782 : tensor<494x512xf32>
    %3784 = stablehlo.convert %3783 : (tensor<494x512xf32>) -> tensor<494x512xbf16>
    %3785 = stablehlo.reshape %arg14 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %3786 = stablehlo.custom_call @tt.mark_argument(%3785) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.mlp.shared_expert.up_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %3787 = stablehlo.reshape %3786 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %3788 = stablehlo.transpose %3787, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %3789 = stablehlo.dot_general %3700, %3788, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %3790 = stablehlo.multiply %3784, %3789 : tensor<494x512xbf16>
    %3791 = stablehlo.reshape %arg13 : (tensor<2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %3792 = stablehlo.custom_call @tt.mark_argument(%3791) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.1.mlp.shared_expert.down_proj.weight"}} : (tensor<1x2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %3793 = stablehlo.reshape %3792 : (tensor<1x2048x512xbf16>) -> tensor<2048x512xbf16>
    %3794 = stablehlo.transpose %3793, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[512,2048]{0,1}"} : (tensor<2048x512xbf16>) -> tensor<512x2048xbf16>
    %3795 = stablehlo.dot_general %3790, %3794, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x512xbf16>, tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %3796 = stablehlo.multiply %3775, %3795 : tensor<494x2048xbf16>
    %3797 = stablehlo.add %3768, %3796 : tensor<494x2048xbf16>
    %3798 = stablehlo.reshape %3797 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %3799 = stablehlo.add %3693, %3798 : tensor<1x494x2048xbf16>
    %3800 = stablehlo.reshape %arg12 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %3801 = stablehlo.custom_call @tt.mark_argument(%3800) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.input_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %3802 = stablehlo.reshape %3801 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %3803 = stablehlo.convert %3802 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %3804 = stablehlo.add %3803, %cst_231 : tensor<2048xf32>
    %3805 = stablehlo.composite "tenstorrent.rms_norm" %3799, %3804 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_8} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %3806 = stablehlo.reshape %3805 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %3807 = stablehlo.reshape %arg57 : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %3808 = stablehlo.custom_call @tt.mark_argument(%3807) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.in_proj_qkv.weight"}} : (tensor<1x8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %3809 = stablehlo.reshape %3808 : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %3810 = stablehlo.transpose %3809, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,8192]{0,1}"} : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %3811 = stablehlo.dot_general %3806, %3810, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<494x8192xbf16>
    %3812 = stablehlo.reshape %3811 : (tensor<494x8192xbf16>) -> tensor<1x494x8192xbf16>
    %3813 = stablehlo.transpose %3812, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,8192,494]{1,2,0}"} : (tensor<1x494x8192xbf16>) -> tensor<1x8192x494xbf16>
    %3814 = stablehlo.reshape %3813 : (tensor<1x8192x494xbf16>) -> tensor<1x8192x1x494xbf16>
    %3815 = stablehlo.custom_call @tt.mark_argument(%arg56) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.conv1d.weight"}} : (tensor<8192x1x4xbf16>) -> tensor<8192x1x4xbf16>
    %3816 = stablehlo.reshape %3815 : (tensor<8192x1x4xbf16>) -> tensor<8192x1x1x4xbf16>
    %3817 = stablehlo.convolution(%3814, %3816) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 8192 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x8192x1x494xbf16>, tensor<8192x1x1x4xbf16>) -> tensor<1x8192x1x497xbf16>
    %3818 = stablehlo.reshape %3817 : (tensor<1x8192x1x497xbf16>) -> tensor<1x8192x497xbf16>
    %3819 = stablehlo.custom_call @tt.sharding_constraint(%3818) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>"}} : (tensor<1x8192x497xbf16>) -> tensor<1x8192x497xbf16>
    %3820 = stablehlo.slice %3819 [0:1, 0:8192, 0:494] : (tensor<1x8192x497xbf16>) -> tensor<1x8192x494xbf16>
    %3821 = stablehlo.convert %3820 : (tensor<1x8192x494xbf16>) -> tensor<1x8192x494xf32>
    %3822 = stablehlo.logistic %3821 : tensor<1x8192x494xf32>
    %3823 = stablehlo.multiply %3821, %3822 : tensor<1x8192x494xf32>
    %3824 = stablehlo.convert %3823 : (tensor<1x8192x494xf32>) -> tensor<1x8192x494xbf16>
    %3825 = stablehlo.transpose %3824, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,494,8192]{1,2,0}"} : (tensor<1x8192x494xbf16>) -> tensor<1x494x8192xbf16>
    %3826 = stablehlo.slice %3825 [0:1, 0:494, 0:2048] : (tensor<1x494x8192xbf16>) -> tensor<1x494x2048xbf16>
    %3827 = stablehlo.reshape %3826 : (tensor<1x494x2048xbf16>) -> tensor<1x494x16x128xbf16>
    %3828 = stablehlo.broadcast_in_dim %3827, dims = [0, 1, 2, 4] : (tensor<1x494x16x128xbf16>) -> tensor<1x494x16x2x128xbf16>
    %3829 = stablehlo.reshape %3828 : (tensor<1x494x16x2x128xbf16>) -> tensor<1x494x32x128xbf16>
    %3830 = stablehlo.multiply %3829, %3829 : tensor<1x494x32x128xbf16>
    %3831 = stablehlo.reduce(%3830 init: %cst_239) applies stablehlo.add across dimensions = [3] : (tensor<1x494x32x128xbf16>, tensor<bf16>) -> tensor<1x494x32xbf16>
    %3832 = stablehlo.reshape %3831 : (tensor<1x494x32xbf16>) -> tensor<1x494x32x1xbf16>
    %3833 = stablehlo.add %3832, %cst_230 : tensor<1x494x32x1xbf16>
    %3834 = stablehlo.rsqrt %3833 : tensor<1x494x32x1xbf16>
    %3835 = stablehlo.reshape %3834 : (tensor<1x494x32x1xbf16>) -> tensor<1x494x32xbf16>
    %3836 = stablehlo.broadcast_in_dim %3835, dims = [0, 1, 2] : (tensor<1x494x32xbf16>) -> tensor<1x494x32x128xbf16>
    %3837 = stablehlo.multiply %3829, %3836 : tensor<1x494x32x128xbf16>
    %3838 = stablehlo.transpose %3837, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %3839 = stablehlo.convert %3838 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %3840 = stablehlo.pad %3839, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %3841 = stablehlo.multiply %3840, %cst_229 : tensor<1x32x512x128xf32>
    %3842 = stablehlo.reshape %3841 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3843 = stablehlo.slice %3842 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3844 = stablehlo.reshape %3843 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3845 = stablehlo.reshape %arg54 : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %3846 = stablehlo.custom_call @tt.mark_argument(%3845) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.A_log"}} : (tensor<1x1x32xbf16>) -> tensor<1x1x32xbf16>
    %3847 = stablehlo.reshape %3846 : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
    %3848 = stablehlo.convert %3847 : (tensor<32xbf16>) -> tensor<32xf32>
    %3849 = stablehlo.exponential %3848 : tensor<32xf32>
    %3850 = stablehlo.negate %3849 : tensor<32xf32>
    %3851 = stablehlo.broadcast_in_dim %3850, dims = [2] : (tensor<32xf32>) -> tensor<1x494x32xf32>
    %3852 = stablehlo.reshape %arg53 : (tensor<32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %3853 = stablehlo.custom_call @tt.mark_argument(%3852) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.in_proj_a.weight"}} : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %3854 = stablehlo.reshape %3853 : (tensor<1x32x2048xbf16>) -> tensor<32x2048xbf16>
    %3855 = stablehlo.transpose %3854, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,32]{0,1}"} : (tensor<32x2048xbf16>) -> tensor<2048x32xbf16>
    %3856 = stablehlo.dot_general %3806, %3855, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x32xbf16>) -> tensor<494x32xbf16>
    %3857 = stablehlo.reshape %3856 : (tensor<494x32xbf16>) -> tensor<1x494x32xbf16>
    %3858 = stablehlo.convert %3857 : (tensor<1x494x32xbf16>) -> tensor<1x494x32xf32>
    %3859 = stablehlo.reshape %arg52 : (tensor<32xbf16>) -> tensor<1x1x32xbf16>
    %3860 = stablehlo.custom_call @tt.mark_argument(%3859) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.dt_bias"}} : (tensor<1x1x32xbf16>) -> tensor<1x1x32xbf16>
    %3861 = stablehlo.reshape %3860 : (tensor<1x1x32xbf16>) -> tensor<32xbf16>
    %3862 = stablehlo.convert %3861 : (tensor<32xbf16>) -> tensor<32xf32>
    %3863 = stablehlo.broadcast_in_dim %3862, dims = [2] : (tensor<32xf32>) -> tensor<1x494x32xf32>
    %3864 = stablehlo.add %3858, %3863 : tensor<1x494x32xf32>
    %3865 = stablehlo.compare  GT, %3864, %cst_228 : (tensor<1x494x32xf32>, tensor<1x494x32xf32>) -> tensor<1x494x32xi1>
    %3866 = stablehlo.exponential %3864 : tensor<1x494x32xf32>
    %3867 = stablehlo.log_plus_one %3866 : tensor<1x494x32xf32>
    %3868 = stablehlo.select %3865, %3864, %3867 : tensor<1x494x32xi1>, tensor<1x494x32xf32>
    %3869 = stablehlo.multiply %3851, %3868 : tensor<1x494x32xf32>
    %3870 = stablehlo.transpose %3869, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,494]{1,2,0}"} : (tensor<1x494x32xf32>) -> tensor<1x32x494xf32>
    %3871 = stablehlo.pad %3870, %cst_240, low = [0, 0, 0], high = [0, 0, 18], interior = [0, 0, 0] : (tensor<1x32x494xf32>, tensor<f32>) -> tensor<1x32x512xf32>
    %3872 = stablehlo.reshape %3871 : (tensor<1x32x512xf32>) -> tensor<1x32x8x64xf32>
    %3873 = "stablehlo.reduce_window"(%3872, %cst_240) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<[[0, 0], [0, 0], [0, 0], [63, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 64>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg75: tensor<f32>, %arg76: tensor<f32>):
      %5568 = stablehlo.add %arg75, %arg76 : tensor<f32>
      stablehlo.return %5568 : tensor<f32>
    }) : (tensor<1x32x8x64xf32>, tensor<f32>) -> tensor<1x32x8x64xf32>
    %3874 = stablehlo.slice %3873 [0:1, 0:32, 7:8, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3875 = stablehlo.reshape %3874 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %3876 = stablehlo.exponential %3875 : tensor<1x32x64x1xf32>
    %3877 = stablehlo.reshape %3876 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3878 = stablehlo.broadcast_in_dim %3877, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3879 = stablehlo.multiply %3844, %3878 : tensor<1x32x64x128xf32>
    %3880 = stablehlo.slice %3873 [0:1, 0:32, 0:1, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %3881 = stablehlo.reshape %3880 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %3882 = stablehlo.slice %3881 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %3883 = stablehlo.reshape %3882 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %3884 = stablehlo.exponential %3883 : tensor<1x32x1x1xf32>
    %3885 = stablehlo.reshape %3884 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %3886 = stablehlo.broadcast_in_dim %3885, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %3887 = stablehlo.multiply %3886, %cst_227 : tensor<1x32x128x128xf32>
    %3888 = stablehlo.slice %3825 [0:1, 0:494, 2048:4096] : (tensor<1x494x8192xbf16>) -> tensor<1x494x2048xbf16>
    %3889 = stablehlo.reshape %3888 : (tensor<1x494x2048xbf16>) -> tensor<1x494x16x128xbf16>
    %3890 = stablehlo.broadcast_in_dim %3889, dims = [0, 1, 2, 4] : (tensor<1x494x16x128xbf16>) -> tensor<1x494x16x2x128xbf16>
    %3891 = stablehlo.reshape %3890 : (tensor<1x494x16x2x128xbf16>) -> tensor<1x494x32x128xbf16>
    %3892 = stablehlo.multiply %3891, %3891 : tensor<1x494x32x128xbf16>
    %3893 = stablehlo.reduce(%3892 init: %cst_239) applies stablehlo.add across dimensions = [3] : (tensor<1x494x32x128xbf16>, tensor<bf16>) -> tensor<1x494x32xbf16>
    %3894 = stablehlo.reshape %3893 : (tensor<1x494x32xbf16>) -> tensor<1x494x32x1xbf16>
    %3895 = stablehlo.add %3894, %cst_230 : tensor<1x494x32x1xbf16>
    %3896 = stablehlo.rsqrt %3895 : tensor<1x494x32x1xbf16>
    %3897 = stablehlo.reshape %3896 : (tensor<1x494x32x1xbf16>) -> tensor<1x494x32xbf16>
    %3898 = stablehlo.broadcast_in_dim %3897, dims = [0, 1, 2] : (tensor<1x494x32xbf16>) -> tensor<1x494x32x128xbf16>
    %3899 = stablehlo.multiply %3891, %3898 : tensor<1x494x32x128xbf16>
    %3900 = stablehlo.transpose %3899, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %3901 = stablehlo.convert %3900 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %3902 = stablehlo.pad %3901, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %3903 = stablehlo.reshape %3902 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3904 = stablehlo.slice %3903 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %3905 = stablehlo.reshape %3904 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %3906 = stablehlo.reshape %3882 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %3907 = stablehlo.broadcast_in_dim %3906, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %3908 = stablehlo.subtract %3907, %3881 : tensor<1x32x64xf32>
    %3909 = stablehlo.exponential %3908 : tensor<1x32x64xf32>
    %3910 = stablehlo.broadcast_in_dim %3909, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %3911 = stablehlo.multiply %3905, %3910 : tensor<1x32x64x128xf32>
    %3912 = stablehlo.transpose %3911, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %3913 = stablehlo.reshape %arg55 : (tensor<32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %3914 = stablehlo.custom_call @tt.mark_argument(%3913) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.in_proj_b.weight"}} : (tensor<1x32x2048xbf16>) -> tensor<1x32x2048xbf16>
    %3915 = stablehlo.reshape %3914 : (tensor<1x32x2048xbf16>) -> tensor<32x2048xbf16>
    %3916 = stablehlo.transpose %3915, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,32]{0,1}"} : (tensor<32x2048xbf16>) -> tensor<2048x32xbf16>
    %3917 = stablehlo.dot_general %3806, %3916, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x32xbf16>) -> tensor<494x32xbf16>
    %3918 = stablehlo.reshape %3917 : (tensor<494x32xbf16>) -> tensor<1x494x32xbf16>
    %3919 = stablehlo.logistic %3918 : tensor<1x494x32xbf16>
    %3920 = stablehlo.transpose %3919, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[1,32,494]{1,2,0}"} : (tensor<1x494x32xbf16>) -> tensor<1x32x494xbf16>
    %3921 = stablehlo.convert %3920 {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "f32[1,32,494]{1,2,0}"} : (tensor<1x32x494xbf16>) -> tensor<1x32x494xf32>
    %3922 = stablehlo.pad %3921, %cst_240, low = [0, 0, 0], high = [0, 0, 18], interior = [0, 0, 0] : (tensor<1x32x494xf32>, tensor<f32>) -> tensor<1x32x512xf32>
    %3923 = stablehlo.broadcast_in_dim %3922, dims = [0, 1, 2] : (tensor<1x32x512xf32>) -> tensor<1x32x512x128xf32>
    %3924 = stablehlo.multiply %3902, %3923 : tensor<1x32x512x128xf32>
    %3925 = stablehlo.reshape %3924 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %3926 = stablehlo.transpose %3903, dims = [0, 1, 2, 4, 3] {result_layout = dense<[3, 4, 2, 1, 0]> : tensor<5xindex>, xla_shape = "f32[1,32,8,128,64]{3,4,2,1,0}"} : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x128x64xf32>
    %3927 = stablehlo.dot_general %3925, %3926, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x128xf32>, tensor<1x32x8x128x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3928 = stablehlo.broadcast_in_dim %3873, dims = [0, 1, 2, 3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3929 = stablehlo.broadcast_in_dim %3873, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3930 = stablehlo.subtract %3928, %3929 : tensor<1x32x8x64x64xf32>
    %3931 = stablehlo.select %645, %3930, %cst_98 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3932 = stablehlo.exponential %3931 : tensor<1x32x8x64x64xf32>
    %3933 = stablehlo.select %645, %3932, %cst_98 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3934 = stablehlo.multiply %3927, %3933 : tensor<1x32x8x64x64xf32>
    %3935 = stablehlo.select %627, %cst_98, %3934 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3936 = stablehlo.negate %3935 : tensor<1x32x8x64x64xf32>
    %3937 = stablehlo.slice %3936 [0:1, 0:32, 0:8, 1:2, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3938 = stablehlo.reshape %3937 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3939 = stablehlo.slice %3938 [0:1, 0:32, 0:8, 0:1] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x1xf32>
    %3940 = stablehlo.reshape %3939 : (tensor<1x32x8x1xf32>) -> tensor<1x32x8x1x1xf32>
    %3941 = stablehlo.slice %3936 [0:1, 0:32, 0:8, 0:1, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3942 = stablehlo.slice %3941 [0:1, 0:32, 0:8, 0:1, 0:1] : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x1x1xf32>
    %3943 = stablehlo.multiply %3940, %3942 : tensor<1x32x8x1x1xf32>
    %3944 = stablehlo.reduce(%3943 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x1x1xf32>, tensor<f32>) -> tensor<1x32x8x1xf32>
    %3945 = stablehlo.add %3939, %3944 : tensor<1x32x8x1xf32>
    %3946 = "stablehlo.gather"(%3945, %670) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x1xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3947 = stablehlo.select %619, %cst_224, %3946 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3948 = stablehlo.select %616, %3947, %3938 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3949 = stablehlo.broadcast_in_dim %3948, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3950 = stablehlo.select %612, %3949, %3936 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3951 = stablehlo.slice %3950 [0:1, 0:32, 0:8, 2:3, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3952 = stablehlo.reshape %3951 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3953 = stablehlo.slice %3952 [0:1, 0:32, 0:8, 0:2] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x2xf32>
    %3954 = stablehlo.broadcast_in_dim %3953, dims = [0, 1, 2, 3] : (tensor<1x32x8x2xf32>) -> tensor<1x32x8x2x2xf32>
    %3955 = stablehlo.slice %3950 [0:1, 0:32, 0:8, 0:2, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x2x64xf32>
    %3956 = stablehlo.slice %3955 [0:1, 0:32, 0:8, 0:2, 0:2] : (tensor<1x32x8x2x64xf32>) -> tensor<1x32x8x2x2xf32>
    %3957 = stablehlo.multiply %3954, %3956 : tensor<1x32x8x2x2xf32>
    %3958 = stablehlo.reduce(%3957 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x2x2xf32>, tensor<f32>) -> tensor<1x32x8x2xf32>
    %3959 = stablehlo.add %3953, %3958 : tensor<1x32x8x2xf32>
    %3960 = "stablehlo.gather"(%3959, %689) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x2xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3961 = stablehlo.select %611, %cst_224, %3960 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3962 = stablehlo.select %608, %3961, %3952 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3963 = stablehlo.broadcast_in_dim %3962, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3964 = stablehlo.select %604, %3963, %3950 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3965 = stablehlo.slice %3964 [0:1, 0:32, 0:8, 3:4, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3966 = stablehlo.reshape %3965 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3967 = stablehlo.slice %3966 [0:1, 0:32, 0:8, 0:3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x3xf32>
    %3968 = stablehlo.broadcast_in_dim %3967, dims = [0, 1, 2, 3] : (tensor<1x32x8x3xf32>) -> tensor<1x32x8x3x3xf32>
    %3969 = stablehlo.slice %3964 [0:1, 0:32, 0:8, 0:3, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x3x64xf32>
    %3970 = stablehlo.slice %3969 [0:1, 0:32, 0:8, 0:3, 0:3] : (tensor<1x32x8x3x64xf32>) -> tensor<1x32x8x3x3xf32>
    %3971 = stablehlo.multiply %3968, %3970 : tensor<1x32x8x3x3xf32>
    %3972 = stablehlo.reduce(%3971 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x3x3xf32>, tensor<f32>) -> tensor<1x32x8x3xf32>
    %3973 = stablehlo.add %3967, %3972 : tensor<1x32x8x3xf32>
    %3974 = "stablehlo.gather"(%3973, %708) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x3xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3975 = stablehlo.select %603, %cst_224, %3974 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3976 = stablehlo.select %600, %3975, %3966 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3977 = stablehlo.broadcast_in_dim %3976, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3978 = stablehlo.select %596, %3977, %3964 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3979 = stablehlo.slice %3978 [0:1, 0:32, 0:8, 4:5, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3980 = stablehlo.reshape %3979 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3981 = stablehlo.slice %3980 [0:1, 0:32, 0:8, 0:4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x4xf32>
    %3982 = stablehlo.broadcast_in_dim %3981, dims = [0, 1, 2, 3] : (tensor<1x32x8x4xf32>) -> tensor<1x32x8x4x4xf32>
    %3983 = stablehlo.slice %3978 [0:1, 0:32, 0:8, 0:4, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x4x64xf32>
    %3984 = stablehlo.slice %3983 [0:1, 0:32, 0:8, 0:4, 0:4] : (tensor<1x32x8x4x64xf32>) -> tensor<1x32x8x4x4xf32>
    %3985 = stablehlo.multiply %3982, %3984 : tensor<1x32x8x4x4xf32>
    %3986 = stablehlo.reduce(%3985 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x4x4xf32>, tensor<f32>) -> tensor<1x32x8x4xf32>
    %3987 = stablehlo.add %3981, %3986 : tensor<1x32x8x4xf32>
    %3988 = "stablehlo.gather"(%3987, %727) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x4xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %3989 = stablehlo.select %595, %cst_224, %3988 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3990 = stablehlo.select %592, %3989, %3980 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %3991 = stablehlo.broadcast_in_dim %3990, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %3992 = stablehlo.select %588, %3991, %3978 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %3993 = stablehlo.slice %3992 [0:1, 0:32, 0:8, 5:6, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %3994 = stablehlo.reshape %3993 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %3995 = stablehlo.slice %3994 [0:1, 0:32, 0:8, 0:5] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x5xf32>
    %3996 = stablehlo.broadcast_in_dim %3995, dims = [0, 1, 2, 3] : (tensor<1x32x8x5xf32>) -> tensor<1x32x8x5x5xf32>
    %3997 = stablehlo.slice %3992 [0:1, 0:32, 0:8, 0:5, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x5x64xf32>
    %3998 = stablehlo.slice %3997 [0:1, 0:32, 0:8, 0:5, 0:5] : (tensor<1x32x8x5x64xf32>) -> tensor<1x32x8x5x5xf32>
    %3999 = stablehlo.multiply %3996, %3998 : tensor<1x32x8x5x5xf32>
    %4000 = stablehlo.reduce(%3999 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x5x5xf32>, tensor<f32>) -> tensor<1x32x8x5xf32>
    %4001 = stablehlo.add %3995, %4000 : tensor<1x32x8x5xf32>
    %4002 = "stablehlo.gather"(%4001, %746) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x5xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4003 = stablehlo.select %587, %cst_224, %4002 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4004 = stablehlo.select %584, %4003, %3994 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4005 = stablehlo.broadcast_in_dim %4004, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4006 = stablehlo.select %580, %4005, %3992 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4007 = stablehlo.slice %4006 [0:1, 0:32, 0:8, 6:7, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4008 = stablehlo.reshape %4007 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4009 = stablehlo.slice %4008 [0:1, 0:32, 0:8, 0:6] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x6xf32>
    %4010 = stablehlo.broadcast_in_dim %4009, dims = [0, 1, 2, 3] : (tensor<1x32x8x6xf32>) -> tensor<1x32x8x6x6xf32>
    %4011 = stablehlo.slice %4006 [0:1, 0:32, 0:8, 0:6, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x6x64xf32>
    %4012 = stablehlo.slice %4011 [0:1, 0:32, 0:8, 0:6, 0:6] : (tensor<1x32x8x6x64xf32>) -> tensor<1x32x8x6x6xf32>
    %4013 = stablehlo.multiply %4010, %4012 : tensor<1x32x8x6x6xf32>
    %4014 = stablehlo.reduce(%4013 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x6x6xf32>, tensor<f32>) -> tensor<1x32x8x6xf32>
    %4015 = stablehlo.add %4009, %4014 : tensor<1x32x8x6xf32>
    %4016 = "stablehlo.gather"(%4015, %765) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x6xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4017 = stablehlo.select %579, %cst_224, %4016 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4018 = stablehlo.select %576, %4017, %4008 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4019 = stablehlo.broadcast_in_dim %4018, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4020 = stablehlo.select %572, %4019, %4006 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4021 = stablehlo.slice %4020 [0:1, 0:32, 0:8, 7:8, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4022 = stablehlo.reshape %4021 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4023 = stablehlo.slice %4022 [0:1, 0:32, 0:8, 0:7] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x7xf32>
    %4024 = stablehlo.broadcast_in_dim %4023, dims = [0, 1, 2, 3] : (tensor<1x32x8x7xf32>) -> tensor<1x32x8x7x7xf32>
    %4025 = stablehlo.slice %4020 [0:1, 0:32, 0:8, 0:7, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x7x64xf32>
    %4026 = stablehlo.slice %4025 [0:1, 0:32, 0:8, 0:7, 0:7] : (tensor<1x32x8x7x64xf32>) -> tensor<1x32x8x7x7xf32>
    %4027 = stablehlo.multiply %4024, %4026 : tensor<1x32x8x7x7xf32>
    %4028 = stablehlo.reduce(%4027 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x7x7xf32>, tensor<f32>) -> tensor<1x32x8x7xf32>
    %4029 = stablehlo.add %4023, %4028 : tensor<1x32x8x7xf32>
    %4030 = "stablehlo.gather"(%4029, %784) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x7xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4031 = stablehlo.select %571, %cst_224, %4030 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4032 = stablehlo.select %568, %4031, %4022 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4033 = stablehlo.broadcast_in_dim %4032, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4034 = stablehlo.select %564, %4033, %4020 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4035 = stablehlo.slice %4034 [0:1, 0:32, 0:8, 8:9, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4036 = stablehlo.reshape %4035 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4037 = stablehlo.slice %4036 [0:1, 0:32, 0:8, 0:8] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x8xf32>
    %4038 = stablehlo.broadcast_in_dim %4037, dims = [0, 1, 2, 3] : (tensor<1x32x8x8xf32>) -> tensor<1x32x8x8x8xf32>
    %4039 = stablehlo.slice %4034 [0:1, 0:32, 0:8, 0:8, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x8x64xf32>
    %4040 = stablehlo.slice %4039 [0:1, 0:32, 0:8, 0:8, 0:8] : (tensor<1x32x8x8x64xf32>) -> tensor<1x32x8x8x8xf32>
    %4041 = stablehlo.multiply %4038, %4040 : tensor<1x32x8x8x8xf32>
    %4042 = stablehlo.reduce(%4041 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x8x8xf32>, tensor<f32>) -> tensor<1x32x8x8xf32>
    %4043 = stablehlo.add %4037, %4042 : tensor<1x32x8x8xf32>
    %4044 = "stablehlo.gather"(%4043, %803) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x8xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4045 = stablehlo.select %563, %cst_224, %4044 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4046 = stablehlo.select %560, %4045, %4036 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4047 = stablehlo.broadcast_in_dim %4046, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4048 = stablehlo.select %556, %4047, %4034 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4049 = stablehlo.slice %4048 [0:1, 0:32, 0:8, 9:10, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4050 = stablehlo.reshape %4049 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4051 = stablehlo.slice %4050 [0:1, 0:32, 0:8, 0:9] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x9xf32>
    %4052 = stablehlo.broadcast_in_dim %4051, dims = [0, 1, 2, 3] : (tensor<1x32x8x9xf32>) -> tensor<1x32x8x9x9xf32>
    %4053 = stablehlo.slice %4048 [0:1, 0:32, 0:8, 0:9, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x9x64xf32>
    %4054 = stablehlo.slice %4053 [0:1, 0:32, 0:8, 0:9, 0:9] : (tensor<1x32x8x9x64xf32>) -> tensor<1x32x8x9x9xf32>
    %4055 = stablehlo.multiply %4052, %4054 : tensor<1x32x8x9x9xf32>
    %4056 = stablehlo.reduce(%4055 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x9x9xf32>, tensor<f32>) -> tensor<1x32x8x9xf32>
    %4057 = stablehlo.add %4051, %4056 : tensor<1x32x8x9xf32>
    %4058 = "stablehlo.gather"(%4057, %822) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x9xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4059 = stablehlo.select %555, %cst_224, %4058 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4060 = stablehlo.select %552, %4059, %4050 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4061 = stablehlo.broadcast_in_dim %4060, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4062 = stablehlo.select %548, %4061, %4048 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4063 = stablehlo.slice %4062 [0:1, 0:32, 0:8, 10:11, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4064 = stablehlo.reshape %4063 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4065 = stablehlo.slice %4064 [0:1, 0:32, 0:8, 0:10] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x10xf32>
    %4066 = stablehlo.broadcast_in_dim %4065, dims = [0, 1, 2, 3] : (tensor<1x32x8x10xf32>) -> tensor<1x32x8x10x10xf32>
    %4067 = stablehlo.slice %4062 [0:1, 0:32, 0:8, 0:10, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x10x64xf32>
    %4068 = stablehlo.slice %4067 [0:1, 0:32, 0:8, 0:10, 0:10] : (tensor<1x32x8x10x64xf32>) -> tensor<1x32x8x10x10xf32>
    %4069 = stablehlo.multiply %4066, %4068 : tensor<1x32x8x10x10xf32>
    %4070 = stablehlo.reduce(%4069 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x10x10xf32>, tensor<f32>) -> tensor<1x32x8x10xf32>
    %4071 = stablehlo.add %4065, %4070 : tensor<1x32x8x10xf32>
    %4072 = "stablehlo.gather"(%4071, %841) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x10xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4073 = stablehlo.select %547, %cst_224, %4072 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4074 = stablehlo.select %544, %4073, %4064 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4075 = stablehlo.broadcast_in_dim %4074, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4076 = stablehlo.select %540, %4075, %4062 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4077 = stablehlo.slice %4076 [0:1, 0:32, 0:8, 11:12, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4078 = stablehlo.reshape %4077 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4079 = stablehlo.slice %4078 [0:1, 0:32, 0:8, 0:11] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x11xf32>
    %4080 = stablehlo.broadcast_in_dim %4079, dims = [0, 1, 2, 3] : (tensor<1x32x8x11xf32>) -> tensor<1x32x8x11x11xf32>
    %4081 = stablehlo.slice %4076 [0:1, 0:32, 0:8, 0:11, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x11x64xf32>
    %4082 = stablehlo.slice %4081 [0:1, 0:32, 0:8, 0:11, 0:11] : (tensor<1x32x8x11x64xf32>) -> tensor<1x32x8x11x11xf32>
    %4083 = stablehlo.multiply %4080, %4082 : tensor<1x32x8x11x11xf32>
    %4084 = stablehlo.reduce(%4083 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x11x11xf32>, tensor<f32>) -> tensor<1x32x8x11xf32>
    %4085 = stablehlo.add %4079, %4084 : tensor<1x32x8x11xf32>
    %4086 = "stablehlo.gather"(%4085, %860) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x11xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4087 = stablehlo.select %539, %cst_224, %4086 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4088 = stablehlo.select %536, %4087, %4078 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4089 = stablehlo.broadcast_in_dim %4088, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4090 = stablehlo.select %532, %4089, %4076 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4091 = stablehlo.slice %4090 [0:1, 0:32, 0:8, 12:13, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4092 = stablehlo.reshape %4091 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4093 = stablehlo.slice %4092 [0:1, 0:32, 0:8, 0:12] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x12xf32>
    %4094 = stablehlo.broadcast_in_dim %4093, dims = [0, 1, 2, 3] : (tensor<1x32x8x12xf32>) -> tensor<1x32x8x12x12xf32>
    %4095 = stablehlo.slice %4090 [0:1, 0:32, 0:8, 0:12, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x12x64xf32>
    %4096 = stablehlo.slice %4095 [0:1, 0:32, 0:8, 0:12, 0:12] : (tensor<1x32x8x12x64xf32>) -> tensor<1x32x8x12x12xf32>
    %4097 = stablehlo.multiply %4094, %4096 : tensor<1x32x8x12x12xf32>
    %4098 = stablehlo.reduce(%4097 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x12x12xf32>, tensor<f32>) -> tensor<1x32x8x12xf32>
    %4099 = stablehlo.add %4093, %4098 : tensor<1x32x8x12xf32>
    %4100 = "stablehlo.gather"(%4099, %879) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x12xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4101 = stablehlo.select %531, %cst_224, %4100 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4102 = stablehlo.select %528, %4101, %4092 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4103 = stablehlo.broadcast_in_dim %4102, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4104 = stablehlo.select %524, %4103, %4090 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4105 = stablehlo.slice %4104 [0:1, 0:32, 0:8, 13:14, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4106 = stablehlo.reshape %4105 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4107 = stablehlo.slice %4106 [0:1, 0:32, 0:8, 0:13] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x13xf32>
    %4108 = stablehlo.broadcast_in_dim %4107, dims = [0, 1, 2, 3] : (tensor<1x32x8x13xf32>) -> tensor<1x32x8x13x13xf32>
    %4109 = stablehlo.slice %4104 [0:1, 0:32, 0:8, 0:13, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x13x64xf32>
    %4110 = stablehlo.slice %4109 [0:1, 0:32, 0:8, 0:13, 0:13] : (tensor<1x32x8x13x64xf32>) -> tensor<1x32x8x13x13xf32>
    %4111 = stablehlo.multiply %4108, %4110 : tensor<1x32x8x13x13xf32>
    %4112 = stablehlo.reduce(%4111 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x13x13xf32>, tensor<f32>) -> tensor<1x32x8x13xf32>
    %4113 = stablehlo.add %4107, %4112 : tensor<1x32x8x13xf32>
    %4114 = "stablehlo.gather"(%4113, %898) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x13xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4115 = stablehlo.select %523, %cst_224, %4114 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4116 = stablehlo.select %520, %4115, %4106 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4117 = stablehlo.broadcast_in_dim %4116, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4118 = stablehlo.select %516, %4117, %4104 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4119 = stablehlo.slice %4118 [0:1, 0:32, 0:8, 14:15, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4120 = stablehlo.reshape %4119 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4121 = stablehlo.slice %4120 [0:1, 0:32, 0:8, 0:14] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x14xf32>
    %4122 = stablehlo.broadcast_in_dim %4121, dims = [0, 1, 2, 3] : (tensor<1x32x8x14xf32>) -> tensor<1x32x8x14x14xf32>
    %4123 = stablehlo.slice %4118 [0:1, 0:32, 0:8, 0:14, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x14x64xf32>
    %4124 = stablehlo.slice %4123 [0:1, 0:32, 0:8, 0:14, 0:14] : (tensor<1x32x8x14x64xf32>) -> tensor<1x32x8x14x14xf32>
    %4125 = stablehlo.multiply %4122, %4124 : tensor<1x32x8x14x14xf32>
    %4126 = stablehlo.reduce(%4125 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x14x14xf32>, tensor<f32>) -> tensor<1x32x8x14xf32>
    %4127 = stablehlo.add %4121, %4126 : tensor<1x32x8x14xf32>
    %4128 = "stablehlo.gather"(%4127, %917) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x14xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4129 = stablehlo.select %515, %cst_224, %4128 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4130 = stablehlo.select %512, %4129, %4120 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4131 = stablehlo.broadcast_in_dim %4130, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4132 = stablehlo.select %508, %4131, %4118 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4133 = stablehlo.slice %4132 [0:1, 0:32, 0:8, 15:16, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4134 = stablehlo.reshape %4133 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4135 = stablehlo.slice %4134 [0:1, 0:32, 0:8, 0:15] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x15xf32>
    %4136 = stablehlo.broadcast_in_dim %4135, dims = [0, 1, 2, 3] : (tensor<1x32x8x15xf32>) -> tensor<1x32x8x15x15xf32>
    %4137 = stablehlo.slice %4132 [0:1, 0:32, 0:8, 0:15, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x15x64xf32>
    %4138 = stablehlo.slice %4137 [0:1, 0:32, 0:8, 0:15, 0:15] : (tensor<1x32x8x15x64xf32>) -> tensor<1x32x8x15x15xf32>
    %4139 = stablehlo.multiply %4136, %4138 : tensor<1x32x8x15x15xf32>
    %4140 = stablehlo.reduce(%4139 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x15x15xf32>, tensor<f32>) -> tensor<1x32x8x15xf32>
    %4141 = stablehlo.add %4135, %4140 : tensor<1x32x8x15xf32>
    %4142 = "stablehlo.gather"(%4141, %936) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x15xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4143 = stablehlo.select %507, %cst_224, %4142 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4144 = stablehlo.select %504, %4143, %4134 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4145 = stablehlo.broadcast_in_dim %4144, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4146 = stablehlo.select %500, %4145, %4132 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4147 = stablehlo.slice %4146 [0:1, 0:32, 0:8, 16:17, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4148 = stablehlo.reshape %4147 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4149 = stablehlo.slice %4148 [0:1, 0:32, 0:8, 0:16] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x16xf32>
    %4150 = stablehlo.broadcast_in_dim %4149, dims = [0, 1, 2, 3] : (tensor<1x32x8x16xf32>) -> tensor<1x32x8x16x16xf32>
    %4151 = stablehlo.slice %4146 [0:1, 0:32, 0:8, 0:16, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x16x64xf32>
    %4152 = stablehlo.slice %4151 [0:1, 0:32, 0:8, 0:16, 0:16] : (tensor<1x32x8x16x64xf32>) -> tensor<1x32x8x16x16xf32>
    %4153 = stablehlo.multiply %4150, %4152 : tensor<1x32x8x16x16xf32>
    %4154 = stablehlo.reduce(%4153 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x16x16xf32>, tensor<f32>) -> tensor<1x32x8x16xf32>
    %4155 = stablehlo.add %4149, %4154 : tensor<1x32x8x16xf32>
    %4156 = "stablehlo.gather"(%4155, %955) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x16xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4157 = stablehlo.select %499, %cst_224, %4156 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4158 = stablehlo.select %496, %4157, %4148 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4159 = stablehlo.broadcast_in_dim %4158, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4160 = stablehlo.select %492, %4159, %4146 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4161 = stablehlo.slice %4160 [0:1, 0:32, 0:8, 17:18, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4162 = stablehlo.reshape %4161 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4163 = stablehlo.slice %4162 [0:1, 0:32, 0:8, 0:17] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x17xf32>
    %4164 = stablehlo.broadcast_in_dim %4163, dims = [0, 1, 2, 3] : (tensor<1x32x8x17xf32>) -> tensor<1x32x8x17x17xf32>
    %4165 = stablehlo.slice %4160 [0:1, 0:32, 0:8, 0:17, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x17x64xf32>
    %4166 = stablehlo.slice %4165 [0:1, 0:32, 0:8, 0:17, 0:17] : (tensor<1x32x8x17x64xf32>) -> tensor<1x32x8x17x17xf32>
    %4167 = stablehlo.multiply %4164, %4166 : tensor<1x32x8x17x17xf32>
    %4168 = stablehlo.reduce(%4167 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x17x17xf32>, tensor<f32>) -> tensor<1x32x8x17xf32>
    %4169 = stablehlo.add %4163, %4168 : tensor<1x32x8x17xf32>
    %4170 = "stablehlo.gather"(%4169, %974) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x17xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4171 = stablehlo.select %491, %cst_224, %4170 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4172 = stablehlo.select %488, %4171, %4162 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4173 = stablehlo.broadcast_in_dim %4172, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4174 = stablehlo.select %484, %4173, %4160 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4175 = stablehlo.slice %4174 [0:1, 0:32, 0:8, 18:19, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4176 = stablehlo.reshape %4175 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4177 = stablehlo.slice %4176 [0:1, 0:32, 0:8, 0:18] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x18xf32>
    %4178 = stablehlo.broadcast_in_dim %4177, dims = [0, 1, 2, 3] : (tensor<1x32x8x18xf32>) -> tensor<1x32x8x18x18xf32>
    %4179 = stablehlo.slice %4174 [0:1, 0:32, 0:8, 0:18, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x18x64xf32>
    %4180 = stablehlo.slice %4179 [0:1, 0:32, 0:8, 0:18, 0:18] : (tensor<1x32x8x18x64xf32>) -> tensor<1x32x8x18x18xf32>
    %4181 = stablehlo.multiply %4178, %4180 : tensor<1x32x8x18x18xf32>
    %4182 = stablehlo.reduce(%4181 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x18x18xf32>, tensor<f32>) -> tensor<1x32x8x18xf32>
    %4183 = stablehlo.add %4177, %4182 : tensor<1x32x8x18xf32>
    %4184 = "stablehlo.gather"(%4183, %993) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x18xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4185 = stablehlo.select %483, %cst_224, %4184 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4186 = stablehlo.select %480, %4185, %4176 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4187 = stablehlo.broadcast_in_dim %4186, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4188 = stablehlo.select %476, %4187, %4174 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4189 = stablehlo.slice %4188 [0:1, 0:32, 0:8, 19:20, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4190 = stablehlo.reshape %4189 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4191 = stablehlo.slice %4190 [0:1, 0:32, 0:8, 0:19] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x19xf32>
    %4192 = stablehlo.broadcast_in_dim %4191, dims = [0, 1, 2, 3] : (tensor<1x32x8x19xf32>) -> tensor<1x32x8x19x19xf32>
    %4193 = stablehlo.slice %4188 [0:1, 0:32, 0:8, 0:19, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x19x64xf32>
    %4194 = stablehlo.slice %4193 [0:1, 0:32, 0:8, 0:19, 0:19] : (tensor<1x32x8x19x64xf32>) -> tensor<1x32x8x19x19xf32>
    %4195 = stablehlo.multiply %4192, %4194 : tensor<1x32x8x19x19xf32>
    %4196 = stablehlo.reduce(%4195 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x19x19xf32>, tensor<f32>) -> tensor<1x32x8x19xf32>
    %4197 = stablehlo.add %4191, %4196 : tensor<1x32x8x19xf32>
    %4198 = "stablehlo.gather"(%4197, %1012) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x19xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4199 = stablehlo.select %475, %cst_224, %4198 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4200 = stablehlo.select %472, %4199, %4190 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4201 = stablehlo.broadcast_in_dim %4200, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4202 = stablehlo.select %468, %4201, %4188 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4203 = stablehlo.slice %4202 [0:1, 0:32, 0:8, 20:21, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4204 = stablehlo.reshape %4203 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4205 = stablehlo.slice %4204 [0:1, 0:32, 0:8, 0:20] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x20xf32>
    %4206 = stablehlo.broadcast_in_dim %4205, dims = [0, 1, 2, 3] : (tensor<1x32x8x20xf32>) -> tensor<1x32x8x20x20xf32>
    %4207 = stablehlo.slice %4202 [0:1, 0:32, 0:8, 0:20, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x20x64xf32>
    %4208 = stablehlo.slice %4207 [0:1, 0:32, 0:8, 0:20, 0:20] : (tensor<1x32x8x20x64xf32>) -> tensor<1x32x8x20x20xf32>
    %4209 = stablehlo.multiply %4206, %4208 : tensor<1x32x8x20x20xf32>
    %4210 = stablehlo.reduce(%4209 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x20x20xf32>, tensor<f32>) -> tensor<1x32x8x20xf32>
    %4211 = stablehlo.add %4205, %4210 : tensor<1x32x8x20xf32>
    %4212 = "stablehlo.gather"(%4211, %1031) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x20xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4213 = stablehlo.select %467, %cst_224, %4212 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4214 = stablehlo.select %464, %4213, %4204 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4215 = stablehlo.broadcast_in_dim %4214, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4216 = stablehlo.select %460, %4215, %4202 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4217 = stablehlo.slice %4216 [0:1, 0:32, 0:8, 21:22, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4218 = stablehlo.reshape %4217 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4219 = stablehlo.slice %4218 [0:1, 0:32, 0:8, 0:21] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x21xf32>
    %4220 = stablehlo.broadcast_in_dim %4219, dims = [0, 1, 2, 3] : (tensor<1x32x8x21xf32>) -> tensor<1x32x8x21x21xf32>
    %4221 = stablehlo.slice %4216 [0:1, 0:32, 0:8, 0:21, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x21x64xf32>
    %4222 = stablehlo.slice %4221 [0:1, 0:32, 0:8, 0:21, 0:21] : (tensor<1x32x8x21x64xf32>) -> tensor<1x32x8x21x21xf32>
    %4223 = stablehlo.multiply %4220, %4222 : tensor<1x32x8x21x21xf32>
    %4224 = stablehlo.reduce(%4223 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x21x21xf32>, tensor<f32>) -> tensor<1x32x8x21xf32>
    %4225 = stablehlo.add %4219, %4224 : tensor<1x32x8x21xf32>
    %4226 = "stablehlo.gather"(%4225, %1050) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x21xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4227 = stablehlo.select %459, %cst_224, %4226 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4228 = stablehlo.select %456, %4227, %4218 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4229 = stablehlo.broadcast_in_dim %4228, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4230 = stablehlo.select %452, %4229, %4216 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4231 = stablehlo.slice %4230 [0:1, 0:32, 0:8, 22:23, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4232 = stablehlo.reshape %4231 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4233 = stablehlo.slice %4232 [0:1, 0:32, 0:8, 0:22] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x22xf32>
    %4234 = stablehlo.broadcast_in_dim %4233, dims = [0, 1, 2, 3] : (tensor<1x32x8x22xf32>) -> tensor<1x32x8x22x22xf32>
    %4235 = stablehlo.slice %4230 [0:1, 0:32, 0:8, 0:22, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x22x64xf32>
    %4236 = stablehlo.slice %4235 [0:1, 0:32, 0:8, 0:22, 0:22] : (tensor<1x32x8x22x64xf32>) -> tensor<1x32x8x22x22xf32>
    %4237 = stablehlo.multiply %4234, %4236 : tensor<1x32x8x22x22xf32>
    %4238 = stablehlo.reduce(%4237 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x22x22xf32>, tensor<f32>) -> tensor<1x32x8x22xf32>
    %4239 = stablehlo.add %4233, %4238 : tensor<1x32x8x22xf32>
    %4240 = "stablehlo.gather"(%4239, %1069) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x22xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4241 = stablehlo.select %451, %cst_224, %4240 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4242 = stablehlo.select %448, %4241, %4232 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4243 = stablehlo.broadcast_in_dim %4242, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4244 = stablehlo.select %444, %4243, %4230 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4245 = stablehlo.slice %4244 [0:1, 0:32, 0:8, 23:24, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4246 = stablehlo.reshape %4245 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4247 = stablehlo.slice %4246 [0:1, 0:32, 0:8, 0:23] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x23xf32>
    %4248 = stablehlo.broadcast_in_dim %4247, dims = [0, 1, 2, 3] : (tensor<1x32x8x23xf32>) -> tensor<1x32x8x23x23xf32>
    %4249 = stablehlo.slice %4244 [0:1, 0:32, 0:8, 0:23, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x23x64xf32>
    %4250 = stablehlo.slice %4249 [0:1, 0:32, 0:8, 0:23, 0:23] : (tensor<1x32x8x23x64xf32>) -> tensor<1x32x8x23x23xf32>
    %4251 = stablehlo.multiply %4248, %4250 : tensor<1x32x8x23x23xf32>
    %4252 = stablehlo.reduce(%4251 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x23x23xf32>, tensor<f32>) -> tensor<1x32x8x23xf32>
    %4253 = stablehlo.add %4247, %4252 : tensor<1x32x8x23xf32>
    %4254 = "stablehlo.gather"(%4253, %1088) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x23xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4255 = stablehlo.select %443, %cst_224, %4254 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4256 = stablehlo.select %440, %4255, %4246 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4257 = stablehlo.broadcast_in_dim %4256, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4258 = stablehlo.select %436, %4257, %4244 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4259 = stablehlo.slice %4258 [0:1, 0:32, 0:8, 24:25, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4260 = stablehlo.reshape %4259 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4261 = stablehlo.slice %4260 [0:1, 0:32, 0:8, 0:24] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x24xf32>
    %4262 = stablehlo.broadcast_in_dim %4261, dims = [0, 1, 2, 3] : (tensor<1x32x8x24xf32>) -> tensor<1x32x8x24x24xf32>
    %4263 = stablehlo.slice %4258 [0:1, 0:32, 0:8, 0:24, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x24x64xf32>
    %4264 = stablehlo.slice %4263 [0:1, 0:32, 0:8, 0:24, 0:24] : (tensor<1x32x8x24x64xf32>) -> tensor<1x32x8x24x24xf32>
    %4265 = stablehlo.multiply %4262, %4264 : tensor<1x32x8x24x24xf32>
    %4266 = stablehlo.reduce(%4265 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x24x24xf32>, tensor<f32>) -> tensor<1x32x8x24xf32>
    %4267 = stablehlo.add %4261, %4266 : tensor<1x32x8x24xf32>
    %4268 = "stablehlo.gather"(%4267, %1107) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x24xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4269 = stablehlo.select %435, %cst_224, %4268 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4270 = stablehlo.select %432, %4269, %4260 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4271 = stablehlo.broadcast_in_dim %4270, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4272 = stablehlo.select %428, %4271, %4258 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4273 = stablehlo.slice %4272 [0:1, 0:32, 0:8, 25:26, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4274 = stablehlo.reshape %4273 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4275 = stablehlo.slice %4274 [0:1, 0:32, 0:8, 0:25] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x25xf32>
    %4276 = stablehlo.broadcast_in_dim %4275, dims = [0, 1, 2, 3] : (tensor<1x32x8x25xf32>) -> tensor<1x32x8x25x25xf32>
    %4277 = stablehlo.slice %4272 [0:1, 0:32, 0:8, 0:25, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x25x64xf32>
    %4278 = stablehlo.slice %4277 [0:1, 0:32, 0:8, 0:25, 0:25] : (tensor<1x32x8x25x64xf32>) -> tensor<1x32x8x25x25xf32>
    %4279 = stablehlo.multiply %4276, %4278 : tensor<1x32x8x25x25xf32>
    %4280 = stablehlo.reduce(%4279 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x25x25xf32>, tensor<f32>) -> tensor<1x32x8x25xf32>
    %4281 = stablehlo.add %4275, %4280 : tensor<1x32x8x25xf32>
    %4282 = "stablehlo.gather"(%4281, %1126) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x25xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4283 = stablehlo.select %427, %cst_224, %4282 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4284 = stablehlo.select %424, %4283, %4274 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4285 = stablehlo.broadcast_in_dim %4284, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4286 = stablehlo.select %420, %4285, %4272 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4287 = stablehlo.slice %4286 [0:1, 0:32, 0:8, 26:27, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4288 = stablehlo.reshape %4287 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4289 = stablehlo.slice %4288 [0:1, 0:32, 0:8, 0:26] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x26xf32>
    %4290 = stablehlo.broadcast_in_dim %4289, dims = [0, 1, 2, 3] : (tensor<1x32x8x26xf32>) -> tensor<1x32x8x26x26xf32>
    %4291 = stablehlo.slice %4286 [0:1, 0:32, 0:8, 0:26, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x26x64xf32>
    %4292 = stablehlo.slice %4291 [0:1, 0:32, 0:8, 0:26, 0:26] : (tensor<1x32x8x26x64xf32>) -> tensor<1x32x8x26x26xf32>
    %4293 = stablehlo.multiply %4290, %4292 : tensor<1x32x8x26x26xf32>
    %4294 = stablehlo.reduce(%4293 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x26x26xf32>, tensor<f32>) -> tensor<1x32x8x26xf32>
    %4295 = stablehlo.add %4289, %4294 : tensor<1x32x8x26xf32>
    %4296 = "stablehlo.gather"(%4295, %1145) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x26xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4297 = stablehlo.select %419, %cst_224, %4296 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4298 = stablehlo.select %416, %4297, %4288 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4299 = stablehlo.broadcast_in_dim %4298, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4300 = stablehlo.select %412, %4299, %4286 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4301 = stablehlo.slice %4300 [0:1, 0:32, 0:8, 27:28, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4302 = stablehlo.reshape %4301 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4303 = stablehlo.slice %4302 [0:1, 0:32, 0:8, 0:27] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x27xf32>
    %4304 = stablehlo.broadcast_in_dim %4303, dims = [0, 1, 2, 3] : (tensor<1x32x8x27xf32>) -> tensor<1x32x8x27x27xf32>
    %4305 = stablehlo.slice %4300 [0:1, 0:32, 0:8, 0:27, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x27x64xf32>
    %4306 = stablehlo.slice %4305 [0:1, 0:32, 0:8, 0:27, 0:27] : (tensor<1x32x8x27x64xf32>) -> tensor<1x32x8x27x27xf32>
    %4307 = stablehlo.multiply %4304, %4306 : tensor<1x32x8x27x27xf32>
    %4308 = stablehlo.reduce(%4307 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x27x27xf32>, tensor<f32>) -> tensor<1x32x8x27xf32>
    %4309 = stablehlo.add %4303, %4308 : tensor<1x32x8x27xf32>
    %4310 = "stablehlo.gather"(%4309, %1164) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x27xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4311 = stablehlo.select %411, %cst_224, %4310 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4312 = stablehlo.select %408, %4311, %4302 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4313 = stablehlo.broadcast_in_dim %4312, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4314 = stablehlo.select %404, %4313, %4300 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4315 = stablehlo.slice %4314 [0:1, 0:32, 0:8, 28:29, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4316 = stablehlo.reshape %4315 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4317 = stablehlo.slice %4316 [0:1, 0:32, 0:8, 0:28] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x28xf32>
    %4318 = stablehlo.broadcast_in_dim %4317, dims = [0, 1, 2, 3] : (tensor<1x32x8x28xf32>) -> tensor<1x32x8x28x28xf32>
    %4319 = stablehlo.slice %4314 [0:1, 0:32, 0:8, 0:28, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x28x64xf32>
    %4320 = stablehlo.slice %4319 [0:1, 0:32, 0:8, 0:28, 0:28] : (tensor<1x32x8x28x64xf32>) -> tensor<1x32x8x28x28xf32>
    %4321 = stablehlo.multiply %4318, %4320 : tensor<1x32x8x28x28xf32>
    %4322 = stablehlo.reduce(%4321 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x28x28xf32>, tensor<f32>) -> tensor<1x32x8x28xf32>
    %4323 = stablehlo.add %4317, %4322 : tensor<1x32x8x28xf32>
    %4324 = "stablehlo.gather"(%4323, %1183) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x28xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4325 = stablehlo.select %403, %cst_224, %4324 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4326 = stablehlo.select %400, %4325, %4316 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4327 = stablehlo.broadcast_in_dim %4326, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4328 = stablehlo.select %396, %4327, %4314 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4329 = stablehlo.slice %4328 [0:1, 0:32, 0:8, 29:30, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4330 = stablehlo.reshape %4329 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4331 = stablehlo.slice %4330 [0:1, 0:32, 0:8, 0:29] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x29xf32>
    %4332 = stablehlo.broadcast_in_dim %4331, dims = [0, 1, 2, 3] : (tensor<1x32x8x29xf32>) -> tensor<1x32x8x29x29xf32>
    %4333 = stablehlo.slice %4328 [0:1, 0:32, 0:8, 0:29, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x29x64xf32>
    %4334 = stablehlo.slice %4333 [0:1, 0:32, 0:8, 0:29, 0:29] : (tensor<1x32x8x29x64xf32>) -> tensor<1x32x8x29x29xf32>
    %4335 = stablehlo.multiply %4332, %4334 : tensor<1x32x8x29x29xf32>
    %4336 = stablehlo.reduce(%4335 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x29x29xf32>, tensor<f32>) -> tensor<1x32x8x29xf32>
    %4337 = stablehlo.add %4331, %4336 : tensor<1x32x8x29xf32>
    %4338 = "stablehlo.gather"(%4337, %1202) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x29xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4339 = stablehlo.select %395, %cst_224, %4338 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4340 = stablehlo.select %392, %4339, %4330 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4341 = stablehlo.broadcast_in_dim %4340, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4342 = stablehlo.select %388, %4341, %4328 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4343 = stablehlo.slice %4342 [0:1, 0:32, 0:8, 30:31, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4344 = stablehlo.reshape %4343 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4345 = stablehlo.slice %4344 [0:1, 0:32, 0:8, 0:30] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x30xf32>
    %4346 = stablehlo.broadcast_in_dim %4345, dims = [0, 1, 2, 3] : (tensor<1x32x8x30xf32>) -> tensor<1x32x8x30x30xf32>
    %4347 = stablehlo.slice %4342 [0:1, 0:32, 0:8, 0:30, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x30x64xf32>
    %4348 = stablehlo.slice %4347 [0:1, 0:32, 0:8, 0:30, 0:30] : (tensor<1x32x8x30x64xf32>) -> tensor<1x32x8x30x30xf32>
    %4349 = stablehlo.multiply %4346, %4348 : tensor<1x32x8x30x30xf32>
    %4350 = stablehlo.reduce(%4349 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x30x30xf32>, tensor<f32>) -> tensor<1x32x8x30xf32>
    %4351 = stablehlo.add %4345, %4350 : tensor<1x32x8x30xf32>
    %4352 = "stablehlo.gather"(%4351, %1221) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x30xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4353 = stablehlo.select %387, %cst_224, %4352 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4354 = stablehlo.select %384, %4353, %4344 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4355 = stablehlo.broadcast_in_dim %4354, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4356 = stablehlo.select %380, %4355, %4342 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4357 = stablehlo.slice %4356 [0:1, 0:32, 0:8, 31:32, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4358 = stablehlo.reshape %4357 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4359 = stablehlo.slice %4358 [0:1, 0:32, 0:8, 0:31] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x31xf32>
    %4360 = stablehlo.broadcast_in_dim %4359, dims = [0, 1, 2, 3] : (tensor<1x32x8x31xf32>) -> tensor<1x32x8x31x31xf32>
    %4361 = stablehlo.slice %4356 [0:1, 0:32, 0:8, 0:31, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x31x64xf32>
    %4362 = stablehlo.slice %4361 [0:1, 0:32, 0:8, 0:31, 0:31] : (tensor<1x32x8x31x64xf32>) -> tensor<1x32x8x31x31xf32>
    %4363 = stablehlo.multiply %4360, %4362 : tensor<1x32x8x31x31xf32>
    %4364 = stablehlo.reduce(%4363 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x31x31xf32>, tensor<f32>) -> tensor<1x32x8x31xf32>
    %4365 = stablehlo.add %4359, %4364 : tensor<1x32x8x31xf32>
    %4366 = "stablehlo.gather"(%4365, %1240) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x31xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4367 = stablehlo.select %379, %cst_224, %4366 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4368 = stablehlo.select %376, %4367, %4358 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4369 = stablehlo.broadcast_in_dim %4368, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4370 = stablehlo.select %372, %4369, %4356 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4371 = stablehlo.slice %4370 [0:1, 0:32, 0:8, 32:33, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4372 = stablehlo.reshape %4371 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4373 = stablehlo.slice %4372 [0:1, 0:32, 0:8, 0:32] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x32xf32>
    %4374 = stablehlo.broadcast_in_dim %4373, dims = [0, 1, 2, 3] : (tensor<1x32x8x32xf32>) -> tensor<1x32x8x32x32xf32>
    %4375 = stablehlo.slice %4370 [0:1, 0:32, 0:8, 0:32, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x32x64xf32>
    %4376 = stablehlo.slice %4375 [0:1, 0:32, 0:8, 0:32, 0:32] : (tensor<1x32x8x32x64xf32>) -> tensor<1x32x8x32x32xf32>
    %4377 = stablehlo.multiply %4374, %4376 : tensor<1x32x8x32x32xf32>
    %4378 = stablehlo.reduce(%4377 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x32x32xf32>, tensor<f32>) -> tensor<1x32x8x32xf32>
    %4379 = stablehlo.add %4373, %4378 : tensor<1x32x8x32xf32>
    %4380 = "stablehlo.gather"(%4379, %1259) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x32xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4381 = stablehlo.select %371, %cst_224, %4380 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4382 = stablehlo.select %368, %4381, %4372 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4383 = stablehlo.broadcast_in_dim %4382, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4384 = stablehlo.select %364, %4383, %4370 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4385 = stablehlo.slice %4384 [0:1, 0:32, 0:8, 33:34, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4386 = stablehlo.reshape %4385 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4387 = stablehlo.slice %4386 [0:1, 0:32, 0:8, 0:33] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x33xf32>
    %4388 = stablehlo.broadcast_in_dim %4387, dims = [0, 1, 2, 3] : (tensor<1x32x8x33xf32>) -> tensor<1x32x8x33x33xf32>
    %4389 = stablehlo.slice %4384 [0:1, 0:32, 0:8, 0:33, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x33x64xf32>
    %4390 = stablehlo.slice %4389 [0:1, 0:32, 0:8, 0:33, 0:33] : (tensor<1x32x8x33x64xf32>) -> tensor<1x32x8x33x33xf32>
    %4391 = stablehlo.multiply %4388, %4390 : tensor<1x32x8x33x33xf32>
    %4392 = stablehlo.reduce(%4391 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x33x33xf32>, tensor<f32>) -> tensor<1x32x8x33xf32>
    %4393 = stablehlo.add %4387, %4392 : tensor<1x32x8x33xf32>
    %4394 = "stablehlo.gather"(%4393, %1278) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x33xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4395 = stablehlo.select %363, %cst_224, %4394 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4396 = stablehlo.select %360, %4395, %4386 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4397 = stablehlo.broadcast_in_dim %4396, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4398 = stablehlo.select %356, %4397, %4384 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4399 = stablehlo.slice %4398 [0:1, 0:32, 0:8, 34:35, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4400 = stablehlo.reshape %4399 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4401 = stablehlo.slice %4400 [0:1, 0:32, 0:8, 0:34] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x34xf32>
    %4402 = stablehlo.broadcast_in_dim %4401, dims = [0, 1, 2, 3] : (tensor<1x32x8x34xf32>) -> tensor<1x32x8x34x34xf32>
    %4403 = stablehlo.slice %4398 [0:1, 0:32, 0:8, 0:34, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x34x64xf32>
    %4404 = stablehlo.slice %4403 [0:1, 0:32, 0:8, 0:34, 0:34] : (tensor<1x32x8x34x64xf32>) -> tensor<1x32x8x34x34xf32>
    %4405 = stablehlo.multiply %4402, %4404 : tensor<1x32x8x34x34xf32>
    %4406 = stablehlo.reduce(%4405 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x34x34xf32>, tensor<f32>) -> tensor<1x32x8x34xf32>
    %4407 = stablehlo.add %4401, %4406 : tensor<1x32x8x34xf32>
    %4408 = "stablehlo.gather"(%4407, %1297) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x34xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4409 = stablehlo.select %355, %cst_224, %4408 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4410 = stablehlo.select %352, %4409, %4400 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4411 = stablehlo.broadcast_in_dim %4410, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4412 = stablehlo.select %348, %4411, %4398 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4413 = stablehlo.slice %4412 [0:1, 0:32, 0:8, 35:36, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4414 = stablehlo.reshape %4413 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4415 = stablehlo.slice %4414 [0:1, 0:32, 0:8, 0:35] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x35xf32>
    %4416 = stablehlo.broadcast_in_dim %4415, dims = [0, 1, 2, 3] : (tensor<1x32x8x35xf32>) -> tensor<1x32x8x35x35xf32>
    %4417 = stablehlo.slice %4412 [0:1, 0:32, 0:8, 0:35, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x35x64xf32>
    %4418 = stablehlo.slice %4417 [0:1, 0:32, 0:8, 0:35, 0:35] : (tensor<1x32x8x35x64xf32>) -> tensor<1x32x8x35x35xf32>
    %4419 = stablehlo.multiply %4416, %4418 : tensor<1x32x8x35x35xf32>
    %4420 = stablehlo.reduce(%4419 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x35x35xf32>, tensor<f32>) -> tensor<1x32x8x35xf32>
    %4421 = stablehlo.add %4415, %4420 : tensor<1x32x8x35xf32>
    %4422 = "stablehlo.gather"(%4421, %1316) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x35xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4423 = stablehlo.select %347, %cst_224, %4422 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4424 = stablehlo.select %344, %4423, %4414 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4425 = stablehlo.broadcast_in_dim %4424, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4426 = stablehlo.select %340, %4425, %4412 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4427 = stablehlo.slice %4426 [0:1, 0:32, 0:8, 36:37, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4428 = stablehlo.reshape %4427 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4429 = stablehlo.slice %4428 [0:1, 0:32, 0:8, 0:36] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x36xf32>
    %4430 = stablehlo.broadcast_in_dim %4429, dims = [0, 1, 2, 3] : (tensor<1x32x8x36xf32>) -> tensor<1x32x8x36x36xf32>
    %4431 = stablehlo.slice %4426 [0:1, 0:32, 0:8, 0:36, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x36x64xf32>
    %4432 = stablehlo.slice %4431 [0:1, 0:32, 0:8, 0:36, 0:36] : (tensor<1x32x8x36x64xf32>) -> tensor<1x32x8x36x36xf32>
    %4433 = stablehlo.multiply %4430, %4432 : tensor<1x32x8x36x36xf32>
    %4434 = stablehlo.reduce(%4433 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x36x36xf32>, tensor<f32>) -> tensor<1x32x8x36xf32>
    %4435 = stablehlo.add %4429, %4434 : tensor<1x32x8x36xf32>
    %4436 = "stablehlo.gather"(%4435, %1335) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x36xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4437 = stablehlo.select %339, %cst_224, %4436 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4438 = stablehlo.select %336, %4437, %4428 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4439 = stablehlo.broadcast_in_dim %4438, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4440 = stablehlo.select %332, %4439, %4426 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4441 = stablehlo.slice %4440 [0:1, 0:32, 0:8, 37:38, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4442 = stablehlo.reshape %4441 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4443 = stablehlo.slice %4442 [0:1, 0:32, 0:8, 0:37] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x37xf32>
    %4444 = stablehlo.broadcast_in_dim %4443, dims = [0, 1, 2, 3] : (tensor<1x32x8x37xf32>) -> tensor<1x32x8x37x37xf32>
    %4445 = stablehlo.slice %4440 [0:1, 0:32, 0:8, 0:37, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x37x64xf32>
    %4446 = stablehlo.slice %4445 [0:1, 0:32, 0:8, 0:37, 0:37] : (tensor<1x32x8x37x64xf32>) -> tensor<1x32x8x37x37xf32>
    %4447 = stablehlo.multiply %4444, %4446 : tensor<1x32x8x37x37xf32>
    %4448 = stablehlo.reduce(%4447 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x37x37xf32>, tensor<f32>) -> tensor<1x32x8x37xf32>
    %4449 = stablehlo.add %4443, %4448 : tensor<1x32x8x37xf32>
    %4450 = "stablehlo.gather"(%4449, %1354) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x37xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4451 = stablehlo.select %331, %cst_224, %4450 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4452 = stablehlo.select %328, %4451, %4442 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4453 = stablehlo.broadcast_in_dim %4452, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4454 = stablehlo.select %324, %4453, %4440 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4455 = stablehlo.slice %4454 [0:1, 0:32, 0:8, 38:39, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4456 = stablehlo.reshape %4455 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4457 = stablehlo.slice %4456 [0:1, 0:32, 0:8, 0:38] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x38xf32>
    %4458 = stablehlo.broadcast_in_dim %4457, dims = [0, 1, 2, 3] : (tensor<1x32x8x38xf32>) -> tensor<1x32x8x38x38xf32>
    %4459 = stablehlo.slice %4454 [0:1, 0:32, 0:8, 0:38, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x38x64xf32>
    %4460 = stablehlo.slice %4459 [0:1, 0:32, 0:8, 0:38, 0:38] : (tensor<1x32x8x38x64xf32>) -> tensor<1x32x8x38x38xf32>
    %4461 = stablehlo.multiply %4458, %4460 : tensor<1x32x8x38x38xf32>
    %4462 = stablehlo.reduce(%4461 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x38x38xf32>, tensor<f32>) -> tensor<1x32x8x38xf32>
    %4463 = stablehlo.add %4457, %4462 : tensor<1x32x8x38xf32>
    %4464 = "stablehlo.gather"(%4463, %1373) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x38xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4465 = stablehlo.select %323, %cst_224, %4464 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4466 = stablehlo.select %320, %4465, %4456 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4467 = stablehlo.broadcast_in_dim %4466, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4468 = stablehlo.select %316, %4467, %4454 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4469 = stablehlo.slice %4468 [0:1, 0:32, 0:8, 39:40, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4470 = stablehlo.reshape %4469 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4471 = stablehlo.slice %4470 [0:1, 0:32, 0:8, 0:39] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x39xf32>
    %4472 = stablehlo.broadcast_in_dim %4471, dims = [0, 1, 2, 3] : (tensor<1x32x8x39xf32>) -> tensor<1x32x8x39x39xf32>
    %4473 = stablehlo.slice %4468 [0:1, 0:32, 0:8, 0:39, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x39x64xf32>
    %4474 = stablehlo.slice %4473 [0:1, 0:32, 0:8, 0:39, 0:39] : (tensor<1x32x8x39x64xf32>) -> tensor<1x32x8x39x39xf32>
    %4475 = stablehlo.multiply %4472, %4474 : tensor<1x32x8x39x39xf32>
    %4476 = stablehlo.reduce(%4475 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x39x39xf32>, tensor<f32>) -> tensor<1x32x8x39xf32>
    %4477 = stablehlo.add %4471, %4476 : tensor<1x32x8x39xf32>
    %4478 = "stablehlo.gather"(%4477, %1392) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x39xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4479 = stablehlo.select %315, %cst_224, %4478 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4480 = stablehlo.select %312, %4479, %4470 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4481 = stablehlo.broadcast_in_dim %4480, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4482 = stablehlo.select %308, %4481, %4468 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4483 = stablehlo.slice %4482 [0:1, 0:32, 0:8, 40:41, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4484 = stablehlo.reshape %4483 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4485 = stablehlo.slice %4484 [0:1, 0:32, 0:8, 0:40] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x40xf32>
    %4486 = stablehlo.broadcast_in_dim %4485, dims = [0, 1, 2, 3] : (tensor<1x32x8x40xf32>) -> tensor<1x32x8x40x40xf32>
    %4487 = stablehlo.slice %4482 [0:1, 0:32, 0:8, 0:40, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x40x64xf32>
    %4488 = stablehlo.slice %4487 [0:1, 0:32, 0:8, 0:40, 0:40] : (tensor<1x32x8x40x64xf32>) -> tensor<1x32x8x40x40xf32>
    %4489 = stablehlo.multiply %4486, %4488 : tensor<1x32x8x40x40xf32>
    %4490 = stablehlo.reduce(%4489 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x40x40xf32>, tensor<f32>) -> tensor<1x32x8x40xf32>
    %4491 = stablehlo.add %4485, %4490 : tensor<1x32x8x40xf32>
    %4492 = "stablehlo.gather"(%4491, %1411) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x40xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4493 = stablehlo.select %307, %cst_224, %4492 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4494 = stablehlo.select %304, %4493, %4484 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4495 = stablehlo.broadcast_in_dim %4494, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4496 = stablehlo.select %300, %4495, %4482 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4497 = stablehlo.slice %4496 [0:1, 0:32, 0:8, 41:42, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4498 = stablehlo.reshape %4497 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4499 = stablehlo.slice %4498 [0:1, 0:32, 0:8, 0:41] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x41xf32>
    %4500 = stablehlo.broadcast_in_dim %4499, dims = [0, 1, 2, 3] : (tensor<1x32x8x41xf32>) -> tensor<1x32x8x41x41xf32>
    %4501 = stablehlo.slice %4496 [0:1, 0:32, 0:8, 0:41, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x41x64xf32>
    %4502 = stablehlo.slice %4501 [0:1, 0:32, 0:8, 0:41, 0:41] : (tensor<1x32x8x41x64xf32>) -> tensor<1x32x8x41x41xf32>
    %4503 = stablehlo.multiply %4500, %4502 : tensor<1x32x8x41x41xf32>
    %4504 = stablehlo.reduce(%4503 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x41x41xf32>, tensor<f32>) -> tensor<1x32x8x41xf32>
    %4505 = stablehlo.add %4499, %4504 : tensor<1x32x8x41xf32>
    %4506 = "stablehlo.gather"(%4505, %1430) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x41xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4507 = stablehlo.select %299, %cst_224, %4506 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4508 = stablehlo.select %296, %4507, %4498 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4509 = stablehlo.broadcast_in_dim %4508, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4510 = stablehlo.select %292, %4509, %4496 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4511 = stablehlo.slice %4510 [0:1, 0:32, 0:8, 42:43, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4512 = stablehlo.reshape %4511 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4513 = stablehlo.slice %4512 [0:1, 0:32, 0:8, 0:42] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x42xf32>
    %4514 = stablehlo.broadcast_in_dim %4513, dims = [0, 1, 2, 3] : (tensor<1x32x8x42xf32>) -> tensor<1x32x8x42x42xf32>
    %4515 = stablehlo.slice %4510 [0:1, 0:32, 0:8, 0:42, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x42x64xf32>
    %4516 = stablehlo.slice %4515 [0:1, 0:32, 0:8, 0:42, 0:42] : (tensor<1x32x8x42x64xf32>) -> tensor<1x32x8x42x42xf32>
    %4517 = stablehlo.multiply %4514, %4516 : tensor<1x32x8x42x42xf32>
    %4518 = stablehlo.reduce(%4517 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x42x42xf32>, tensor<f32>) -> tensor<1x32x8x42xf32>
    %4519 = stablehlo.add %4513, %4518 : tensor<1x32x8x42xf32>
    %4520 = "stablehlo.gather"(%4519, %1449) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x42xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4521 = stablehlo.select %291, %cst_224, %4520 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4522 = stablehlo.select %288, %4521, %4512 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4523 = stablehlo.broadcast_in_dim %4522, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4524 = stablehlo.select %284, %4523, %4510 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4525 = stablehlo.slice %4524 [0:1, 0:32, 0:8, 43:44, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4526 = stablehlo.reshape %4525 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4527 = stablehlo.slice %4526 [0:1, 0:32, 0:8, 0:43] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x43xf32>
    %4528 = stablehlo.broadcast_in_dim %4527, dims = [0, 1, 2, 3] : (tensor<1x32x8x43xf32>) -> tensor<1x32x8x43x43xf32>
    %4529 = stablehlo.slice %4524 [0:1, 0:32, 0:8, 0:43, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x43x64xf32>
    %4530 = stablehlo.slice %4529 [0:1, 0:32, 0:8, 0:43, 0:43] : (tensor<1x32x8x43x64xf32>) -> tensor<1x32x8x43x43xf32>
    %4531 = stablehlo.multiply %4528, %4530 : tensor<1x32x8x43x43xf32>
    %4532 = stablehlo.reduce(%4531 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x43x43xf32>, tensor<f32>) -> tensor<1x32x8x43xf32>
    %4533 = stablehlo.add %4527, %4532 : tensor<1x32x8x43xf32>
    %4534 = "stablehlo.gather"(%4533, %1468) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x43xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4535 = stablehlo.select %283, %cst_224, %4534 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4536 = stablehlo.select %280, %4535, %4526 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4537 = stablehlo.broadcast_in_dim %4536, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4538 = stablehlo.select %276, %4537, %4524 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4539 = stablehlo.slice %4538 [0:1, 0:32, 0:8, 44:45, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4540 = stablehlo.reshape %4539 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4541 = stablehlo.slice %4540 [0:1, 0:32, 0:8, 0:44] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x44xf32>
    %4542 = stablehlo.broadcast_in_dim %4541, dims = [0, 1, 2, 3] : (tensor<1x32x8x44xf32>) -> tensor<1x32x8x44x44xf32>
    %4543 = stablehlo.slice %4538 [0:1, 0:32, 0:8, 0:44, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x44x64xf32>
    %4544 = stablehlo.slice %4543 [0:1, 0:32, 0:8, 0:44, 0:44] : (tensor<1x32x8x44x64xf32>) -> tensor<1x32x8x44x44xf32>
    %4545 = stablehlo.multiply %4542, %4544 : tensor<1x32x8x44x44xf32>
    %4546 = stablehlo.reduce(%4545 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x44x44xf32>, tensor<f32>) -> tensor<1x32x8x44xf32>
    %4547 = stablehlo.add %4541, %4546 : tensor<1x32x8x44xf32>
    %4548 = "stablehlo.gather"(%4547, %1487) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x44xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4549 = stablehlo.select %275, %cst_224, %4548 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4550 = stablehlo.select %272, %4549, %4540 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4551 = stablehlo.broadcast_in_dim %4550, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4552 = stablehlo.select %268, %4551, %4538 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4553 = stablehlo.slice %4552 [0:1, 0:32, 0:8, 45:46, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4554 = stablehlo.reshape %4553 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4555 = stablehlo.slice %4554 [0:1, 0:32, 0:8, 0:45] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x45xf32>
    %4556 = stablehlo.broadcast_in_dim %4555, dims = [0, 1, 2, 3] : (tensor<1x32x8x45xf32>) -> tensor<1x32x8x45x45xf32>
    %4557 = stablehlo.slice %4552 [0:1, 0:32, 0:8, 0:45, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x45x64xf32>
    %4558 = stablehlo.slice %4557 [0:1, 0:32, 0:8, 0:45, 0:45] : (tensor<1x32x8x45x64xf32>) -> tensor<1x32x8x45x45xf32>
    %4559 = stablehlo.multiply %4556, %4558 : tensor<1x32x8x45x45xf32>
    %4560 = stablehlo.reduce(%4559 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x45x45xf32>, tensor<f32>) -> tensor<1x32x8x45xf32>
    %4561 = stablehlo.add %4555, %4560 : tensor<1x32x8x45xf32>
    %4562 = "stablehlo.gather"(%4561, %1506) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x45xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4563 = stablehlo.select %267, %cst_224, %4562 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4564 = stablehlo.select %264, %4563, %4554 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4565 = stablehlo.broadcast_in_dim %4564, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4566 = stablehlo.select %260, %4565, %4552 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4567 = stablehlo.slice %4566 [0:1, 0:32, 0:8, 46:47, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4568 = stablehlo.reshape %4567 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4569 = stablehlo.slice %4568 [0:1, 0:32, 0:8, 0:46] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x46xf32>
    %4570 = stablehlo.broadcast_in_dim %4569, dims = [0, 1, 2, 3] : (tensor<1x32x8x46xf32>) -> tensor<1x32x8x46x46xf32>
    %4571 = stablehlo.slice %4566 [0:1, 0:32, 0:8, 0:46, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x46x64xf32>
    %4572 = stablehlo.slice %4571 [0:1, 0:32, 0:8, 0:46, 0:46] : (tensor<1x32x8x46x64xf32>) -> tensor<1x32x8x46x46xf32>
    %4573 = stablehlo.multiply %4570, %4572 : tensor<1x32x8x46x46xf32>
    %4574 = stablehlo.reduce(%4573 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x46x46xf32>, tensor<f32>) -> tensor<1x32x8x46xf32>
    %4575 = stablehlo.add %4569, %4574 : tensor<1x32x8x46xf32>
    %4576 = "stablehlo.gather"(%4575, %1525) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x46xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4577 = stablehlo.select %259, %cst_224, %4576 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4578 = stablehlo.select %256, %4577, %4568 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4579 = stablehlo.broadcast_in_dim %4578, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4580 = stablehlo.select %252, %4579, %4566 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4581 = stablehlo.slice %4580 [0:1, 0:32, 0:8, 47:48, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4582 = stablehlo.reshape %4581 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4583 = stablehlo.slice %4582 [0:1, 0:32, 0:8, 0:47] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x47xf32>
    %4584 = stablehlo.broadcast_in_dim %4583, dims = [0, 1, 2, 3] : (tensor<1x32x8x47xf32>) -> tensor<1x32x8x47x47xf32>
    %4585 = stablehlo.slice %4580 [0:1, 0:32, 0:8, 0:47, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x47x64xf32>
    %4586 = stablehlo.slice %4585 [0:1, 0:32, 0:8, 0:47, 0:47] : (tensor<1x32x8x47x64xf32>) -> tensor<1x32x8x47x47xf32>
    %4587 = stablehlo.multiply %4584, %4586 : tensor<1x32x8x47x47xf32>
    %4588 = stablehlo.reduce(%4587 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x47x47xf32>, tensor<f32>) -> tensor<1x32x8x47xf32>
    %4589 = stablehlo.add %4583, %4588 : tensor<1x32x8x47xf32>
    %4590 = "stablehlo.gather"(%4589, %1544) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x47xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4591 = stablehlo.select %251, %cst_224, %4590 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4592 = stablehlo.select %248, %4591, %4582 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4593 = stablehlo.broadcast_in_dim %4592, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4594 = stablehlo.select %244, %4593, %4580 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4595 = stablehlo.slice %4594 [0:1, 0:32, 0:8, 48:49, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4596 = stablehlo.reshape %4595 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4597 = stablehlo.slice %4596 [0:1, 0:32, 0:8, 0:48] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x48xf32>
    %4598 = stablehlo.broadcast_in_dim %4597, dims = [0, 1, 2, 3] : (tensor<1x32x8x48xf32>) -> tensor<1x32x8x48x48xf32>
    %4599 = stablehlo.slice %4594 [0:1, 0:32, 0:8, 0:48, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x48x64xf32>
    %4600 = stablehlo.slice %4599 [0:1, 0:32, 0:8, 0:48, 0:48] : (tensor<1x32x8x48x64xf32>) -> tensor<1x32x8x48x48xf32>
    %4601 = stablehlo.multiply %4598, %4600 : tensor<1x32x8x48x48xf32>
    %4602 = stablehlo.reduce(%4601 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x48x48xf32>, tensor<f32>) -> tensor<1x32x8x48xf32>
    %4603 = stablehlo.add %4597, %4602 : tensor<1x32x8x48xf32>
    %4604 = "stablehlo.gather"(%4603, %1563) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x48xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4605 = stablehlo.select %243, %cst_224, %4604 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4606 = stablehlo.select %240, %4605, %4596 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4607 = stablehlo.broadcast_in_dim %4606, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4608 = stablehlo.select %236, %4607, %4594 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4609 = stablehlo.slice %4608 [0:1, 0:32, 0:8, 49:50, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4610 = stablehlo.reshape %4609 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4611 = stablehlo.slice %4610 [0:1, 0:32, 0:8, 0:49] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x49xf32>
    %4612 = stablehlo.broadcast_in_dim %4611, dims = [0, 1, 2, 3] : (tensor<1x32x8x49xf32>) -> tensor<1x32x8x49x49xf32>
    %4613 = stablehlo.slice %4608 [0:1, 0:32, 0:8, 0:49, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x49x64xf32>
    %4614 = stablehlo.slice %4613 [0:1, 0:32, 0:8, 0:49, 0:49] : (tensor<1x32x8x49x64xf32>) -> tensor<1x32x8x49x49xf32>
    %4615 = stablehlo.multiply %4612, %4614 : tensor<1x32x8x49x49xf32>
    %4616 = stablehlo.reduce(%4615 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x49x49xf32>, tensor<f32>) -> tensor<1x32x8x49xf32>
    %4617 = stablehlo.add %4611, %4616 : tensor<1x32x8x49xf32>
    %4618 = "stablehlo.gather"(%4617, %1582) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x49xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4619 = stablehlo.select %235, %cst_224, %4618 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4620 = stablehlo.select %232, %4619, %4610 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4621 = stablehlo.broadcast_in_dim %4620, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4622 = stablehlo.select %228, %4621, %4608 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4623 = stablehlo.slice %4622 [0:1, 0:32, 0:8, 50:51, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4624 = stablehlo.reshape %4623 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4625 = stablehlo.slice %4624 [0:1, 0:32, 0:8, 0:50] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x50xf32>
    %4626 = stablehlo.broadcast_in_dim %4625, dims = [0, 1, 2, 3] : (tensor<1x32x8x50xf32>) -> tensor<1x32x8x50x50xf32>
    %4627 = stablehlo.slice %4622 [0:1, 0:32, 0:8, 0:50, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x50x64xf32>
    %4628 = stablehlo.slice %4627 [0:1, 0:32, 0:8, 0:50, 0:50] : (tensor<1x32x8x50x64xf32>) -> tensor<1x32x8x50x50xf32>
    %4629 = stablehlo.multiply %4626, %4628 : tensor<1x32x8x50x50xf32>
    %4630 = stablehlo.reduce(%4629 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x50x50xf32>, tensor<f32>) -> tensor<1x32x8x50xf32>
    %4631 = stablehlo.add %4625, %4630 : tensor<1x32x8x50xf32>
    %4632 = "stablehlo.gather"(%4631, %1601) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x50xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4633 = stablehlo.select %227, %cst_224, %4632 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4634 = stablehlo.select %224, %4633, %4624 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4635 = stablehlo.broadcast_in_dim %4634, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4636 = stablehlo.select %220, %4635, %4622 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4637 = stablehlo.slice %4636 [0:1, 0:32, 0:8, 51:52, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4638 = stablehlo.reshape %4637 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4639 = stablehlo.slice %4638 [0:1, 0:32, 0:8, 0:51] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x51xf32>
    %4640 = stablehlo.broadcast_in_dim %4639, dims = [0, 1, 2, 3] : (tensor<1x32x8x51xf32>) -> tensor<1x32x8x51x51xf32>
    %4641 = stablehlo.slice %4636 [0:1, 0:32, 0:8, 0:51, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x51x64xf32>
    %4642 = stablehlo.slice %4641 [0:1, 0:32, 0:8, 0:51, 0:51] : (tensor<1x32x8x51x64xf32>) -> tensor<1x32x8x51x51xf32>
    %4643 = stablehlo.multiply %4640, %4642 : tensor<1x32x8x51x51xf32>
    %4644 = stablehlo.reduce(%4643 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x51x51xf32>, tensor<f32>) -> tensor<1x32x8x51xf32>
    %4645 = stablehlo.add %4639, %4644 : tensor<1x32x8x51xf32>
    %4646 = "stablehlo.gather"(%4645, %1620) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x51xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4647 = stablehlo.select %219, %cst_224, %4646 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4648 = stablehlo.select %216, %4647, %4638 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4649 = stablehlo.broadcast_in_dim %4648, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4650 = stablehlo.select %212, %4649, %4636 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4651 = stablehlo.slice %4650 [0:1, 0:32, 0:8, 52:53, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4652 = stablehlo.reshape %4651 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4653 = stablehlo.slice %4652 [0:1, 0:32, 0:8, 0:52] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x52xf32>
    %4654 = stablehlo.broadcast_in_dim %4653, dims = [0, 1, 2, 3] : (tensor<1x32x8x52xf32>) -> tensor<1x32x8x52x52xf32>
    %4655 = stablehlo.slice %4650 [0:1, 0:32, 0:8, 0:52, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x52x64xf32>
    %4656 = stablehlo.slice %4655 [0:1, 0:32, 0:8, 0:52, 0:52] : (tensor<1x32x8x52x64xf32>) -> tensor<1x32x8x52x52xf32>
    %4657 = stablehlo.multiply %4654, %4656 : tensor<1x32x8x52x52xf32>
    %4658 = stablehlo.reduce(%4657 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x52x52xf32>, tensor<f32>) -> tensor<1x32x8x52xf32>
    %4659 = stablehlo.add %4653, %4658 : tensor<1x32x8x52xf32>
    %4660 = "stablehlo.gather"(%4659, %1639) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x52xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4661 = stablehlo.select %211, %cst_224, %4660 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4662 = stablehlo.select %208, %4661, %4652 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4663 = stablehlo.broadcast_in_dim %4662, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4664 = stablehlo.select %204, %4663, %4650 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4665 = stablehlo.slice %4664 [0:1, 0:32, 0:8, 53:54, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4666 = stablehlo.reshape %4665 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4667 = stablehlo.slice %4666 [0:1, 0:32, 0:8, 0:53] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x53xf32>
    %4668 = stablehlo.broadcast_in_dim %4667, dims = [0, 1, 2, 3] : (tensor<1x32x8x53xf32>) -> tensor<1x32x8x53x53xf32>
    %4669 = stablehlo.slice %4664 [0:1, 0:32, 0:8, 0:53, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x53x64xf32>
    %4670 = stablehlo.slice %4669 [0:1, 0:32, 0:8, 0:53, 0:53] : (tensor<1x32x8x53x64xf32>) -> tensor<1x32x8x53x53xf32>
    %4671 = stablehlo.multiply %4668, %4670 : tensor<1x32x8x53x53xf32>
    %4672 = stablehlo.reduce(%4671 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x53x53xf32>, tensor<f32>) -> tensor<1x32x8x53xf32>
    %4673 = stablehlo.add %4667, %4672 : tensor<1x32x8x53xf32>
    %4674 = "stablehlo.gather"(%4673, %1658) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x53xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4675 = stablehlo.select %203, %cst_224, %4674 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4676 = stablehlo.select %200, %4675, %4666 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4677 = stablehlo.broadcast_in_dim %4676, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4678 = stablehlo.select %196, %4677, %4664 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4679 = stablehlo.slice %4678 [0:1, 0:32, 0:8, 54:55, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4680 = stablehlo.reshape %4679 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4681 = stablehlo.slice %4680 [0:1, 0:32, 0:8, 0:54] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x54xf32>
    %4682 = stablehlo.broadcast_in_dim %4681, dims = [0, 1, 2, 3] : (tensor<1x32x8x54xf32>) -> tensor<1x32x8x54x54xf32>
    %4683 = stablehlo.slice %4678 [0:1, 0:32, 0:8, 0:54, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x54x64xf32>
    %4684 = stablehlo.slice %4683 [0:1, 0:32, 0:8, 0:54, 0:54] : (tensor<1x32x8x54x64xf32>) -> tensor<1x32x8x54x54xf32>
    %4685 = stablehlo.multiply %4682, %4684 : tensor<1x32x8x54x54xf32>
    %4686 = stablehlo.reduce(%4685 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x54x54xf32>, tensor<f32>) -> tensor<1x32x8x54xf32>
    %4687 = stablehlo.add %4681, %4686 : tensor<1x32x8x54xf32>
    %4688 = "stablehlo.gather"(%4687, %1677) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x54xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4689 = stablehlo.select %195, %cst_224, %4688 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4690 = stablehlo.select %192, %4689, %4680 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4691 = stablehlo.broadcast_in_dim %4690, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4692 = stablehlo.select %188, %4691, %4678 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4693 = stablehlo.slice %4692 [0:1, 0:32, 0:8, 55:56, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4694 = stablehlo.reshape %4693 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4695 = stablehlo.slice %4694 [0:1, 0:32, 0:8, 0:55] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x55xf32>
    %4696 = stablehlo.broadcast_in_dim %4695, dims = [0, 1, 2, 3] : (tensor<1x32x8x55xf32>) -> tensor<1x32x8x55x55xf32>
    %4697 = stablehlo.slice %4692 [0:1, 0:32, 0:8, 0:55, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x55x64xf32>
    %4698 = stablehlo.slice %4697 [0:1, 0:32, 0:8, 0:55, 0:55] : (tensor<1x32x8x55x64xf32>) -> tensor<1x32x8x55x55xf32>
    %4699 = stablehlo.multiply %4696, %4698 : tensor<1x32x8x55x55xf32>
    %4700 = stablehlo.reduce(%4699 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x55x55xf32>, tensor<f32>) -> tensor<1x32x8x55xf32>
    %4701 = stablehlo.add %4695, %4700 : tensor<1x32x8x55xf32>
    %4702 = "stablehlo.gather"(%4701, %1696) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x55xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4703 = stablehlo.select %187, %cst_224, %4702 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4704 = stablehlo.select %184, %4703, %4694 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4705 = stablehlo.broadcast_in_dim %4704, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4706 = stablehlo.select %180, %4705, %4692 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4707 = stablehlo.slice %4706 [0:1, 0:32, 0:8, 56:57, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4708 = stablehlo.reshape %4707 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4709 = stablehlo.slice %4708 [0:1, 0:32, 0:8, 0:56] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x56xf32>
    %4710 = stablehlo.broadcast_in_dim %4709, dims = [0, 1, 2, 3] : (tensor<1x32x8x56xf32>) -> tensor<1x32x8x56x56xf32>
    %4711 = stablehlo.slice %4706 [0:1, 0:32, 0:8, 0:56, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x56x64xf32>
    %4712 = stablehlo.slice %4711 [0:1, 0:32, 0:8, 0:56, 0:56] : (tensor<1x32x8x56x64xf32>) -> tensor<1x32x8x56x56xf32>
    %4713 = stablehlo.multiply %4710, %4712 : tensor<1x32x8x56x56xf32>
    %4714 = stablehlo.reduce(%4713 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x56x56xf32>, tensor<f32>) -> tensor<1x32x8x56xf32>
    %4715 = stablehlo.add %4709, %4714 : tensor<1x32x8x56xf32>
    %4716 = "stablehlo.gather"(%4715, %1715) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x56xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4717 = stablehlo.select %179, %cst_224, %4716 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4718 = stablehlo.select %176, %4717, %4708 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4719 = stablehlo.broadcast_in_dim %4718, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4720 = stablehlo.select %172, %4719, %4706 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4721 = stablehlo.slice %4720 [0:1, 0:32, 0:8, 57:58, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4722 = stablehlo.reshape %4721 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4723 = stablehlo.slice %4722 [0:1, 0:32, 0:8, 0:57] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x57xf32>
    %4724 = stablehlo.broadcast_in_dim %4723, dims = [0, 1, 2, 3] : (tensor<1x32x8x57xf32>) -> tensor<1x32x8x57x57xf32>
    %4725 = stablehlo.slice %4720 [0:1, 0:32, 0:8, 0:57, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x57x64xf32>
    %4726 = stablehlo.slice %4725 [0:1, 0:32, 0:8, 0:57, 0:57] : (tensor<1x32x8x57x64xf32>) -> tensor<1x32x8x57x57xf32>
    %4727 = stablehlo.multiply %4724, %4726 : tensor<1x32x8x57x57xf32>
    %4728 = stablehlo.reduce(%4727 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x57x57xf32>, tensor<f32>) -> tensor<1x32x8x57xf32>
    %4729 = stablehlo.add %4723, %4728 : tensor<1x32x8x57xf32>
    %4730 = "stablehlo.gather"(%4729, %1734) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x57xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4731 = stablehlo.select %171, %cst_224, %4730 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4732 = stablehlo.select %168, %4731, %4722 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4733 = stablehlo.broadcast_in_dim %4732, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4734 = stablehlo.select %164, %4733, %4720 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4735 = stablehlo.slice %4734 [0:1, 0:32, 0:8, 58:59, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4736 = stablehlo.reshape %4735 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4737 = stablehlo.slice %4736 [0:1, 0:32, 0:8, 0:58] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x58xf32>
    %4738 = stablehlo.broadcast_in_dim %4737, dims = [0, 1, 2, 3] : (tensor<1x32x8x58xf32>) -> tensor<1x32x8x58x58xf32>
    %4739 = stablehlo.slice %4734 [0:1, 0:32, 0:8, 0:58, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x58x64xf32>
    %4740 = stablehlo.slice %4739 [0:1, 0:32, 0:8, 0:58, 0:58] : (tensor<1x32x8x58x64xf32>) -> tensor<1x32x8x58x58xf32>
    %4741 = stablehlo.multiply %4738, %4740 : tensor<1x32x8x58x58xf32>
    %4742 = stablehlo.reduce(%4741 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x58x58xf32>, tensor<f32>) -> tensor<1x32x8x58xf32>
    %4743 = stablehlo.add %4737, %4742 : tensor<1x32x8x58xf32>
    %4744 = "stablehlo.gather"(%4743, %1753) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x58xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4745 = stablehlo.select %163, %cst_224, %4744 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4746 = stablehlo.select %160, %4745, %4736 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4747 = stablehlo.broadcast_in_dim %4746, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4748 = stablehlo.select %156, %4747, %4734 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4749 = stablehlo.slice %4748 [0:1, 0:32, 0:8, 59:60, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4750 = stablehlo.reshape %4749 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4751 = stablehlo.slice %4750 [0:1, 0:32, 0:8, 0:59] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x59xf32>
    %4752 = stablehlo.broadcast_in_dim %4751, dims = [0, 1, 2, 3] : (tensor<1x32x8x59xf32>) -> tensor<1x32x8x59x59xf32>
    %4753 = stablehlo.slice %4748 [0:1, 0:32, 0:8, 0:59, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x59x64xf32>
    %4754 = stablehlo.slice %4753 [0:1, 0:32, 0:8, 0:59, 0:59] : (tensor<1x32x8x59x64xf32>) -> tensor<1x32x8x59x59xf32>
    %4755 = stablehlo.multiply %4752, %4754 : tensor<1x32x8x59x59xf32>
    %4756 = stablehlo.reduce(%4755 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x59x59xf32>, tensor<f32>) -> tensor<1x32x8x59xf32>
    %4757 = stablehlo.add %4751, %4756 : tensor<1x32x8x59xf32>
    %4758 = "stablehlo.gather"(%4757, %1772) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x59xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4759 = stablehlo.select %155, %cst_224, %4758 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4760 = stablehlo.select %152, %4759, %4750 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4761 = stablehlo.broadcast_in_dim %4760, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4762 = stablehlo.select %148, %4761, %4748 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4763 = stablehlo.slice %4762 [0:1, 0:32, 0:8, 60:61, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4764 = stablehlo.reshape %4763 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4765 = stablehlo.slice %4764 [0:1, 0:32, 0:8, 0:60] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x60xf32>
    %4766 = stablehlo.broadcast_in_dim %4765, dims = [0, 1, 2, 3] : (tensor<1x32x8x60xf32>) -> tensor<1x32x8x60x60xf32>
    %4767 = stablehlo.slice %4762 [0:1, 0:32, 0:8, 0:60, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x60x64xf32>
    %4768 = stablehlo.slice %4767 [0:1, 0:32, 0:8, 0:60, 0:60] : (tensor<1x32x8x60x64xf32>) -> tensor<1x32x8x60x60xf32>
    %4769 = stablehlo.multiply %4766, %4768 : tensor<1x32x8x60x60xf32>
    %4770 = stablehlo.reduce(%4769 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x60x60xf32>, tensor<f32>) -> tensor<1x32x8x60xf32>
    %4771 = stablehlo.add %4765, %4770 : tensor<1x32x8x60xf32>
    %4772 = "stablehlo.gather"(%4771, %1791) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x60xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4773 = stablehlo.select %147, %cst_224, %4772 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4774 = stablehlo.select %144, %4773, %4764 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4775 = stablehlo.broadcast_in_dim %4774, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4776 = stablehlo.select %140, %4775, %4762 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4777 = stablehlo.slice %4776 [0:1, 0:32, 0:8, 61:62, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4778 = stablehlo.reshape %4777 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4779 = stablehlo.slice %4778 [0:1, 0:32, 0:8, 0:61] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x61xf32>
    %4780 = stablehlo.broadcast_in_dim %4779, dims = [0, 1, 2, 3] : (tensor<1x32x8x61xf32>) -> tensor<1x32x8x61x61xf32>
    %4781 = stablehlo.slice %4776 [0:1, 0:32, 0:8, 0:61, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x61x64xf32>
    %4782 = stablehlo.slice %4781 [0:1, 0:32, 0:8, 0:61, 0:61] : (tensor<1x32x8x61x64xf32>) -> tensor<1x32x8x61x61xf32>
    %4783 = stablehlo.multiply %4780, %4782 : tensor<1x32x8x61x61xf32>
    %4784 = stablehlo.reduce(%4783 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x61x61xf32>, tensor<f32>) -> tensor<1x32x8x61xf32>
    %4785 = stablehlo.add %4779, %4784 : tensor<1x32x8x61xf32>
    %4786 = "stablehlo.gather"(%4785, %1810) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x61xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4787 = stablehlo.select %139, %cst_224, %4786 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4788 = stablehlo.select %136, %4787, %4778 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4789 = stablehlo.broadcast_in_dim %4788, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4790 = stablehlo.select %132, %4789, %4776 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4791 = stablehlo.slice %4790 [0:1, 0:32, 0:8, 62:63, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4792 = stablehlo.reshape %4791 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4793 = stablehlo.slice %4792 [0:1, 0:32, 0:8, 0:62] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x62xf32>
    %4794 = stablehlo.broadcast_in_dim %4793, dims = [0, 1, 2, 3] : (tensor<1x32x8x62xf32>) -> tensor<1x32x8x62x62xf32>
    %4795 = stablehlo.slice %4790 [0:1, 0:32, 0:8, 0:62, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x62x64xf32>
    %4796 = stablehlo.slice %4795 [0:1, 0:32, 0:8, 0:62, 0:62] : (tensor<1x32x8x62x64xf32>) -> tensor<1x32x8x62x62xf32>
    %4797 = stablehlo.multiply %4794, %4796 : tensor<1x32x8x62x62xf32>
    %4798 = stablehlo.reduce(%4797 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x62x62xf32>, tensor<f32>) -> tensor<1x32x8x62xf32>
    %4799 = stablehlo.add %4793, %4798 : tensor<1x32x8x62xf32>
    %4800 = "stablehlo.gather"(%4799, %1829) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x62xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4801 = stablehlo.select %131, %cst_224, %4800 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4802 = stablehlo.select %128, %4801, %4792 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4803 = stablehlo.broadcast_in_dim %4802, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4804 = stablehlo.select %124, %4803, %4790 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4805 = stablehlo.slice %4804 [0:1, 0:32, 0:8, 63:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x1x64xf32>
    %4806 = stablehlo.reshape %4805 : (tensor<1x32x8x1x64xf32>) -> tensor<1x32x8x64xf32>
    %4807 = stablehlo.slice %4806 [0:1, 0:32, 0:8, 0:63] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x63xf32>
    %4808 = stablehlo.broadcast_in_dim %4807, dims = [0, 1, 2, 3] : (tensor<1x32x8x63xf32>) -> tensor<1x32x8x63x63xf32>
    %4809 = stablehlo.slice %4804 [0:1, 0:32, 0:8, 0:63, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x8x63x64xf32>
    %4810 = stablehlo.slice %4809 [0:1, 0:32, 0:8, 0:63, 0:63] : (tensor<1x32x8x63x64xf32>) -> tensor<1x32x8x63x63xf32>
    %4811 = stablehlo.multiply %4808, %4810 : tensor<1x32x8x63x63xf32>
    %4812 = stablehlo.reduce(%4811 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x32x8x63x63xf32>, tensor<f32>) -> tensor<1x32x8x63xf32>
    %4813 = stablehlo.add %4807, %4812 : tensor<1x32x8x63xf32>
    %4814 = "stablehlo.gather"(%4813, %1848) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 32, 8, 1>}> : (tensor<1x32x8x63xf32>, tensor<64x1xi64>) -> tensor<1x32x8x64xf32>
    %4815 = stablehlo.select %123, %cst_224, %4814 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4816 = stablehlo.select %120, %4815, %4806 : tensor<1x32x8x64xi1>, tensor<1x32x8x64xf32>
    %4817 = stablehlo.broadcast_in_dim %4816, dims = [0, 1, 2, 4] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x64xf32>
    %4818 = stablehlo.select %115, %4817, %4804 : tensor<1x32x8x64x64xi1>, tensor<1x32x8x64x64xf32>
    %4819 = stablehlo.add %4818, %1858 : tensor<1x32x8x64x64xf32>
    %4820 = stablehlo.slice %3825 [0:1, 0:494, 4096:8192] : (tensor<1x494x8192xbf16>) -> tensor<1x494x4096xbf16>
    %4821 = stablehlo.reshape %4820 : (tensor<1x494x4096xbf16>) -> tensor<1x494x32x128xbf16>
    %4822 = stablehlo.transpose %4821, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,32,494,128]{3,1,2,0}"} : (tensor<1x494x32x128xbf16>) -> tensor<1x32x494x128xbf16>
    %4823 = stablehlo.convert %4822 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,494,128]{3,1,2,0}"} : (tensor<1x32x494x128xbf16>) -> tensor<1x32x494x128xf32>
    %4824 = stablehlo.pad %4823, %cst_240, low = [0, 0, 0, 0], high = [0, 0, 18, 0], interior = [0, 0, 0, 0] : (tensor<1x32x494x128xf32>, tensor<f32>) -> tensor<1x32x512x128xf32>
    %4825 = stablehlo.multiply %4824, %3923 : tensor<1x32x512x128xf32>
    %4826 = stablehlo.reshape %4825 : (tensor<1x32x512x128xf32>) -> tensor<1x32x8x64x128xf32>
    %4827 = stablehlo.dot_general %4819, %4826, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x64xf32>, tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %4828 = stablehlo.slice %4827 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4829 = stablehlo.reshape %4828 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4830 = stablehlo.exponential %3873 : tensor<1x32x8x64xf32>
    %4831 = stablehlo.broadcast_in_dim %4830, dims = [0, 1, 2, 3] : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x64x128xf32>
    %4832 = stablehlo.multiply %3925, %4831 : tensor<1x32x8x64x128xf32>
    %4833 = stablehlo.dot_general %4819, %4832, batching_dims = [0, 1, 2] x [0, 1, 2], contracting_dims = [4] x [3], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x8x64x64xf32>, tensor<1x32x8x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %4834 = stablehlo.slice %4833 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4835 = stablehlo.reshape %4834 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4836 = stablehlo.dot_general %4835, %cst_227, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4837 = stablehlo.subtract %4829, %4836 : tensor<1x32x64x128xf32>
    %4838 = stablehlo.dot_general %3912, %4837, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %4839 = stablehlo.add %3887, %4838 : tensor<1x32x128x128xf32>
    %4840 = stablehlo.slice %3873 [0:1, 0:32, 1:2, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %4841 = stablehlo.reshape %4840 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %4842 = stablehlo.slice %4841 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %4843 = stablehlo.reshape %4842 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %4844 = stablehlo.exponential %4843 : tensor<1x32x1x1xf32>
    %4845 = stablehlo.reshape %4844 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %4846 = stablehlo.broadcast_in_dim %4845, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %4847 = stablehlo.multiply %4839, %4846 : tensor<1x32x128x128xf32>
    %4848 = stablehlo.slice %3903 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4849 = stablehlo.reshape %4848 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4850 = stablehlo.reshape %4842 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %4851 = stablehlo.broadcast_in_dim %4850, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %4852 = stablehlo.subtract %4851, %4841 : tensor<1x32x64xf32>
    %4853 = stablehlo.exponential %4852 : tensor<1x32x64xf32>
    %4854 = stablehlo.broadcast_in_dim %4853, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %4855 = stablehlo.multiply %4849, %4854 : tensor<1x32x64x128xf32>
    %4856 = stablehlo.transpose %4855, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %4857 = stablehlo.slice %4827 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4858 = stablehlo.reshape %4857 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4859 = stablehlo.slice %4833 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4860 = stablehlo.reshape %4859 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4861 = stablehlo.dot_general %4860, %4839, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4862 = stablehlo.subtract %4858, %4861 : tensor<1x32x64x128xf32>
    %4863 = stablehlo.dot_general %4856, %4862, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %4864 = stablehlo.add %4847, %4863 : tensor<1x32x128x128xf32>
    %4865 = stablehlo.slice %3873 [0:1, 0:32, 2:3, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %4866 = stablehlo.reshape %4865 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %4867 = stablehlo.slice %4866 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %4868 = stablehlo.reshape %4867 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %4869 = stablehlo.exponential %4868 : tensor<1x32x1x1xf32>
    %4870 = stablehlo.reshape %4869 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %4871 = stablehlo.broadcast_in_dim %4870, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %4872 = stablehlo.multiply %4864, %4871 : tensor<1x32x128x128xf32>
    %4873 = stablehlo.slice %3903 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4874 = stablehlo.reshape %4873 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4875 = stablehlo.reshape %4867 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %4876 = stablehlo.broadcast_in_dim %4875, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %4877 = stablehlo.subtract %4876, %4866 : tensor<1x32x64xf32>
    %4878 = stablehlo.exponential %4877 : tensor<1x32x64xf32>
    %4879 = stablehlo.broadcast_in_dim %4878, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %4880 = stablehlo.multiply %4874, %4879 : tensor<1x32x64x128xf32>
    %4881 = stablehlo.transpose %4880, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %4882 = stablehlo.slice %4827 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4883 = stablehlo.reshape %4882 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4884 = stablehlo.slice %4833 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4885 = stablehlo.reshape %4884 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4886 = stablehlo.dot_general %4885, %4864, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4887 = stablehlo.subtract %4883, %4886 : tensor<1x32x64x128xf32>
    %4888 = stablehlo.dot_general %4881, %4887, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %4889 = stablehlo.add %4872, %4888 : tensor<1x32x128x128xf32>
    %4890 = stablehlo.slice %3873 [0:1, 0:32, 3:4, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %4891 = stablehlo.reshape %4890 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %4892 = stablehlo.slice %4891 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %4893 = stablehlo.reshape %4892 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %4894 = stablehlo.exponential %4893 : tensor<1x32x1x1xf32>
    %4895 = stablehlo.reshape %4894 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %4896 = stablehlo.broadcast_in_dim %4895, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %4897 = stablehlo.multiply %4889, %4896 : tensor<1x32x128x128xf32>
    %4898 = stablehlo.slice %3903 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4899 = stablehlo.reshape %4898 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4900 = stablehlo.reshape %4892 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %4901 = stablehlo.broadcast_in_dim %4900, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %4902 = stablehlo.subtract %4901, %4891 : tensor<1x32x64xf32>
    %4903 = stablehlo.exponential %4902 : tensor<1x32x64xf32>
    %4904 = stablehlo.broadcast_in_dim %4903, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %4905 = stablehlo.multiply %4899, %4904 : tensor<1x32x64x128xf32>
    %4906 = stablehlo.transpose %4905, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %4907 = stablehlo.slice %4827 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4908 = stablehlo.reshape %4907 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4909 = stablehlo.slice %4833 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4910 = stablehlo.reshape %4909 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4911 = stablehlo.dot_general %4910, %4889, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4912 = stablehlo.subtract %4908, %4911 : tensor<1x32x64x128xf32>
    %4913 = stablehlo.dot_general %4906, %4912, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %4914 = stablehlo.add %4897, %4913 : tensor<1x32x128x128xf32>
    %4915 = stablehlo.slice %3873 [0:1, 0:32, 4:5, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %4916 = stablehlo.reshape %4915 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %4917 = stablehlo.slice %4916 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %4918 = stablehlo.reshape %4917 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %4919 = stablehlo.exponential %4918 : tensor<1x32x1x1xf32>
    %4920 = stablehlo.reshape %4919 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %4921 = stablehlo.broadcast_in_dim %4920, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %4922 = stablehlo.multiply %4914, %4921 : tensor<1x32x128x128xf32>
    %4923 = stablehlo.slice %3903 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4924 = stablehlo.reshape %4923 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4925 = stablehlo.reshape %4917 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %4926 = stablehlo.broadcast_in_dim %4925, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %4927 = stablehlo.subtract %4926, %4916 : tensor<1x32x64xf32>
    %4928 = stablehlo.exponential %4927 : tensor<1x32x64xf32>
    %4929 = stablehlo.broadcast_in_dim %4928, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %4930 = stablehlo.multiply %4924, %4929 : tensor<1x32x64x128xf32>
    %4931 = stablehlo.transpose %4930, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %4932 = stablehlo.slice %4827 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4933 = stablehlo.reshape %4932 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4934 = stablehlo.slice %4833 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4935 = stablehlo.reshape %4934 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4936 = stablehlo.dot_general %4935, %4914, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4937 = stablehlo.subtract %4933, %4936 : tensor<1x32x64x128xf32>
    %4938 = stablehlo.dot_general %4931, %4937, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %4939 = stablehlo.add %4922, %4938 : tensor<1x32x128x128xf32>
    %4940 = stablehlo.slice %3873 [0:1, 0:32, 5:6, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %4941 = stablehlo.reshape %4940 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %4942 = stablehlo.slice %4941 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %4943 = stablehlo.reshape %4942 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %4944 = stablehlo.exponential %4943 : tensor<1x32x1x1xf32>
    %4945 = stablehlo.reshape %4944 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %4946 = stablehlo.broadcast_in_dim %4945, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %4947 = stablehlo.multiply %4939, %4946 : tensor<1x32x128x128xf32>
    %4948 = stablehlo.slice %3903 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4949 = stablehlo.reshape %4948 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4950 = stablehlo.reshape %4942 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %4951 = stablehlo.broadcast_in_dim %4950, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %4952 = stablehlo.subtract %4951, %4941 : tensor<1x32x64xf32>
    %4953 = stablehlo.exponential %4952 : tensor<1x32x64xf32>
    %4954 = stablehlo.broadcast_in_dim %4953, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %4955 = stablehlo.multiply %4949, %4954 : tensor<1x32x64x128xf32>
    %4956 = stablehlo.transpose %4955, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %4957 = stablehlo.slice %4827 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4958 = stablehlo.reshape %4957 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4959 = stablehlo.slice %4833 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4960 = stablehlo.reshape %4959 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4961 = stablehlo.dot_general %4960, %4939, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4962 = stablehlo.subtract %4958, %4961 : tensor<1x32x64x128xf32>
    %4963 = stablehlo.dot_general %4956, %4962, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %4964 = stablehlo.add %4947, %4963 : tensor<1x32x128x128xf32>
    %4965 = stablehlo.slice %3873 [0:1, 0:32, 6:7, 0:64] : (tensor<1x32x8x64xf32>) -> tensor<1x32x1x64xf32>
    %4966 = stablehlo.reshape %4965 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64xf32>
    %4967 = stablehlo.slice %4966 [0:1, 0:32, 63:64] : (tensor<1x32x64xf32>) -> tensor<1x32x1xf32>
    %4968 = stablehlo.reshape %4967 : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %4969 = stablehlo.exponential %4968 : tensor<1x32x1x1xf32>
    %4970 = stablehlo.reshape %4969 : (tensor<1x32x1x1xf32>) -> tensor<1x32xf32>
    %4971 = stablehlo.broadcast_in_dim %4970, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x128x128xf32>
    %4972 = stablehlo.multiply %4964, %4971 : tensor<1x32x128x128xf32>
    %4973 = stablehlo.slice %3903 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4974 = stablehlo.reshape %4973 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4975 = stablehlo.reshape %4967 : (tensor<1x32x1xf32>) -> tensor<1x32xf32>
    %4976 = stablehlo.broadcast_in_dim %4975, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32x64xf32>
    %4977 = stablehlo.subtract %4976, %4966 : tensor<1x32x64xf32>
    %4978 = stablehlo.exponential %4977 : tensor<1x32x64xf32>
    %4979 = stablehlo.broadcast_in_dim %4978, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %4980 = stablehlo.multiply %4974, %4979 : tensor<1x32x64x128xf32>
    %4981 = stablehlo.transpose %4980, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %4982 = stablehlo.slice %4827 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4983 = stablehlo.reshape %4982 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4984 = stablehlo.slice %4833 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4985 = stablehlo.reshape %4984 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4986 = stablehlo.dot_general %4985, %4964, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4987 = stablehlo.subtract %4983, %4986 : tensor<1x32x64x128xf32>
    %4988 = stablehlo.dot_general %4981, %4987, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %4989 = stablehlo.add %4972, %4988 : tensor<1x32x128x128xf32>
    %4990 = stablehlo.dot_general %3879, %4989, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %4991 = stablehlo.slice %3903 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %4992 = stablehlo.reshape %4991 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %4993 = stablehlo.transpose %4992, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %4994 = stablehlo.dot_general %3844, %4993, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %4995 = stablehlo.slice %3933 [0:1, 0:32, 7:8, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %4996 = stablehlo.reshape %4995 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %4997 = stablehlo.multiply %4994, %4996 : tensor<1x32x64x64xf32>
    %4998 = stablehlo.select %2034, %cst_31, %4997 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %4999 = stablehlo.slice %4827 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5000 = stablehlo.reshape %4999 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5001 = stablehlo.slice %4833 [0:1, 0:32, 7:8, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5002 = stablehlo.reshape %5001 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5003 = stablehlo.dot_general %5002, %4989, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5004 = stablehlo.subtract %5000, %5003 : tensor<1x32x64x128xf32>
    %5005 = stablehlo.dot_general %4998, %5004, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5006 = stablehlo.add %4990, %5005 : tensor<1x32x64x128xf32>
    %5007 = stablehlo.broadcast_in_dim %5006, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5008 = stablehlo.slice %3842 [0:1, 0:32, 6:7, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5009 = stablehlo.reshape %5008 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5010 = stablehlo.reshape %4965 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %5011 = stablehlo.exponential %5010 : tensor<1x32x64x1xf32>
    %5012 = stablehlo.reshape %5011 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %5013 = stablehlo.broadcast_in_dim %5012, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %5014 = stablehlo.multiply %5009, %5013 : tensor<1x32x64x128xf32>
    %5015 = stablehlo.dot_general %5014, %4964, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5016 = stablehlo.transpose %4974, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %5017 = stablehlo.dot_general %5009, %5016, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %5018 = stablehlo.slice %3933 [0:1, 0:32, 6:7, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %5019 = stablehlo.reshape %5018 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %5020 = stablehlo.multiply %5017, %5019 : tensor<1x32x64x64xf32>
    %5021 = stablehlo.select %2034, %cst_31, %5020 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %5022 = stablehlo.dot_general %5021, %4987, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5023 = stablehlo.add %5015, %5022 : tensor<1x32x64x128xf32>
    %5024 = stablehlo.broadcast_in_dim %5023, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5025 = stablehlo.slice %3842 [0:1, 0:32, 5:6, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5026 = stablehlo.reshape %5025 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5027 = stablehlo.reshape %4940 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %5028 = stablehlo.exponential %5027 : tensor<1x32x64x1xf32>
    %5029 = stablehlo.reshape %5028 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %5030 = stablehlo.broadcast_in_dim %5029, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %5031 = stablehlo.multiply %5026, %5030 : tensor<1x32x64x128xf32>
    %5032 = stablehlo.dot_general %5031, %4939, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5033 = stablehlo.transpose %4949, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %5034 = stablehlo.dot_general %5026, %5033, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %5035 = stablehlo.slice %3933 [0:1, 0:32, 5:6, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %5036 = stablehlo.reshape %5035 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %5037 = stablehlo.multiply %5034, %5036 : tensor<1x32x64x64xf32>
    %5038 = stablehlo.select %2034, %cst_31, %5037 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %5039 = stablehlo.dot_general %5038, %4962, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5040 = stablehlo.add %5032, %5039 : tensor<1x32x64x128xf32>
    %5041 = stablehlo.broadcast_in_dim %5040, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5042 = stablehlo.slice %3842 [0:1, 0:32, 4:5, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5043 = stablehlo.reshape %5042 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5044 = stablehlo.reshape %4915 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %5045 = stablehlo.exponential %5044 : tensor<1x32x64x1xf32>
    %5046 = stablehlo.reshape %5045 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %5047 = stablehlo.broadcast_in_dim %5046, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %5048 = stablehlo.multiply %5043, %5047 : tensor<1x32x64x128xf32>
    %5049 = stablehlo.dot_general %5048, %4914, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5050 = stablehlo.transpose %4924, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %5051 = stablehlo.dot_general %5043, %5050, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %5052 = stablehlo.slice %3933 [0:1, 0:32, 4:5, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %5053 = stablehlo.reshape %5052 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %5054 = stablehlo.multiply %5051, %5053 : tensor<1x32x64x64xf32>
    %5055 = stablehlo.select %2034, %cst_31, %5054 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %5056 = stablehlo.dot_general %5055, %4937, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5057 = stablehlo.add %5049, %5056 : tensor<1x32x64x128xf32>
    %5058 = stablehlo.broadcast_in_dim %5057, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5059 = stablehlo.slice %3842 [0:1, 0:32, 3:4, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5060 = stablehlo.reshape %5059 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5061 = stablehlo.reshape %4890 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %5062 = stablehlo.exponential %5061 : tensor<1x32x64x1xf32>
    %5063 = stablehlo.reshape %5062 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %5064 = stablehlo.broadcast_in_dim %5063, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %5065 = stablehlo.multiply %5060, %5064 : tensor<1x32x64x128xf32>
    %5066 = stablehlo.dot_general %5065, %4889, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5067 = stablehlo.transpose %4899, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %5068 = stablehlo.dot_general %5060, %5067, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %5069 = stablehlo.slice %3933 [0:1, 0:32, 3:4, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %5070 = stablehlo.reshape %5069 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %5071 = stablehlo.multiply %5068, %5070 : tensor<1x32x64x64xf32>
    %5072 = stablehlo.select %2034, %cst_31, %5071 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %5073 = stablehlo.dot_general %5072, %4912, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5074 = stablehlo.add %5066, %5073 : tensor<1x32x64x128xf32>
    %5075 = stablehlo.broadcast_in_dim %5074, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5076 = stablehlo.slice %3842 [0:1, 0:32, 2:3, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5077 = stablehlo.reshape %5076 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5078 = stablehlo.reshape %4865 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %5079 = stablehlo.exponential %5078 : tensor<1x32x64x1xf32>
    %5080 = stablehlo.reshape %5079 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %5081 = stablehlo.broadcast_in_dim %5080, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %5082 = stablehlo.multiply %5077, %5081 : tensor<1x32x64x128xf32>
    %5083 = stablehlo.dot_general %5082, %4864, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5084 = stablehlo.transpose %4874, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %5085 = stablehlo.dot_general %5077, %5084, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %5086 = stablehlo.slice %3933 [0:1, 0:32, 2:3, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %5087 = stablehlo.reshape %5086 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %5088 = stablehlo.multiply %5085, %5087 : tensor<1x32x64x64xf32>
    %5089 = stablehlo.select %2034, %cst_31, %5088 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %5090 = stablehlo.dot_general %5089, %4887, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5091 = stablehlo.add %5083, %5090 : tensor<1x32x64x128xf32>
    %5092 = stablehlo.broadcast_in_dim %5091, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5093 = stablehlo.slice %3842 [0:1, 0:32, 1:2, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5094 = stablehlo.reshape %5093 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5095 = stablehlo.reshape %4840 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %5096 = stablehlo.exponential %5095 : tensor<1x32x64x1xf32>
    %5097 = stablehlo.reshape %5096 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %5098 = stablehlo.broadcast_in_dim %5097, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %5099 = stablehlo.multiply %5094, %5098 : tensor<1x32x64x128xf32>
    %5100 = stablehlo.dot_general %5099, %4839, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5101 = stablehlo.transpose %4849, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %5102 = stablehlo.dot_general %5094, %5101, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %5103 = stablehlo.slice %3933 [0:1, 0:32, 1:2, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %5104 = stablehlo.reshape %5103 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %5105 = stablehlo.multiply %5102, %5104 : tensor<1x32x64x64xf32>
    %5106 = stablehlo.select %2034, %cst_31, %5105 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %5107 = stablehlo.dot_general %5106, %4862, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5108 = stablehlo.add %5100, %5107 : tensor<1x32x64x128xf32>
    %5109 = stablehlo.broadcast_in_dim %5108, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5110 = stablehlo.slice %3842 [0:1, 0:32, 0:1, 0:64, 0:128] : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x1x64x128xf32>
    %5111 = stablehlo.reshape %5110 : (tensor<1x32x1x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5112 = stablehlo.reshape %3880 : (tensor<1x32x1x64xf32>) -> tensor<1x32x64x1xf32>
    %5113 = stablehlo.exponential %5112 : tensor<1x32x64x1xf32>
    %5114 = stablehlo.reshape %5113 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %5115 = stablehlo.broadcast_in_dim %5114, dims = [0, 1, 2] : (tensor<1x32x64xf32>) -> tensor<1x32x64x128xf32>
    %5116 = stablehlo.multiply %5111, %5115 : tensor<1x32x64x128xf32>
    %5117 = stablehlo.dot_general %5116, %cst_227, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x64x128xf32>
    %5118 = stablehlo.transpose %3905, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,32,128,64]{2,3,1,0}"} : (tensor<1x32x64x128xf32>) -> tensor<1x32x128x64xf32>
    %5119 = stablehlo.dot_general %5111, %5118, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x64x64xf32>
    %5120 = stablehlo.slice %3933 [0:1, 0:32, 0:1, 0:64, 0:64] : (tensor<1x32x8x64x64xf32>) -> tensor<1x32x1x64x64xf32>
    %5121 = stablehlo.reshape %5120 : (tensor<1x32x1x64x64xf32>) -> tensor<1x32x64x64xf32>
    %5122 = stablehlo.multiply %5119, %5121 : tensor<1x32x64x64xf32>
    %5123 = stablehlo.select %2034, %cst_31, %5122 : tensor<1x32x64x64xi1>, tensor<1x32x64x64xf32>
    %5124 = stablehlo.dot_general %5123, %4837, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x32x64x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x64x128xf32>
    %5125 = stablehlo.add %5117, %5124 : tensor<1x32x64x128xf32>
    %5126 = stablehlo.broadcast_in_dim %5125, dims = [0, 1, 3, 4] : (tensor<1x32x64x128xf32>) -> tensor<1x32x8x64x128xf32>
    %5127 = stablehlo.select %2160, %5126, %cst_23 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5128 = stablehlo.select %2142, %5109, %5127 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5129 = stablehlo.select %2124, %5092, %5128 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5130 = stablehlo.select %2106, %5075, %5129 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5131 = stablehlo.select %2088, %5058, %5130 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5132 = stablehlo.select %2070, %5041, %5131 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5133 = stablehlo.select %2052, %5024, %5132 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5134 = stablehlo.select %1, %5007, %5133 : tensor<1x32x8x64x128xi1>, tensor<1x32x8x64x128xf32>
    %5135 = stablehlo.reshape %5134 : (tensor<1x32x8x64x128xf32>) -> tensor<1x32x512x128xf32>
    %5136 = stablehlo.slice %5135 [0:1, 0:32, 0:494, 0:128] : (tensor<1x32x512x128xf32>) -> tensor<1x32x494x128xf32>
    %5137 = stablehlo.transpose %5136, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,494,32,128]{3,1,2,0}"} : (tensor<1x32x494x128xf32>) -> tensor<1x494x32x128xf32>
    %5138 = stablehlo.convert %5137 {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,494,32,128]{3,1,2,0}"} : (tensor<1x494x32x128xf32>) -> tensor<1x494x32x128xbf16>
    %5139 = stablehlo.reshape %5138 : (tensor<1x494x32x128xbf16>) -> tensor<15808x128xbf16>
    %5140 = stablehlo.reshape %arg51 : (tensor<128xbf16>) -> tensor<1x1x128xbf16>
    %5141 = stablehlo.custom_call @tt.mark_argument(%5140) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.norm.weight"}} : (tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16>
    %5142 = stablehlo.reshape %5141 : (tensor<1x1x128xbf16>) -> tensor<128xbf16>
    %5143 = stablehlo.composite "tenstorrent.rms_norm" %5139, %5142 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<128> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_9} : (tensor<15808x128xbf16>, tensor<128xbf16>) -> tensor<15808x128xbf16>
    %5144 = stablehlo.convert %5143 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %5145 = stablehlo.reshape %arg11 : (tensor<4096x2048xbf16>) -> tensor<1x4096x2048xbf16>
    %5146 = stablehlo.custom_call @tt.mark_argument(%5145) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.in_proj_z.weight"}} : (tensor<1x4096x2048xbf16>) -> tensor<1x4096x2048xbf16>
    %5147 = stablehlo.reshape %5146 : (tensor<1x4096x2048xbf16>) -> tensor<4096x2048xbf16>
    %5148 = stablehlo.transpose %5147, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,4096]{0,1}"} : (tensor<4096x2048xbf16>) -> tensor<2048x4096xbf16>
    %5149 = stablehlo.dot_general %3806, %5148, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x4096xbf16>) -> tensor<494x4096xbf16>
    %5150 = stablehlo.reshape %5149 : (tensor<494x4096xbf16>) -> tensor<15808x128xbf16>
    %5151 = stablehlo.convert %5150 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %5152 = stablehlo.logistic %5151 : tensor<15808x128xf32>
    %5153 = stablehlo.multiply %5151, %5152 : tensor<15808x128xf32>
    %5154 = stablehlo.multiply %5144, %5153 : tensor<15808x128xf32>
    %5155 = stablehlo.convert %5154 : (tensor<15808x128xf32>) -> tensor<15808x128xbf16>
    %5156 = stablehlo.reshape %5155 : (tensor<15808x128xbf16>) -> tensor<494x4096xbf16>
    %5157 = stablehlo.reshape %arg10 : (tensor<2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %5158 = stablehlo.custom_call @tt.mark_argument(%5157) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.linear_attn.out_proj.weight"}} : (tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %5159 = stablehlo.reshape %5158 : (tensor<1x2048x4096xbf16>) -> tensor<2048x4096xbf16>
    %5160 = stablehlo.transpose %5159, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[4096,2048]{0,1}"} : (tensor<2048x4096xbf16>) -> tensor<4096x2048xbf16>
    %5161 = stablehlo.dot_general %5156, %5160, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<494x2048xbf16>
    %5162 = stablehlo.reshape %5161 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %5163 = stablehlo.add %3799, %5162 : tensor<1x494x2048xbf16>
    %5164 = stablehlo.reshape %arg9 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %5165 = stablehlo.custom_call @tt.mark_argument(%5164) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.post_attention_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5166 = stablehlo.reshape %5165 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %5167 = stablehlo.convert %5166 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %5168 = stablehlo.add %5167, %cst_231 : tensor<2048xf32>
    %5169 = stablehlo.composite "tenstorrent.rms_norm" %5163, %5168 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_12} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %5170 = stablehlo.reshape %5169 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %5171 = stablehlo.reshape %arg60 : (tensor<256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %5172 = stablehlo.custom_call @tt.mark_argument(%5171) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.mlp.gate.weight"}} : (tensor<1x256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %5173 = stablehlo.reshape %5172 : (tensor<1x256x2048xbf16>) -> tensor<256x2048xbf16>
    %5174 = stablehlo.transpose %5173, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,256]{0,1}"} : (tensor<256x2048xbf16>) -> tensor<2048x256xbf16>
    %5175 = stablehlo.dot_general %5170, %5174, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x256xbf16>) -> tensor<494x256xbf16>
    %5176 = stablehlo.convert %5175 : (tensor<494x256xbf16>) -> tensor<494x256xf32>
    %5177 = stablehlo.reduce(%5176 init: %cst_236) applies stablehlo.maximum across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %5178 = stablehlo.broadcast_in_dim %5177, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %5179 = stablehlo.subtract %5176, %5178 : tensor<494x256xf32>
    %5180 = stablehlo.exponential %5179 : tensor<494x256xf32>
    %5181 = stablehlo.reduce(%5180 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %5182 = stablehlo.broadcast_in_dim %5181, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %5183 = stablehlo.divide %5180, %5182 : tensor<494x256xf32>
    %5184:2 = stablehlo.composite "tenstorrent.topk" %5183 {composite_attributes = {dim = -1 : i64, k = 8 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk.impl_1} : (tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>)
    %5185 = stablehlo.reduce(%5184#0 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x8xf32>, tensor<f32>) -> tensor<494xf32>
    %5186 = stablehlo.broadcast_in_dim %5185, dims = [0] : (tensor<494xf32>) -> tensor<494x8xf32>
    %5187 = stablehlo.divide %5184#0, %5186 : tensor<494x8xf32>
    %5188 = stablehlo.concatenate %5187, %cst_22, dim = 0 : (tensor<494x8xf32>, tensor<18x8xf32>) -> tensor<512x8xf32>
    %5189 = stablehlo.convert %5188 : (tensor<512x8xf32>) -> tensor<512x8xbf16>
    %5190 = stablehlo.reshape %5189 : (tensor<512x8xbf16>) -> tensor<512x1x8xbf16>
    %5191 = stablehlo.concatenate %5184#1, %c_21, dim = 0 : (tensor<494x8xi64>, tensor<18x8xi64>) -> tensor<512x8xi64>
    %5192 = stablehlo.broadcast_in_dim %5191, dims = [0, 1] : (tensor<512x8xi64>) -> tensor<512x8x256xi64>
    %5193 = stablehlo.compare  EQ, %5192, %2244 : (tensor<512x8x256xi64>, tensor<512x8x256xi64>) -> tensor<512x8x256xi1>
    %5194 = stablehlo.convert %5193 : (tensor<512x8x256xi1>) -> tensor<512x8x256xbf16>
    %5195 = stablehlo.dot_general %5190, %5194, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x1x8xbf16>, tensor<512x8x256xbf16>) -> tensor<512x1x256xbf16>
    %5196 = stablehlo.reshape %5195 : (tensor<512x1x256xbf16>) -> tensor<1x512x256xbf16>
    %5197 = stablehlo.concatenate %5196, %5196, %5196, %5196, %5196, %5196, %5196, %5196, dim = 1 : (tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>) -> tensor<1x4096x256xbf16>
    %5198 = stablehlo.reshape %5197 : (tensor<1x4096x256xbf16>) -> tensor<1x1x4096x256xbf16>
    %5199 = stablehlo.concatenate %5170, %cst_18, dim = 0 : (tensor<494x2048xbf16>, tensor<18x2048xbf16>) -> tensor<512x2048xbf16>
    %5200 = stablehlo.reshape %5199 : (tensor<512x2048xbf16>) -> tensor<1x1x512x2048xbf16>
    %5201 = stablehlo.reshape %5191 : (tensor<512x8xi64>) -> tensor<1x1x512x8xi64>
    %5202 = stablehlo.custom_call @tt.all_to_all_dispatch(%5200, %5201, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8"}, xla_shape = "(bf16[1,8,512,2048]{3,2,1,0}, s64[1,8,512,8]{3,2,1,0})"} : (tensor<1x1x512x2048xbf16>, tensor<1x1x512x8xi64>, tensor<1x1x256x8xui16>) -> tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>
    %5203 = stablehlo.get_tuple_element %5202[1] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x8xi64>
    %5204 = stablehlo.reshape %5203 : (tensor<1x8x512x8xi64>) -> tensor<1x1x4096x8xi64>
    %5205 = stablehlo.custom_call @tt.moe_expert_token_remap(%5198, %2258, %5204) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {num_devices = "8", reduction_size = "32"}, xla_shape = "(bf16[1,1,4096,256]{3,2,1,0}, bf16[1,1,128,256]{3,2,1,0})"} : (tensor<1x1x4096x256xbf16>, tensor<1x1x256x8xui16>, tensor<1x1x4096x8xi64>) -> tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>
    %5206 = stablehlo.get_tuple_element %5202[0] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x2048xbf16>
    %5207 = stablehlo.reshape %5206 : (tensor<1x8x512x2048xbf16>) -> tensor<8x16x32x2048xbf16>
    %5208 = stablehlo.custom_call @tt.mark_argument(%arg62) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.mlp.experts.gate_up_proj"}} : (tensor<256x1024x2048xbf16>) -> tensor<256x1024x2048xbf16>
    %5209 = stablehlo.transpose %5208, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,2048,1024]{1,2,0}"} : (tensor<256x1024x2048xbf16>) -> tensor<256x2048x1024xbf16>
    %5210 = stablehlo.reshape %5209 : (tensor<256x2048x1024xbf16>) -> tensor<1x256x2048x1024xbf16>
    %5211 = stablehlo.get_tuple_element %5205[1] : (tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>) -> tensor<1x1x128x256xbf16>
    %5212 = stablehlo.reshape %5211 : (tensor<1x1x128x256xbf16>) -> tensor<8x16x1x256xbf16>
    %5213 = stablehlo.custom_call @tt.sparse_matmul(%5207, %5210, %5212) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True", nnz = "0"}} : (tensor<8x16x32x2048xbf16>, tensor<1x256x2048x1024xbf16>, tensor<8x16x1x256xbf16>) -> tensor<8x16x1x256x32x1024xbf16>
    %5214 = stablehlo.reshape %5213 : (tensor<8x16x1x256x32x1024xbf16>) -> tensor<8x16x256x32x1024xbf16>
    %5215 = stablehlo.slice %5214 [0:8, 0:16, 0:256, 0:32, 0:512] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %5216 = stablehlo.convert %5215 : (tensor<8x16x256x32x512xbf16>) -> tensor<8x16x256x32x512xf32>
    %5217 = stablehlo.logistic %5216 : tensor<8x16x256x32x512xf32>
    %5218 = stablehlo.multiply %5216, %5217 : tensor<8x16x256x32x512xf32>
    %5219 = stablehlo.convert %5218 : (tensor<8x16x256x32x512xf32>) -> tensor<8x16x256x32x512xbf16>
    %5220 = stablehlo.slice %5214 [0:8, 0:16, 0:256, 0:32, 512:1024] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %5221 = stablehlo.multiply %5219, %5220 : tensor<8x16x256x32x512xbf16>
    %5222 = stablehlo.reshape %5221 : (tensor<8x16x256x32x512xbf16>) -> tensor<128x256x32x512xbf16>
    %5223 = stablehlo.custom_call @tt.mark_argument(%arg61) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.mlp.experts.down_proj"}} : (tensor<256x2048x512xbf16>) -> tensor<256x2048x512xbf16>
    %5224 = stablehlo.transpose %5223, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,512,2048]{1,2,0}"} : (tensor<256x2048x512xbf16>) -> tensor<256x512x2048xbf16>
    %5225 = stablehlo.reshape %5224 : (tensor<256x512x2048xbf16>) -> tensor<1x256x512x2048xbf16>
    %5226 = stablehlo.custom_call @tt.sparse_matmul(%5222, %5225, %5211) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "True", is_input_b_sparse = "False", nnz = "0"}} : (tensor<128x256x32x512xbf16>, tensor<1x256x512x2048xbf16>, tensor<1x1x128x256xbf16>) -> tensor<128x256x32x2048xbf16>
    %5227 = stablehlo.transpose %5226, dims = [1, 0, 2, 3] {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, xla_shape = "bf16[256,128,32,2048]{3,2,0,1}"} : (tensor<128x256x32x2048xbf16>) -> tensor<256x128x32x2048xbf16>
    %5228 = stablehlo.reshape %5227 : (tensor<256x128x32x2048xbf16>) -> tensor<256x1x4096x2048xbf16>
    %5229 = stablehlo.custom_call @tt.all_to_all_combine(%5228, %5204, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8", num_experts_per_tok = "8", output_shard_dim = "2"}} : (tensor<256x1x4096x2048xbf16>, tensor<1x1x4096x8xi64>, tensor<1x1x256x8xui16>) -> tensor<8x1x512x2048xbf16>
    %5230 = stablehlo.transpose %5188, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[8,512]{0,1}"} : (tensor<512x8xf32>) -> tensor<8x512xf32>
    %5231 = stablehlo.reshape %5230 : (tensor<8x512xf32>) -> tensor<8x1x512x1xf32>
    %5232 = stablehlo.convert %5231 : (tensor<8x1x512x1xf32>) -> tensor<8x1x512x1xbf16>
    %5233 = stablehlo.reshape %5232 : (tensor<8x1x512x1xbf16>) -> tensor<8x1x512xbf16>
    %5234 = stablehlo.broadcast_in_dim %5233, dims = [0, 1, 2] : (tensor<8x1x512xbf16>) -> tensor<8x1x512x2048xbf16>
    %5235 = stablehlo.multiply %5229, %5234 : tensor<8x1x512x2048xbf16>
    %5236 = stablehlo.reduce(%5235 init: %cst_239) applies stablehlo.add across dimensions = [0] : (tensor<8x1x512x2048xbf16>, tensor<bf16>) -> tensor<1x512x2048xbf16>
    %5237 = stablehlo.reshape %5236 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5238 = stablehlo.slice %5237 [0:494, 0:2048] : (tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %5239 = stablehlo.reshape %arg59 : (tensor<1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5240 = stablehlo.custom_call @tt.mark_argument(%5239) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.mlp.shared_expert_gate.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5241 = stablehlo.reshape %5240 : (tensor<1x1x2048xbf16>) -> tensor<2048x1xbf16>
    %5242 = stablehlo.dot_general %5170, %5241, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x1xbf16>) -> tensor<494x1xbf16>
    %5243 = stablehlo.logistic %5242 : tensor<494x1xbf16>
    %5244 = stablehlo.reshape %5243 : (tensor<494x1xbf16>) -> tensor<494xbf16>
    %5245 = stablehlo.broadcast_in_dim %5244, dims = [0] : (tensor<494xbf16>) -> tensor<494x2048xbf16>
    %5246 = stablehlo.reshape %arg58 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5247 = stablehlo.custom_call @tt.mark_argument(%5246) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.mlp.shared_expert.gate_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5248 = stablehlo.reshape %5247 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5249 = stablehlo.transpose %5248, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %5250 = stablehlo.dot_general %5170, %5249, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %5251 = stablehlo.convert %5250 : (tensor<494x512xbf16>) -> tensor<494x512xf32>
    %5252 = stablehlo.logistic %5251 : tensor<494x512xf32>
    %5253 = stablehlo.multiply %5251, %5252 : tensor<494x512xf32>
    %5254 = stablehlo.convert %5253 : (tensor<494x512xf32>) -> tensor<494x512xbf16>
    %5255 = stablehlo.reshape %arg8 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5256 = stablehlo.custom_call @tt.mark_argument(%5255) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.mlp.shared_expert.up_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5257 = stablehlo.reshape %5256 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5258 = stablehlo.transpose %5257, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %5259 = stablehlo.dot_general %5170, %5258, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %5260 = stablehlo.multiply %5254, %5259 : tensor<494x512xbf16>
    %5261 = stablehlo.reshape %arg7 : (tensor<2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %5262 = stablehlo.custom_call @tt.mark_argument(%5261) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.2.mlp.shared_expert.down_proj.weight"}} : (tensor<1x2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %5263 = stablehlo.reshape %5262 : (tensor<1x2048x512xbf16>) -> tensor<2048x512xbf16>
    %5264 = stablehlo.transpose %5263, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[512,2048]{0,1}"} : (tensor<2048x512xbf16>) -> tensor<512x2048xbf16>
    %5265 = stablehlo.dot_general %5260, %5264, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x512xbf16>, tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %5266 = stablehlo.multiply %5245, %5265 : tensor<494x2048xbf16>
    %5267 = stablehlo.add %5238, %5266 : tensor<494x2048xbf16>
    %5268 = stablehlo.reshape %5267 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %5269 = stablehlo.add %5163, %5268 : tensor<1x494x2048xbf16>
    %5270 = stablehlo.reshape %arg6 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %5271 = stablehlo.custom_call @tt.mark_argument(%5270) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.input_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5272 = stablehlo.reshape %5271 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %5273 = stablehlo.convert %5272 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %5274 = stablehlo.add %5273, %cst_231 : tensor<2048xf32>
    %5275 = stablehlo.composite "tenstorrent.rms_norm" %5269, %5274 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_10} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %5276 = stablehlo.reshape %5275 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %5277 = stablehlo.reshape %arg5 : (tensor<8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %5278 = stablehlo.custom_call @tt.mark_argument(%5277) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.self_attn.q_proj.weight"}} : (tensor<1x8192x2048xbf16>) -> tensor<1x8192x2048xbf16>
    %5279 = stablehlo.reshape %5278 : (tensor<1x8192x2048xbf16>) -> tensor<8192x2048xbf16>
    %5280 = stablehlo.transpose %5279, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,8192]{0,1}"} : (tensor<8192x2048xbf16>) -> tensor<2048x8192xbf16>
    %5281 = stablehlo.dot_general %5276, %5280, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x8192xbf16>) -> tensor<494x8192xbf16>
    %5282 = stablehlo.reshape %5281 : (tensor<494x8192xbf16>) -> tensor<1x494x16x512xbf16>
    %5283 = stablehlo.slice %5282 [0:1, 0:494, 0:16, 0:256] : (tensor<1x494x16x512xbf16>) -> tensor<1x494x16x256xbf16>
    %5284 = stablehlo.reshape %arg69 : (tensor<256xbf16>) -> tensor<1x1x256xbf16>
    %5285 = stablehlo.custom_call @tt.mark_argument(%5284) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.self_attn.q_norm.weight"}} : (tensor<1x1x256xbf16>) -> tensor<1x1x256xbf16>
    %5286 = stablehlo.reshape %5285 : (tensor<1x1x256xbf16>) -> tensor<256xbf16>
    %5287 = stablehlo.convert %5286 : (tensor<256xbf16>) -> tensor<256xf32>
    %5288 = stablehlo.add %5287, %cst_17 : tensor<256xf32>
    %5289 = stablehlo.composite "tenstorrent.rms_norm" %5283, %5288 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<256> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_11} : (tensor<1x494x16x256xbf16>, tensor<256xf32>) -> tensor<1x494x16x256xbf16>
    %5290 = stablehlo.transpose %5289, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,16,494,256]{3,1,2,0}"} : (tensor<1x494x16x256xbf16>) -> tensor<1x16x494x256xbf16>
    %5291 = stablehlo.slice %5290 [0:1, 0:16, 0:494, 0:64] : (tensor<1x16x494x256xbf16>) -> tensor<1x16x494x64xbf16>
    %5292 = stablehlo.broadcast_in_dim %c_16, dims = [0, 1] : (tensor<3x1xi1>) -> tensor<3x1x494x32xi1>
    %5293 = stablehlo.broadcast_in_dim %arg33, dims = [] : (tensor<i1>) -> tensor<32xi1>
    %5294 = stablehlo.and %5293, %c_15 : tensor<32xi1>
    %5295 = stablehlo.and %5294, %c_14 : tensor<32xi1>
    %5296 = stablehlo.and %5295, %c_13 : tensor<32xi1>
    %5297 = stablehlo.reshape %5296 : (tensor<32xi1>) -> tensor<1x1x32xi1>
    %5298 = stablehlo.reshape %5296 : (tensor<32xi1>) -> tensor<1x32xi1>
    %5299 = stablehlo.broadcast_in_dim %5298, dims = [0, 2] : (tensor<1x32xi1>) -> tensor<1x494x32xi1>
    %5300 = stablehlo.not %5297 : tensor<1x1x32xi1>
    %5301 = stablehlo.reshape %5300 : (tensor<1x1x32xi1>) -> tensor<1x32xi1>
    %5302 = stablehlo.broadcast_in_dim %5301, dims = [0, 2] : (tensor<1x32xi1>) -> tensor<1x494x32xi1>
    %5303 = stablehlo.and %5293, %c_11 : tensor<32xi1>
    %5304 = stablehlo.and %5303, %c_10 : tensor<32xi1>
    %5305 = stablehlo.reshape %5304 : (tensor<32xi1>) -> tensor<1x1x32xi1>
    %5306 = stablehlo.reshape %5304 : (tensor<32xi1>) -> tensor<1x32xi1>
    %5307 = stablehlo.broadcast_in_dim %5306, dims = [0, 2] : (tensor<1x32xi1>) -> tensor<1x494x32xi1>
    %5308 = stablehlo.not %5305 : tensor<1x1x32xi1>
    %5309 = stablehlo.reshape %5308 : (tensor<1x1x32xi1>) -> tensor<1x32xi1>
    %5310 = stablehlo.broadcast_in_dim %5309, dims = [0, 2] : (tensor<1x32xi1>) -> tensor<1x494x32xi1>
    %5311 = stablehlo.reshape %arg68 : (tensor<32xf32>) -> tensor<1x1x32xf32>
    %5312 = stablehlo.custom_call @tt.mark_argument(%5311) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "constant", ttir.name = "rotary_emb.inv_freq"}} : (tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %5313 = stablehlo.reshape %5312 : (tensor<1x1x32xf32>) -> tensor<1x32x1xf32>
    %5314 = stablehlo.broadcast_in_dim %5313, dims = [1, 2, 3] : (tensor<1x32x1xf32>) -> tensor<3x1x32x1xf32>
    %5315 = stablehlo.reshape %5314 : (tensor<3x1x32x1xf32>) -> tensor<3x32x1xf32>
    %5316 = stablehlo.custom_call @tt.mark_argument(%arg67) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "input", ttir.name = "arg71_1"}} : (tensor<3x1x494xi64>) -> tensor<3x1x494xi64>
    %5317 = stablehlo.reshape %5316 : (tensor<3x1x494xi64>) -> tensor<3x1x1x494xi64>
    %5318 = stablehlo.convert %5317 : (tensor<3x1x1x494xi64>) -> tensor<3x1x1x494xf32>
    %5319 = stablehlo.reshape %5318 : (tensor<3x1x1x494xf32>) -> tensor<3x1x494xf32>
    %5320 = stablehlo.dot_general %5315, %5319, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x32x1xf32>, tensor<3x1x494xf32>) -> tensor<3x32x494xf32>
    %5321 = stablehlo.reshape %5320 : (tensor<3x32x494xf32>) -> tensor<3x1x32x494xf32>
    %5322 = stablehlo.transpose %5321, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[3,1,494,32]{2,3,1,0}"} : (tensor<3x1x32x494xf32>) -> tensor<3x1x494x32xf32>
    %5323 = stablehlo.slice %5322 [1:2, 0:1, 0:494, 0:32] : (tensor<3x1x494x32xf32>) -> tensor<1x1x494x32xf32>
    %5324 = stablehlo.reshape %5323 : (tensor<1x1x494x32xf32>) -> tensor<1x494x32xf32>
    %5325 = stablehlo.slice %5324 [0:1, 0:494, 1:32:3] : (tensor<1x494x32xf32>) -> tensor<1x494x11xf32>
    %5326 = stablehlo.floor %cst_8 : tensor<32xf32>
    %5327 = stablehlo.convert %5326 : (tensor<32xf32>) -> tensor<32xi64>
    %5328 = stablehlo.clamp %c_9, %5327, %c_7 : tensor<32xi64>
    %5329 = stablehlo.compare  LT, %5328, %c_9 : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
    %5330 = stablehlo.add %5328, %c_6 : tensor<32xi64>
    %5331 = stablehlo.select %5329, %5330, %5328 : tensor<32xi1>, tensor<32xi64>
    %5332 = stablehlo.reshape %5331 : (tensor<32xi64>) -> tensor<32x1xi64>
    %5333 = "stablehlo.gather"(%5325, %5332) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 494, 1>}> : (tensor<1x494x11xf32>, tensor<32x1xi64>) -> tensor<1x494x32xf32>
    %5334 = stablehlo.select %5310, %cst_12, %5333 : tensor<1x494x32xi1>, tensor<1x494x32xf32>
    %5335 = stablehlo.slice %5322 [0:1, 0:1, 0:494, 0:32] : (tensor<3x1x494x32xf32>) -> tensor<1x1x494x32xf32>
    %5336 = stablehlo.reshape %5335 : (tensor<1x1x494x32xf32>) -> tensor<1x494x32xf32>
    %5337 = stablehlo.select %5307, %5334, %5336 : tensor<1x494x32xi1>, tensor<1x494x32xf32>
    %5338 = stablehlo.broadcast_in_dim %5337, dims = [1, 2, 3] : (tensor<1x494x32xf32>) -> tensor<3x1x494x32xf32>
    %5339 = stablehlo.select %5292, %5338, %5322 : tensor<3x1x494x32xi1>, tensor<3x1x494x32xf32>
    %5340 = stablehlo.slice %5339 [2:3, 0:1, 0:494, 0:32] : (tensor<3x1x494x32xf32>) -> tensor<1x1x494x32xf32>
    %5341 = stablehlo.reshape %5340 : (tensor<1x1x494x32xf32>) -> tensor<1x494x32xf32>
    %5342 = stablehlo.slice %5341 [0:1, 0:494, 2:30:3] : (tensor<1x494x32xf32>) -> tensor<1x494x10xf32>
    %5343 = stablehlo.floor %cst_5 : tensor<32xf32>
    %5344 = stablehlo.convert %5343 : (tensor<32xf32>) -> tensor<32xi64>
    %5345 = stablehlo.clamp %c_9, %5344, %c : tensor<32xi64>
    %5346 = stablehlo.compare  LT, %5345, %c_9 : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
    %5347 = stablehlo.add %5345, %c_7 : tensor<32xi64>
    %5348 = stablehlo.select %5346, %5347, %5345 : tensor<32xi1>, tensor<32xi64>
    %5349 = stablehlo.reshape %5348 : (tensor<32xi64>) -> tensor<32x1xi64>
    %5350 = "stablehlo.gather"(%5342, %5349) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 494, 1>}> : (tensor<1x494x10xf32>, tensor<32x1xi64>) -> tensor<1x494x32xf32>
    %5351 = stablehlo.select %5302, %cst_12, %5350 : tensor<1x494x32xi1>, tensor<1x494x32xf32>
    %5352 = stablehlo.slice %5339 [0:1, 0:1, 0:494, 0:32] : (tensor<3x1x494x32xf32>) -> tensor<1x1x494x32xf32>
    %5353 = stablehlo.reshape %5352 : (tensor<1x1x494x32xf32>) -> tensor<1x494x32xf32>
    %5354 = stablehlo.select %5299, %5351, %5353 : tensor<1x494x32xi1>, tensor<1x494x32xf32>
    %5355 = stablehlo.broadcast_in_dim %5354, dims = [1, 2, 3] : (tensor<1x494x32xf32>) -> tensor<3x1x494x32xf32>
    %5356 = stablehlo.select %5292, %5355, %5339 : tensor<3x1x494x32xi1>, tensor<3x1x494x32xf32>
    %5357 = stablehlo.slice %5356 [0:1, 0:1, 0:494, 0:32] : (tensor<3x1x494x32xf32>) -> tensor<1x1x494x32xf32>
    %5358 = stablehlo.reshape %5357 : (tensor<1x1x494x32xf32>) -> tensor<1x494x32xf32>
    %5359 = stablehlo.concatenate %5358, %5358, dim = 2 : (tensor<1x494x32xf32>, tensor<1x494x32xf32>) -> tensor<1x494x64xf32>
    %5360 = stablehlo.cosine %5359 : tensor<1x494x64xf32>
    %5361 = stablehlo.convert %5360 : (tensor<1x494x64xf32>) -> tensor<1x494x64xbf16>
    %5362 = stablehlo.broadcast_in_dim %5361, dims = [0, 2, 3] : (tensor<1x494x64xbf16>) -> tensor<1x16x494x64xbf16>
    %5363 = stablehlo.multiply %5291, %5362 : tensor<1x16x494x64xbf16>
    %5364 = stablehlo.slice %5291 [0:1, 0:16, 0:494, 32:64] : (tensor<1x16x494x64xbf16>) -> tensor<1x16x494x32xbf16>
    %5365 = stablehlo.negate %5364 : tensor<1x16x494x32xbf16>
    %5366 = stablehlo.slice %5291 [0:1, 0:16, 0:494, 0:32] : (tensor<1x16x494x64xbf16>) -> tensor<1x16x494x32xbf16>
    %5367 = stablehlo.concatenate %5365, %5366, dim = 3 : (tensor<1x16x494x32xbf16>, tensor<1x16x494x32xbf16>) -> tensor<1x16x494x64xbf16>
    %5368 = stablehlo.sine %5359 : tensor<1x494x64xf32>
    %5369 = stablehlo.convert %5368 : (tensor<1x494x64xf32>) -> tensor<1x494x64xbf16>
    %5370 = stablehlo.broadcast_in_dim %5369, dims = [0, 2, 3] : (tensor<1x494x64xbf16>) -> tensor<1x16x494x64xbf16>
    %5371 = stablehlo.multiply %5367, %5370 : tensor<1x16x494x64xbf16>
    %5372 = stablehlo.add %5363, %5371 : tensor<1x16x494x64xbf16>
    %5373 = stablehlo.slice %5290 [0:1, 0:16, 0:494, 64:256] : (tensor<1x16x494x256xbf16>) -> tensor<1x16x494x192xbf16>
    %5374 = stablehlo.concatenate %5372, %5373, dim = 3 : (tensor<1x16x494x64xbf16>, tensor<1x16x494x192xbf16>) -> tensor<1x16x494x256xbf16>
    %5375 = stablehlo.convert %5374 : (tensor<1x16x494x256xbf16>) -> tensor<1x16x494x256xf32>
    %5376 = stablehlo.multiply %5375, %cst_4 : tensor<1x16x494x256xf32>
    %5377 = stablehlo.reshape %arg66 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5378 = stablehlo.custom_call @tt.mark_argument(%5377) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.self_attn.k_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5379 = stablehlo.reshape %5378 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5380 = stablehlo.transpose %5379, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %5381 = stablehlo.dot_general %5276, %5380, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %5382 = stablehlo.reshape %5381 : (tensor<494x512xbf16>) -> tensor<1x494x2x256xbf16>
    %5383 = stablehlo.reshape %arg65 : (tensor<256xbf16>) -> tensor<1x1x256xbf16>
    %5384 = stablehlo.custom_call @tt.mark_argument(%5383) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.self_attn.k_norm.weight"}} : (tensor<1x1x256xbf16>) -> tensor<1x1x256xbf16>
    %5385 = stablehlo.reshape %5384 : (tensor<1x1x256xbf16>) -> tensor<256xbf16>
    %5386 = stablehlo.convert %5385 : (tensor<256xbf16>) -> tensor<256xf32>
    %5387 = stablehlo.add %5386, %cst_17 : tensor<256xf32>
    %5388 = stablehlo.composite "tenstorrent.rms_norm" %5382, %5387 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<256> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_1} : (tensor<1x494x2x256xbf16>, tensor<256xf32>) -> tensor<1x494x2x256xbf16>
    %5389 = stablehlo.transpose %5388, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,2,494,256]{3,1,2,0}"} : (tensor<1x494x2x256xbf16>) -> tensor<1x2x494x256xbf16>
    %5390 = stablehlo.slice %5389 [0:1, 0:2, 0:494, 0:64] : (tensor<1x2x494x256xbf16>) -> tensor<1x2x494x64xbf16>
    %5391 = stablehlo.broadcast_in_dim %5361, dims = [0, 2, 3] : (tensor<1x494x64xbf16>) -> tensor<1x2x494x64xbf16>
    %5392 = stablehlo.multiply %5390, %5391 : tensor<1x2x494x64xbf16>
    %5393 = stablehlo.slice %5390 [0:1, 0:2, 0:494, 32:64] : (tensor<1x2x494x64xbf16>) -> tensor<1x2x494x32xbf16>
    %5394 = stablehlo.negate %5393 : tensor<1x2x494x32xbf16>
    %5395 = stablehlo.slice %5390 [0:1, 0:2, 0:494, 0:32] : (tensor<1x2x494x64xbf16>) -> tensor<1x2x494x32xbf16>
    %5396 = stablehlo.concatenate %5394, %5395, dim = 3 : (tensor<1x2x494x32xbf16>, tensor<1x2x494x32xbf16>) -> tensor<1x2x494x64xbf16>
    %5397 = stablehlo.broadcast_in_dim %5369, dims = [0, 2, 3] : (tensor<1x494x64xbf16>) -> tensor<1x2x494x64xbf16>
    %5398 = stablehlo.multiply %5396, %5397 : tensor<1x2x494x64xbf16>
    %5399 = stablehlo.add %5392, %5398 : tensor<1x2x494x64xbf16>
    %5400 = stablehlo.slice %5389 [0:1, 0:2, 0:494, 64:256] : (tensor<1x2x494x256xbf16>) -> tensor<1x2x494x192xbf16>
    %5401 = stablehlo.concatenate %5399, %5400, dim = 3 : (tensor<1x2x494x64xbf16>, tensor<1x2x494x192xbf16>) -> tensor<1x2x494x256xbf16>
    %5402 = stablehlo.broadcast_in_dim %5401, dims = [0, 1, 3, 4] : (tensor<1x2x494x256xbf16>) -> tensor<1x2x8x494x256xbf16>
    %5403 = stablehlo.reshape %5402 : (tensor<1x2x8x494x256xbf16>) -> tensor<1x16x494x256xbf16>
    %5404 = stablehlo.convert %5403 : (tensor<1x16x494x256xbf16>) -> tensor<1x16x494x256xf32>
    %5405 = stablehlo.transpose %5404, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,16,256,494]{2,3,1,0}"} : (tensor<1x16x494x256xf32>) -> tensor<1x16x256x494xf32>
    %5406 = stablehlo.multiply %5405, %cst_3 : tensor<1x16x256x494xf32>
    %5407 = stablehlo.dot_general %5376, %5406, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x16x494x256xf32>, tensor<1x16x256x494xf32>) -> tensor<1x16x494x494xf32>
    %5408 = stablehlo.custom_call @tt.mark_argument(%arg64) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "input", ttir.name = "arg73_1"}} : (tensor<1x1x494x494xi1>) -> tensor<1x1x494x494xi1>
    %5409 = stablehlo.select %5408, %cst_2, %cst_1 : tensor<1x1x494x494xi1>, tensor<1x1x494x494xbf16>
    %5410 = stablehlo.convert %5409 : (tensor<1x1x494x494xbf16>) -> tensor<1x1x494x494xf32>
    %5411 = stablehlo.reshape %5410 : (tensor<1x1x494x494xf32>) -> tensor<1x494x494xf32>
    %5412 = stablehlo.broadcast_in_dim %5411, dims = [0, 2, 3] : (tensor<1x494x494xf32>) -> tensor<1x16x494x494xf32>
    %5413 = stablehlo.add %5407, %5412 : tensor<1x16x494x494xf32>
    %5414 = stablehlo.convert %5413 : (tensor<1x16x494x494xf32>) -> tensor<1x16x494x494xf64>
    %5415 = stablehlo.compare  EQ, %5414, %cst_0 : (tensor<1x16x494x494xf64>, tensor<1x16x494x494xf64>) -> tensor<1x16x494x494xi1>
    %5416 = stablehlo.not %5415 : tensor<1x16x494x494xi1>
    %5417 = stablehlo.reduce(%5416 init: %c_234) across dimensions = [3] : (tensor<1x16x494x494xi1>, tensor<i1>) -> tensor<1x16x494xi1>
     reducer(%arg75: tensor<i1>, %arg76: tensor<i1>)  {
      %5568 = stablehlo.or %arg75, %arg76 : tensor<i1>
      %5569 = stablehlo.select %5568, %c_233, %c_234 : tensor<i1>, tensor<i1>
      stablehlo.return %5569 : tensor<i1>
    }
    %5418 = stablehlo.reshape %5417 : (tensor<1x16x494xi1>) -> tensor<1x16x494x1xi1>
    %5419 = stablehlo.not %5418 : tensor<1x16x494x1xi1>
    %5420 = stablehlo.reshape %5419 : (tensor<1x16x494x1xi1>) -> tensor<1x16x494xi1>
    %5421 = stablehlo.broadcast_in_dim %5420, dims = [0, 1, 2] : (tensor<1x16x494xi1>) -> tensor<1x16x494x494xi1>
    %5422 = stablehlo.reduce(%5413 init: %cst_236) applies stablehlo.maximum across dimensions = [3] : (tensor<1x16x494x494xf32>, tensor<f32>) -> tensor<1x16x494xf32>
    %5423 = stablehlo.broadcast_in_dim %5422, dims = [0, 1, 2] : (tensor<1x16x494xf32>) -> tensor<1x16x494x494xf32>
    %5424 = stablehlo.subtract %5413, %5423 : tensor<1x16x494x494xf32>
    %5425 = stablehlo.exponential %5424 : tensor<1x16x494x494xf32>
    %5426 = stablehlo.reduce(%5425 init: %cst_240) applies stablehlo.add across dimensions = [3] : (tensor<1x16x494x494xf32>, tensor<f32>) -> tensor<1x16x494xf32>
    %5427 = stablehlo.broadcast_in_dim %5426, dims = [0, 1, 2] : (tensor<1x16x494xf32>) -> tensor<1x16x494x494xf32>
    %5428 = stablehlo.divide %5425, %5427 : tensor<1x16x494x494xf32>
    %5429 = stablehlo.select %5421, %cst, %5428 : tensor<1x16x494x494xi1>, tensor<1x16x494x494xf32>
    %5430 = stablehlo.reshape %arg63 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5431 = stablehlo.custom_call @tt.mark_argument(%5430) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.self_attn.v_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5432 = stablehlo.reshape %5431 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5433 = stablehlo.transpose %5432, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %5434 = stablehlo.dot_general %5276, %5433, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %5435 = stablehlo.reshape %5434 : (tensor<494x512xbf16>) -> tensor<1x494x2x256xbf16>
    %5436 = stablehlo.transpose %5435, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,2,494,256]{3,1,2,0}"} : (tensor<1x494x2x256xbf16>) -> tensor<1x2x494x256xbf16>
    %5437 = stablehlo.broadcast_in_dim %5436, dims = [0, 1, 3, 4] : (tensor<1x2x494x256xbf16>) -> tensor<1x2x8x494x256xbf16>
    %5438 = stablehlo.reshape %5437 : (tensor<1x2x8x494x256xbf16>) -> tensor<1x16x494x256xbf16>
    %5439 = stablehlo.convert %5438 : (tensor<1x16x494x256xbf16>) -> tensor<1x16x494x256xf32>
    %5440 = stablehlo.dot_general %5429, %5439, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<1x16x494x494xf32>, tensor<1x16x494x256xf32>) -> tensor<1x16x494x256xf32>
    %5441 = stablehlo.convert %5440 : (tensor<1x16x494x256xf32>) -> tensor<1x16x494x256xbf16>
    %5442 = stablehlo.transpose %5441, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "bf16[1,494,16,256]{3,1,2,0}"} : (tensor<1x16x494x256xbf16>) -> tensor<1x494x16x256xbf16>
    %5443 = stablehlo.reshape %5442 : (tensor<1x494x16x256xbf16>) -> tensor<1x494x4096xbf16>
    %5444 = stablehlo.slice %5282 [0:1, 0:494, 0:16, 256:512] : (tensor<1x494x16x512xbf16>) -> tensor<1x494x16x256xbf16>
    %5445 = stablehlo.reshape %5444 : (tensor<1x494x16x256xbf16>) -> tensor<1x494x4096xbf16>
    %5446 = stablehlo.logistic %5445 : tensor<1x494x4096xbf16>
    %5447 = stablehlo.multiply %5443, %5446 : tensor<1x494x4096xbf16>
    %5448 = stablehlo.reshape %5447 : (tensor<1x494x4096xbf16>) -> tensor<494x4096xbf16>
    %5449 = stablehlo.reshape %arg4 : (tensor<2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %5450 = stablehlo.custom_call @tt.mark_argument(%5449) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.self_attn.o_proj.weight"}} : (tensor<1x2048x4096xbf16>) -> tensor<1x2048x4096xbf16>
    %5451 = stablehlo.reshape %5450 : (tensor<1x2048x4096xbf16>) -> tensor<2048x4096xbf16>
    %5452 = stablehlo.transpose %5451, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[4096,2048]{0,1}"} : (tensor<2048x4096xbf16>) -> tensor<4096x2048xbf16>
    %5453 = stablehlo.dot_general %5448, %5452, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x4096xbf16>, tensor<4096x2048xbf16>) -> tensor<494x2048xbf16>
    %5454 = stablehlo.reshape %5453 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %5455 = stablehlo.add %5269, %5454 : tensor<1x494x2048xbf16>
    %5456 = stablehlo.reshape %arg3 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %5457 = stablehlo.custom_call @tt.mark_argument(%5456) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.post_attention_layernorm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5458 = stablehlo.reshape %5457 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %5459 = stablehlo.convert %5458 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %5460 = stablehlo.add %5459, %cst_231 : tensor<2048xf32>
    %5461 = stablehlo.composite "tenstorrent.rms_norm" %5455, %5460 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl_0} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    %5462 = stablehlo.reshape %5461 : (tensor<1x494x2048xbf16>) -> tensor<494x2048xbf16>
    %5463 = stablehlo.reshape %arg72 : (tensor<256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %5464 = stablehlo.custom_call @tt.mark_argument(%5463) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.mlp.gate.weight"}} : (tensor<1x256x2048xbf16>) -> tensor<1x256x2048xbf16>
    %5465 = stablehlo.reshape %5464 : (tensor<1x256x2048xbf16>) -> tensor<256x2048xbf16>
    %5466 = stablehlo.transpose %5465, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,256]{0,1}"} : (tensor<256x2048xbf16>) -> tensor<2048x256xbf16>
    %5467 = stablehlo.dot_general %5462, %5466, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x256xbf16>) -> tensor<494x256xbf16>
    %5468 = stablehlo.convert %5467 : (tensor<494x256xbf16>) -> tensor<494x256xf32>
    %5469 = stablehlo.reduce(%5468 init: %cst_236) applies stablehlo.maximum across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %5470 = stablehlo.broadcast_in_dim %5469, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %5471 = stablehlo.subtract %5468, %5470 : tensor<494x256xf32>
    %5472 = stablehlo.exponential %5471 : tensor<494x256xf32>
    %5473 = stablehlo.reduce(%5472 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x256xf32>, tensor<f32>) -> tensor<494xf32>
    %5474 = stablehlo.broadcast_in_dim %5473, dims = [0] : (tensor<494xf32>) -> tensor<494x256xf32>
    %5475 = stablehlo.divide %5472, %5474 : tensor<494x256xf32>
    %5476:2 = stablehlo.composite "tenstorrent.topk" %5475 {composite_attributes = {dim = -1 : i64, k = 8 : i64, largest = true, sorted = true}, decomposition = @tenstorrent.topk.impl} : (tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>)
    %5477 = stablehlo.reduce(%5476#0 init: %cst_240) applies stablehlo.add across dimensions = [1] : (tensor<494x8xf32>, tensor<f32>) -> tensor<494xf32>
    %5478 = stablehlo.broadcast_in_dim %5477, dims = [0] : (tensor<494xf32>) -> tensor<494x8xf32>
    %5479 = stablehlo.divide %5476#0, %5478 : tensor<494x8xf32>
    %5480 = stablehlo.concatenate %5479, %cst_22, dim = 0 : (tensor<494x8xf32>, tensor<18x8xf32>) -> tensor<512x8xf32>
    %5481 = stablehlo.convert %5480 : (tensor<512x8xf32>) -> tensor<512x8xbf16>
    %5482 = stablehlo.reshape %5481 : (tensor<512x8xbf16>) -> tensor<512x1x8xbf16>
    %5483 = stablehlo.concatenate %5476#1, %c_21, dim = 0 : (tensor<494x8xi64>, tensor<18x8xi64>) -> tensor<512x8xi64>
    %5484 = stablehlo.broadcast_in_dim %5483, dims = [0, 1] : (tensor<512x8xi64>) -> tensor<512x8x256xi64>
    %5485 = stablehlo.compare  EQ, %5484, %2244 : (tensor<512x8x256xi64>, tensor<512x8x256xi64>) -> tensor<512x8x256xi1>
    %5486 = stablehlo.convert %5485 : (tensor<512x8x256xi1>) -> tensor<512x8x256xbf16>
    %5487 = stablehlo.dot_general %5482, %5486, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<512x1x8xbf16>, tensor<512x8x256xbf16>) -> tensor<512x1x256xbf16>
    %5488 = stablehlo.reshape %5487 : (tensor<512x1x256xbf16>) -> tensor<1x512x256xbf16>
    %5489 = stablehlo.concatenate %5488, %5488, %5488, %5488, %5488, %5488, %5488, %5488, dim = 1 : (tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>, tensor<1x512x256xbf16>) -> tensor<1x4096x256xbf16>
    %5490 = stablehlo.reshape %5489 : (tensor<1x4096x256xbf16>) -> tensor<1x1x4096x256xbf16>
    %5491 = stablehlo.concatenate %5462, %cst_18, dim = 0 : (tensor<494x2048xbf16>, tensor<18x2048xbf16>) -> tensor<512x2048xbf16>
    %5492 = stablehlo.reshape %5491 : (tensor<512x2048xbf16>) -> tensor<1x1x512x2048xbf16>
    %5493 = stablehlo.reshape %5483 : (tensor<512x8xi64>) -> tensor<1x1x512x8xi64>
    %5494 = stablehlo.custom_call @tt.all_to_all_dispatch(%5492, %5493, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8"}, xla_shape = "(bf16[1,8,512,2048]{3,2,1,0}, s64[1,8,512,8]{3,2,1,0})"} : (tensor<1x1x512x2048xbf16>, tensor<1x1x512x8xi64>, tensor<1x1x256x8xui16>) -> tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>
    %5495 = stablehlo.get_tuple_element %5494[1] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x8xi64>
    %5496 = stablehlo.reshape %5495 : (tensor<1x8x512x8xi64>) -> tensor<1x1x4096x8xi64>
    %5497 = stablehlo.custom_call @tt.moe_expert_token_remap(%5490, %2258, %5496) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {num_devices = "8", reduction_size = "32"}, xla_shape = "(bf16[1,1,4096,256]{3,2,1,0}, bf16[1,1,128,256]{3,2,1,0})"} : (tensor<1x1x4096x256xbf16>, tensor<1x1x256x8xui16>, tensor<1x1x4096x8xi64>) -> tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>
    %5498 = stablehlo.get_tuple_element %5494[0] : (tuple<tensor<1x8x512x2048xbf16>, tensor<1x8x512x8xi64>>) -> tensor<1x8x512x2048xbf16>
    %5499 = stablehlo.reshape %5498 : (tensor<1x8x512x2048xbf16>) -> tensor<8x16x32x2048xbf16>
    %5500 = stablehlo.custom_call @tt.mark_argument(%arg74) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.mlp.experts.gate_up_proj"}} : (tensor<256x1024x2048xbf16>) -> tensor<256x1024x2048xbf16>
    %5501 = stablehlo.transpose %5500, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,2048,1024]{1,2,0}"} : (tensor<256x1024x2048xbf16>) -> tensor<256x2048x1024xbf16>
    %5502 = stablehlo.reshape %5501 : (tensor<256x2048x1024xbf16>) -> tensor<1x256x2048x1024xbf16>
    %5503 = stablehlo.get_tuple_element %5497[1] : (tuple<tensor<1x1x4096x256xbf16>, tensor<1x1x128x256xbf16>>) -> tensor<1x1x128x256xbf16>
    %5504 = stablehlo.reshape %5503 : (tensor<1x1x128x256xbf16>) -> tensor<8x16x1x256xbf16>
    %5505 = stablehlo.custom_call @tt.sparse_matmul(%5499, %5502, %5504) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "False", is_input_b_sparse = "True", nnz = "0"}} : (tensor<8x16x32x2048xbf16>, tensor<1x256x2048x1024xbf16>, tensor<8x16x1x256xbf16>) -> tensor<8x16x1x256x32x1024xbf16>
    %5506 = stablehlo.reshape %5505 : (tensor<8x16x1x256x32x1024xbf16>) -> tensor<8x16x256x32x1024xbf16>
    %5507 = stablehlo.slice %5506 [0:8, 0:16, 0:256, 0:32, 0:512] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %5508 = stablehlo.convert %5507 : (tensor<8x16x256x32x512xbf16>) -> tensor<8x16x256x32x512xf32>
    %5509 = stablehlo.logistic %5508 : tensor<8x16x256x32x512xf32>
    %5510 = stablehlo.multiply %5508, %5509 : tensor<8x16x256x32x512xf32>
    %5511 = stablehlo.convert %5510 : (tensor<8x16x256x32x512xf32>) -> tensor<8x16x256x32x512xbf16>
    %5512 = stablehlo.slice %5506 [0:8, 0:16, 0:256, 0:32, 512:1024] : (tensor<8x16x256x32x1024xbf16>) -> tensor<8x16x256x32x512xbf16>
    %5513 = stablehlo.multiply %5511, %5512 : tensor<8x16x256x32x512xbf16>
    %5514 = stablehlo.reshape %5513 : (tensor<8x16x256x32x512xbf16>) -> tensor<128x256x32x512xbf16>
    %5515 = stablehlo.custom_call @tt.mark_argument(%arg73) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.mlp.experts.down_proj"}} : (tensor<256x2048x512xbf16>) -> tensor<256x2048x512xbf16>
    %5516 = stablehlo.transpose %5515, dims = [0, 2, 1] {result_layout = dense<[1, 2, 0]> : tensor<3xindex>, xla_shape = "bf16[256,512,2048]{1,2,0}"} : (tensor<256x2048x512xbf16>) -> tensor<256x512x2048xbf16>
    %5517 = stablehlo.reshape %5516 : (tensor<256x512x2048xbf16>) -> tensor<1x256x512x2048xbf16>
    %5518 = stablehlo.custom_call @tt.sparse_matmul(%5514, %5517, %5503) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {is_input_a_sparse = "True", is_input_b_sparse = "False", nnz = "0"}} : (tensor<128x256x32x512xbf16>, tensor<1x256x512x2048xbf16>, tensor<1x1x128x256xbf16>) -> tensor<128x256x32x2048xbf16>
    %5519 = stablehlo.transpose %5518, dims = [1, 0, 2, 3] {result_layout = dense<[3, 2, 0, 1]> : tensor<4xindex>, xla_shape = "bf16[256,128,32,2048]{3,2,0,1}"} : (tensor<128x256x32x2048xbf16>) -> tensor<256x128x32x2048xbf16>
    %5520 = stablehlo.reshape %5519 : (tensor<256x128x32x2048xbf16>) -> tensor<256x1x4096x2048xbf16>
    %5521 = stablehlo.custom_call @tt.all_to_all_combine(%5520, %5496, %2258) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {cluster_axis = "1", num_devices = "8", num_experts_per_tok = "8", output_shard_dim = "2"}} : (tensor<256x1x4096x2048xbf16>, tensor<1x1x4096x8xi64>, tensor<1x1x256x8xui16>) -> tensor<8x1x512x2048xbf16>
    %5522 = stablehlo.transpose %5480, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[8,512]{0,1}"} : (tensor<512x8xf32>) -> tensor<8x512xf32>
    %5523 = stablehlo.reshape %5522 : (tensor<8x512xf32>) -> tensor<8x1x512x1xf32>
    %5524 = stablehlo.convert %5523 : (tensor<8x1x512x1xf32>) -> tensor<8x1x512x1xbf16>
    %5525 = stablehlo.reshape %5524 : (tensor<8x1x512x1xbf16>) -> tensor<8x1x512xbf16>
    %5526 = stablehlo.broadcast_in_dim %5525, dims = [0, 1, 2] : (tensor<8x1x512xbf16>) -> tensor<8x1x512x2048xbf16>
    %5527 = stablehlo.multiply %5521, %5526 : tensor<8x1x512x2048xbf16>
    %5528 = stablehlo.reduce(%5527 init: %cst_239) applies stablehlo.add across dimensions = [0] : (tensor<8x1x512x2048xbf16>, tensor<bf16>) -> tensor<1x512x2048xbf16>
    %5529 = stablehlo.reshape %5528 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5530 = stablehlo.slice %5529 [0:494, 0:2048] : (tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %5531 = stablehlo.reshape %arg71 : (tensor<1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5532 = stablehlo.custom_call @tt.mark_argument(%5531) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.mlp.shared_expert_gate.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5533 = stablehlo.reshape %5532 : (tensor<1x1x2048xbf16>) -> tensor<2048x1xbf16>
    %5534 = stablehlo.dot_general %5462, %5533, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x1xbf16>) -> tensor<494x1xbf16>
    %5535 = stablehlo.logistic %5534 : tensor<494x1xbf16>
    %5536 = stablehlo.reshape %5535 : (tensor<494x1xbf16>) -> tensor<494xbf16>
    %5537 = stablehlo.broadcast_in_dim %5536, dims = [0] : (tensor<494xbf16>) -> tensor<494x2048xbf16>
    %5538 = stablehlo.reshape %arg70 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5539 = stablehlo.custom_call @tt.mark_argument(%5538) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.mlp.shared_expert.gate_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5540 = stablehlo.reshape %5539 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5541 = stablehlo.transpose %5540, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %5542 = stablehlo.dot_general %5462, %5541, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %5543 = stablehlo.convert %5542 : (tensor<494x512xbf16>) -> tensor<494x512xf32>
    %5544 = stablehlo.logistic %5543 : tensor<494x512xf32>
    %5545 = stablehlo.multiply %5543, %5544 : tensor<494x512xf32>
    %5546 = stablehlo.convert %5545 : (tensor<494x512xf32>) -> tensor<494x512xbf16>
    %5547 = stablehlo.reshape %arg2 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5548 = stablehlo.custom_call @tt.mark_argument(%5547) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.mlp.shared_expert.up_proj.weight"}} : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xbf16>
    %5549 = stablehlo.reshape %5548 : (tensor<1x512x2048xbf16>) -> tensor<512x2048xbf16>
    %5550 = stablehlo.transpose %5549, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,512]{0,1}"} : (tensor<512x2048xbf16>) -> tensor<2048x512xbf16>
    %5551 = stablehlo.dot_general %5462, %5550, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x2048xbf16>, tensor<2048x512xbf16>) -> tensor<494x512xbf16>
    %5552 = stablehlo.multiply %5546, %5551 : tensor<494x512xbf16>
    %5553 = stablehlo.reshape %arg1 : (tensor<2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %5554 = stablehlo.custom_call @tt.mark_argument(%5553) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "layers.3.mlp.shared_expert.down_proj.weight"}} : (tensor<1x2048x512xbf16>) -> tensor<1x2048x512xbf16>
    %5555 = stablehlo.reshape %5554 : (tensor<1x2048x512xbf16>) -> tensor<2048x512xbf16>
    %5556 = stablehlo.transpose %5555, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[512,2048]{0,1}"} : (tensor<2048x512xbf16>) -> tensor<512x2048xbf16>
    %5557 = stablehlo.dot_general %5552, %5556, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<494x512xbf16>, tensor<512x2048xbf16>) -> tensor<494x2048xbf16>
    %5558 = stablehlo.multiply %5537, %5557 : tensor<494x2048xbf16>
    %5559 = stablehlo.add %5530, %5558 : tensor<494x2048xbf16>
    %5560 = stablehlo.reshape %5559 : (tensor<494x2048xbf16>) -> tensor<1x494x2048xbf16>
    %5561 = stablehlo.add %5455, %5560 : tensor<1x494x2048xbf16>
    %5562 = stablehlo.reshape %arg0 : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
    %5563 = stablehlo.custom_call @tt.mark_argument(%5562) {api_version = 0 : i32, backend_config = "", mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "norm.weight"}} : (tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    %5564 = stablehlo.reshape %5563 : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
    %5565 = stablehlo.convert %5564 : (tensor<2048xbf16>) -> tensor<2048xf32>
    %5566 = stablehlo.add %5565, %cst_231 : tensor<2048xf32>
    %5567 = stablehlo.composite "tenstorrent.rms_norm" %5561, %5566 {composite_attributes = {epsilon = 9.99999997E-7 : f32, normalized_shape = dense<2048> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl} : (tensor<1x494x2048xbf16>, tensor<2048xf32>) -> tensor<1x494x2048xbf16>
    return %5567 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_0(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.topk.impl(%arg0: tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>) {
    %0 = stablehlo.iota dim = 0 : tensor<256xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<256xi32>) -> tensor<494x256xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64, is_stable = false}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %6 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    }) : (tensor<494x256xf32>, tensor<494x256xi32>) -> (tensor<494x256xf32>, tensor<494x256xi32>)
    %3 = stablehlo.slice %2#0 [0:494, 0:8] : (tensor<494x256xf32>) -> tensor<494x8xf32>
    %4 = stablehlo.slice %2#1 [0:494, 0:8] : (tensor<494x256xi32>) -> tensor<494x8xi32>
    %5 = stablehlo.convert %4 : (tensor<494x8xi32>) -> tensor<494x8xi64>
    return %3, %5 : tensor<494x8xf32>, tensor<494x8xi64>
  }
  func.func private @tenstorrent.rms_norm.impl_1(%arg0: tensor<1x494x2x256xbf16>, %arg1: tensor<256xf32>) -> tensor<1x494x2x256xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x2x1xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<1x494x2xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2x256xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2x256xbf16>) -> tensor<1x494x2x256xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2x256xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x494x2x256xf32>, tensor<f32>) -> tensor<1x494x2xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494x2xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494x2xf32>) -> tensor<1x494x2x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x2x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x2x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x2x1xf32>) -> tensor<1x494x2xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x494x2xf32>) -> tensor<1x494x2x256xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<1x494x2x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2x256xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2x256xf32>) -> tensor<1x494x2x256xbf16>
    return %12 : tensor<1x494x2x256xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_2(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_3(%arg0: tensor<15808x128xbf16>, %arg1: tensor<128xbf16>) -> tensor<15808x128xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<15808x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<15808xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<15808x128xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<15808x128xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<15808x128xf32>, tensor<f32>) -> tensor<15808xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<15808xf32>
    %4 = stablehlo.reshape %3 : (tensor<15808xf32>) -> tensor<15808x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<15808x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<15808x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<15808x1xf32>) -> tensor<15808xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<15808xf32>) -> tensor<15808x128xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<15808x128xf32>
    %10 = stablehlo.convert %arg1 : (tensor<128xbf16>) -> tensor<128xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<128xf32>) -> tensor<15808x128xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<15808x128xf32>
    %13 = stablehlo.convert %12 : (tensor<15808x128xf32>) -> tensor<15808x128xbf16>
    return %13 : tensor<15808x128xbf16>
  }
  func.func private @tenstorrent.topk.impl_0(%arg0: tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>) {
    %0 = stablehlo.iota dim = 0 : tensor<256xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<256xi32>) -> tensor<494x256xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64, is_stable = false}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %6 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    }) : (tensor<494x256xf32>, tensor<494x256xi32>) -> (tensor<494x256xf32>, tensor<494x256xi32>)
    %3 = stablehlo.slice %2#0 [0:494, 0:8] : (tensor<494x256xf32>) -> tensor<494x8xf32>
    %4 = stablehlo.slice %2#1 [0:494, 0:8] : (tensor<494x256xi32>) -> tensor<494x8xi32>
    %5 = stablehlo.convert %4 : (tensor<494x8xi32>) -> tensor<494x8xi64>
    return %3, %5 : tensor<494x8xf32>, tensor<494x8xi64>
  }
  func.func private @tenstorrent.topk.impl_1(%arg0: tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>) {
    %0 = stablehlo.iota dim = 0 : tensor<256xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<256xi32>) -> tensor<494x256xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64, is_stable = false}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %6 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    }) : (tensor<494x256xf32>, tensor<494x256xi32>) -> (tensor<494x256xf32>, tensor<494x256xi32>)
    %3 = stablehlo.slice %2#0 [0:494, 0:8] : (tensor<494x256xf32>) -> tensor<494x8xf32>
    %4 = stablehlo.slice %2#1 [0:494, 0:8] : (tensor<494x256xi32>) -> tensor<494x8xi32>
    %5 = stablehlo.convert %4 : (tensor<494x8xi32>) -> tensor<494x8xi64>
    return %3, %5 : tensor<494x8xf32>, tensor<494x8xi64>
  }
  func.func private @tenstorrent.rms_norm.impl_4(%arg0: tensor<15808x128xbf16>, %arg1: tensor<128xbf16>) -> tensor<15808x128xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<15808x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<15808xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<15808x128xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<15808x128xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<15808x128xf32>, tensor<f32>) -> tensor<15808xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<15808xf32>
    %4 = stablehlo.reshape %3 : (tensor<15808xf32>) -> tensor<15808x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<15808x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<15808x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<15808x1xf32>) -> tensor<15808xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<15808xf32>) -> tensor<15808x128xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<15808x128xf32>
    %10 = stablehlo.convert %arg1 : (tensor<128xbf16>) -> tensor<128xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<128xf32>) -> tensor<15808x128xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<15808x128xf32>
    %13 = stablehlo.convert %12 : (tensor<15808x128xf32>) -> tensor<15808x128xbf16>
    return %13 : tensor<15808x128xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_5(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_6(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_7(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_8(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_9(%arg0: tensor<15808x128xbf16>, %arg1: tensor<128xbf16>) -> tensor<15808x128xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<15808x1xf32>
    %cst_0 = stablehlo.constant dense<7.812500e-03> : tensor<15808xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<15808x128xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<15808x128xbf16>) -> tensor<15808x128xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<15808x128xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<15808x128xf32>, tensor<f32>) -> tensor<15808xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<15808xf32>
    %4 = stablehlo.reshape %3 : (tensor<15808xf32>) -> tensor<15808x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<15808x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<15808x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<15808x1xf32>) -> tensor<15808xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<15808xf32>) -> tensor<15808x128xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<15808x128xf32>
    %10 = stablehlo.convert %arg1 : (tensor<128xbf16>) -> tensor<128xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<128xf32>) -> tensor<15808x128xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<15808x128xf32>
    %13 = stablehlo.convert %12 : (tensor<15808x128xf32>) -> tensor<15808x128xbf16>
    return %13 : tensor<15808x128xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_10(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
  func.func private @tenstorrent.rms_norm.impl_11(%arg0: tensor<1x494x16x256xbf16>, %arg1: tensor<256xf32>) -> tensor<1x494x16x256xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x16x1xf32>
    %cst_0 = stablehlo.constant dense<3.906250e-03> : tensor<1x494x16xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x16x256xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x16x256xbf16>) -> tensor<1x494x16x256xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x16x256xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [3] : (tensor<1x494x16x256xf32>, tensor<f32>) -> tensor<1x494x16xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494x16xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494x16xf32>) -> tensor<1x494x16x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x16x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x16x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x16x1xf32>) -> tensor<1x494x16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x494x16xf32>) -> tensor<1x494x16x256xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x16x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<256xf32>) -> tensor<1x494x16x256xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x16x256xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x16x256xf32>) -> tensor<1x494x16x256xbf16>
    return %12 : tensor<1x494x16x256xbf16>
  }
  func.func private @tenstorrent.topk.impl_2(%arg0: tensor<494x256xf32>) -> (tensor<494x8xf32>, tensor<494x8xi64>) {
    %0 = stablehlo.iota dim = 0 : tensor<256xi32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<256xi32>) -> tensor<494x256xi32>
    %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64, is_stable = false}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
      %6 = stablehlo.compare  GT, %arg1, %arg2,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %6 : tensor<i1>
    }) : (tensor<494x256xf32>, tensor<494x256xi32>) -> (tensor<494x256xf32>, tensor<494x256xi32>)
    %3 = stablehlo.slice %2#0 [0:494, 0:8] : (tensor<494x256xf32>) -> tensor<494x8xf32>
    %4 = stablehlo.slice %2#1 [0:494, 0:8] : (tensor<494x256xi32>) -> tensor<494x8xi32>
    %5 = stablehlo.convert %4 : (tensor<494x8xi32>) -> tensor<494x8xi64>
    return %3, %5 : tensor<494x8xf32>, tensor<494x8xi64>
  }
  func.func private @tenstorrent.rms_norm.impl_12(%arg0: tensor<1x494x2048xbf16>, %arg1: tensor<2048xf32>) -> tensor<1x494x2048xbf16> {
    %cst = stablehlo.constant dense<9.99999997E-7> : tensor<1x494x1xf32>
    %cst_0 = stablehlo.constant dense<4.8828125E-4> : tensor<1x494xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<1x494x2048xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x494x2048xbf16>) -> tensor<1x494x2048xf32>
    %1 = stablehlo.power %0, %cst_1 : tensor<1x494x2048xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x494x2048xf32>, tensor<f32>) -> tensor<1x494xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x494xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x494xf32>) -> tensor<1x494x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<1x494x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<1x494x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x494x1xf32>) -> tensor<1x494xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x494xf32>) -> tensor<1x494x2048xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<1x494x2048xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<2048xf32>) -> tensor<1x494x2048xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<1x494x2048xf32>
    %12 = stablehlo.convert %11 : (tensor<1x494x2048xf32>) -> tensor<1x494x2048xbf16>
    return %12 : tensor<1x494x2048xbf16>
  }
}
