#loc = loc("YOLOv8":0:0)
module @YOLOv8 {
  func.func @forward(%arg0: tensor<1x3x640x640xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_1"} loc("YOLOv8":0:0), %arg1: tensor<1x2x8400xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_add_1644"} loc("YOLOv8":0:0), %arg2: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_3"} loc("YOLOv8":0:0), %arg3: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_3_fork_clone2431"} loc("YOLOv8":0:0), %arg4: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_19"} loc("YOLOv8":0:0), %arg5: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_19_fork_clone2349"} loc("YOLOv8":0:0), %arg6: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_35"} loc("YOLOv8":0:0), %arg7: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_35_fork_clone2112"} loc("YOLOv8":0:0), %arg8: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_53"} loc("YOLOv8":0:0), %arg9: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_53_fork_clone2354"} loc("YOLOv8":0:0), %arg10: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_69"} loc("YOLOv8":0:0), %arg11: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_69_fork_clone2117"} loc("YOLOv8":0:0), %arg12: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_86"} loc("YOLOv8":0:0), %arg13: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_86_fork_clone2359"} loc("YOLOv8":0:0), %arg14: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_102"} loc("YOLOv8":0:0), %arg15: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_102_fork_clone2122"} loc("YOLOv8":0:0), %arg16: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_119"} loc("YOLOv8":0:0), %arg17: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_119_fork_clone2364"} loc("YOLOv8":0:0), %arg18: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_135"} loc("YOLOv8":0:0), %arg19: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_135_fork_clone2127"} loc("YOLOv8":0:0), %arg20: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_153"} loc("YOLOv8":0:0), %arg21: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_153_fork_clone1730"} loc("YOLOv8":0:0), %arg22: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_169"} loc("YOLOv8":0:0), %arg23: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_169_fork_clone1508"} loc("YOLOv8":0:0), %arg24: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_185"} loc("YOLOv8":0:0), %arg25: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_185_fork_clone1250"} loc("YOLOv8":0:0), %arg26: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_203"} loc("YOLOv8":0:0), %arg27: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_203_fork_clone1513"} loc("YOLOv8":0:0), %arg28: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_219"} loc("YOLOv8":0:0), %arg29: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_219_fork_clone1255"} loc("YOLOv8":0:0), %arg30: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_236"} loc("YOLOv8":0:0), %arg31: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_236_fork_clone1518"} loc("YOLOv8":0:0), %arg32: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_252"} loc("YOLOv8":0:0), %arg33: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_252_fork_clone1260"} loc("YOLOv8":0:0), %arg34: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_269"} loc("YOLOv8":0:0), %arg35: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_269_fork_clone1523"} loc("YOLOv8":0:0), %arg36: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_285"} loc("YOLOv8":0:0), %arg37: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_285_fork_clone1265"} loc("YOLOv8":0:0), %arg38: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_302"} loc("YOLOv8":0:0), %arg39: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_302_fork_clone1528"} loc("YOLOv8":0:0), %arg40: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_318"} loc("YOLOv8":0:0), %arg41: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_318_fork_clone1270"} loc("YOLOv8":0:0), %arg42: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_335"} loc("YOLOv8":0:0), %arg43: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_335_fork_clone1533"} loc("YOLOv8":0:0), %arg44: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_351"} loc("YOLOv8":0:0), %arg45: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_351_fork_clone1275"} loc("YOLOv8":0:0), %arg46: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_368"} loc("YOLOv8":0:0), %arg47: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_368_fork_clone1538"} loc("YOLOv8":0:0), %arg48: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_384"} loc("YOLOv8":0:0), %arg49: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_384_fork_clone1280"} loc("YOLOv8":0:0), %arg50: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_402"} loc("YOLOv8":0:0), %arg51: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_402_fork_clone951"} loc("YOLOv8":0:0), %arg52: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_418"} loc("YOLOv8":0:0), %arg53: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_418_fork_clone2190"} loc("YOLOv8":0:0), %arg54: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_434"} loc("YOLOv8":0:0), %arg55: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_434_fork_clone1899"} loc("YOLOv8":0:0), %arg56: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_452"} loc("YOLOv8":0:0), %arg57: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_452_fork_clone2195"} loc("YOLOv8":0:0), %arg58: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_468"} loc("YOLOv8":0:0), %arg59: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_468_fork_clone1904"} loc("YOLOv8":0:0), %arg60: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_485"} loc("YOLOv8":0:0), %arg61: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_485_fork_clone2200"} loc("YOLOv8":0:0), %arg62: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_501"} loc("YOLOv8":0:0), %arg63: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_501_fork_clone1909"} loc("YOLOv8":0:0), %arg64: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_518"} loc("YOLOv8":0:0), %arg65: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_518_fork_clone2205"} loc("YOLOv8":0:0), %arg66: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_534"} loc("YOLOv8":0:0), %arg67: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_534_fork_clone1914"} loc("YOLOv8":0:0), %arg68: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_551"} loc("YOLOv8":0:0), %arg69: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_551_fork_clone2210"} loc("YOLOv8":0:0), %arg70: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_567"} loc("YOLOv8":0:0), %arg71: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_567_fork_clone1919"} loc("YOLOv8":0:0), %arg72: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_584"} loc("YOLOv8":0:0), %arg73: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_584_fork_clone2215"} loc("YOLOv8":0:0), %arg74: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_600"} loc("YOLOv8":0:0), %arg75: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_600_fork_clone1924"} loc("YOLOv8":0:0), %arg76: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_617"} loc("YOLOv8":0:0), %arg77: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_617_fork_clone2220"} loc("YOLOv8":0:0), %arg78: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_633"} loc("YOLOv8":0:0), %arg79: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_633_fork_clone1929"} loc("YOLOv8":0:0), %arg80: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_651"} loc("YOLOv8":0:0), %arg81: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_651_fork_clone1596"} loc("YOLOv8":0:0), %arg82: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_667"} loc("YOLOv8":0:0), %arg83: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_667_fork_clone2090"} loc("YOLOv8":0:0), %arg84: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_683"} loc("YOLOv8":0:0), %arg85: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_683_fork_clone1810"} loc("YOLOv8":0:0), %arg86: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_701"} loc("YOLOv8":0:0), %arg87: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_701_fork_clone2095"} loc("YOLOv8":0:0), %arg88: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_717"} loc("YOLOv8":0:0), %arg89: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_717_fork_clone1815"} loc("YOLOv8":0:0), %arg90: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_734"} loc("YOLOv8":0:0), %arg91: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_734_fork_clone2100"} loc("YOLOv8":0:0), %arg92: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_750"} loc("YOLOv8":0:0), %arg93: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_750_fork_clone1820"} loc("YOLOv8":0:0), %arg94: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_767"} loc("YOLOv8":0:0), %arg95: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_767_fork_clone2105"} loc("YOLOv8":0:0), %arg96: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_783"} loc("YOLOv8":0:0), %arg97: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_783_fork_clone1825"} loc("YOLOv8":0:0), %arg98: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_801"} loc("YOLOv8":0:0), %arg99: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_801_fork_clone1501"} loc("YOLOv8":0:0), %arg100: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_817"} loc("YOLOv8":0:0), %arg101: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_817_fork_clone1245"} loc("YOLOv8":0:0), %arg102: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_837"} loc("YOLOv8":0:0), %arg103: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_837_fork_clone996"} loc("YOLOv8":0:0), %arg104: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_855"} loc("YOLOv8":0:0), %arg105: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_855_fork_clone1285"} loc("YOLOv8":0:0), %arg106: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_873"} loc("YOLOv8":0:0), %arg107: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_873_fork_clone1481"} loc("YOLOv8":0:0), %arg108: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_889"} loc("YOLOv8":0:0), %arg109: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_889_fork_clone1230"} loc("YOLOv8":0:0), %arg110: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_905"} loc("YOLOv8":0:0), %arg111: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_905_fork_clone1486"} loc("YOLOv8":0:0), %arg112: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_921"} loc("YOLOv8":0:0), %arg113: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_921_fork_clone1235"} loc("YOLOv8":0:0), %arg114: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_937"} loc("YOLOv8":0:0), %arg115: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_937_fork_clone1491"} loc("YOLOv8":0:0), %arg116: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_953"} loc("YOLOv8":0:0), %arg117: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_953_fork_clone1240"} loc("YOLOv8":0:0), %arg118: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_970"} loc("YOLOv8":0:0), %arg119: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_970_fork_clone974"} loc("YOLOv8":0:0), %arg120: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_988"} loc("YOLOv8":0:0), %arg121: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_988_fork_clone631"} loc("YOLOv8":0:0), %arg122: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1006"} loc("YOLOv8":0:0), %arg123: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1006_fork_clone833"} loc("YOLOv8":0:0), %arg124: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1022"} loc("YOLOv8":0:0), %arg125: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1022_fork_clone578"} loc("YOLOv8":0:0), %arg126: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1038"} loc("YOLOv8":0:0), %arg127: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1038_fork_clone838"} loc("YOLOv8":0:0), %arg128: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1054"} loc("YOLOv8":0:0), %arg129: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1054_fork_clone583"} loc("YOLOv8":0:0), %arg130: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1070"} loc("YOLOv8":0:0), %arg131: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1070_fork_clone843"} loc("YOLOv8":0:0), %arg132: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1086"} loc("YOLOv8":0:0), %arg133: tensor<1x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1086_fork_clone588"} loc("YOLOv8":0:0), %arg134: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1103"} loc("YOLOv8":0:0), %arg135: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1103_fork_clone388"} loc("YOLOv8":0:0), %arg136: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1119"} loc("YOLOv8":0:0), %arg137: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1119_fork_clone274"} loc("YOLOv8":0:0), %arg138: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1135"} loc("YOLOv8":0:0), %arg139: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1135_fork_clone142"} loc("YOLOv8":0:0), %arg140: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1155"} loc("YOLOv8":0:0), %arg141: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1155_fork_clone279"} loc("YOLOv8":0:0), %arg142: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1171"} loc("YOLOv8":0:0), %arg143: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1171_fork_clone147"} loc("YOLOv8":0:0), %arg144: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1193"} loc("YOLOv8":0:0), %arg145: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1193_fork_clone966"} loc("YOLOv8":0:0), %arg146: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1210"} loc("YOLOv8":0:0), %arg147: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1210_fork_clone651"} loc("YOLOv8":0:0), %arg148: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1228"} loc("YOLOv8":0:0), %arg149: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1228_fork_clone855"} loc("YOLOv8":0:0), %arg150: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1244"} loc("YOLOv8":0:0), %arg151: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1244_fork_clone597"} loc("YOLOv8":0:0), %arg152: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1260"} loc("YOLOv8":0:0), %arg153: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1260_fork_clone860"} loc("YOLOv8":0:0), %arg154: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1276"} loc("YOLOv8":0:0), %arg155: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1276_fork_clone602"} loc("YOLOv8":0:0), %arg156: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1292"} loc("YOLOv8":0:0), %arg157: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1292_fork_clone865"} loc("YOLOv8":0:0), %arg158: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1308"} loc("YOLOv8":0:0), %arg159: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1308_fork_clone607"} loc("YOLOv8":0:0), %arg160: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1325"} loc("YOLOv8":0:0), %arg161: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1325_fork_clone400"} loc("YOLOv8":0:0), %arg162: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1341"} loc("YOLOv8":0:0), %arg163: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1341_fork_clone286"} loc("YOLOv8":0:0), %arg164: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1357"} loc("YOLOv8":0:0), %arg165: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1357_fork_clone152"} loc("YOLOv8":0:0), %arg166: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1377"} loc("YOLOv8":0:0), %arg167: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1377_fork_clone291"} loc("YOLOv8":0:0), %arg168: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1393"} loc("YOLOv8":0:0), %arg169: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1393_fork_clone157"} loc("YOLOv8":0:0), %arg170: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1415"} loc("YOLOv8":0:0), %arg171: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1415_fork_clone989"} loc("YOLOv8":0:0), %arg172: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1432"} loc("YOLOv8":0:0), %arg173: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1432_fork_clone671"} loc("YOLOv8":0:0), %arg174: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1450"} loc("YOLOv8":0:0), %arg175: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1450_fork_clone877"} loc("YOLOv8":0:0), %arg176: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1466"} loc("YOLOv8":0:0), %arg177: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1466_fork_clone616"} loc("YOLOv8":0:0), %arg178: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1482"} loc("YOLOv8":0:0), %arg179: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1482_fork_clone882"} loc("YOLOv8":0:0), %arg180: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1498"} loc("YOLOv8":0:0), %arg181: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1498_fork_clone621"} loc("YOLOv8":0:0), %arg182: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1514"} loc("YOLOv8":0:0), %arg183: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1514_fork_clone887"} loc("YOLOv8":0:0), %arg184: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1530"} loc("YOLOv8":0:0), %arg185: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1530_fork_clone626"} loc("YOLOv8":0:0), %arg186: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1547"} loc("YOLOv8":0:0), %arg187: tensor<1x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1547_fork_clone412"} loc("YOLOv8":0:0), %arg188: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1563"} loc("YOLOv8":0:0), %arg189: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1563_fork_clone298"} loc("YOLOv8":0:0), %arg190: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1579"} loc("YOLOv8":0:0), %arg191: tensor<1x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1579_fork_clone162"} loc("YOLOv8":0:0), %arg192: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1599"} loc("YOLOv8":0:0), %arg193: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1599_fork_clone303"} loc("YOLOv8":0:0), %arg194: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1615"} loc("YOLOv8":0:0), %arg195: tensor<1x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_0_divide_1615_fork_clone167"} loc("YOLOv8":0:0), %arg196: tensor<1x16x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_conv2d_1639"} loc("YOLOv8":0:0), %arg197: tensor<1xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_divide_1646"} loc("YOLOv8":0:0), %arg198: tensor<1x8400xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>, ttir.name = "input_1_multiply_1649"} loc("YOLOv8":0:0), %arg199: tensor<80x3x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.0.conv.weight"} loc("YOLOv8":0:0), %arg200: tensor<160x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.1.conv.weight"} loc("YOLOv8":0:0), %arg201: tensor<160x160x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg202: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg203: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg204: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg205: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg206: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg207: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg208: tensor<160x400x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg209: tensor<320x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.3.conv.weight"} loc("YOLOv8":0:0), %arg210: tensor<320x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.cv1.conv.weight"} loc("YOLOv8":0:0), %arg211: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg212: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg213: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg214: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg215: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg216: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg217: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.3.cv1.conv.weight"} loc("YOLOv8":0:0), %arg218: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.3.cv2.conv.weight"} loc("YOLOv8":0:0), %arg219: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.4.cv1.conv.weight"} loc("YOLOv8":0:0), %arg220: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.4.cv2.conv.weight"} loc("YOLOv8":0:0), %arg221: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.5.cv1.conv.weight"} loc("YOLOv8":0:0), %arg222: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.m.5.cv2.conv.weight"} loc("YOLOv8":0:0), %arg223: tensor<320x1280x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.4.cv2.conv.weight"} loc("YOLOv8":0:0), %arg224: tensor<640x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.5.conv.weight"} loc("YOLOv8":0:0), %arg225: tensor<640x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.cv1.conv.weight"} loc("YOLOv8":0:0), %arg226: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg227: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg228: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg229: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg230: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg231: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg232: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.3.cv1.conv.weight"} loc("YOLOv8":0:0), %arg233: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.3.cv2.conv.weight"} loc("YOLOv8":0:0), %arg234: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.4.cv1.conv.weight"} loc("YOLOv8":0:0), %arg235: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.4.cv2.conv.weight"} loc("YOLOv8":0:0), %arg236: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.5.cv1.conv.weight"} loc("YOLOv8":0:0), %arg237: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.m.5.cv2.conv.weight"} loc("YOLOv8":0:0), %arg238: tensor<640x2560x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.6.cv2.conv.weight"} loc("YOLOv8":0:0), %arg239: tensor<640x640x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.7.conv.weight"} loc("YOLOv8":0:0), %arg240: tensor<640x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.cv1.conv.weight"} loc("YOLOv8":0:0), %arg241: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg242: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg243: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg244: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg245: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg246: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg247: tensor<640x1600x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.8.cv2.conv.weight"} loc("YOLOv8":0:0), %arg248: tensor<320x640x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.9.cv1.conv.weight"} loc("YOLOv8":0:0), %arg249: tensor<640x1280x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.9.cv2.conv.weight"} loc("YOLOv8":0:0), %arg250: tensor<640x1280x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.cv1.conv.weight"} loc("YOLOv8":0:0), %arg251: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg252: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg253: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg254: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg255: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg256: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg257: tensor<640x1600x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.12.cv2.conv.weight"} loc("YOLOv8":0:0), %arg258: tensor<320x960x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.cv1.conv.weight"} loc("YOLOv8":0:0), %arg259: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg260: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg261: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg262: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg263: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg264: tensor<160x160x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg265: tensor<320x800x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.15.cv2.conv.weight"} loc("YOLOv8":0:0), %arg266: tensor<80x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.0.0.conv.weight"} loc("YOLOv8":0:0), %arg267: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.0.1.conv.weight"} loc("YOLOv8":0:0), %arg268: tensor<64x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.0.2.weight"} loc("YOLOv8":0:0), %arg269: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.0.2.bias"} loc("YOLOv8":0:0), %arg270: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.0.0.conv.weight"} loc("YOLOv8":0:0), %arg271: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.0.1.conv.weight"} loc("YOLOv8":0:0), %arg272: tensor<80x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.0.2.weight"} loc("YOLOv8":0:0), %arg273: tensor<1x1x1x80xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.0.2.bias"} loc("YOLOv8":0:0), %arg274: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.16.conv.weight"} loc("YOLOv8":0:0), %arg275: tensor<640x960x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.cv1.conv.weight"} loc("YOLOv8":0:0), %arg276: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg277: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg278: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg279: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg280: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg281: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg282: tensor<640x1600x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.18.cv2.conv.weight"} loc("YOLOv8":0:0), %arg283: tensor<80x640x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.1.0.conv.weight"} loc("YOLOv8":0:0), %arg284: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.1.1.conv.weight"} loc("YOLOv8":0:0), %arg285: tensor<64x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.1.2.weight"} loc("YOLOv8":0:0), %arg286: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.1.2.bias"} loc("YOLOv8":0:0), %arg287: tensor<320x640x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.1.0.conv.weight"} loc("YOLOv8":0:0), %arg288: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.1.1.conv.weight"} loc("YOLOv8":0:0), %arg289: tensor<80x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.1.2.weight"} loc("YOLOv8":0:0), %arg290: tensor<1x1x1x80xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.1.2.bias"} loc("YOLOv8":0:0), %arg291: tensor<640x640x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.19.conv.weight"} loc("YOLOv8":0:0), %arg292: tensor<640x1280x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.cv1.conv.weight"} loc("YOLOv8":0:0), %arg293: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.m.0.cv1.conv.weight"} loc("YOLOv8":0:0), %arg294: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.m.0.cv2.conv.weight"} loc("YOLOv8":0:0), %arg295: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.m.1.cv1.conv.weight"} loc("YOLOv8":0:0), %arg296: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.m.1.cv2.conv.weight"} loc("YOLOv8":0:0), %arg297: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.m.2.cv1.conv.weight"} loc("YOLOv8":0:0), %arg298: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.m.2.cv2.conv.weight"} loc("YOLOv8":0:0), %arg299: tensor<640x1600x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.21.cv2.conv.weight"} loc("YOLOv8":0:0), %arg300: tensor<80x640x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.2.0.conv.weight"} loc("YOLOv8":0:0), %arg301: tensor<80x80x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.2.1.conv.weight"} loc("YOLOv8":0:0), %arg302: tensor<64x80x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.2.2.weight"} loc("YOLOv8":0:0), %arg303: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv2.2.2.bias"} loc("YOLOv8":0:0), %arg304: tensor<320x640x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.2.0.conv.weight"} loc("YOLOv8":0:0), %arg305: tensor<320x320x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.2.1.conv.weight"} loc("YOLOv8":0:0), %arg306: tensor<80x320x1x1xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.2.2.weight"} loc("YOLOv8":0:0), %arg307: tensor<1x1x1x80xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttir.name = "model.model.22.cv3.2.2.bias"} loc("YOLOv8":0:0)) -> (tensor<1x84x8400xbf16> {ttir.name = "YOLOv8.output_concatenate_1652"}, tensor<1x144x80x80xbf16> {ttir.name = "YOLOv8.output_concatenate_1188"}, tensor<1x144x40x40xbf16> {ttir.name = "YOLOv8.output_concatenate_1410"}, tensor<1x144x20x20xbf16> {ttir.name = "YOLOv8.output_concatenate_1632"}) {
    %0 = "ttir.transpose"(%arg0) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x3x640x640xbf16>) -> tensor<1x640x3x640xbf16>
    %1 = "ttir.transpose"(%0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x640x3x640xbf16>) -> tensor<1x640x640x3xbf16>
    %2 = "ttir.conv2d"(%1, %arg199) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x640x640x3xbf16>, tensor<80x3x3x3xbf16>) -> tensor<1x320x320x80xbf16>
    %3 = "ttir.transpose"(%2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x320x320x80xbf16>) -> tensor<1x320x80x320xbf16>
    %4 = "ttir.transpose"(%3) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x320xbf16>) -> tensor<1x80x320x320xbf16>
    %5 = "ttir.multiply"(%4, %arg2) : (tensor<1x80x320x320xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x320x320xbf16>
    %6 = "ttir.add"(%5, %arg3) : (tensor<1x80x320x320xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x320x320xbf16>
    %7 = "ttir.sigmoid"(%6) : (tensor<1x80x320x320xbf16>) -> tensor<1x80x320x320xbf16>
    %8 = "ttir.multiply"(%6, %7) : (tensor<1x80x320x320xbf16>, tensor<1x80x320x320xbf16>) -> tensor<1x80x320x320xbf16>
    %9 = "ttir.transpose"(%8) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x320xbf16>) -> tensor<1x320x80x320xbf16>
    %10 = "ttir.transpose"(%9) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x320x80x320xbf16>) -> tensor<1x320x320x80xbf16>
    %11 = "ttir.conv2d"(%10, %arg200) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x320x320x80xbf16>, tensor<160x80x3x3xbf16>) -> tensor<1x160x160x160xbf16>
    %12 = "ttir.transpose"(%11) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %13 = "ttir.transpose"(%12) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %14 = "ttir.multiply"(%13, %arg4) : (tensor<1x160x160x160xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %15 = "ttir.add"(%14, %arg5) : (tensor<1x160x160x160xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %16 = "ttir.sigmoid"(%15) : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %17 = "ttir.multiply"(%15, %16) : (tensor<1x160x160x160xbf16>, tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %18 = "ttir.transpose"(%17) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %19 = "ttir.transpose"(%18) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %20 = "ttir.conv2d"(%19, %arg201) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x160xbf16>, tensor<160x160x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %21 = "ttir.transpose"(%20) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %22 = "ttir.transpose"(%21) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %23 = "ttir.multiply"(%22, %arg6) : (tensor<1x160x160x160xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %24 = "ttir.add"(%23, %arg7) : (tensor<1x160x160x160xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %25 = "ttir.sigmoid"(%24) : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %26 = "ttir.multiply"(%24, %25) : (tensor<1x160x160x160xbf16>, tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %27 = "ttir.index"(%26) <{begin = 0 : i32, dim = 1 : i32, end = 80 : i32, step = 1 : i32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %28 = "ttir.index"(%26) <{begin = 80 : i32, dim = 1 : i32, end = 160 : i32, step = 1 : i32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %29 = "ttir.transpose"(%28) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x160xbf16>) -> tensor<1x160x80x160xbf16>
    %30 = "ttir.transpose"(%29) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x160x160x80xbf16>
    %31 = "ttir.conv2d"(%30, %arg202) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x160x160x80xbf16>
    %32 = "ttir.transpose"(%31) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x80xbf16>) -> tensor<1x160x80x160xbf16>
    %33 = "ttir.transpose"(%32) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x80x160x160xbf16>
    %34 = "ttir.multiply"(%33, %arg8) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %35 = "ttir.add"(%34, %arg9) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %36 = "ttir.sigmoid"(%35) : (tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %37 = "ttir.multiply"(%35, %36) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %38 = "ttir.transpose"(%37) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x160xbf16>) -> tensor<1x160x80x160xbf16>
    %39 = "ttir.transpose"(%38) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x160x160x80xbf16>
    %40 = "ttir.conv2d"(%39, %arg203) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x160x160x80xbf16>
    %41 = "ttir.transpose"(%40) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x80xbf16>) -> tensor<1x160x80x160xbf16>
    %42 = "ttir.transpose"(%41) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x80x160x160xbf16>
    %43 = "ttir.multiply"(%42, %arg10) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %44 = "ttir.add"(%43, %arg11) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %45 = "ttir.sigmoid"(%44) : (tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %46 = "ttir.multiply"(%44, %45) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %47 = "ttir.add"(%28, %46) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %48 = "ttir.transpose"(%47) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x160xbf16>) -> tensor<1x160x80x160xbf16>
    %49 = "ttir.transpose"(%48) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x160x160x80xbf16>
    %50 = "ttir.conv2d"(%49, %arg204) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x160x160x80xbf16>
    %51 = "ttir.transpose"(%50) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x80xbf16>) -> tensor<1x160x80x160xbf16>
    %52 = "ttir.transpose"(%51) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x80x160x160xbf16>
    %53 = "ttir.multiply"(%52, %arg12) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %54 = "ttir.add"(%53, %arg13) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %55 = "ttir.sigmoid"(%54) : (tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %56 = "ttir.multiply"(%54, %55) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %57 = "ttir.transpose"(%56) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x160xbf16>) -> tensor<1x160x80x160xbf16>
    %58 = "ttir.transpose"(%57) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x160x160x80xbf16>
    %59 = "ttir.conv2d"(%58, %arg205) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x160x160x80xbf16>
    %60 = "ttir.transpose"(%59) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x80xbf16>) -> tensor<1x160x80x160xbf16>
    %61 = "ttir.transpose"(%60) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x80x160x160xbf16>
    %62 = "ttir.multiply"(%61, %arg14) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %63 = "ttir.add"(%62, %arg15) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %64 = "ttir.sigmoid"(%63) : (tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %65 = "ttir.multiply"(%63, %64) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %66 = "ttir.add"(%47, %65) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %67 = "ttir.transpose"(%66) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x160xbf16>) -> tensor<1x160x80x160xbf16>
    %68 = "ttir.transpose"(%67) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x160x160x80xbf16>
    %69 = "ttir.conv2d"(%68, %arg206) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x160x160x80xbf16>
    %70 = "ttir.transpose"(%69) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x80xbf16>) -> tensor<1x160x80x160xbf16>
    %71 = "ttir.transpose"(%70) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x80x160x160xbf16>
    %72 = "ttir.multiply"(%71, %arg16) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %73 = "ttir.add"(%72, %arg17) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %74 = "ttir.sigmoid"(%73) : (tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %75 = "ttir.multiply"(%73, %74) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %76 = "ttir.transpose"(%75) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x160xbf16>) -> tensor<1x160x80x160xbf16>
    %77 = "ttir.transpose"(%76) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x160x160x80xbf16>
    %78 = "ttir.conv2d"(%77, %arg207) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x160x160x80xbf16>
    %79 = "ttir.transpose"(%78) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x80xbf16>) -> tensor<1x160x80x160xbf16>
    %80 = "ttir.transpose"(%79) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x160xbf16>) -> tensor<1x80x160x160xbf16>
    %81 = "ttir.multiply"(%80, %arg18) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %82 = "ttir.add"(%81, %arg19) : (tensor<1x80x160x160xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x160x160xbf16>
    %83 = "ttir.sigmoid"(%82) : (tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %84 = "ttir.multiply"(%82, %83) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %85 = "ttir.add"(%66, %84) : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x80x160x160xbf16>
    %86 = "ttir.concat"(%27, %28, %47, %66, %85) <{dim = -3 : si32}> : (tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>, tensor<1x80x160x160xbf16>) -> tensor<1x400x160x160xbf16>
    %87 = "ttir.transpose"(%86) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x400x160x160xbf16>) -> tensor<1x160x400x160xbf16>
    %88 = "ttir.transpose"(%87) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x400x160xbf16>) -> tensor<1x160x160x400xbf16>
    %89 = "ttir.conv2d"(%88, %arg208) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x160x160x400xbf16>, tensor<160x400x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %90 = "ttir.transpose"(%89) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %91 = "ttir.transpose"(%90) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %92 = "ttir.multiply"(%91, %arg20) : (tensor<1x160x160x160xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %93 = "ttir.add"(%92, %arg21) : (tensor<1x160x160x160xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x160x160xbf16>
    %94 = "ttir.sigmoid"(%93) : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %95 = "ttir.multiply"(%93, %94) : (tensor<1x160x160x160xbf16>, tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %96 = "ttir.transpose"(%95) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %97 = "ttir.transpose"(%96) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x160x160x160xbf16>) -> tensor<1x160x160x160xbf16>
    %98 = "ttir.conv2d"(%97, %arg209) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x160x160x160xbf16>, tensor<320x160x3x3xbf16>) -> tensor<1x80x80x320xbf16>
    %99 = "ttir.transpose"(%98) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x320xbf16>) -> tensor<1x80x320x80xbf16>
    %100 = "ttir.transpose"(%99) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x320x80x80xbf16>
    %101 = "ttir.multiply"(%100, %arg22) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %102 = "ttir.add"(%101, %arg23) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %103 = "ttir.sigmoid"(%102) : (tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %104 = "ttir.multiply"(%102, %103) : (tensor<1x320x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %105 = "ttir.transpose"(%104) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x80x320x80xbf16>
    %106 = "ttir.transpose"(%105) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x80x80x320xbf16>
    %107 = "ttir.conv2d"(%106, %arg210) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x320xbf16>, tensor<320x320x1x1xbf16>) -> tensor<1x80x80x320xbf16>
    %108 = "ttir.transpose"(%107) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x320xbf16>) -> tensor<1x80x320x80xbf16>
    %109 = "ttir.transpose"(%108) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x320x80x80xbf16>
    %110 = "ttir.multiply"(%109, %arg24) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %111 = "ttir.add"(%110, %arg25) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %112 = "ttir.sigmoid"(%111) : (tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %113 = "ttir.multiply"(%111, %112) : (tensor<1x320x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %114 = "ttir.index"(%113) <{begin = 0 : i32, dim = 1 : i32, end = 160 : i32, step = 1 : i32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %115 = "ttir.index"(%113) <{begin = 160 : i32, dim = 1 : i32, end = 320 : i32, step = 1 : i32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %116 = "ttir.transpose"(%115) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %117 = "ttir.transpose"(%116) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %118 = "ttir.conv2d"(%117, %arg211) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %119 = "ttir.transpose"(%118) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %120 = "ttir.transpose"(%119) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %121 = "ttir.multiply"(%120, %arg26) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %122 = "ttir.add"(%121, %arg27) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %123 = "ttir.sigmoid"(%122) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %124 = "ttir.multiply"(%122, %123) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %125 = "ttir.transpose"(%124) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %126 = "ttir.transpose"(%125) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %127 = "ttir.conv2d"(%126, %arg212) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %128 = "ttir.transpose"(%127) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %129 = "ttir.transpose"(%128) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %130 = "ttir.multiply"(%129, %arg28) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %131 = "ttir.add"(%130, %arg29) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %132 = "ttir.sigmoid"(%131) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %133 = "ttir.multiply"(%131, %132) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %134 = "ttir.add"(%115, %133) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %135 = "ttir.transpose"(%134) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %136 = "ttir.transpose"(%135) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %137 = "ttir.conv2d"(%136, %arg213) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %138 = "ttir.transpose"(%137) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %139 = "ttir.transpose"(%138) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %140 = "ttir.multiply"(%139, %arg30) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %141 = "ttir.add"(%140, %arg31) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %142 = "ttir.sigmoid"(%141) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %143 = "ttir.multiply"(%141, %142) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %144 = "ttir.transpose"(%143) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %145 = "ttir.transpose"(%144) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %146 = "ttir.conv2d"(%145, %arg214) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %147 = "ttir.transpose"(%146) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %148 = "ttir.transpose"(%147) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %149 = "ttir.multiply"(%148, %arg32) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %150 = "ttir.add"(%149, %arg33) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %151 = "ttir.sigmoid"(%150) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %152 = "ttir.multiply"(%150, %151) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %153 = "ttir.add"(%134, %152) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %154 = "ttir.transpose"(%153) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %155 = "ttir.transpose"(%154) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %156 = "ttir.conv2d"(%155, %arg215) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %157 = "ttir.transpose"(%156) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %158 = "ttir.transpose"(%157) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %159 = "ttir.multiply"(%158, %arg34) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %160 = "ttir.add"(%159, %arg35) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %161 = "ttir.sigmoid"(%160) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %162 = "ttir.multiply"(%160, %161) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %163 = "ttir.transpose"(%162) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %164 = "ttir.transpose"(%163) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %165 = "ttir.conv2d"(%164, %arg216) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %166 = "ttir.transpose"(%165) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %167 = "ttir.transpose"(%166) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %168 = "ttir.multiply"(%167, %arg36) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %169 = "ttir.add"(%168, %arg37) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %170 = "ttir.sigmoid"(%169) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %171 = "ttir.multiply"(%169, %170) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %172 = "ttir.add"(%153, %171) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %173 = "ttir.transpose"(%172) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %174 = "ttir.transpose"(%173) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %175 = "ttir.conv2d"(%174, %arg217) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %176 = "ttir.transpose"(%175) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %177 = "ttir.transpose"(%176) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %178 = "ttir.multiply"(%177, %arg38) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %179 = "ttir.add"(%178, %arg39) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %180 = "ttir.sigmoid"(%179) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %181 = "ttir.multiply"(%179, %180) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %182 = "ttir.transpose"(%181) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %183 = "ttir.transpose"(%182) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %184 = "ttir.conv2d"(%183, %arg218) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %185 = "ttir.transpose"(%184) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %186 = "ttir.transpose"(%185) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %187 = "ttir.multiply"(%186, %arg40) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %188 = "ttir.add"(%187, %arg41) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %189 = "ttir.sigmoid"(%188) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %190 = "ttir.multiply"(%188, %189) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %191 = "ttir.add"(%172, %190) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %192 = "ttir.transpose"(%191) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %193 = "ttir.transpose"(%192) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %194 = "ttir.conv2d"(%193, %arg219) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %195 = "ttir.transpose"(%194) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %196 = "ttir.transpose"(%195) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %197 = "ttir.multiply"(%196, %arg42) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %198 = "ttir.add"(%197, %arg43) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %199 = "ttir.sigmoid"(%198) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %200 = "ttir.multiply"(%198, %199) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %201 = "ttir.transpose"(%200) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %202 = "ttir.transpose"(%201) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %203 = "ttir.conv2d"(%202, %arg220) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %204 = "ttir.transpose"(%203) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %205 = "ttir.transpose"(%204) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %206 = "ttir.multiply"(%205, %arg44) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %207 = "ttir.add"(%206, %arg45) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %208 = "ttir.sigmoid"(%207) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %209 = "ttir.multiply"(%207, %208) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %210 = "ttir.add"(%191, %209) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %211 = "ttir.transpose"(%210) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %212 = "ttir.transpose"(%211) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %213 = "ttir.conv2d"(%212, %arg221) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %214 = "ttir.transpose"(%213) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %215 = "ttir.transpose"(%214) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %216 = "ttir.multiply"(%215, %arg46) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %217 = "ttir.add"(%216, %arg47) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %218 = "ttir.sigmoid"(%217) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %219 = "ttir.multiply"(%217, %218) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %220 = "ttir.transpose"(%219) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %221 = "ttir.transpose"(%220) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %222 = "ttir.conv2d"(%221, %arg222) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %223 = "ttir.transpose"(%222) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %224 = "ttir.transpose"(%223) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %225 = "ttir.multiply"(%224, %arg48) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %226 = "ttir.add"(%225, %arg49) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %227 = "ttir.sigmoid"(%226) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %228 = "ttir.multiply"(%226, %227) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %229 = "ttir.add"(%210, %228) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %230 = "ttir.concat"(%114, %115, %134, %153, %172, %191, %210, %229) <{dim = -3 : si32}> : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x1280x80x80xbf16>
    %231 = "ttir.transpose"(%230) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1280x80x80xbf16>) -> tensor<1x80x1280x80xbf16>
    %232 = "ttir.transpose"(%231) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x1280x80xbf16>) -> tensor<1x80x80x1280xbf16>
    %233 = "ttir.conv2d"(%232, %arg223) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x1280xbf16>, tensor<320x1280x1x1xbf16>) -> tensor<1x80x80x320xbf16>
    %234 = "ttir.transpose"(%233) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x320xbf16>) -> tensor<1x80x320x80xbf16>
    %235 = "ttir.transpose"(%234) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x320x80x80xbf16>
    %236 = "ttir.multiply"(%235, %arg50) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %237 = "ttir.add"(%236, %arg51) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %238 = "ttir.sigmoid"(%237) : (tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %239 = "ttir.multiply"(%237, %238) : (tensor<1x320x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %240 = "ttir.transpose"(%239) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x80x320x80xbf16>
    %241 = "ttir.transpose"(%240) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x80x80x320xbf16>
    %242 = "ttir.conv2d"(%241, %arg224) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x80x80x320xbf16>, tensor<640x320x3x3xbf16>) -> tensor<1x40x40x640xbf16>
    %243 = "ttir.transpose"(%242) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %244 = "ttir.transpose"(%243) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %245 = "ttir.multiply"(%244, %arg52) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %246 = "ttir.add"(%245, %arg53) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %247 = "ttir.sigmoid"(%246) : (tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %248 = "ttir.multiply"(%246, %247) : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %249 = "ttir.transpose"(%248) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x40x640x40xbf16>
    %250 = "ttir.transpose"(%249) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x40x40x640xbf16>
    %251 = "ttir.conv2d"(%250, %arg225) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x640xbf16>, tensor<640x640x1x1xbf16>) -> tensor<1x40x40x640xbf16>
    %252 = "ttir.transpose"(%251) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %253 = "ttir.transpose"(%252) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %254 = "ttir.multiply"(%253, %arg54) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %255 = "ttir.add"(%254, %arg55) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %256 = "ttir.sigmoid"(%255) : (tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %257 = "ttir.multiply"(%255, %256) : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %258 = "ttir.index"(%257) <{begin = 0 : i32, dim = 1 : i32, end = 320 : i32, step = 1 : i32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %259 = "ttir.index"(%257) <{begin = 320 : i32, dim = 1 : i32, end = 640 : i32, step = 1 : i32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %260 = "ttir.transpose"(%259) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %261 = "ttir.transpose"(%260) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %262 = "ttir.conv2d"(%261, %arg226) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %263 = "ttir.transpose"(%262) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %264 = "ttir.transpose"(%263) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %265 = "ttir.multiply"(%264, %arg56) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %266 = "ttir.add"(%265, %arg57) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %267 = "ttir.sigmoid"(%266) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %268 = "ttir.multiply"(%266, %267) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %269 = "ttir.transpose"(%268) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %270 = "ttir.transpose"(%269) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %271 = "ttir.conv2d"(%270, %arg227) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %272 = "ttir.transpose"(%271) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %273 = "ttir.transpose"(%272) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %274 = "ttir.multiply"(%273, %arg58) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %275 = "ttir.add"(%274, %arg59) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %276 = "ttir.sigmoid"(%275) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %277 = "ttir.multiply"(%275, %276) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %278 = "ttir.add"(%259, %277) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %279 = "ttir.transpose"(%278) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %280 = "ttir.transpose"(%279) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %281 = "ttir.conv2d"(%280, %arg228) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %282 = "ttir.transpose"(%281) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %283 = "ttir.transpose"(%282) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %284 = "ttir.multiply"(%283, %arg60) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %285 = "ttir.add"(%284, %arg61) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %286 = "ttir.sigmoid"(%285) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %287 = "ttir.multiply"(%285, %286) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %288 = "ttir.transpose"(%287) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %289 = "ttir.transpose"(%288) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %290 = "ttir.conv2d"(%289, %arg229) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %291 = "ttir.transpose"(%290) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %292 = "ttir.transpose"(%291) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %293 = "ttir.multiply"(%292, %arg62) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %294 = "ttir.add"(%293, %arg63) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %295 = "ttir.sigmoid"(%294) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %296 = "ttir.multiply"(%294, %295) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %297 = "ttir.add"(%278, %296) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %298 = "ttir.transpose"(%297) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %299 = "ttir.transpose"(%298) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %300 = "ttir.conv2d"(%299, %arg230) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %301 = "ttir.transpose"(%300) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %302 = "ttir.transpose"(%301) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %303 = "ttir.multiply"(%302, %arg64) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %304 = "ttir.add"(%303, %arg65) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %305 = "ttir.sigmoid"(%304) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %306 = "ttir.multiply"(%304, %305) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %307 = "ttir.transpose"(%306) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %308 = "ttir.transpose"(%307) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %309 = "ttir.conv2d"(%308, %arg231) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %310 = "ttir.transpose"(%309) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %311 = "ttir.transpose"(%310) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %312 = "ttir.multiply"(%311, %arg66) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %313 = "ttir.add"(%312, %arg67) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %314 = "ttir.sigmoid"(%313) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %315 = "ttir.multiply"(%313, %314) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %316 = "ttir.add"(%297, %315) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %317 = "ttir.transpose"(%316) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %318 = "ttir.transpose"(%317) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %319 = "ttir.conv2d"(%318, %arg232) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %320 = "ttir.transpose"(%319) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %321 = "ttir.transpose"(%320) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %322 = "ttir.multiply"(%321, %arg68) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %323 = "ttir.add"(%322, %arg69) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %324 = "ttir.sigmoid"(%323) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %325 = "ttir.multiply"(%323, %324) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %326 = "ttir.transpose"(%325) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %327 = "ttir.transpose"(%326) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %328 = "ttir.conv2d"(%327, %arg233) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %329 = "ttir.transpose"(%328) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %330 = "ttir.transpose"(%329) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %331 = "ttir.multiply"(%330, %arg70) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %332 = "ttir.add"(%331, %arg71) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %333 = "ttir.sigmoid"(%332) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %334 = "ttir.multiply"(%332, %333) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %335 = "ttir.add"(%316, %334) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %336 = "ttir.transpose"(%335) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %337 = "ttir.transpose"(%336) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %338 = "ttir.conv2d"(%337, %arg234) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %339 = "ttir.transpose"(%338) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %340 = "ttir.transpose"(%339) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %341 = "ttir.multiply"(%340, %arg72) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %342 = "ttir.add"(%341, %arg73) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %343 = "ttir.sigmoid"(%342) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %344 = "ttir.multiply"(%342, %343) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %345 = "ttir.transpose"(%344) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %346 = "ttir.transpose"(%345) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %347 = "ttir.conv2d"(%346, %arg235) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %348 = "ttir.transpose"(%347) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %349 = "ttir.transpose"(%348) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %350 = "ttir.multiply"(%349, %arg74) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %351 = "ttir.add"(%350, %arg75) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %352 = "ttir.sigmoid"(%351) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %353 = "ttir.multiply"(%351, %352) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %354 = "ttir.add"(%335, %353) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %355 = "ttir.transpose"(%354) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %356 = "ttir.transpose"(%355) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %357 = "ttir.conv2d"(%356, %arg236) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %358 = "ttir.transpose"(%357) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %359 = "ttir.transpose"(%358) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %360 = "ttir.multiply"(%359, %arg76) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %361 = "ttir.add"(%360, %arg77) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %362 = "ttir.sigmoid"(%361) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %363 = "ttir.multiply"(%361, %362) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %364 = "ttir.transpose"(%363) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %365 = "ttir.transpose"(%364) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %366 = "ttir.conv2d"(%365, %arg237) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %367 = "ttir.transpose"(%366) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %368 = "ttir.transpose"(%367) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %369 = "ttir.multiply"(%368, %arg78) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %370 = "ttir.add"(%369, %arg79) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %371 = "ttir.sigmoid"(%370) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %372 = "ttir.multiply"(%370, %371) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %373 = "ttir.add"(%354, %372) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %374 = "ttir.concat"(%258, %259, %278, %297, %316, %335, %354, %373) <{dim = -3 : si32}> : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x2560x40x40xbf16>
    %375 = "ttir.transpose"(%374) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x2560x40x40xbf16>) -> tensor<1x40x2560x40xbf16>
    %376 = "ttir.transpose"(%375) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x2560x40xbf16>) -> tensor<1x40x40x2560xbf16>
    %377 = "ttir.conv2d"(%376, %arg238) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x2560xbf16>, tensor<640x2560x1x1xbf16>) -> tensor<1x40x40x640xbf16>
    %378 = "ttir.transpose"(%377) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %379 = "ttir.transpose"(%378) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %380 = "ttir.multiply"(%379, %arg80) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %381 = "ttir.add"(%380, %arg81) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %382 = "ttir.sigmoid"(%381) : (tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %383 = "ttir.multiply"(%381, %382) : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %384 = "ttir.transpose"(%383) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x40x640x40xbf16>
    %385 = "ttir.transpose"(%384) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x40x40x640xbf16>
    %386 = "ttir.conv2d"(%385, %arg239) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x40x40x640xbf16>, tensor<640x640x3x3xbf16>) -> tensor<1x20x20x640xbf16>
    %387 = "ttir.transpose"(%386) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x640xbf16>) -> tensor<1x20x640x20xbf16>
    %388 = "ttir.transpose"(%387) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x640x20x20xbf16>
    %389 = "ttir.multiply"(%388, %arg82) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %390 = "ttir.add"(%389, %arg83) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %391 = "ttir.sigmoid"(%390) : (tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %392 = "ttir.multiply"(%390, %391) : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %393 = "ttir.transpose"(%392) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x20x640x20xbf16>
    %394 = "ttir.transpose"(%393) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x20x20x640xbf16>
    %395 = "ttir.conv2d"(%394, %arg240) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x640xbf16>, tensor<640x640x1x1xbf16>) -> tensor<1x20x20x640xbf16>
    %396 = "ttir.transpose"(%395) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x640xbf16>) -> tensor<1x20x640x20xbf16>
    %397 = "ttir.transpose"(%396) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x640x20x20xbf16>
    %398 = "ttir.multiply"(%397, %arg84) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %399 = "ttir.add"(%398, %arg85) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %400 = "ttir.sigmoid"(%399) : (tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %401 = "ttir.multiply"(%399, %400) : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %402 = "ttir.index"(%401) <{begin = 0 : i32, dim = 1 : i32, end = 320 : i32, step = 1 : i32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %403 = "ttir.index"(%401) <{begin = 320 : i32, dim = 1 : i32, end = 640 : i32, step = 1 : i32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %404 = "ttir.transpose"(%403) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %405 = "ttir.transpose"(%404) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %406 = "ttir.conv2d"(%405, %arg241) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %407 = "ttir.transpose"(%406) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %408 = "ttir.transpose"(%407) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %409 = "ttir.multiply"(%408, %arg86) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %410 = "ttir.add"(%409, %arg87) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %411 = "ttir.sigmoid"(%410) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %412 = "ttir.multiply"(%410, %411) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %413 = "ttir.transpose"(%412) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %414 = "ttir.transpose"(%413) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %415 = "ttir.conv2d"(%414, %arg242) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %416 = "ttir.transpose"(%415) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %417 = "ttir.transpose"(%416) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %418 = "ttir.multiply"(%417, %arg88) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %419 = "ttir.add"(%418, %arg89) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %420 = "ttir.sigmoid"(%419) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %421 = "ttir.multiply"(%419, %420) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %422 = "ttir.add"(%403, %421) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %423 = "ttir.transpose"(%422) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %424 = "ttir.transpose"(%423) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %425 = "ttir.conv2d"(%424, %arg243) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %426 = "ttir.transpose"(%425) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %427 = "ttir.transpose"(%426) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %428 = "ttir.multiply"(%427, %arg90) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %429 = "ttir.add"(%428, %arg91) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %430 = "ttir.sigmoid"(%429) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %431 = "ttir.multiply"(%429, %430) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %432 = "ttir.transpose"(%431) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %433 = "ttir.transpose"(%432) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %434 = "ttir.conv2d"(%433, %arg244) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %435 = "ttir.transpose"(%434) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %436 = "ttir.transpose"(%435) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %437 = "ttir.multiply"(%436, %arg92) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %438 = "ttir.add"(%437, %arg93) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %439 = "ttir.sigmoid"(%438) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %440 = "ttir.multiply"(%438, %439) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %441 = "ttir.add"(%422, %440) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %442 = "ttir.transpose"(%441) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %443 = "ttir.transpose"(%442) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %444 = "ttir.conv2d"(%443, %arg245) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %445 = "ttir.transpose"(%444) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %446 = "ttir.transpose"(%445) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %447 = "ttir.multiply"(%446, %arg94) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %448 = "ttir.add"(%447, %arg95) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %449 = "ttir.sigmoid"(%448) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %450 = "ttir.multiply"(%448, %449) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %451 = "ttir.transpose"(%450) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %452 = "ttir.transpose"(%451) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %453 = "ttir.conv2d"(%452, %arg246) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %454 = "ttir.transpose"(%453) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %455 = "ttir.transpose"(%454) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %456 = "ttir.multiply"(%455, %arg96) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %457 = "ttir.add"(%456, %arg97) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %458 = "ttir.sigmoid"(%457) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %459 = "ttir.multiply"(%457, %458) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %460 = "ttir.add"(%441, %459) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %461 = "ttir.concat"(%402, %403, %422, %441, %460) <{dim = -3 : si32}> : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x1600x20x20xbf16>
    %462 = "ttir.transpose"(%461) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1600x20x20xbf16>) -> tensor<1x20x1600x20xbf16>
    %463 = "ttir.transpose"(%462) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x1600x20xbf16>) -> tensor<1x20x20x1600xbf16>
    %464 = "ttir.conv2d"(%463, %arg247) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x1600xbf16>, tensor<640x1600x1x1xbf16>) -> tensor<1x20x20x640xbf16>
    %465 = "ttir.transpose"(%464) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x640xbf16>) -> tensor<1x20x640x20xbf16>
    %466 = "ttir.transpose"(%465) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x640x20x20xbf16>
    %467 = "ttir.multiply"(%466, %arg98) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %468 = "ttir.add"(%467, %arg99) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %469 = "ttir.sigmoid"(%468) : (tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %470 = "ttir.multiply"(%468, %469) : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %471 = "ttir.transpose"(%470) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x20x640x20xbf16>
    %472 = "ttir.transpose"(%471) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x20x20x640xbf16>
    %473 = "ttir.conv2d"(%472, %arg248) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x640xbf16>, tensor<320x640x1x1xbf16>) -> tensor<1x20x20x320xbf16>
    %474 = "ttir.transpose"(%473) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %475 = "ttir.transpose"(%474) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %476 = "ttir.multiply"(%475, %arg100) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %477 = "ttir.add"(%476, %arg101) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %478 = "ttir.sigmoid"(%477) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %479 = "ttir.multiply"(%477, %478) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %480 = "ttir.transpose"(%479) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %481 = "ttir.transpose"(%480) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %482 = "ttir.max_pool2d"(%481) <{ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 5, 5>, padding = array<i32: 2, 2, 2, 2>, stride = array<i32: 1, 1>}> {channel_last = true} : (tensor<1x20x20x320xbf16>) -> tensor<1x20x20x320xbf16>
    %483 = "ttir.transpose"(%482) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %484 = "ttir.transpose"(%483) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %485 = "ttir.transpose"(%484) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %486 = "ttir.transpose"(%485) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %487 = "ttir.max_pool2d"(%486) <{ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 5, 5>, padding = array<i32: 2, 2, 2, 2>, stride = array<i32: 1, 1>}> {channel_last = true} : (tensor<1x20x20x320xbf16>) -> tensor<1x20x20x320xbf16>
    %488 = "ttir.transpose"(%487) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %489 = "ttir.transpose"(%488) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %490 = "ttir.transpose"(%489) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %491 = "ttir.transpose"(%490) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %492 = "ttir.max_pool2d"(%491) <{ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 5, 5>, padding = array<i32: 2, 2, 2, 2>, stride = array<i32: 1, 1>}> {channel_last = true} : (tensor<1x20x20x320xbf16>) -> tensor<1x20x20x320xbf16>
    %493 = "ttir.transpose"(%492) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %494 = "ttir.transpose"(%493) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %495 = "ttir.concat"(%479, %484, %489, %494) <{dim = -3 : si32}> : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x1280x20x20xbf16>
    %496 = "ttir.transpose"(%495) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1280x20x20xbf16>) -> tensor<1x20x1280x20xbf16>
    %497 = "ttir.transpose"(%496) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x1280x20xbf16>) -> tensor<1x20x20x1280xbf16>
    %498 = "ttir.conv2d"(%497, %arg249) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x1280xbf16>, tensor<640x1280x1x1xbf16>) -> tensor<1x20x20x640xbf16>
    %499 = "ttir.transpose"(%498) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x640xbf16>) -> tensor<1x20x640x20xbf16>
    %500 = "ttir.transpose"(%499) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x640x20x20xbf16>
    %501 = "ttir.multiply"(%500, %arg102) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %502 = "ttir.add"(%501, %arg103) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %503 = "ttir.sigmoid"(%502) : (tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %504 = "ttir.multiply"(%502, %503) : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %505 = "ttir.transpose"(%504) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x20x640x20xbf16>
    %506 = "ttir.transpose"(%505) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x20x20x640xbf16>
    %507 = "ttir.upsample2d"(%506) <{mode = "nearest", scale_factor = 2 : si32}> {channel_last = true} : (tensor<1x20x20x640xbf16>) -> tensor<1x40x40x640xbf16>
    %508 = "ttir.transpose"(%507) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %509 = "ttir.transpose"(%508) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %510 = "ttir.concat"(%509, %383) <{dim = -3 : si32}> : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x1280x40x40xbf16>
    %511 = "ttir.transpose"(%510) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1280x40x40xbf16>) -> tensor<1x40x1280x40xbf16>
    %512 = "ttir.transpose"(%511) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x1280x40xbf16>) -> tensor<1x40x40x1280xbf16>
    %513 = "ttir.conv2d"(%512, %arg250) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x1280xbf16>, tensor<640x1280x1x1xbf16>) -> tensor<1x40x40x640xbf16>
    %514 = "ttir.transpose"(%513) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %515 = "ttir.transpose"(%514) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %516 = "ttir.multiply"(%515, %arg104) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %517 = "ttir.add"(%516, %arg105) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %518 = "ttir.sigmoid"(%517) : (tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %519 = "ttir.multiply"(%517, %518) : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %520 = "ttir.index"(%519) <{begin = 0 : i32, dim = 1 : i32, end = 320 : i32, step = 1 : i32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %521 = "ttir.index"(%519) <{begin = 320 : i32, dim = 1 : i32, end = 640 : i32, step = 1 : i32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %522 = "ttir.transpose"(%521) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %523 = "ttir.transpose"(%522) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %524 = "ttir.conv2d"(%523, %arg251) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %525 = "ttir.transpose"(%524) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %526 = "ttir.transpose"(%525) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %527 = "ttir.multiply"(%526, %arg106) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %528 = "ttir.add"(%527, %arg107) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %529 = "ttir.sigmoid"(%528) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %530 = "ttir.multiply"(%528, %529) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %531 = "ttir.transpose"(%530) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %532 = "ttir.transpose"(%531) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %533 = "ttir.conv2d"(%532, %arg252) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %534 = "ttir.transpose"(%533) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %535 = "ttir.transpose"(%534) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %536 = "ttir.multiply"(%535, %arg108) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %537 = "ttir.add"(%536, %arg109) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %538 = "ttir.sigmoid"(%537) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %539 = "ttir.multiply"(%537, %538) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %540 = "ttir.transpose"(%539) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %541 = "ttir.transpose"(%540) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %542 = "ttir.conv2d"(%541, %arg253) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %543 = "ttir.transpose"(%542) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %544 = "ttir.transpose"(%543) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %545 = "ttir.multiply"(%544, %arg110) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %546 = "ttir.add"(%545, %arg111) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %547 = "ttir.sigmoid"(%546) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %548 = "ttir.multiply"(%546, %547) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %549 = "ttir.transpose"(%548) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %550 = "ttir.transpose"(%549) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %551 = "ttir.conv2d"(%550, %arg254) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %552 = "ttir.transpose"(%551) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %553 = "ttir.transpose"(%552) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %554 = "ttir.multiply"(%553, %arg112) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %555 = "ttir.add"(%554, %arg113) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %556 = "ttir.sigmoid"(%555) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %557 = "ttir.multiply"(%555, %556) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %558 = "ttir.transpose"(%557) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %559 = "ttir.transpose"(%558) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %560 = "ttir.conv2d"(%559, %arg255) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %561 = "ttir.transpose"(%560) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %562 = "ttir.transpose"(%561) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %563 = "ttir.multiply"(%562, %arg114) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %564 = "ttir.add"(%563, %arg115) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %565 = "ttir.sigmoid"(%564) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %566 = "ttir.multiply"(%564, %565) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %567 = "ttir.transpose"(%566) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %568 = "ttir.transpose"(%567) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %569 = "ttir.conv2d"(%568, %arg256) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %570 = "ttir.transpose"(%569) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %571 = "ttir.transpose"(%570) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %572 = "ttir.multiply"(%571, %arg116) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %573 = "ttir.add"(%572, %arg117) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %574 = "ttir.sigmoid"(%573) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %575 = "ttir.multiply"(%573, %574) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %576 = "ttir.concat"(%520, %521, %539, %557, %575) <{dim = -3 : si32}> : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x1600x40x40xbf16>
    %577 = "ttir.transpose"(%576) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1600x40x40xbf16>) -> tensor<1x40x1600x40xbf16>
    %578 = "ttir.transpose"(%577) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x1600x40xbf16>) -> tensor<1x40x40x1600xbf16>
    %579 = "ttir.conv2d"(%578, %arg257) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x1600xbf16>, tensor<640x1600x1x1xbf16>) -> tensor<1x40x40x640xbf16>
    %580 = "ttir.transpose"(%579) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %581 = "ttir.transpose"(%580) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %582 = "ttir.multiply"(%581, %arg118) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %583 = "ttir.add"(%582, %arg119) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %584 = "ttir.sigmoid"(%583) : (tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %585 = "ttir.multiply"(%583, %584) : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %586 = "ttir.transpose"(%585) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x40x640x40xbf16>
    %587 = "ttir.transpose"(%586) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x40x40x640xbf16>
    %588 = "ttir.upsample2d"(%587) <{mode = "nearest", scale_factor = 2 : si32}> {channel_last = true} : (tensor<1x40x40x640xbf16>) -> tensor<1x80x80x640xbf16>
    %589 = "ttir.transpose"(%588) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x640xbf16>) -> tensor<1x80x640x80xbf16>
    %590 = "ttir.transpose"(%589) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x640x80xbf16>) -> tensor<1x640x80x80xbf16>
    %591 = "ttir.concat"(%590, %239) <{dim = -3 : si32}> : (tensor<1x640x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x960x80x80xbf16>
    %592 = "ttir.transpose"(%591) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x960x80x80xbf16>) -> tensor<1x80x960x80xbf16>
    %593 = "ttir.transpose"(%592) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x960x80xbf16>) -> tensor<1x80x80x960xbf16>
    %594 = "ttir.conv2d"(%593, %arg258) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x960xbf16>, tensor<320x960x1x1xbf16>) -> tensor<1x80x80x320xbf16>
    %595 = "ttir.transpose"(%594) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x320xbf16>) -> tensor<1x80x320x80xbf16>
    %596 = "ttir.transpose"(%595) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x320x80x80xbf16>
    %597 = "ttir.multiply"(%596, %arg120) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %598 = "ttir.add"(%597, %arg121) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %599 = "ttir.sigmoid"(%598) : (tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %600 = "ttir.multiply"(%598, %599) : (tensor<1x320x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %601 = "ttir.index"(%600) <{begin = 0 : i32, dim = 1 : i32, end = 160 : i32, step = 1 : i32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %602 = "ttir.index"(%600) <{begin = 160 : i32, dim = 1 : i32, end = 320 : i32, step = 1 : i32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %603 = "ttir.transpose"(%602) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %604 = "ttir.transpose"(%603) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %605 = "ttir.conv2d"(%604, %arg259) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %606 = "ttir.transpose"(%605) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %607 = "ttir.transpose"(%606) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %608 = "ttir.multiply"(%607, %arg122) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %609 = "ttir.add"(%608, %arg123) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %610 = "ttir.sigmoid"(%609) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %611 = "ttir.multiply"(%609, %610) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %612 = "ttir.transpose"(%611) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %613 = "ttir.transpose"(%612) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %614 = "ttir.conv2d"(%613, %arg260) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %615 = "ttir.transpose"(%614) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %616 = "ttir.transpose"(%615) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %617 = "ttir.multiply"(%616, %arg124) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %618 = "ttir.add"(%617, %arg125) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %619 = "ttir.sigmoid"(%618) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %620 = "ttir.multiply"(%618, %619) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %621 = "ttir.transpose"(%620) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %622 = "ttir.transpose"(%621) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %623 = "ttir.conv2d"(%622, %arg261) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %624 = "ttir.transpose"(%623) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %625 = "ttir.transpose"(%624) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %626 = "ttir.multiply"(%625, %arg126) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %627 = "ttir.add"(%626, %arg127) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %628 = "ttir.sigmoid"(%627) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %629 = "ttir.multiply"(%627, %628) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %630 = "ttir.transpose"(%629) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %631 = "ttir.transpose"(%630) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %632 = "ttir.conv2d"(%631, %arg262) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %633 = "ttir.transpose"(%632) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %634 = "ttir.transpose"(%633) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %635 = "ttir.multiply"(%634, %arg128) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %636 = "ttir.add"(%635, %arg129) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %637 = "ttir.sigmoid"(%636) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %638 = "ttir.multiply"(%636, %637) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %639 = "ttir.transpose"(%638) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %640 = "ttir.transpose"(%639) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %641 = "ttir.conv2d"(%640, %arg263) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %642 = "ttir.transpose"(%641) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %643 = "ttir.transpose"(%642) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %644 = "ttir.multiply"(%643, %arg130) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %645 = "ttir.add"(%644, %arg131) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %646 = "ttir.sigmoid"(%645) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %647 = "ttir.multiply"(%645, %646) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %648 = "ttir.transpose"(%647) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x160x80x80xbf16>) -> tensor<1x80x160x80xbf16>
    %649 = "ttir.transpose"(%648) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x80x80x160xbf16>
    %650 = "ttir.conv2d"(%649, %arg264) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x160xbf16>, tensor<160x160x3x3xbf16>) -> tensor<1x80x80x160xbf16>
    %651 = "ttir.transpose"(%650) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x160xbf16>) -> tensor<1x80x160x80xbf16>
    %652 = "ttir.transpose"(%651) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x160x80xbf16>) -> tensor<1x160x80x80xbf16>
    %653 = "ttir.multiply"(%652, %arg132) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %654 = "ttir.add"(%653, %arg133) : (tensor<1x160x80x80xbf16>, tensor<1x160x1x1xbf16>) -> tensor<1x160x80x80xbf16>
    %655 = "ttir.sigmoid"(%654) : (tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %656 = "ttir.multiply"(%654, %655) : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x160x80x80xbf16>
    %657 = "ttir.concat"(%601, %602, %620, %638, %656) <{dim = -3 : si32}> : (tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>, tensor<1x160x80x80xbf16>) -> tensor<1x800x80x80xbf16>
    %658 = "ttir.transpose"(%657) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x800x80x80xbf16>) -> tensor<1x80x800x80xbf16>
    %659 = "ttir.transpose"(%658) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x800x80xbf16>) -> tensor<1x80x80x800xbf16>
    %660 = "ttir.conv2d"(%659, %arg265) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x800xbf16>, tensor<320x800x1x1xbf16>) -> tensor<1x80x80x320xbf16>
    %661 = "ttir.transpose"(%660) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x320xbf16>) -> tensor<1x80x320x80xbf16>
    %662 = "ttir.transpose"(%661) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x320x80x80xbf16>
    %663 = "ttir.multiply"(%662, %arg134) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %664 = "ttir.add"(%663, %arg135) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %665 = "ttir.sigmoid"(%664) : (tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %666 = "ttir.multiply"(%664, %665) : (tensor<1x320x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %667 = "ttir.transpose"(%666) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x80x320x80xbf16>
    %668 = "ttir.transpose"(%667) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x80x80x320xbf16>
    %669 = "ttir.conv2d"(%668, %arg266) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x320xbf16>, tensor<80x320x3x3xbf16>) -> tensor<1x80x80x80xbf16>
    %670 = "ttir.transpose"(%669) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %671 = "ttir.transpose"(%670) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %672 = "ttir.multiply"(%671, %arg136) : (tensor<1x80x80x80xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x80x80xbf16>
    %673 = "ttir.add"(%672, %arg137) : (tensor<1x80x80x80xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x80x80xbf16>
    %674 = "ttir.sigmoid"(%673) : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %675 = "ttir.multiply"(%673, %674) : (tensor<1x80x80x80xbf16>, tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %676 = "ttir.transpose"(%675) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %677 = "ttir.transpose"(%676) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %678 = "ttir.conv2d"(%677, %arg267) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x80x80x80xbf16>
    %679 = "ttir.transpose"(%678) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %680 = "ttir.transpose"(%679) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %681 = "ttir.multiply"(%680, %arg138) : (tensor<1x80x80x80xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x80x80xbf16>
    %682 = "ttir.add"(%681, %arg139) : (tensor<1x80x80x80xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x80x80xbf16>
    %683 = "ttir.sigmoid"(%682) : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %684 = "ttir.multiply"(%682, %683) : (tensor<1x80x80x80xbf16>, tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %685 = "ttir.transpose"(%684) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %686 = "ttir.transpose"(%685) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %687 = "ttir.conv2d"(%686, %arg268, %arg269) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x80xbf16>, tensor<64x80x1x1xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x80x80x64xbf16>
    %688 = "ttir.transpose"(%687) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x64xbf16>) -> tensor<1x80x64x80xbf16>
    %689 = "ttir.transpose"(%688) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x64x80xbf16>) -> tensor<1x64x80x80xbf16>
    %690 = "ttir.transpose"(%666) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x80x320x80xbf16>
    %691 = "ttir.transpose"(%690) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x80x80x320xbf16>
    %692 = "ttir.conv2d"(%691, %arg270) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x80x80x320xbf16>
    %693 = "ttir.transpose"(%692) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x320xbf16>) -> tensor<1x80x320x80xbf16>
    %694 = "ttir.transpose"(%693) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x320x80x80xbf16>
    %695 = "ttir.multiply"(%694, %arg140) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %696 = "ttir.add"(%695, %arg141) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %697 = "ttir.sigmoid"(%696) : (tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %698 = "ttir.multiply"(%696, %697) : (tensor<1x320x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %699 = "ttir.transpose"(%698) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x80x320x80xbf16>
    %700 = "ttir.transpose"(%699) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x80x80x320xbf16>
    %701 = "ttir.conv2d"(%700, %arg271) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x80x80x320xbf16>
    %702 = "ttir.transpose"(%701) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x320xbf16>) -> tensor<1x80x320x80xbf16>
    %703 = "ttir.transpose"(%702) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x320x80x80xbf16>
    %704 = "ttir.multiply"(%703, %arg142) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %705 = "ttir.add"(%704, %arg143) : (tensor<1x320x80x80xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x80x80xbf16>
    %706 = "ttir.sigmoid"(%705) : (tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %707 = "ttir.multiply"(%705, %706) : (tensor<1x320x80x80xbf16>, tensor<1x320x80x80xbf16>) -> tensor<1x320x80x80xbf16>
    %708 = "ttir.transpose"(%707) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x80x320x80xbf16>
    %709 = "ttir.transpose"(%708) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x80x80x320xbf16>
    %710 = "ttir.conv2d"(%709, %arg272, %arg273) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x80x80x320xbf16>, tensor<80x320x1x1xbf16>, tensor<1x1x1x80xbf16>) -> tensor<1x80x80x80xbf16>
    %711 = "ttir.transpose"(%710) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %712 = "ttir.transpose"(%711) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x80x80xbf16>) -> tensor<1x80x80x80xbf16>
    %713 = "ttir.concat"(%689, %712) <{dim = -3 : si32}> : (tensor<1x64x80x80xbf16>, tensor<1x80x80x80xbf16>) -> tensor<1x144x80x80xbf16>
    %714 = "ttir.reshape"(%713) <{shape = [1 : i32, 144 : i32, 6400 : i32]}> : (tensor<1x144x80x80xbf16>) -> tensor<1x144x6400xbf16>
    %715 = "ttir.transpose"(%666) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x80x80xbf16>) -> tensor<1x80x320x80xbf16>
    %716 = "ttir.transpose"(%715) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x80x320x80xbf16>) -> tensor<1x80x80x320xbf16>
    %717 = "ttir.conv2d"(%716, %arg274) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x80x80x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %718 = "ttir.transpose"(%717) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %719 = "ttir.transpose"(%718) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %720 = "ttir.multiply"(%719, %arg144) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %721 = "ttir.add"(%720, %arg145) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %722 = "ttir.sigmoid"(%721) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %723 = "ttir.multiply"(%721, %722) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %724 = "ttir.concat"(%723, %585) <{dim = -3 : si32}> : (tensor<1x320x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x960x40x40xbf16>
    %725 = "ttir.transpose"(%724) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x960x40x40xbf16>) -> tensor<1x40x960x40xbf16>
    %726 = "ttir.transpose"(%725) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x960x40xbf16>) -> tensor<1x40x40x960xbf16>
    %727 = "ttir.conv2d"(%726, %arg275) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x960xbf16>, tensor<640x960x1x1xbf16>) -> tensor<1x40x40x640xbf16>
    %728 = "ttir.transpose"(%727) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %729 = "ttir.transpose"(%728) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %730 = "ttir.multiply"(%729, %arg146) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %731 = "ttir.add"(%730, %arg147) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %732 = "ttir.sigmoid"(%731) : (tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %733 = "ttir.multiply"(%731, %732) : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %734 = "ttir.index"(%733) <{begin = 0 : i32, dim = 1 : i32, end = 320 : i32, step = 1 : i32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %735 = "ttir.index"(%733) <{begin = 320 : i32, dim = 1 : i32, end = 640 : i32, step = 1 : i32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %736 = "ttir.transpose"(%735) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %737 = "ttir.transpose"(%736) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %738 = "ttir.conv2d"(%737, %arg276) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %739 = "ttir.transpose"(%738) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %740 = "ttir.transpose"(%739) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %741 = "ttir.multiply"(%740, %arg148) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %742 = "ttir.add"(%741, %arg149) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %743 = "ttir.sigmoid"(%742) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %744 = "ttir.multiply"(%742, %743) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %745 = "ttir.transpose"(%744) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %746 = "ttir.transpose"(%745) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %747 = "ttir.conv2d"(%746, %arg277) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %748 = "ttir.transpose"(%747) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %749 = "ttir.transpose"(%748) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %750 = "ttir.multiply"(%749, %arg150) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %751 = "ttir.add"(%750, %arg151) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %752 = "ttir.sigmoid"(%751) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %753 = "ttir.multiply"(%751, %752) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %754 = "ttir.transpose"(%753) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %755 = "ttir.transpose"(%754) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %756 = "ttir.conv2d"(%755, %arg278) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %757 = "ttir.transpose"(%756) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %758 = "ttir.transpose"(%757) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %759 = "ttir.multiply"(%758, %arg152) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %760 = "ttir.add"(%759, %arg153) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %761 = "ttir.sigmoid"(%760) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %762 = "ttir.multiply"(%760, %761) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %763 = "ttir.transpose"(%762) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %764 = "ttir.transpose"(%763) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %765 = "ttir.conv2d"(%764, %arg279) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %766 = "ttir.transpose"(%765) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %767 = "ttir.transpose"(%766) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %768 = "ttir.multiply"(%767, %arg154) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %769 = "ttir.add"(%768, %arg155) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %770 = "ttir.sigmoid"(%769) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %771 = "ttir.multiply"(%769, %770) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %772 = "ttir.transpose"(%771) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %773 = "ttir.transpose"(%772) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %774 = "ttir.conv2d"(%773, %arg280) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %775 = "ttir.transpose"(%774) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %776 = "ttir.transpose"(%775) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %777 = "ttir.multiply"(%776, %arg156) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %778 = "ttir.add"(%777, %arg157) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %779 = "ttir.sigmoid"(%778) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %780 = "ttir.multiply"(%778, %779) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %781 = "ttir.transpose"(%780) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %782 = "ttir.transpose"(%781) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %783 = "ttir.conv2d"(%782, %arg281) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %784 = "ttir.transpose"(%783) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %785 = "ttir.transpose"(%784) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %786 = "ttir.multiply"(%785, %arg158) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %787 = "ttir.add"(%786, %arg159) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %788 = "ttir.sigmoid"(%787) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %789 = "ttir.multiply"(%787, %788) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %790 = "ttir.concat"(%734, %735, %753, %771, %789) <{dim = -3 : si32}> : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x1600x40x40xbf16>
    %791 = "ttir.transpose"(%790) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1600x40x40xbf16>) -> tensor<1x40x1600x40xbf16>
    %792 = "ttir.transpose"(%791) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x1600x40xbf16>) -> tensor<1x40x40x1600xbf16>
    %793 = "ttir.conv2d"(%792, %arg282) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x1600xbf16>, tensor<640x1600x1x1xbf16>) -> tensor<1x40x40x640xbf16>
    %794 = "ttir.transpose"(%793) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x640xbf16>) -> tensor<1x40x640x40xbf16>
    %795 = "ttir.transpose"(%794) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x640x40x40xbf16>
    %796 = "ttir.multiply"(%795, %arg160) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %797 = "ttir.add"(%796, %arg161) : (tensor<1x640x40x40xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x40x40xbf16>
    %798 = "ttir.sigmoid"(%797) : (tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %799 = "ttir.multiply"(%797, %798) : (tensor<1x640x40x40xbf16>, tensor<1x640x40x40xbf16>) -> tensor<1x640x40x40xbf16>
    %800 = "ttir.transpose"(%799) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x40x640x40xbf16>
    %801 = "ttir.transpose"(%800) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x40x40x640xbf16>
    %802 = "ttir.conv2d"(%801, %arg283) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x640xbf16>, tensor<80x640x3x3xbf16>) -> tensor<1x40x40x80xbf16>
    %803 = "ttir.transpose"(%802) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x80xbf16>) -> tensor<1x40x80x40xbf16>
    %804 = "ttir.transpose"(%803) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x80x40xbf16>) -> tensor<1x80x40x40xbf16>
    %805 = "ttir.multiply"(%804, %arg162) : (tensor<1x80x40x40xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x40x40xbf16>
    %806 = "ttir.add"(%805, %arg163) : (tensor<1x80x40x40xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x40x40xbf16>
    %807 = "ttir.sigmoid"(%806) : (tensor<1x80x40x40xbf16>) -> tensor<1x80x40x40xbf16>
    %808 = "ttir.multiply"(%806, %807) : (tensor<1x80x40x40xbf16>, tensor<1x80x40x40xbf16>) -> tensor<1x80x40x40xbf16>
    %809 = "ttir.transpose"(%808) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x40x40xbf16>) -> tensor<1x40x80x40xbf16>
    %810 = "ttir.transpose"(%809) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x80x40xbf16>) -> tensor<1x40x40x80xbf16>
    %811 = "ttir.conv2d"(%810, %arg284) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x40x40x80xbf16>
    %812 = "ttir.transpose"(%811) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x80xbf16>) -> tensor<1x40x80x40xbf16>
    %813 = "ttir.transpose"(%812) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x80x40xbf16>) -> tensor<1x80x40x40xbf16>
    %814 = "ttir.multiply"(%813, %arg164) : (tensor<1x80x40x40xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x40x40xbf16>
    %815 = "ttir.add"(%814, %arg165) : (tensor<1x80x40x40xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x40x40xbf16>
    %816 = "ttir.sigmoid"(%815) : (tensor<1x80x40x40xbf16>) -> tensor<1x80x40x40xbf16>
    %817 = "ttir.multiply"(%815, %816) : (tensor<1x80x40x40xbf16>, tensor<1x80x40x40xbf16>) -> tensor<1x80x40x40xbf16>
    %818 = "ttir.transpose"(%817) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x40x40xbf16>) -> tensor<1x40x80x40xbf16>
    %819 = "ttir.transpose"(%818) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x80x40xbf16>) -> tensor<1x40x40x80xbf16>
    %820 = "ttir.conv2d"(%819, %arg285, %arg286) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x80xbf16>, tensor<64x80x1x1xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x40x40x64xbf16>
    %821 = "ttir.transpose"(%820) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x64xbf16>) -> tensor<1x40x64x40xbf16>
    %822 = "ttir.transpose"(%821) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x64x40xbf16>) -> tensor<1x64x40x40xbf16>
    %823 = "ttir.transpose"(%799) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x40x640x40xbf16>
    %824 = "ttir.transpose"(%823) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x40x40x640xbf16>
    %825 = "ttir.conv2d"(%824, %arg287) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x640xbf16>, tensor<320x640x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %826 = "ttir.transpose"(%825) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %827 = "ttir.transpose"(%826) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %828 = "ttir.multiply"(%827, %arg166) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %829 = "ttir.add"(%828, %arg167) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %830 = "ttir.sigmoid"(%829) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %831 = "ttir.multiply"(%829, %830) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %832 = "ttir.transpose"(%831) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %833 = "ttir.transpose"(%832) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %834 = "ttir.conv2d"(%833, %arg288) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x40x40x320xbf16>
    %835 = "ttir.transpose"(%834) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x320xbf16>) -> tensor<1x40x320x40xbf16>
    %836 = "ttir.transpose"(%835) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x320x40x40xbf16>
    %837 = "ttir.multiply"(%836, %arg168) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %838 = "ttir.add"(%837, %arg169) : (tensor<1x320x40x40xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x40x40xbf16>
    %839 = "ttir.sigmoid"(%838) : (tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %840 = "ttir.multiply"(%838, %839) : (tensor<1x320x40x40xbf16>, tensor<1x320x40x40xbf16>) -> tensor<1x320x40x40xbf16>
    %841 = "ttir.transpose"(%840) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x40x40xbf16>) -> tensor<1x40x320x40xbf16>
    %842 = "ttir.transpose"(%841) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x320x40xbf16>) -> tensor<1x40x40x320xbf16>
    %843 = "ttir.conv2d"(%842, %arg289, %arg290) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x40x40x320xbf16>, tensor<80x320x1x1xbf16>, tensor<1x1x1x80xbf16>) -> tensor<1x40x40x80xbf16>
    %844 = "ttir.transpose"(%843) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x40x80xbf16>) -> tensor<1x40x80x40xbf16>
    %845 = "ttir.transpose"(%844) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x40x80x40xbf16>) -> tensor<1x80x40x40xbf16>
    %846 = "ttir.concat"(%822, %845) <{dim = -3 : si32}> : (tensor<1x64x40x40xbf16>, tensor<1x80x40x40xbf16>) -> tensor<1x144x40x40xbf16>
    %847 = "ttir.reshape"(%846) <{shape = [1 : i32, 144 : i32, 1600 : i32]}> : (tensor<1x144x40x40xbf16>) -> tensor<1x144x1600xbf16>
    %848 = "ttir.transpose"(%799) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x40x40xbf16>) -> tensor<1x40x640x40xbf16>
    %849 = "ttir.transpose"(%848) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x40x640x40xbf16>) -> tensor<1x40x40x640xbf16>
    %850 = "ttir.conv2d"(%849, %arg291) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> {channel_last = 1 : si32} : (tensor<1x40x40x640xbf16>, tensor<640x640x3x3xbf16>) -> tensor<1x20x20x640xbf16>
    %851 = "ttir.transpose"(%850) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x640xbf16>) -> tensor<1x20x640x20xbf16>
    %852 = "ttir.transpose"(%851) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x640x20x20xbf16>
    %853 = "ttir.multiply"(%852, %arg170) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %854 = "ttir.add"(%853, %arg171) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %855 = "ttir.sigmoid"(%854) : (tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %856 = "ttir.multiply"(%854, %855) : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %857 = "ttir.concat"(%856, %504) <{dim = -3 : si32}> : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x1280x20x20xbf16>
    %858 = "ttir.transpose"(%857) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1280x20x20xbf16>) -> tensor<1x20x1280x20xbf16>
    %859 = "ttir.transpose"(%858) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x1280x20xbf16>) -> tensor<1x20x20x1280xbf16>
    %860 = "ttir.conv2d"(%859, %arg292) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x1280xbf16>, tensor<640x1280x1x1xbf16>) -> tensor<1x20x20x640xbf16>
    %861 = "ttir.transpose"(%860) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x640xbf16>) -> tensor<1x20x640x20xbf16>
    %862 = "ttir.transpose"(%861) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x640x20x20xbf16>
    %863 = "ttir.multiply"(%862, %arg172) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %864 = "ttir.add"(%863, %arg173) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %865 = "ttir.sigmoid"(%864) : (tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %866 = "ttir.multiply"(%864, %865) : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %867 = "ttir.index"(%866) <{begin = 0 : i32, dim = 1 : i32, end = 320 : i32, step = 1 : i32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %868 = "ttir.index"(%866) <{begin = 320 : i32, dim = 1 : i32, end = 640 : i32, step = 1 : i32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %869 = "ttir.transpose"(%868) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %870 = "ttir.transpose"(%869) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %871 = "ttir.conv2d"(%870, %arg293) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %872 = "ttir.transpose"(%871) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %873 = "ttir.transpose"(%872) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %874 = "ttir.multiply"(%873, %arg174) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %875 = "ttir.add"(%874, %arg175) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %876 = "ttir.sigmoid"(%875) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %877 = "ttir.multiply"(%875, %876) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %878 = "ttir.transpose"(%877) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %879 = "ttir.transpose"(%878) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %880 = "ttir.conv2d"(%879, %arg294) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %881 = "ttir.transpose"(%880) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %882 = "ttir.transpose"(%881) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %883 = "ttir.multiply"(%882, %arg176) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %884 = "ttir.add"(%883, %arg177) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %885 = "ttir.sigmoid"(%884) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %886 = "ttir.multiply"(%884, %885) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %887 = "ttir.transpose"(%886) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %888 = "ttir.transpose"(%887) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %889 = "ttir.conv2d"(%888, %arg295) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %890 = "ttir.transpose"(%889) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %891 = "ttir.transpose"(%890) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %892 = "ttir.multiply"(%891, %arg178) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %893 = "ttir.add"(%892, %arg179) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %894 = "ttir.sigmoid"(%893) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %895 = "ttir.multiply"(%893, %894) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %896 = "ttir.transpose"(%895) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %897 = "ttir.transpose"(%896) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %898 = "ttir.conv2d"(%897, %arg296) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %899 = "ttir.transpose"(%898) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %900 = "ttir.transpose"(%899) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %901 = "ttir.multiply"(%900, %arg180) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %902 = "ttir.add"(%901, %arg181) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %903 = "ttir.sigmoid"(%902) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %904 = "ttir.multiply"(%902, %903) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %905 = "ttir.transpose"(%904) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %906 = "ttir.transpose"(%905) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %907 = "ttir.conv2d"(%906, %arg297) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %908 = "ttir.transpose"(%907) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %909 = "ttir.transpose"(%908) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %910 = "ttir.multiply"(%909, %arg182) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %911 = "ttir.add"(%910, %arg183) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %912 = "ttir.sigmoid"(%911) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %913 = "ttir.multiply"(%911, %912) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %914 = "ttir.transpose"(%913) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %915 = "ttir.transpose"(%914) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %916 = "ttir.conv2d"(%915, %arg298) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %917 = "ttir.transpose"(%916) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %918 = "ttir.transpose"(%917) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %919 = "ttir.multiply"(%918, %arg184) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %920 = "ttir.add"(%919, %arg185) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %921 = "ttir.sigmoid"(%920) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %922 = "ttir.multiply"(%920, %921) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %923 = "ttir.concat"(%867, %868, %886, %904, %922) <{dim = -3 : si32}> : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x1600x20x20xbf16>
    %924 = "ttir.transpose"(%923) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x1600x20x20xbf16>) -> tensor<1x20x1600x20xbf16>
    %925 = "ttir.transpose"(%924) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x1600x20xbf16>) -> tensor<1x20x20x1600xbf16>
    %926 = "ttir.conv2d"(%925, %arg299) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x1600xbf16>, tensor<640x1600x1x1xbf16>) -> tensor<1x20x20x640xbf16>
    %927 = "ttir.transpose"(%926) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x640xbf16>) -> tensor<1x20x640x20xbf16>
    %928 = "ttir.transpose"(%927) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x640x20x20xbf16>
    %929 = "ttir.multiply"(%928, %arg186) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %930 = "ttir.add"(%929, %arg187) : (tensor<1x640x20x20xbf16>, tensor<1x640x1x1xbf16>) -> tensor<1x640x20x20xbf16>
    %931 = "ttir.sigmoid"(%930) : (tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %932 = "ttir.multiply"(%930, %931) : (tensor<1x640x20x20xbf16>, tensor<1x640x20x20xbf16>) -> tensor<1x640x20x20xbf16>
    %933 = "ttir.transpose"(%932) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x20x640x20xbf16>
    %934 = "ttir.transpose"(%933) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x20x20x640xbf16>
    %935 = "ttir.conv2d"(%934, %arg300) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x640xbf16>, tensor<80x640x3x3xbf16>) -> tensor<1x20x20x80xbf16>
    %936 = "ttir.transpose"(%935) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x80xbf16>) -> tensor<1x20x80x20xbf16>
    %937 = "ttir.transpose"(%936) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x80x20xbf16>) -> tensor<1x80x20x20xbf16>
    %938 = "ttir.multiply"(%937, %arg188) : (tensor<1x80x20x20xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x20x20xbf16>
    %939 = "ttir.add"(%938, %arg189) : (tensor<1x80x20x20xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x20x20xbf16>
    %940 = "ttir.sigmoid"(%939) : (tensor<1x80x20x20xbf16>) -> tensor<1x80x20x20xbf16>
    %941 = "ttir.multiply"(%939, %940) : (tensor<1x80x20x20xbf16>, tensor<1x80x20x20xbf16>) -> tensor<1x80x20x20xbf16>
    %942 = "ttir.transpose"(%941) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x20x20xbf16>) -> tensor<1x20x80x20xbf16>
    %943 = "ttir.transpose"(%942) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x80x20xbf16>) -> tensor<1x20x20x80xbf16>
    %944 = "ttir.conv2d"(%943, %arg301) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x80xbf16>, tensor<80x80x3x3xbf16>) -> tensor<1x20x20x80xbf16>
    %945 = "ttir.transpose"(%944) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x80xbf16>) -> tensor<1x20x80x20xbf16>
    %946 = "ttir.transpose"(%945) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x80x20xbf16>) -> tensor<1x80x20x20xbf16>
    %947 = "ttir.multiply"(%946, %arg190) : (tensor<1x80x20x20xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x20x20xbf16>
    %948 = "ttir.add"(%947, %arg191) : (tensor<1x80x20x20xbf16>, tensor<1x80x1x1xbf16>) -> tensor<1x80x20x20xbf16>
    %949 = "ttir.sigmoid"(%948) : (tensor<1x80x20x20xbf16>) -> tensor<1x80x20x20xbf16>
    %950 = "ttir.multiply"(%948, %949) : (tensor<1x80x20x20xbf16>, tensor<1x80x20x20xbf16>) -> tensor<1x80x20x20xbf16>
    %951 = "ttir.transpose"(%950) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x80x20x20xbf16>) -> tensor<1x20x80x20xbf16>
    %952 = "ttir.transpose"(%951) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x80x20xbf16>) -> tensor<1x20x20x80xbf16>
    %953 = "ttir.conv2d"(%952, %arg302, %arg303) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x80xbf16>, tensor<64x80x1x1xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x20x20x64xbf16>
    %954 = "ttir.transpose"(%953) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x64xbf16>) -> tensor<1x20x64x20xbf16>
    %955 = "ttir.transpose"(%954) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x64x20xbf16>) -> tensor<1x64x20x20xbf16>
    %956 = "ttir.transpose"(%932) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x640x20x20xbf16>) -> tensor<1x20x640x20xbf16>
    %957 = "ttir.transpose"(%956) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x640x20xbf16>) -> tensor<1x20x20x640xbf16>
    %958 = "ttir.conv2d"(%957, %arg304) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x640xbf16>, tensor<320x640x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %959 = "ttir.transpose"(%958) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %960 = "ttir.transpose"(%959) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %961 = "ttir.multiply"(%960, %arg192) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %962 = "ttir.add"(%961, %arg193) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %963 = "ttir.sigmoid"(%962) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %964 = "ttir.multiply"(%962, %963) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %965 = "ttir.transpose"(%964) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %966 = "ttir.transpose"(%965) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %967 = "ttir.conv2d"(%966, %arg305) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<320x320x3x3xbf16>) -> tensor<1x20x20x320xbf16>
    %968 = "ttir.transpose"(%967) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x320xbf16>) -> tensor<1x20x320x20xbf16>
    %969 = "ttir.transpose"(%968) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x320x20x20xbf16>
    %970 = "ttir.multiply"(%969, %arg194) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %971 = "ttir.add"(%970, %arg195) : (tensor<1x320x20x20xbf16>, tensor<1x320x1x1xbf16>) -> tensor<1x320x20x20xbf16>
    %972 = "ttir.sigmoid"(%971) : (tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %973 = "ttir.multiply"(%971, %972) : (tensor<1x320x20x20xbf16>, tensor<1x320x20x20xbf16>) -> tensor<1x320x20x20xbf16>
    %974 = "ttir.transpose"(%973) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x320x20x20xbf16>) -> tensor<1x20x320x20xbf16>
    %975 = "ttir.transpose"(%974) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x320x20xbf16>) -> tensor<1x20x20x320xbf16>
    %976 = "ttir.conv2d"(%975, %arg306, %arg307) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x20x20x320xbf16>, tensor<80x320x1x1xbf16>, tensor<1x1x1x80xbf16>) -> tensor<1x20x20x80xbf16>
    %977 = "ttir.transpose"(%976) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x20x20x80xbf16>) -> tensor<1x20x80x20xbf16>
    %978 = "ttir.transpose"(%977) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x20x80x20xbf16>) -> tensor<1x80x20x20xbf16>
    %979 = "ttir.concat"(%955, %978) <{dim = -3 : si32}> : (tensor<1x64x20x20xbf16>, tensor<1x80x20x20xbf16>) -> tensor<1x144x20x20xbf16>
    %980 = "ttir.reshape"(%979) <{shape = [1 : i32, 144 : i32, 400 : i32]}> : (tensor<1x144x20x20xbf16>) -> tensor<1x144x400xbf16>
    %981 = "ttir.concat"(%714, %847, %980) <{dim = -1 : si32}> : (tensor<1x144x6400xbf16>, tensor<1x144x1600xbf16>, tensor<1x144x400xbf16>) -> tensor<1x144x8400xbf16>
    %982 = "ttir.index"(%981) <{begin = 0 : i32, dim = 1 : i32, end = 64 : i32, step = 1 : i32}> : (tensor<1x144x8400xbf16>) -> tensor<1x64x8400xbf16>
    %983 = "ttir.reshape"(%982) <{shape = [1 : i32, 4 : i32, 16 : i32, 8400 : i32]}> : (tensor<1x64x8400xbf16>) -> tensor<1x4x16x8400xbf16>
    %984 = "ttir.transpose"(%983) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x4x16x8400xbf16>) -> tensor<1x16x4x8400xbf16>
    %985 = "ttir.softmax"(%984) <{dimension = 1 : si32}> : (tensor<1x16x4x8400xbf16>) -> tensor<1x16x4x8400xbf16>
    %986 = "ttir.transpose"(%985) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x16x4x8400xbf16>) -> tensor<1x4x16x8400xbf16>
    %987 = "ttir.transpose"(%986) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4x16x8400xbf16>) -> tensor<1x4x8400x16xbf16>
    %988 = "ttir.conv2d"(%987, %arg196) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x4x8400x16xbf16>, tensor<1x16x1x1xbf16>) -> tensor<1x4x8400x1xbf16>
    %989 = "ttir.transpose"(%988) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4x8400x1xbf16>) -> tensor<1x4x1x8400xbf16>
    %990 = "ttir.transpose"(%989) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x4x1x8400xbf16>) -> tensor<1x1x4x8400xbf16>
    %991 = "ttir.squeeze"(%990) <{dim = 0 : si32}> : (tensor<1x1x4x8400xbf16>) -> tensor<1x4x8400xbf16>
    %992 = "ttir.index"(%991) <{begin = 0 : i32, dim = 1 : i32, end = 2 : i32, step = 1 : i32}> : (tensor<1x4x8400xbf16>) -> tensor<1x2x8400xbf16>
    %993 = "ttir.subtract"(%arg1, %992) : (tensor<1x2x8400xbf16>, tensor<1x2x8400xbf16>) -> tensor<1x2x8400xbf16>
    %994 = "ttir.index"(%991) <{begin = 2 : i32, dim = 1 : i32, end = 4 : i32, step = 1 : i32}> : (tensor<1x4x8400xbf16>) -> tensor<1x2x8400xbf16>
    %995 = "ttir.add"(%arg1, %994) : (tensor<1x2x8400xbf16>, tensor<1x2x8400xbf16>) -> tensor<1x2x8400xbf16>
    %996 = "ttir.add"(%993, %995) : (tensor<1x2x8400xbf16>, tensor<1x2x8400xbf16>) -> tensor<1x2x8400xbf16>
    %997 = "ttir.div"(%996, %arg197) : (tensor<1x2x8400xbf16>, tensor<1xbf16>) -> tensor<1x2x8400xbf16>
    %998 = "ttir.subtract"(%995, %993) : (tensor<1x2x8400xbf16>, tensor<1x2x8400xbf16>) -> tensor<1x2x8400xbf16>
    %999 = "ttir.concat"(%997, %998) <{dim = -2 : si32}> : (tensor<1x2x8400xbf16>, tensor<1x2x8400xbf16>) -> tensor<1x4x8400xbf16>
    %1000 = "ttir.multiply"(%999, %arg198) : (tensor<1x4x8400xbf16>, tensor<1x8400xbf16>) -> tensor<1x4x8400xbf16>
    %1001 = "ttir.index"(%981) <{begin = 64 : i32, dim = 1 : i32, end = 144 : i32, step = 1 : i32}> : (tensor<1x144x8400xbf16>) -> tensor<1x80x8400xbf16>
    %1002 = "ttir.sigmoid"(%1001) : (tensor<1x80x8400xbf16>) -> tensor<1x80x8400xbf16>
    %1003 = "ttir.concat"(%1000, %1002) <{dim = -2 : si32}> : (tensor<1x4x8400xbf16>, tensor<1x80x8400xbf16>) -> tensor<1x84x8400xbf16>
    return %1003, %713, %846, %979 : tensor<1x84x8400xbf16>, tensor<1x144x80x80xbf16>, tensor<1x144x40x40xbf16>, tensor<1x144x20x20xbf16> loc(#loc196)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("test.benchmark.utils.YoloWrapper::")
#loc2 = loc("multiply_7")
#loc3 = loc("add_13")
#loc4 = loc("multiply_23")
#loc5 = loc("add_29")
#loc6 = loc("multiply_39")
#loc7 = loc("add_45")
#loc8 = loc("multiply_57")
#loc9 = loc("add_63")
#loc10 = loc("multiply_73")
#loc11 = loc("add_79")
#loc12 = loc("multiply_90")
#loc13 = loc("add_96")
#loc14 = loc("multiply_106")
#loc15 = loc("add_112")
#loc16 = loc("multiply_123")
#loc17 = loc("add_129")
#loc18 = loc("multiply_139")
#loc19 = loc("add_145")
#loc20 = loc("multiply_157")
#loc21 = loc("add_163")
#loc22 = loc("multiply_173")
#loc23 = loc("add_179")
#loc24 = loc("multiply_189")
#loc25 = loc("add_195")
#loc26 = loc("multiply_207")
#loc27 = loc("add_213")
#loc28 = loc("multiply_223")
#loc29 = loc("add_229")
#loc30 = loc("multiply_240")
#loc31 = loc("add_246")
#loc32 = loc("multiply_256")
#loc33 = loc("add_262")
#loc34 = loc("multiply_273")
#loc35 = loc("add_279")
#loc36 = loc("multiply_289")
#loc37 = loc("add_295")
#loc38 = loc("multiply_306")
#loc39 = loc("add_312")
#loc40 = loc("multiply_322")
#loc41 = loc("add_328")
#loc42 = loc("multiply_339")
#loc43 = loc("add_345")
#loc44 = loc("multiply_355")
#loc45 = loc("add_361")
#loc46 = loc("multiply_372")
#loc47 = loc("add_378")
#loc48 = loc("multiply_388")
#loc49 = loc("add_394")
#loc50 = loc("multiply_406")
#loc51 = loc("add_412")
#loc52 = loc("multiply_422")
#loc53 = loc("add_428")
#loc54 = loc("multiply_438")
#loc55 = loc("add_444")
#loc56 = loc("multiply_456")
#loc57 = loc("add_462")
#loc58 = loc("multiply_472")
#loc59 = loc("add_478")
#loc60 = loc("multiply_489")
#loc61 = loc("add_495")
#loc62 = loc("multiply_505")
#loc63 = loc("add_511")
#loc64 = loc("multiply_522")
#loc65 = loc("add_528")
#loc66 = loc("multiply_538")
#loc67 = loc("add_544")
#loc68 = loc("multiply_555")
#loc69 = loc("add_561")
#loc70 = loc("multiply_571")
#loc71 = loc("add_577")
#loc72 = loc("multiply_588")
#loc73 = loc("add_594")
#loc74 = loc("multiply_604")
#loc75 = loc("add_610")
#loc76 = loc("multiply_621")
#loc77 = loc("add_627")
#loc78 = loc("multiply_637")
#loc79 = loc("add_643")
#loc80 = loc("multiply_655")
#loc81 = loc("add_661")
#loc82 = loc("multiply_671")
#loc83 = loc("add_677")
#loc84 = loc("multiply_687")
#loc85 = loc("add_693")
#loc86 = loc("multiply_705")
#loc87 = loc("add_711")
#loc88 = loc("multiply_721")
#loc89 = loc("add_727")
#loc90 = loc("multiply_738")
#loc91 = loc("add_744")
#loc92 = loc("multiply_754")
#loc93 = loc("add_760")
#loc94 = loc("multiply_771")
#loc95 = loc("add_777")
#loc96 = loc("multiply_787")
#loc97 = loc("add_793")
#loc98 = loc("multiply_805")
#loc99 = loc("add_811")
#loc100 = loc("multiply_821")
#loc101 = loc("add_827")
#loc102 = loc("multiply_841")
#loc103 = loc("add_847")
#loc104 = loc("multiply_859")
#loc105 = loc("add_865")
#loc106 = loc("multiply_877")
#loc107 = loc("add_883")
#loc108 = loc("multiply_893")
#loc109 = loc("add_899")
#loc110 = loc("multiply_909")
#loc111 = loc("add_915")
#loc112 = loc("multiply_925")
#loc113 = loc("add_931")
#loc114 = loc("multiply_941")
#loc115 = loc("add_947")
#loc116 = loc("multiply_957")
#loc117 = loc("add_963")
#loc118 = loc("multiply_974")
#loc119 = loc("add_980")
#loc120 = loc("multiply_992")
#loc121 = loc("add_998")
#loc122 = loc("multiply_1010")
#loc123 = loc("add_1016")
#loc124 = loc("multiply_1026")
#loc125 = loc("add_1032")
#loc126 = loc("multiply_1042")
#loc127 = loc("add_1048")
#loc128 = loc("multiply_1058")
#loc129 = loc("add_1064")
#loc130 = loc("multiply_1074")
#loc131 = loc("add_1080")
#loc132 = loc("multiply_1090")
#loc133 = loc("add_1096")
#loc134 = loc("multiply_1107")
#loc135 = loc("add_1113")
#loc136 = loc("multiply_1123")
#loc137 = loc("add_1129")
#loc138 = loc("multiply_1139")
#loc139 = loc("add_1145")
#loc140 = loc("multiply_1159")
#loc141 = loc("add_1165")
#loc142 = loc("multiply_1175")
#loc143 = loc("add_1181")
#loc144 = loc("multiply_1197")
#loc145 = loc("add_1203")
#loc146 = loc("multiply_1214")
#loc147 = loc("add_1220")
#loc148 = loc("multiply_1232")
#loc149 = loc("add_1238")
#loc150 = loc("multiply_1248")
#loc151 = loc("add_1254")
#loc152 = loc("multiply_1264")
#loc153 = loc("add_1270")
#loc154 = loc("multiply_1280")
#loc155 = loc("add_1286")
#loc156 = loc("multiply_1296")
#loc157 = loc("add_1302")
#loc158 = loc("multiply_1312")
#loc159 = loc("add_1318")
#loc160 = loc("multiply_1329")
#loc161 = loc("add_1335")
#loc162 = loc("multiply_1345")
#loc163 = loc("add_1351")
#loc164 = loc("multiply_1361")
#loc165 = loc("add_1367")
#loc166 = loc("multiply_1381")
#loc167 = loc("add_1387")
#loc168 = loc("multiply_1397")
#loc169 = loc("add_1403")
#loc170 = loc("multiply_1419")
#loc171 = loc("add_1425")
#loc172 = loc("multiply_1436")
#loc173 = loc("add_1442")
#loc174 = loc("multiply_1454")
#loc175 = loc("add_1460")
#loc176 = loc("multiply_1470")
#loc177 = loc("add_1476")
#loc178 = loc("multiply_1486")
#loc179 = loc("add_1492")
#loc180 = loc("multiply_1502")
#loc181 = loc("add_1508")
#loc182 = loc("multiply_1518")
#loc183 = loc("add_1524")
#loc184 = loc("multiply_1534")
#loc185 = loc("add_1540")
#loc186 = loc("multiply_1551")
#loc187 = loc("add_1557")
#loc188 = loc("multiply_1567")
#loc189 = loc("add_1573")
#loc190 = loc("multiply_1583")
#loc191 = loc("add_1589")
#loc192 = loc("multiply_1603")
#loc193 = loc("add_1609")
#loc194 = loc("multiply_1619")
#loc195 = loc("add_1625")
#loc196 = loc(unknown)
#loc197 = loc("ultralytics.nn.tasks.DetectionModel::model"(#loc1))
#loc198 = loc("ultralytics.nn.modules.conv.Conv::0"(#loc197))
#loc199 = loc("ultralytics.nn.modules.conv.Conv::1"(#loc197))
#loc200 = loc("ultralytics.nn.modules.block.C2f::2"(#loc197))
#loc201 = loc("ultralytics.nn.modules.conv.Conv::3"(#loc197))
#loc202 = loc("ultralytics.nn.modules.block.C2f::4"(#loc197))
#loc203 = loc("ultralytics.nn.modules.conv.Conv::5"(#loc197))
#loc204 = loc("ultralytics.nn.modules.block.C2f::6"(#loc197))
#loc205 = loc("ultralytics.nn.modules.conv.Conv::7"(#loc197))
#loc206 = loc("ultralytics.nn.modules.block.C2f::8"(#loc197))
#loc207 = loc("ultralytics.nn.modules.block.SPPF::9"(#loc197))
#loc208 = loc("torch.nn.modules.upsampling.Upsample::10"(#loc197))
#loc209 = loc("ultralytics.nn.modules.conv.Concat::11"(#loc197))
#loc210 = loc("ultralytics.nn.modules.block.C2f::12"(#loc197))
#loc211 = loc("torch.nn.modules.upsampling.Upsample::13"(#loc197))
#loc212 = loc("ultralytics.nn.modules.conv.Concat::14"(#loc197))
#loc213 = loc("ultralytics.nn.modules.block.C2f::15"(#loc197))
#loc214 = loc("ultralytics.nn.modules.head.Detect::22"(#loc197))
#loc215 = loc("ultralytics.nn.modules.conv.Conv::16"(#loc197))
#loc216 = loc("ultralytics.nn.modules.conv.Concat::17"(#loc197))
#loc217 = loc("ultralytics.nn.modules.block.C2f::18"(#loc197))
#loc218 = loc("ultralytics.nn.modules.conv.Conv::19"(#loc197))
#loc219 = loc("ultralytics.nn.modules.conv.Concat::20"(#loc197))
#loc220 = loc("ultralytics.nn.modules.block.C2f::21"(#loc197))
#loc221 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc198))
#loc222 = loc("torch.nn.modules.activation.SiLU::"(#loc198))
#loc223 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc199))
#loc224 = loc("sigmoid_30"(#loc199))
#loc225 = loc("multiply_31"(#loc199))
#loc226 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc200))
#loc227 = loc("index_48"(#loc200))
#loc228 = loc("index_49"(#loc200))
#loc229 = loc("ultralytics.nn.modules.block.Bottleneck::m.0"(#loc200))
#loc230 = loc("ultralytics.nn.modules.block.Bottleneck::m.1"(#loc200))
#loc231 = loc("ultralytics.nn.modules.block.Bottleneck::m.2"(#loc200))
#loc232 = loc("concatenate_149"(#loc200))
#loc233 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc200))
#loc234 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc201))
#loc235 = loc("sigmoid_180"(#loc201))
#loc236 = loc("multiply_181"(#loc201))
#loc237 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc202))
#loc238 = loc("index_198"(#loc202))
#loc239 = loc("index_199"(#loc202))
#loc240 = loc("ultralytics.nn.modules.block.Bottleneck::0"(#loc202))
#loc241 = loc("ultralytics.nn.modules.block.Bottleneck::1"(#loc202))
#loc242 = loc("ultralytics.nn.modules.block.Bottleneck::2"(#loc202))
#loc243 = loc("ultralytics.nn.modules.block.Bottleneck::3"(#loc202))
#loc244 = loc("ultralytics.nn.modules.block.Bottleneck::4"(#loc202))
#loc245 = loc("ultralytics.nn.modules.block.Bottleneck::5"(#loc202))
#loc246 = loc("concatenate_398"(#loc202))
#loc247 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc202))
#loc248 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc203))
#loc249 = loc("sigmoid_429"(#loc203))
#loc250 = loc("multiply_430"(#loc203))
#loc251 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc204))
#loc252 = loc("index_447"(#loc204))
#loc253 = loc("index_448"(#loc204))
#loc254 = loc("ultralytics.nn.modules.block.Bottleneck::0"(#loc204))
#loc255 = loc("ultralytics.nn.modules.block.Bottleneck::1"(#loc204))
#loc256 = loc("ultralytics.nn.modules.block.Bottleneck::2"(#loc204))
#loc257 = loc("ultralytics.nn.modules.block.Bottleneck::3"(#loc204))
#loc258 = loc("ultralytics.nn.modules.block.Bottleneck::4"(#loc204))
#loc259 = loc("ultralytics.nn.modules.block.Bottleneck::5"(#loc204))
#loc260 = loc("concatenate_647"(#loc204))
#loc261 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc204))
#loc262 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc205))
#loc263 = loc("sigmoid_678"(#loc205))
#loc264 = loc("multiply_679"(#loc205))
#loc265 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc206))
#loc266 = loc("index_696"(#loc206))
#loc267 = loc("index_697"(#loc206))
#loc268 = loc("ultralytics.nn.modules.block.Bottleneck::0"(#loc206))
#loc269 = loc("ultralytics.nn.modules.block.Bottleneck::1"(#loc206))
#loc270 = loc("ultralytics.nn.modules.block.Bottleneck::2"(#loc206))
#loc271 = loc("concatenate_797"(#loc206))
#loc272 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc206))
#loc273 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc207))
#loc274 = loc("torch.nn.modules.pooling.MaxPool2d::m"(#loc207))
#loc275 = loc("max_pool2d_831.dc.transpose.0"(#loc207))
#loc276 = loc("max_pool2d_831.dc.transpose.1"(#loc207))
#loc277 = loc("max_pool2d_831.dc.max_pool2d.2"(#loc207))
#loc278 = loc("max_pool2d_831.dc.transpose.3"(#loc207))
#loc279 = loc("max_pool2d_831.dc.transpose.4"(#loc207))
#loc280 = loc("max_pool2d_832.dc.transpose.0"(#loc207))
#loc281 = loc("max_pool2d_832.dc.transpose.1"(#loc207))
#loc282 = loc("max_pool2d_832.dc.max_pool2d.2"(#loc207))
#loc283 = loc("max_pool2d_832.dc.transpose.3"(#loc207))
#loc284 = loc("max_pool2d_832.dc.transpose.4"(#loc207))
#loc285 = loc("concatenate_833"(#loc207))
#loc286 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc207))
#loc287 = loc("resize2d_850.dc.transpose.0"(#loc208))
#loc288 = loc("resize2d_850.dc.transpose.1"(#loc208))
#loc289 = loc("resize2d_850.dc.upsample2d.2"(#loc208))
#loc290 = loc("resize2d_850.dc.transpose.3"(#loc208))
#loc291 = loc("resize2d_850.dc.transpose.4"(#loc208))
#loc292 = loc("concatenate_851"(#loc209))
#loc293 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc210))
#loc294 = loc("index_868"(#loc210))
#loc295 = loc("index_869"(#loc210))
#loc296 = loc("ultralytics.nn.modules.block.Bottleneck::0"(#loc210))
#loc297 = loc("ultralytics.nn.modules.block.Bottleneck::1"(#loc210))
#loc298 = loc("ultralytics.nn.modules.block.Bottleneck::2"(#loc210))
#loc299 = loc("concatenate_966"(#loc210))
#loc300 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc210))
#loc301 = loc("resize2d_983.dc.transpose.0"(#loc211))
#loc302 = loc("resize2d_983.dc.transpose.1"(#loc211))
#loc303 = loc("resize2d_983.dc.upsample2d.2"(#loc211))
#loc304 = loc("resize2d_983.dc.transpose.3"(#loc211))
#loc305 = loc("resize2d_983.dc.transpose.4"(#loc211))
#loc306 = loc("concatenate_984"(#loc212))
#loc307 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc213))
#loc308 = loc("index_1001"(#loc213))
#loc309 = loc("index_1002"(#loc213))
#loc310 = loc("ultralytics.nn.modules.block.Bottleneck::0"(#loc213))
#loc311 = loc("ultralytics.nn.modules.block.Bottleneck::1"(#loc213))
#loc312 = loc("ultralytics.nn.modules.block.Bottleneck::2"(#loc213))
#loc313 = loc("concatenate_1099"(#loc213))
#loc314 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc213))
#loc315 = loc("torch.nn.modules.container.Sequential::0"(#loc214))
#loc316 = loc("concatenate_1188"(#loc214))
#loc317 = loc("reshape_1189"(#loc214))
#loc318 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc215))
#loc319 = loc("sigmoid_1204"(#loc215))
#loc320 = loc("multiply_1205"(#loc215))
#loc321 = loc("concatenate_1206"(#loc216))
#loc322 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc217))
#loc323 = loc("index_1223"(#loc217))
#loc324 = loc("index_1224"(#loc217))
#loc325 = loc("ultralytics.nn.modules.block.Bottleneck::0"(#loc217))
#loc326 = loc("ultralytics.nn.modules.block.Bottleneck::1"(#loc217))
#loc327 = loc("ultralytics.nn.modules.block.Bottleneck::2"(#loc217))
#loc328 = loc("concatenate_1321"(#loc217))
#loc329 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc217))
#loc330 = loc("torch.nn.modules.container.Sequential::1"(#loc214))
#loc331 = loc("concatenate_1410"(#loc214))
#loc332 = loc("reshape_1411"(#loc214))
#loc333 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc218))
#loc334 = loc("sigmoid_1426"(#loc218))
#loc335 = loc("multiply_1427"(#loc218))
#loc336 = loc("concatenate_1428"(#loc219))
#loc337 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc220))
#loc338 = loc("index_1445"(#loc220))
#loc339 = loc("index_1446"(#loc220))
#loc340 = loc("ultralytics.nn.modules.block.Bottleneck::0"(#loc220))
#loc341 = loc("ultralytics.nn.modules.block.Bottleneck::1"(#loc220))
#loc342 = loc("ultralytics.nn.modules.block.Bottleneck::2"(#loc220))
#loc343 = loc("concatenate_1543"(#loc220))
#loc344 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc220))
#loc345 = loc("torch.nn.modules.container.Sequential::2"(#loc214))
#loc346 = loc("concatenate_1632"(#loc214))
#loc347 = loc("reshape_1633"(#loc214))
#loc348 = loc("concatenate_1634"(#loc214))
#loc349 = loc("index_1635"(#loc214))
#loc350 = loc("ultralytics.nn.modules.block.DFL::dfl"(#loc214))
#loc351 = loc("index_1641"(#loc214))
#loc352 = loc("subtract_1642"(#loc214))
#loc353 = loc("index_1643"(#loc214))
#loc354 = loc("add_1644"(#loc214))
#loc355 = loc("add_1645"(#loc214))
#loc356 = loc("divide_1646"(#loc214))
#loc357 = loc("subtract_1647"(#loc214))
#loc358 = loc("concatenate_1648"(#loc214))
#loc359 = loc("multiply_1649"(#loc214))
#loc360 = loc("index_1650"(#loc214))
#loc361 = loc("sigmoid_1651"(#loc214))
#loc362 = loc("concatenate_1652"(#loc214))
#loc363 = loc("conv2d_0.dc.transpose.0"(#loc221))
#loc364 = loc("conv2d_0.dc.transpose.1"(#loc221))
#loc365 = loc("conv2d_0.dc.conv2d.2"(#loc221))
#loc366 = loc("conv2d_0.dc.transpose.3"(#loc221))
#loc367 = loc("conv2d_0.dc.transpose.4"(#loc221))
#loc368 = loc("sigmoid_14"(#loc222))
#loc369 = loc("multiply_15"(#loc222))
#loc370 = loc("conv2d_16.dc.transpose.0"(#loc223))
#loc371 = loc("conv2d_16.dc.transpose.1"(#loc223))
#loc372 = loc("conv2d_16.dc.conv2d.2"(#loc223))
#loc373 = loc("conv2d_16.dc.transpose.3"(#loc223))
#loc374 = loc("conv2d_16.dc.transpose.4"(#loc223))
#loc375 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc226))
#loc376 = loc("sigmoid_46"(#loc226))
#loc377 = loc("multiply_47"(#loc226))
#loc378 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc229))
#loc379 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc229))
#loc380 = loc("add_82"(#loc229))
#loc381 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc230))
#loc382 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc230))
#loc383 = loc("add_115"(#loc230))
#loc384 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc231))
#loc385 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc231))
#loc386 = loc("add_148"(#loc231))
#loc387 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc233))
#loc388 = loc("sigmoid_164"(#loc233))
#loc389 = loc("multiply_165"(#loc233))
#loc390 = loc("conv2d_166.dc.transpose.0"(#loc234))
#loc391 = loc("conv2d_166.dc.transpose.1"(#loc234))
#loc392 = loc("conv2d_166.dc.conv2d.2"(#loc234))
#loc393 = loc("conv2d_166.dc.transpose.3"(#loc234))
#loc394 = loc("conv2d_166.dc.transpose.4"(#loc234))
#loc395 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc237))
#loc396 = loc("sigmoid_196"(#loc237))
#loc397 = loc("multiply_197"(#loc237))
#loc398 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc240))
#loc399 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc240))
#loc400 = loc("add_232"(#loc240))
#loc401 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc241))
#loc402 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc241))
#loc403 = loc("add_265"(#loc241))
#loc404 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc242))
#loc405 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc242))
#loc406 = loc("add_298"(#loc242))
#loc407 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc243))
#loc408 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc243))
#loc409 = loc("add_331"(#loc243))
#loc410 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc244))
#loc411 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc244))
#loc412 = loc("add_364"(#loc244))
#loc413 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc245))
#loc414 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc245))
#loc415 = loc("add_397"(#loc245))
#loc416 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc247))
#loc417 = loc("sigmoid_413"(#loc247))
#loc418 = loc("multiply_414"(#loc247))
#loc419 = loc("conv2d_415.dc.transpose.0"(#loc248))
#loc420 = loc("conv2d_415.dc.transpose.1"(#loc248))
#loc421 = loc("conv2d_415.dc.conv2d.2"(#loc248))
#loc422 = loc("conv2d_415.dc.transpose.3"(#loc248))
#loc423 = loc("conv2d_415.dc.transpose.4"(#loc248))
#loc424 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc251))
#loc425 = loc("sigmoid_445"(#loc251))
#loc426 = loc("multiply_446"(#loc251))
#loc427 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc254))
#loc428 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc254))
#loc429 = loc("add_481"(#loc254))
#loc430 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc255))
#loc431 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc255))
#loc432 = loc("add_514"(#loc255))
#loc433 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc256))
#loc434 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc256))
#loc435 = loc("add_547"(#loc256))
#loc436 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc257))
#loc437 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc257))
#loc438 = loc("add_580"(#loc257))
#loc439 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc258))
#loc440 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc258))
#loc441 = loc("add_613"(#loc258))
#loc442 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc259))
#loc443 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc259))
#loc444 = loc("add_646"(#loc259))
#loc445 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc261))
#loc446 = loc("sigmoid_662"(#loc261))
#loc447 = loc("multiply_663"(#loc261))
#loc448 = loc("conv2d_664.dc.transpose.0"(#loc262))
#loc449 = loc("conv2d_664.dc.transpose.1"(#loc262))
#loc450 = loc("conv2d_664.dc.conv2d.2"(#loc262))
#loc451 = loc("conv2d_664.dc.transpose.3"(#loc262))
#loc452 = loc("conv2d_664.dc.transpose.4"(#loc262))
#loc453 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc265))
#loc454 = loc("sigmoid_694"(#loc265))
#loc455 = loc("multiply_695"(#loc265))
#loc456 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc268))
#loc457 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc268))
#loc458 = loc("add_730"(#loc268))
#loc459 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc269))
#loc460 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc269))
#loc461 = loc("add_763"(#loc269))
#loc462 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc270))
#loc463 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc270))
#loc464 = loc("add_796"(#loc270))
#loc465 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc272))
#loc466 = loc("sigmoid_812"(#loc272))
#loc467 = loc("multiply_813"(#loc272))
#loc468 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc273))
#loc469 = loc("sigmoid_828"(#loc273))
#loc470 = loc("multiply_829"(#loc273))
#loc471 = loc("max_pool2d_830.dc.transpose.0"(#loc274))
#loc472 = loc("max_pool2d_830.dc.transpose.1"(#loc274))
#loc473 = loc("max_pool2d_830.dc.max_pool2d.2"(#loc274))
#loc474 = loc("max_pool2d_830.dc.transpose.3"(#loc274))
#loc475 = loc("max_pool2d_830.dc.transpose.4"(#loc274))
#loc476 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc286))
#loc477 = loc("sigmoid_848"(#loc286))
#loc478 = loc("multiply_849"(#loc286))
#loc479 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc293))
#loc480 = loc("sigmoid_866"(#loc293))
#loc481 = loc("multiply_867"(#loc293))
#loc482 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc296))
#loc483 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc296))
#loc484 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc297))
#loc485 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc297))
#loc486 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc298))
#loc487 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc298))
#loc488 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc300))
#loc489 = loc("sigmoid_981"(#loc300))
#loc490 = loc("multiply_982"(#loc300))
#loc491 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc307))
#loc492 = loc("sigmoid_999"(#loc307))
#loc493 = loc("multiply_1000"(#loc307))
#loc494 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc310))
#loc495 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc310))
#loc496 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc311))
#loc497 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc311))
#loc498 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc312))
#loc499 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc312))
#loc500 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc314))
#loc501 = loc("sigmoid_1114"(#loc314))
#loc502 = loc("multiply_1115"(#loc314))
#loc503 = loc("ultralytics.nn.modules.conv.Conv::0"(#loc315))
#loc504 = loc("ultralytics.nn.modules.conv.Conv::1"(#loc315))
#loc505 = loc("torch.nn.modules.conv.Conv2d::2"(#loc315))
#loc506 = loc("conv2d_1190.dc.transpose.0"(#loc318))
#loc507 = loc("conv2d_1190.dc.transpose.1"(#loc318))
#loc508 = loc("conv2d_1190.dc.conv2d.2"(#loc318))
#loc509 = loc("conv2d_1190.dc.transpose.3"(#loc318))
#loc510 = loc("conv2d_1190.dc.transpose.4"(#loc318))
#loc511 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc322))
#loc512 = loc("sigmoid_1221"(#loc322))
#loc513 = loc("multiply_1222"(#loc322))
#loc514 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc325))
#loc515 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc325))
#loc516 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc326))
#loc517 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc326))
#loc518 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc327))
#loc519 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc327))
#loc520 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc329))
#loc521 = loc("sigmoid_1336"(#loc329))
#loc522 = loc("multiply_1337"(#loc329))
#loc523 = loc("ultralytics.nn.modules.conv.Conv::0"(#loc330))
#loc524 = loc("ultralytics.nn.modules.conv.Conv::1"(#loc330))
#loc525 = loc("torch.nn.modules.conv.Conv2d::2"(#loc330))
#loc526 = loc("conv2d_1412.dc.transpose.0"(#loc333))
#loc527 = loc("conv2d_1412.dc.transpose.1"(#loc333))
#loc528 = loc("conv2d_1412.dc.conv2d.2"(#loc333))
#loc529 = loc("conv2d_1412.dc.transpose.3"(#loc333))
#loc530 = loc("conv2d_1412.dc.transpose.4"(#loc333))
#loc531 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc337))
#loc532 = loc("sigmoid_1443"(#loc337))
#loc533 = loc("multiply_1444"(#loc337))
#loc534 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc340))
#loc535 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc340))
#loc536 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc341))
#loc537 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc341))
#loc538 = loc("ultralytics.nn.modules.conv.Conv::cv1"(#loc342))
#loc539 = loc("ultralytics.nn.modules.conv.Conv::cv2"(#loc342))
#loc540 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc344))
#loc541 = loc("sigmoid_1558"(#loc344))
#loc542 = loc("multiply_1559"(#loc344))
#loc543 = loc("ultralytics.nn.modules.conv.Conv::0"(#loc345))
#loc544 = loc("ultralytics.nn.modules.conv.Conv::1"(#loc345))
#loc545 = loc("torch.nn.modules.conv.Conv2d::2"(#loc345))
#loc546 = loc("reshape_1636"(#loc350))
#loc547 = loc("transpose_1637"(#loc350))
#loc548 = loc("softmax_1638"(#loc350))
#loc549 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc350))
#loc550 = loc("reshape_1640.dc.squeeze.0"(#loc350))
#loc551 = loc("conv2d_32.dc.transpose.0"(#loc375))
#loc552 = loc("conv2d_32.dc.transpose.1"(#loc375))
#loc553 = loc("conv2d_32.dc.conv2d.2"(#loc375))
#loc554 = loc("conv2d_32.dc.transpose.3"(#loc375))
#loc555 = loc("conv2d_32.dc.transpose.4"(#loc375))
#loc556 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc378))
#loc557 = loc("sigmoid_64"(#loc378))
#loc558 = loc("multiply_65"(#loc378))
#loc559 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc379))
#loc560 = loc("sigmoid_80"(#loc379))
#loc561 = loc("multiply_81"(#loc379))
#loc562 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc381))
#loc563 = loc("sigmoid_97"(#loc381))
#loc564 = loc("multiply_98"(#loc381))
#loc565 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc382))
#loc566 = loc("sigmoid_113"(#loc382))
#loc567 = loc("multiply_114"(#loc382))
#loc568 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc384))
#loc569 = loc("sigmoid_130"(#loc384))
#loc570 = loc("multiply_131"(#loc384))
#loc571 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc385))
#loc572 = loc("sigmoid_146"(#loc385))
#loc573 = loc("multiply_147"(#loc385))
#loc574 = loc("conv2d_150.dc.transpose.0"(#loc387))
#loc575 = loc("conv2d_150.dc.transpose.1"(#loc387))
#loc576 = loc("conv2d_150.dc.conv2d.2"(#loc387))
#loc577 = loc("conv2d_150.dc.transpose.3"(#loc387))
#loc578 = loc("conv2d_150.dc.transpose.4"(#loc387))
#loc579 = loc("conv2d_182.dc.transpose.0"(#loc395))
#loc580 = loc("conv2d_182.dc.transpose.1"(#loc395))
#loc581 = loc("conv2d_182.dc.conv2d.2"(#loc395))
#loc582 = loc("conv2d_182.dc.transpose.3"(#loc395))
#loc583 = loc("conv2d_182.dc.transpose.4"(#loc395))
#loc584 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc398))
#loc585 = loc("sigmoid_214"(#loc398))
#loc586 = loc("multiply_215"(#loc398))
#loc587 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc399))
#loc588 = loc("sigmoid_230"(#loc399))
#loc589 = loc("multiply_231"(#loc399))
#loc590 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc401))
#loc591 = loc("sigmoid_247"(#loc401))
#loc592 = loc("multiply_248"(#loc401))
#loc593 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc402))
#loc594 = loc("sigmoid_263"(#loc402))
#loc595 = loc("multiply_264"(#loc402))
#loc596 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc404))
#loc597 = loc("sigmoid_280"(#loc404))
#loc598 = loc("multiply_281"(#loc404))
#loc599 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc405))
#loc600 = loc("sigmoid_296"(#loc405))
#loc601 = loc("multiply_297"(#loc405))
#loc602 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc407))
#loc603 = loc("sigmoid_313"(#loc407))
#loc604 = loc("multiply_314"(#loc407))
#loc605 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc408))
#loc606 = loc("sigmoid_329"(#loc408))
#loc607 = loc("multiply_330"(#loc408))
#loc608 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc410))
#loc609 = loc("sigmoid_346"(#loc410))
#loc610 = loc("multiply_347"(#loc410))
#loc611 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc411))
#loc612 = loc("sigmoid_362"(#loc411))
#loc613 = loc("multiply_363"(#loc411))
#loc614 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc413))
#loc615 = loc("sigmoid_379"(#loc413))
#loc616 = loc("multiply_380"(#loc413))
#loc617 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc414))
#loc618 = loc("sigmoid_395"(#loc414))
#loc619 = loc("multiply_396"(#loc414))
#loc620 = loc("conv2d_399.dc.transpose.0"(#loc416))
#loc621 = loc("conv2d_399.dc.transpose.1"(#loc416))
#loc622 = loc("conv2d_399.dc.conv2d.2"(#loc416))
#loc623 = loc("conv2d_399.dc.transpose.3"(#loc416))
#loc624 = loc("conv2d_399.dc.transpose.4"(#loc416))
#loc625 = loc("conv2d_431.dc.transpose.0"(#loc424))
#loc626 = loc("conv2d_431.dc.transpose.1"(#loc424))
#loc627 = loc("conv2d_431.dc.conv2d.2"(#loc424))
#loc628 = loc("conv2d_431.dc.transpose.3"(#loc424))
#loc629 = loc("conv2d_431.dc.transpose.4"(#loc424))
#loc630 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc427))
#loc631 = loc("sigmoid_463"(#loc427))
#loc632 = loc("multiply_464"(#loc427))
#loc633 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc428))
#loc634 = loc("sigmoid_479"(#loc428))
#loc635 = loc("multiply_480"(#loc428))
#loc636 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc430))
#loc637 = loc("sigmoid_496"(#loc430))
#loc638 = loc("multiply_497"(#loc430))
#loc639 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc431))
#loc640 = loc("sigmoid_512"(#loc431))
#loc641 = loc("multiply_513"(#loc431))
#loc642 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc433))
#loc643 = loc("sigmoid_529"(#loc433))
#loc644 = loc("multiply_530"(#loc433))
#loc645 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc434))
#loc646 = loc("sigmoid_545"(#loc434))
#loc647 = loc("multiply_546"(#loc434))
#loc648 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc436))
#loc649 = loc("sigmoid_562"(#loc436))
#loc650 = loc("multiply_563"(#loc436))
#loc651 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc437))
#loc652 = loc("sigmoid_578"(#loc437))
#loc653 = loc("multiply_579"(#loc437))
#loc654 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc439))
#loc655 = loc("sigmoid_595"(#loc439))
#loc656 = loc("multiply_596"(#loc439))
#loc657 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc440))
#loc658 = loc("sigmoid_611"(#loc440))
#loc659 = loc("multiply_612"(#loc440))
#loc660 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc442))
#loc661 = loc("sigmoid_628"(#loc442))
#loc662 = loc("multiply_629"(#loc442))
#loc663 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc443))
#loc664 = loc("sigmoid_644"(#loc443))
#loc665 = loc("multiply_645"(#loc443))
#loc666 = loc("conv2d_648.dc.transpose.0"(#loc445))
#loc667 = loc("conv2d_648.dc.transpose.1"(#loc445))
#loc668 = loc("conv2d_648.dc.conv2d.2"(#loc445))
#loc669 = loc("conv2d_648.dc.transpose.3"(#loc445))
#loc670 = loc("conv2d_648.dc.transpose.4"(#loc445))
#loc671 = loc("conv2d_680.dc.transpose.0"(#loc453))
#loc672 = loc("conv2d_680.dc.transpose.1"(#loc453))
#loc673 = loc("conv2d_680.dc.conv2d.2"(#loc453))
#loc674 = loc("conv2d_680.dc.transpose.3"(#loc453))
#loc675 = loc("conv2d_680.dc.transpose.4"(#loc453))
#loc676 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc456))
#loc677 = loc("sigmoid_712"(#loc456))
#loc678 = loc("multiply_713"(#loc456))
#loc679 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc457))
#loc680 = loc("sigmoid_728"(#loc457))
#loc681 = loc("multiply_729"(#loc457))
#loc682 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc459))
#loc683 = loc("sigmoid_745"(#loc459))
#loc684 = loc("multiply_746"(#loc459))
#loc685 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc460))
#loc686 = loc("sigmoid_761"(#loc460))
#loc687 = loc("multiply_762"(#loc460))
#loc688 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc462))
#loc689 = loc("sigmoid_778"(#loc462))
#loc690 = loc("multiply_779"(#loc462))
#loc691 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc463))
#loc692 = loc("sigmoid_794"(#loc463))
#loc693 = loc("multiply_795"(#loc463))
#loc694 = loc("conv2d_798.dc.transpose.0"(#loc465))
#loc695 = loc("conv2d_798.dc.transpose.1"(#loc465))
#loc696 = loc("conv2d_798.dc.conv2d.2"(#loc465))
#loc697 = loc("conv2d_798.dc.transpose.3"(#loc465))
#loc698 = loc("conv2d_798.dc.transpose.4"(#loc465))
#loc699 = loc("conv2d_814.dc.transpose.0"(#loc468))
#loc700 = loc("conv2d_814.dc.transpose.1"(#loc468))
#loc701 = loc("conv2d_814.dc.conv2d.2"(#loc468))
#loc702 = loc("conv2d_814.dc.transpose.3"(#loc468))
#loc703 = loc("conv2d_814.dc.transpose.4"(#loc468))
#loc704 = loc("conv2d_834.dc.transpose.0"(#loc476))
#loc705 = loc("conv2d_834.dc.transpose.1"(#loc476))
#loc706 = loc("conv2d_834.dc.conv2d.2"(#loc476))
#loc707 = loc("conv2d_834.dc.transpose.3"(#loc476))
#loc708 = loc("conv2d_834.dc.transpose.4"(#loc476))
#loc709 = loc("conv2d_852.dc.transpose.0"(#loc479))
#loc710 = loc("conv2d_852.dc.transpose.1"(#loc479))
#loc711 = loc("conv2d_852.dc.conv2d.2"(#loc479))
#loc712 = loc("conv2d_852.dc.transpose.3"(#loc479))
#loc713 = loc("conv2d_852.dc.transpose.4"(#loc479))
#loc714 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc482))
#loc715 = loc("sigmoid_884"(#loc482))
#loc716 = loc("multiply_885"(#loc482))
#loc717 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc483))
#loc718 = loc("sigmoid_900"(#loc483))
#loc719 = loc("multiply_901"(#loc483))
#loc720 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc484))
#loc721 = loc("sigmoid_916"(#loc484))
#loc722 = loc("multiply_917"(#loc484))
#loc723 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc485))
#loc724 = loc("sigmoid_932"(#loc485))
#loc725 = loc("multiply_933"(#loc485))
#loc726 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc486))
#loc727 = loc("sigmoid_948"(#loc486))
#loc728 = loc("multiply_949"(#loc486))
#loc729 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc487))
#loc730 = loc("sigmoid_964"(#loc487))
#loc731 = loc("multiply_965"(#loc487))
#loc732 = loc("conv2d_967.dc.transpose.0"(#loc488))
#loc733 = loc("conv2d_967.dc.transpose.1"(#loc488))
#loc734 = loc("conv2d_967.dc.conv2d.2"(#loc488))
#loc735 = loc("conv2d_967.dc.transpose.3"(#loc488))
#loc736 = loc("conv2d_967.dc.transpose.4"(#loc488))
#loc737 = loc("conv2d_985.dc.transpose.0"(#loc491))
#loc738 = loc("conv2d_985.dc.transpose.1"(#loc491))
#loc739 = loc("conv2d_985.dc.conv2d.2"(#loc491))
#loc740 = loc("conv2d_985.dc.transpose.3"(#loc491))
#loc741 = loc("conv2d_985.dc.transpose.4"(#loc491))
#loc742 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc494))
#loc743 = loc("sigmoid_1017"(#loc494))
#loc744 = loc("multiply_1018"(#loc494))
#loc745 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc495))
#loc746 = loc("sigmoid_1033"(#loc495))
#loc747 = loc("multiply_1034"(#loc495))
#loc748 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc496))
#loc749 = loc("sigmoid_1049"(#loc496))
#loc750 = loc("multiply_1050"(#loc496))
#loc751 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc497))
#loc752 = loc("sigmoid_1065"(#loc497))
#loc753 = loc("multiply_1066"(#loc497))
#loc754 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc498))
#loc755 = loc("sigmoid_1081"(#loc498))
#loc756 = loc("multiply_1082"(#loc498))
#loc757 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc499))
#loc758 = loc("sigmoid_1097"(#loc499))
#loc759 = loc("multiply_1098"(#loc499))
#loc760 = loc("conv2d_1100.dc.transpose.0"(#loc500))
#loc761 = loc("conv2d_1100.dc.transpose.1"(#loc500))
#loc762 = loc("conv2d_1100.dc.conv2d.2"(#loc500))
#loc763 = loc("conv2d_1100.dc.transpose.3"(#loc500))
#loc764 = loc("conv2d_1100.dc.transpose.4"(#loc500))
#loc765 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc503))
#loc766 = loc("sigmoid_1130"(#loc503))
#loc767 = loc("multiply_1131"(#loc503))
#loc768 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc504))
#loc769 = loc("sigmoid_1146"(#loc504))
#loc770 = loc("multiply_1147"(#loc504))
#loc771 = loc("conv2d_1148.dc.transpose.0"(#loc505))
#loc772 = loc("conv2d_1148.dc.transpose.1"(#loc505))
#loc773 = loc("conv2d_1148.dc.conv2d.4"(#loc505))
#loc774 = loc("conv2d_1148.dc.transpose.5"(#loc505))
#loc775 = loc("conv2d_1148.dc.transpose.6"(#loc505))
#loc776 = loc("sigmoid_1166"(#loc503))
#loc777 = loc("multiply_1167"(#loc503))
#loc778 = loc("sigmoid_1182"(#loc504))
#loc779 = loc("multiply_1183"(#loc504))
#loc780 = loc("conv2d_1184.dc.transpose.0"(#loc505))
#loc781 = loc("conv2d_1184.dc.transpose.1"(#loc505))
#loc782 = loc("conv2d_1184.dc.conv2d.4"(#loc505))
#loc783 = loc("conv2d_1184.dc.transpose.5"(#loc505))
#loc784 = loc("conv2d_1184.dc.transpose.6"(#loc505))
#loc785 = loc("conv2d_1207.dc.transpose.0"(#loc511))
#loc786 = loc("conv2d_1207.dc.transpose.1"(#loc511))
#loc787 = loc("conv2d_1207.dc.conv2d.2"(#loc511))
#loc788 = loc("conv2d_1207.dc.transpose.3"(#loc511))
#loc789 = loc("conv2d_1207.dc.transpose.4"(#loc511))
#loc790 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc514))
#loc791 = loc("sigmoid_1239"(#loc514))
#loc792 = loc("multiply_1240"(#loc514))
#loc793 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc515))
#loc794 = loc("sigmoid_1255"(#loc515))
#loc795 = loc("multiply_1256"(#loc515))
#loc796 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc516))
#loc797 = loc("sigmoid_1271"(#loc516))
#loc798 = loc("multiply_1272"(#loc516))
#loc799 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc517))
#loc800 = loc("sigmoid_1287"(#loc517))
#loc801 = loc("multiply_1288"(#loc517))
#loc802 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc518))
#loc803 = loc("sigmoid_1303"(#loc518))
#loc804 = loc("multiply_1304"(#loc518))
#loc805 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc519))
#loc806 = loc("sigmoid_1319"(#loc519))
#loc807 = loc("multiply_1320"(#loc519))
#loc808 = loc("conv2d_1322.dc.transpose.0"(#loc520))
#loc809 = loc("conv2d_1322.dc.transpose.1"(#loc520))
#loc810 = loc("conv2d_1322.dc.conv2d.2"(#loc520))
#loc811 = loc("conv2d_1322.dc.transpose.3"(#loc520))
#loc812 = loc("conv2d_1322.dc.transpose.4"(#loc520))
#loc813 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc523))
#loc814 = loc("sigmoid_1352"(#loc523))
#loc815 = loc("multiply_1353"(#loc523))
#loc816 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc524))
#loc817 = loc("sigmoid_1368"(#loc524))
#loc818 = loc("multiply_1369"(#loc524))
#loc819 = loc("conv2d_1370.dc.transpose.0"(#loc525))
#loc820 = loc("conv2d_1370.dc.transpose.1"(#loc525))
#loc821 = loc("conv2d_1370.dc.conv2d.4"(#loc525))
#loc822 = loc("conv2d_1370.dc.transpose.5"(#loc525))
#loc823 = loc("conv2d_1370.dc.transpose.6"(#loc525))
#loc824 = loc("sigmoid_1388"(#loc523))
#loc825 = loc("multiply_1389"(#loc523))
#loc826 = loc("sigmoid_1404"(#loc524))
#loc827 = loc("multiply_1405"(#loc524))
#loc828 = loc("conv2d_1406.dc.transpose.0"(#loc525))
#loc829 = loc("conv2d_1406.dc.transpose.1"(#loc525))
#loc830 = loc("conv2d_1406.dc.conv2d.4"(#loc525))
#loc831 = loc("conv2d_1406.dc.transpose.5"(#loc525))
#loc832 = loc("conv2d_1406.dc.transpose.6"(#loc525))
#loc833 = loc("conv2d_1429.dc.transpose.0"(#loc531))
#loc834 = loc("conv2d_1429.dc.transpose.1"(#loc531))
#loc835 = loc("conv2d_1429.dc.conv2d.2"(#loc531))
#loc836 = loc("conv2d_1429.dc.transpose.3"(#loc531))
#loc837 = loc("conv2d_1429.dc.transpose.4"(#loc531))
#loc838 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc534))
#loc839 = loc("sigmoid_1461"(#loc534))
#loc840 = loc("multiply_1462"(#loc534))
#loc841 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc535))
#loc842 = loc("sigmoid_1477"(#loc535))
#loc843 = loc("multiply_1478"(#loc535))
#loc844 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc536))
#loc845 = loc("sigmoid_1493"(#loc536))
#loc846 = loc("multiply_1494"(#loc536))
#loc847 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc537))
#loc848 = loc("sigmoid_1509"(#loc537))
#loc849 = loc("multiply_1510"(#loc537))
#loc850 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc538))
#loc851 = loc("sigmoid_1525"(#loc538))
#loc852 = loc("multiply_1526"(#loc538))
#loc853 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc539))
#loc854 = loc("sigmoid_1541"(#loc539))
#loc855 = loc("multiply_1542"(#loc539))
#loc856 = loc("conv2d_1544.dc.transpose.0"(#loc540))
#loc857 = loc("conv2d_1544.dc.transpose.1"(#loc540))
#loc858 = loc("conv2d_1544.dc.conv2d.2"(#loc540))
#loc859 = loc("conv2d_1544.dc.transpose.3"(#loc540))
#loc860 = loc("conv2d_1544.dc.transpose.4"(#loc540))
#loc861 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc543))
#loc862 = loc("sigmoid_1574"(#loc543))
#loc863 = loc("multiply_1575"(#loc543))
#loc864 = loc("torch.nn.modules.conv.Conv2d::conv"(#loc544))
#loc865 = loc("sigmoid_1590"(#loc544))
#loc866 = loc("multiply_1591"(#loc544))
#loc867 = loc("conv2d_1592.dc.transpose.0"(#loc545))
#loc868 = loc("conv2d_1592.dc.transpose.1"(#loc545))
#loc869 = loc("conv2d_1592.dc.conv2d.4"(#loc545))
#loc870 = loc("conv2d_1592.dc.transpose.5"(#loc545))
#loc871 = loc("conv2d_1592.dc.transpose.6"(#loc545))
#loc872 = loc("sigmoid_1610"(#loc543))
#loc873 = loc("multiply_1611"(#loc543))
#loc874 = loc("sigmoid_1626"(#loc544))
#loc875 = loc("multiply_1627"(#loc544))
#loc876 = loc("conv2d_1628.dc.transpose.0"(#loc545))
#loc877 = loc("conv2d_1628.dc.transpose.1"(#loc545))
#loc878 = loc("conv2d_1628.dc.conv2d.4"(#loc545))
#loc879 = loc("conv2d_1628.dc.transpose.5"(#loc545))
#loc880 = loc("conv2d_1628.dc.transpose.6"(#loc545))
#loc881 = loc("conv2d_1639.dc.transpose.0"(#loc549))
#loc882 = loc("conv2d_1639.dc.transpose.1"(#loc549))
#loc883 = loc("conv2d_1639.dc.conv2d.2"(#loc549))
#loc884 = loc("conv2d_1639.dc.transpose.3"(#loc549))
#loc885 = loc("conv2d_1639.dc.transpose.4"(#loc549))
#loc886 = loc("conv2d_50.dc.transpose.0"(#loc556))
#loc887 = loc("conv2d_50.dc.transpose.1"(#loc556))
#loc888 = loc("conv2d_50.dc.conv2d.2"(#loc556))
#loc889 = loc("conv2d_50.dc.transpose.3"(#loc556))
#loc890 = loc("conv2d_50.dc.transpose.4"(#loc556))
#loc891 = loc("conv2d_66.dc.transpose.0"(#loc559))
#loc892 = loc("conv2d_66.dc.transpose.1"(#loc559))
#loc893 = loc("conv2d_66.dc.conv2d.2"(#loc559))
#loc894 = loc("conv2d_66.dc.transpose.3"(#loc559))
#loc895 = loc("conv2d_66.dc.transpose.4"(#loc559))
#loc896 = loc("conv2d_83.dc.transpose.0"(#loc562))
#loc897 = loc("conv2d_83.dc.transpose.1"(#loc562))
#loc898 = loc("conv2d_83.dc.conv2d.2"(#loc562))
#loc899 = loc("conv2d_83.dc.transpose.3"(#loc562))
#loc900 = loc("conv2d_83.dc.transpose.4"(#loc562))
#loc901 = loc("conv2d_99.dc.transpose.0"(#loc565))
#loc902 = loc("conv2d_99.dc.transpose.1"(#loc565))
#loc903 = loc("conv2d_99.dc.conv2d.2"(#loc565))
#loc904 = loc("conv2d_99.dc.transpose.3"(#loc565))
#loc905 = loc("conv2d_99.dc.transpose.4"(#loc565))
#loc906 = loc("conv2d_116.dc.transpose.0"(#loc568))
#loc907 = loc("conv2d_116.dc.transpose.1"(#loc568))
#loc908 = loc("conv2d_116.dc.conv2d.2"(#loc568))
#loc909 = loc("conv2d_116.dc.transpose.3"(#loc568))
#loc910 = loc("conv2d_116.dc.transpose.4"(#loc568))
#loc911 = loc("conv2d_132.dc.transpose.0"(#loc571))
#loc912 = loc("conv2d_132.dc.transpose.1"(#loc571))
#loc913 = loc("conv2d_132.dc.conv2d.2"(#loc571))
#loc914 = loc("conv2d_132.dc.transpose.3"(#loc571))
#loc915 = loc("conv2d_132.dc.transpose.4"(#loc571))
#loc916 = loc("conv2d_200.dc.transpose.0"(#loc584))
#loc917 = loc("conv2d_200.dc.transpose.1"(#loc584))
#loc918 = loc("conv2d_200.dc.conv2d.2"(#loc584))
#loc919 = loc("conv2d_200.dc.transpose.3"(#loc584))
#loc920 = loc("conv2d_200.dc.transpose.4"(#loc584))
#loc921 = loc("conv2d_216.dc.transpose.0"(#loc587))
#loc922 = loc("conv2d_216.dc.transpose.1"(#loc587))
#loc923 = loc("conv2d_216.dc.conv2d.2"(#loc587))
#loc924 = loc("conv2d_216.dc.transpose.3"(#loc587))
#loc925 = loc("conv2d_216.dc.transpose.4"(#loc587))
#loc926 = loc("conv2d_233.dc.transpose.0"(#loc590))
#loc927 = loc("conv2d_233.dc.transpose.1"(#loc590))
#loc928 = loc("conv2d_233.dc.conv2d.2"(#loc590))
#loc929 = loc("conv2d_233.dc.transpose.3"(#loc590))
#loc930 = loc("conv2d_233.dc.transpose.4"(#loc590))
#loc931 = loc("conv2d_249.dc.transpose.0"(#loc593))
#loc932 = loc("conv2d_249.dc.transpose.1"(#loc593))
#loc933 = loc("conv2d_249.dc.conv2d.2"(#loc593))
#loc934 = loc("conv2d_249.dc.transpose.3"(#loc593))
#loc935 = loc("conv2d_249.dc.transpose.4"(#loc593))
#loc936 = loc("conv2d_266.dc.transpose.0"(#loc596))
#loc937 = loc("conv2d_266.dc.transpose.1"(#loc596))
#loc938 = loc("conv2d_266.dc.conv2d.2"(#loc596))
#loc939 = loc("conv2d_266.dc.transpose.3"(#loc596))
#loc940 = loc("conv2d_266.dc.transpose.4"(#loc596))
#loc941 = loc("conv2d_282.dc.transpose.0"(#loc599))
#loc942 = loc("conv2d_282.dc.transpose.1"(#loc599))
#loc943 = loc("conv2d_282.dc.conv2d.2"(#loc599))
#loc944 = loc("conv2d_282.dc.transpose.3"(#loc599))
#loc945 = loc("conv2d_282.dc.transpose.4"(#loc599))
#loc946 = loc("conv2d_299.dc.transpose.0"(#loc602))
#loc947 = loc("conv2d_299.dc.transpose.1"(#loc602))
#loc948 = loc("conv2d_299.dc.conv2d.2"(#loc602))
#loc949 = loc("conv2d_299.dc.transpose.3"(#loc602))
#loc950 = loc("conv2d_299.dc.transpose.4"(#loc602))
#loc951 = loc("conv2d_315.dc.transpose.0"(#loc605))
#loc952 = loc("conv2d_315.dc.transpose.1"(#loc605))
#loc953 = loc("conv2d_315.dc.conv2d.2"(#loc605))
#loc954 = loc("conv2d_315.dc.transpose.3"(#loc605))
#loc955 = loc("conv2d_315.dc.transpose.4"(#loc605))
#loc956 = loc("conv2d_332.dc.transpose.0"(#loc608))
#loc957 = loc("conv2d_332.dc.transpose.1"(#loc608))
#loc958 = loc("conv2d_332.dc.conv2d.2"(#loc608))
#loc959 = loc("conv2d_332.dc.transpose.3"(#loc608))
#loc960 = loc("conv2d_332.dc.transpose.4"(#loc608))
#loc961 = loc("conv2d_348.dc.transpose.0"(#loc611))
#loc962 = loc("conv2d_348.dc.transpose.1"(#loc611))
#loc963 = loc("conv2d_348.dc.conv2d.2"(#loc611))
#loc964 = loc("conv2d_348.dc.transpose.3"(#loc611))
#loc965 = loc("conv2d_348.dc.transpose.4"(#loc611))
#loc966 = loc("conv2d_365.dc.transpose.0"(#loc614))
#loc967 = loc("conv2d_365.dc.transpose.1"(#loc614))
#loc968 = loc("conv2d_365.dc.conv2d.2"(#loc614))
#loc969 = loc("conv2d_365.dc.transpose.3"(#loc614))
#loc970 = loc("conv2d_365.dc.transpose.4"(#loc614))
#loc971 = loc("conv2d_381.dc.transpose.0"(#loc617))
#loc972 = loc("conv2d_381.dc.transpose.1"(#loc617))
#loc973 = loc("conv2d_381.dc.conv2d.2"(#loc617))
#loc974 = loc("conv2d_381.dc.transpose.3"(#loc617))
#loc975 = loc("conv2d_381.dc.transpose.4"(#loc617))
#loc976 = loc("conv2d_449.dc.transpose.0"(#loc630))
#loc977 = loc("conv2d_449.dc.transpose.1"(#loc630))
#loc978 = loc("conv2d_449.dc.conv2d.2"(#loc630))
#loc979 = loc("conv2d_449.dc.transpose.3"(#loc630))
#loc980 = loc("conv2d_449.dc.transpose.4"(#loc630))
#loc981 = loc("conv2d_465.dc.transpose.0"(#loc633))
#loc982 = loc("conv2d_465.dc.transpose.1"(#loc633))
#loc983 = loc("conv2d_465.dc.conv2d.2"(#loc633))
#loc984 = loc("conv2d_465.dc.transpose.3"(#loc633))
#loc985 = loc("conv2d_465.dc.transpose.4"(#loc633))
#loc986 = loc("conv2d_482.dc.transpose.0"(#loc636))
#loc987 = loc("conv2d_482.dc.transpose.1"(#loc636))
#loc988 = loc("conv2d_482.dc.conv2d.2"(#loc636))
#loc989 = loc("conv2d_482.dc.transpose.3"(#loc636))
#loc990 = loc("conv2d_482.dc.transpose.4"(#loc636))
#loc991 = loc("conv2d_498.dc.transpose.0"(#loc639))
#loc992 = loc("conv2d_498.dc.transpose.1"(#loc639))
#loc993 = loc("conv2d_498.dc.conv2d.2"(#loc639))
#loc994 = loc("conv2d_498.dc.transpose.3"(#loc639))
#loc995 = loc("conv2d_498.dc.transpose.4"(#loc639))
#loc996 = loc("conv2d_515.dc.transpose.0"(#loc642))
#loc997 = loc("conv2d_515.dc.transpose.1"(#loc642))
#loc998 = loc("conv2d_515.dc.conv2d.2"(#loc642))
#loc999 = loc("conv2d_515.dc.transpose.3"(#loc642))
#loc1000 = loc("conv2d_515.dc.transpose.4"(#loc642))
#loc1001 = loc("conv2d_531.dc.transpose.0"(#loc645))
#loc1002 = loc("conv2d_531.dc.transpose.1"(#loc645))
#loc1003 = loc("conv2d_531.dc.conv2d.2"(#loc645))
#loc1004 = loc("conv2d_531.dc.transpose.3"(#loc645))
#loc1005 = loc("conv2d_531.dc.transpose.4"(#loc645))
#loc1006 = loc("conv2d_548.dc.transpose.0"(#loc648))
#loc1007 = loc("conv2d_548.dc.transpose.1"(#loc648))
#loc1008 = loc("conv2d_548.dc.conv2d.2"(#loc648))
#loc1009 = loc("conv2d_548.dc.transpose.3"(#loc648))
#loc1010 = loc("conv2d_548.dc.transpose.4"(#loc648))
#loc1011 = loc("conv2d_564.dc.transpose.0"(#loc651))
#loc1012 = loc("conv2d_564.dc.transpose.1"(#loc651))
#loc1013 = loc("conv2d_564.dc.conv2d.2"(#loc651))
#loc1014 = loc("conv2d_564.dc.transpose.3"(#loc651))
#loc1015 = loc("conv2d_564.dc.transpose.4"(#loc651))
#loc1016 = loc("conv2d_581.dc.transpose.0"(#loc654))
#loc1017 = loc("conv2d_581.dc.transpose.1"(#loc654))
#loc1018 = loc("conv2d_581.dc.conv2d.2"(#loc654))
#loc1019 = loc("conv2d_581.dc.transpose.3"(#loc654))
#loc1020 = loc("conv2d_581.dc.transpose.4"(#loc654))
#loc1021 = loc("conv2d_597.dc.transpose.0"(#loc657))
#loc1022 = loc("conv2d_597.dc.transpose.1"(#loc657))
#loc1023 = loc("conv2d_597.dc.conv2d.2"(#loc657))
#loc1024 = loc("conv2d_597.dc.transpose.3"(#loc657))
#loc1025 = loc("conv2d_597.dc.transpose.4"(#loc657))
#loc1026 = loc("conv2d_614.dc.transpose.0"(#loc660))
#loc1027 = loc("conv2d_614.dc.transpose.1"(#loc660))
#loc1028 = loc("conv2d_614.dc.conv2d.2"(#loc660))
#loc1029 = loc("conv2d_614.dc.transpose.3"(#loc660))
#loc1030 = loc("conv2d_614.dc.transpose.4"(#loc660))
#loc1031 = loc("conv2d_630.dc.transpose.0"(#loc663))
#loc1032 = loc("conv2d_630.dc.transpose.1"(#loc663))
#loc1033 = loc("conv2d_630.dc.conv2d.2"(#loc663))
#loc1034 = loc("conv2d_630.dc.transpose.3"(#loc663))
#loc1035 = loc("conv2d_630.dc.transpose.4"(#loc663))
#loc1036 = loc("conv2d_698.dc.transpose.0"(#loc676))
#loc1037 = loc("conv2d_698.dc.transpose.1"(#loc676))
#loc1038 = loc("conv2d_698.dc.conv2d.2"(#loc676))
#loc1039 = loc("conv2d_698.dc.transpose.3"(#loc676))
#loc1040 = loc("conv2d_698.dc.transpose.4"(#loc676))
#loc1041 = loc("conv2d_714.dc.transpose.0"(#loc679))
#loc1042 = loc("conv2d_714.dc.transpose.1"(#loc679))
#loc1043 = loc("conv2d_714.dc.conv2d.2"(#loc679))
#loc1044 = loc("conv2d_714.dc.transpose.3"(#loc679))
#loc1045 = loc("conv2d_714.dc.transpose.4"(#loc679))
#loc1046 = loc("conv2d_731.dc.transpose.0"(#loc682))
#loc1047 = loc("conv2d_731.dc.transpose.1"(#loc682))
#loc1048 = loc("conv2d_731.dc.conv2d.2"(#loc682))
#loc1049 = loc("conv2d_731.dc.transpose.3"(#loc682))
#loc1050 = loc("conv2d_731.dc.transpose.4"(#loc682))
#loc1051 = loc("conv2d_747.dc.transpose.0"(#loc685))
#loc1052 = loc("conv2d_747.dc.transpose.1"(#loc685))
#loc1053 = loc("conv2d_747.dc.conv2d.2"(#loc685))
#loc1054 = loc("conv2d_747.dc.transpose.3"(#loc685))
#loc1055 = loc("conv2d_747.dc.transpose.4"(#loc685))
#loc1056 = loc("conv2d_764.dc.transpose.0"(#loc688))
#loc1057 = loc("conv2d_764.dc.transpose.1"(#loc688))
#loc1058 = loc("conv2d_764.dc.conv2d.2"(#loc688))
#loc1059 = loc("conv2d_764.dc.transpose.3"(#loc688))
#loc1060 = loc("conv2d_764.dc.transpose.4"(#loc688))
#loc1061 = loc("conv2d_780.dc.transpose.0"(#loc691))
#loc1062 = loc("conv2d_780.dc.transpose.1"(#loc691))
#loc1063 = loc("conv2d_780.dc.conv2d.2"(#loc691))
#loc1064 = loc("conv2d_780.dc.transpose.3"(#loc691))
#loc1065 = loc("conv2d_780.dc.transpose.4"(#loc691))
#loc1066 = loc("conv2d_870.dc.transpose.0"(#loc714))
#loc1067 = loc("conv2d_870.dc.transpose.1"(#loc714))
#loc1068 = loc("conv2d_870.dc.conv2d.2"(#loc714))
#loc1069 = loc("conv2d_870.dc.transpose.3"(#loc714))
#loc1070 = loc("conv2d_870.dc.transpose.4"(#loc714))
#loc1071 = loc("conv2d_886.dc.transpose.0"(#loc717))
#loc1072 = loc("conv2d_886.dc.transpose.1"(#loc717))
#loc1073 = loc("conv2d_886.dc.conv2d.2"(#loc717))
#loc1074 = loc("conv2d_886.dc.transpose.3"(#loc717))
#loc1075 = loc("conv2d_886.dc.transpose.4"(#loc717))
#loc1076 = loc("conv2d_902.dc.transpose.0"(#loc720))
#loc1077 = loc("conv2d_902.dc.transpose.1"(#loc720))
#loc1078 = loc("conv2d_902.dc.conv2d.2"(#loc720))
#loc1079 = loc("conv2d_902.dc.transpose.3"(#loc720))
#loc1080 = loc("conv2d_902.dc.transpose.4"(#loc720))
#loc1081 = loc("conv2d_918.dc.transpose.0"(#loc723))
#loc1082 = loc("conv2d_918.dc.transpose.1"(#loc723))
#loc1083 = loc("conv2d_918.dc.conv2d.2"(#loc723))
#loc1084 = loc("conv2d_918.dc.transpose.3"(#loc723))
#loc1085 = loc("conv2d_918.dc.transpose.4"(#loc723))
#loc1086 = loc("conv2d_934.dc.transpose.0"(#loc726))
#loc1087 = loc("conv2d_934.dc.transpose.1"(#loc726))
#loc1088 = loc("conv2d_934.dc.conv2d.2"(#loc726))
#loc1089 = loc("conv2d_934.dc.transpose.3"(#loc726))
#loc1090 = loc("conv2d_934.dc.transpose.4"(#loc726))
#loc1091 = loc("conv2d_950.dc.transpose.0"(#loc729))
#loc1092 = loc("conv2d_950.dc.transpose.1"(#loc729))
#loc1093 = loc("conv2d_950.dc.conv2d.2"(#loc729))
#loc1094 = loc("conv2d_950.dc.transpose.3"(#loc729))
#loc1095 = loc("conv2d_950.dc.transpose.4"(#loc729))
#loc1096 = loc("conv2d_1003.dc.transpose.0"(#loc742))
#loc1097 = loc("conv2d_1003.dc.transpose.1"(#loc742))
#loc1098 = loc("conv2d_1003.dc.conv2d.2"(#loc742))
#loc1099 = loc("conv2d_1003.dc.transpose.3"(#loc742))
#loc1100 = loc("conv2d_1003.dc.transpose.4"(#loc742))
#loc1101 = loc("conv2d_1019.dc.transpose.0"(#loc745))
#loc1102 = loc("conv2d_1019.dc.transpose.1"(#loc745))
#loc1103 = loc("conv2d_1019.dc.conv2d.2"(#loc745))
#loc1104 = loc("conv2d_1019.dc.transpose.3"(#loc745))
#loc1105 = loc("conv2d_1019.dc.transpose.4"(#loc745))
#loc1106 = loc("conv2d_1035.dc.transpose.0"(#loc748))
#loc1107 = loc("conv2d_1035.dc.transpose.1"(#loc748))
#loc1108 = loc("conv2d_1035.dc.conv2d.2"(#loc748))
#loc1109 = loc("conv2d_1035.dc.transpose.3"(#loc748))
#loc1110 = loc("conv2d_1035.dc.transpose.4"(#loc748))
#loc1111 = loc("conv2d_1051.dc.transpose.0"(#loc751))
#loc1112 = loc("conv2d_1051.dc.transpose.1"(#loc751))
#loc1113 = loc("conv2d_1051.dc.conv2d.2"(#loc751))
#loc1114 = loc("conv2d_1051.dc.transpose.3"(#loc751))
#loc1115 = loc("conv2d_1051.dc.transpose.4"(#loc751))
#loc1116 = loc("conv2d_1067.dc.transpose.0"(#loc754))
#loc1117 = loc("conv2d_1067.dc.transpose.1"(#loc754))
#loc1118 = loc("conv2d_1067.dc.conv2d.2"(#loc754))
#loc1119 = loc("conv2d_1067.dc.transpose.3"(#loc754))
#loc1120 = loc("conv2d_1067.dc.transpose.4"(#loc754))
#loc1121 = loc("conv2d_1083.dc.transpose.0"(#loc757))
#loc1122 = loc("conv2d_1083.dc.transpose.1"(#loc757))
#loc1123 = loc("conv2d_1083.dc.conv2d.2"(#loc757))
#loc1124 = loc("conv2d_1083.dc.transpose.3"(#loc757))
#loc1125 = loc("conv2d_1083.dc.transpose.4"(#loc757))
#loc1126 = loc("conv2d_1116.dc.transpose.0"(#loc765))
#loc1127 = loc("conv2d_1116.dc.transpose.1"(#loc765))
#loc1128 = loc("conv2d_1116.dc.conv2d.2"(#loc765))
#loc1129 = loc("conv2d_1116.dc.transpose.3"(#loc765))
#loc1130 = loc("conv2d_1116.dc.transpose.4"(#loc765))
#loc1131 = loc("conv2d_1132.dc.transpose.0"(#loc768))
#loc1132 = loc("conv2d_1132.dc.transpose.1"(#loc768))
#loc1133 = loc("conv2d_1132.dc.conv2d.2"(#loc768))
#loc1134 = loc("conv2d_1132.dc.transpose.3"(#loc768))
#loc1135 = loc("conv2d_1132.dc.transpose.4"(#loc768))
#loc1136 = loc("conv2d_1152.dc.transpose.0"(#loc765))
#loc1137 = loc("conv2d_1152.dc.transpose.1"(#loc765))
#loc1138 = loc("conv2d_1152.dc.conv2d.2"(#loc765))
#loc1139 = loc("conv2d_1152.dc.transpose.3"(#loc765))
#loc1140 = loc("conv2d_1152.dc.transpose.4"(#loc765))
#loc1141 = loc("conv2d_1168.dc.transpose.0"(#loc768))
#loc1142 = loc("conv2d_1168.dc.transpose.1"(#loc768))
#loc1143 = loc("conv2d_1168.dc.conv2d.2"(#loc768))
#loc1144 = loc("conv2d_1168.dc.transpose.3"(#loc768))
#loc1145 = loc("conv2d_1168.dc.transpose.4"(#loc768))
#loc1146 = loc("conv2d_1225.dc.transpose.0"(#loc790))
#loc1147 = loc("conv2d_1225.dc.transpose.1"(#loc790))
#loc1148 = loc("conv2d_1225.dc.conv2d.2"(#loc790))
#loc1149 = loc("conv2d_1225.dc.transpose.3"(#loc790))
#loc1150 = loc("conv2d_1225.dc.transpose.4"(#loc790))
#loc1151 = loc("conv2d_1241.dc.transpose.0"(#loc793))
#loc1152 = loc("conv2d_1241.dc.transpose.1"(#loc793))
#loc1153 = loc("conv2d_1241.dc.conv2d.2"(#loc793))
#loc1154 = loc("conv2d_1241.dc.transpose.3"(#loc793))
#loc1155 = loc("conv2d_1241.dc.transpose.4"(#loc793))
#loc1156 = loc("conv2d_1257.dc.transpose.0"(#loc796))
#loc1157 = loc("conv2d_1257.dc.transpose.1"(#loc796))
#loc1158 = loc("conv2d_1257.dc.conv2d.2"(#loc796))
#loc1159 = loc("conv2d_1257.dc.transpose.3"(#loc796))
#loc1160 = loc("conv2d_1257.dc.transpose.4"(#loc796))
#loc1161 = loc("conv2d_1273.dc.transpose.0"(#loc799))
#loc1162 = loc("conv2d_1273.dc.transpose.1"(#loc799))
#loc1163 = loc("conv2d_1273.dc.conv2d.2"(#loc799))
#loc1164 = loc("conv2d_1273.dc.transpose.3"(#loc799))
#loc1165 = loc("conv2d_1273.dc.transpose.4"(#loc799))
#loc1166 = loc("conv2d_1289.dc.transpose.0"(#loc802))
#loc1167 = loc("conv2d_1289.dc.transpose.1"(#loc802))
#loc1168 = loc("conv2d_1289.dc.conv2d.2"(#loc802))
#loc1169 = loc("conv2d_1289.dc.transpose.3"(#loc802))
#loc1170 = loc("conv2d_1289.dc.transpose.4"(#loc802))
#loc1171 = loc("conv2d_1305.dc.transpose.0"(#loc805))
#loc1172 = loc("conv2d_1305.dc.transpose.1"(#loc805))
#loc1173 = loc("conv2d_1305.dc.conv2d.2"(#loc805))
#loc1174 = loc("conv2d_1305.dc.transpose.3"(#loc805))
#loc1175 = loc("conv2d_1305.dc.transpose.4"(#loc805))
#loc1176 = loc("conv2d_1338.dc.transpose.0"(#loc813))
#loc1177 = loc("conv2d_1338.dc.transpose.1"(#loc813))
#loc1178 = loc("conv2d_1338.dc.conv2d.2"(#loc813))
#loc1179 = loc("conv2d_1338.dc.transpose.3"(#loc813))
#loc1180 = loc("conv2d_1338.dc.transpose.4"(#loc813))
#loc1181 = loc("conv2d_1354.dc.transpose.0"(#loc816))
#loc1182 = loc("conv2d_1354.dc.transpose.1"(#loc816))
#loc1183 = loc("conv2d_1354.dc.conv2d.2"(#loc816))
#loc1184 = loc("conv2d_1354.dc.transpose.3"(#loc816))
#loc1185 = loc("conv2d_1354.dc.transpose.4"(#loc816))
#loc1186 = loc("conv2d_1374.dc.transpose.0"(#loc813))
#loc1187 = loc("conv2d_1374.dc.transpose.1"(#loc813))
#loc1188 = loc("conv2d_1374.dc.conv2d.2"(#loc813))
#loc1189 = loc("conv2d_1374.dc.transpose.3"(#loc813))
#loc1190 = loc("conv2d_1374.dc.transpose.4"(#loc813))
#loc1191 = loc("conv2d_1390.dc.transpose.0"(#loc816))
#loc1192 = loc("conv2d_1390.dc.transpose.1"(#loc816))
#loc1193 = loc("conv2d_1390.dc.conv2d.2"(#loc816))
#loc1194 = loc("conv2d_1390.dc.transpose.3"(#loc816))
#loc1195 = loc("conv2d_1390.dc.transpose.4"(#loc816))
#loc1196 = loc("conv2d_1447.dc.transpose.0"(#loc838))
#loc1197 = loc("conv2d_1447.dc.transpose.1"(#loc838))
#loc1198 = loc("conv2d_1447.dc.conv2d.2"(#loc838))
#loc1199 = loc("conv2d_1447.dc.transpose.3"(#loc838))
#loc1200 = loc("conv2d_1447.dc.transpose.4"(#loc838))
#loc1201 = loc("conv2d_1463.dc.transpose.0"(#loc841))
#loc1202 = loc("conv2d_1463.dc.transpose.1"(#loc841))
#loc1203 = loc("conv2d_1463.dc.conv2d.2"(#loc841))
#loc1204 = loc("conv2d_1463.dc.transpose.3"(#loc841))
#loc1205 = loc("conv2d_1463.dc.transpose.4"(#loc841))
#loc1206 = loc("conv2d_1479.dc.transpose.0"(#loc844))
#loc1207 = loc("conv2d_1479.dc.transpose.1"(#loc844))
#loc1208 = loc("conv2d_1479.dc.conv2d.2"(#loc844))
#loc1209 = loc("conv2d_1479.dc.transpose.3"(#loc844))
#loc1210 = loc("conv2d_1479.dc.transpose.4"(#loc844))
#loc1211 = loc("conv2d_1495.dc.transpose.0"(#loc847))
#loc1212 = loc("conv2d_1495.dc.transpose.1"(#loc847))
#loc1213 = loc("conv2d_1495.dc.conv2d.2"(#loc847))
#loc1214 = loc("conv2d_1495.dc.transpose.3"(#loc847))
#loc1215 = loc("conv2d_1495.dc.transpose.4"(#loc847))
#loc1216 = loc("conv2d_1511.dc.transpose.0"(#loc850))
#loc1217 = loc("conv2d_1511.dc.transpose.1"(#loc850))
#loc1218 = loc("conv2d_1511.dc.conv2d.2"(#loc850))
#loc1219 = loc("conv2d_1511.dc.transpose.3"(#loc850))
#loc1220 = loc("conv2d_1511.dc.transpose.4"(#loc850))
#loc1221 = loc("conv2d_1527.dc.transpose.0"(#loc853))
#loc1222 = loc("conv2d_1527.dc.transpose.1"(#loc853))
#loc1223 = loc("conv2d_1527.dc.conv2d.2"(#loc853))
#loc1224 = loc("conv2d_1527.dc.transpose.3"(#loc853))
#loc1225 = loc("conv2d_1527.dc.transpose.4"(#loc853))
#loc1226 = loc("conv2d_1560.dc.transpose.0"(#loc861))
#loc1227 = loc("conv2d_1560.dc.transpose.1"(#loc861))
#loc1228 = loc("conv2d_1560.dc.conv2d.2"(#loc861))
#loc1229 = loc("conv2d_1560.dc.transpose.3"(#loc861))
#loc1230 = loc("conv2d_1560.dc.transpose.4"(#loc861))
#loc1231 = loc("conv2d_1576.dc.transpose.0"(#loc864))
#loc1232 = loc("conv2d_1576.dc.transpose.1"(#loc864))
#loc1233 = loc("conv2d_1576.dc.conv2d.2"(#loc864))
#loc1234 = loc("conv2d_1576.dc.transpose.3"(#loc864))
#loc1235 = loc("conv2d_1576.dc.transpose.4"(#loc864))
#loc1236 = loc("conv2d_1596.dc.transpose.0"(#loc861))
#loc1237 = loc("conv2d_1596.dc.transpose.1"(#loc861))
#loc1238 = loc("conv2d_1596.dc.conv2d.2"(#loc861))
#loc1239 = loc("conv2d_1596.dc.transpose.3"(#loc861))
#loc1240 = loc("conv2d_1596.dc.transpose.4"(#loc861))
#loc1241 = loc("conv2d_1612.dc.transpose.0"(#loc864))
#loc1242 = loc("conv2d_1612.dc.transpose.1"(#loc864))
#loc1243 = loc("conv2d_1612.dc.conv2d.2"(#loc864))
#loc1244 = loc("conv2d_1612.dc.transpose.3"(#loc864))
#loc1245 = loc("conv2d_1612.dc.transpose.4"(#loc864))
