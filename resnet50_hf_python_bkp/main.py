import time
import ttnn
import my_get_device
def forward_const_eval_0(v1): 
  v2 = v1[0]
  v3 = ttnn.permute(v2, [0, 2, 3, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  v4 = [v3]
  return v4

def forward_const_eval_1(v1): 
  v2 = v1[0]
  v3 = ttnn.reshape(v2, [1, 1, 2048, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v4 = ttnn.permute(v3, [0, 1, 3, 2], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  ttnn.deallocate(v3, False)
  v5 = ttnn.repeat(v4, ttnn.Shape([8, 1, 49, 1]))
  ttnn.deallocate(v4, False)
  v6 = ttnn.reshape(v5, [1, 1, 392, 2048], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v5, False)
  v7 = [v6]
  return v7

def forward_const_eval_2(v1): 
  v2 = v1[0]
  v3 = ttnn.permute(v2, [0, 2, 3, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  v4 = [v3]
  return v4

def forward_const_eval_3(v1): 
  v2 = v1[0]
  v3 = ttnn.reshape(v2, [1, 1000], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v4 = [v3]
  return v4

def forward_const_eval_4(v1): 
  v2 = v1[0]
  v3 = ttnn.reshape(v2, [1, 1, 2048, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v4 = ttnn.permute(v3, [0, 1, 3, 2], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  ttnn.deallocate(v3, False)
  v5 = ttnn.repeat(v4, ttnn.Shape([8, 1, 49, 1]))
  ttnn.deallocate(v4, False)
  v6 = ttnn.reshape(v5, [1, 1, 392, 2048], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v5, False)
  v7 = [v6]
  return v7

def forward_const_eval_5(v1): 
  v2 = v1[0]
  v3 = ttnn.permute(v2, [0, 2, 3, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  v4 = [v3]
  return v4

def forward_const_eval_6(v1): 
  v2 = v1[0]
  v3 = ttnn.permute(v2, [0, 2, 3, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  v4 = [v3]
  return v4

def forward_const_eval_7(v1): 
  v2 = v1[0]
  v3 = ttnn.permute(v2, [0, 2, 3, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  v4 = [v3]
  return v4

def forward_const_eval_8(v1): 
  v2 = v1[0]
  v3 = ttnn.permute(v2, [0, 2, 3, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  v4 = [v3]
  return v4

def forward(v1): 
  v2 = v1[0]
  v3 = v1[1]
  v4 = v1[2]
  v5 = v1[3]
  v6 = v1[4]
  v7 = v1[5]
  v8 = v1[6]
  v9 = v1[7]
  v10 = v1[8]
  v11 = v1[9]
  v12 = v1[10]
  v13 = v1[11]
  v14 = v1[12]
  v15 = v1[13]
  v16 = v1[14]
  v17 = v1[15]
  v18 = v1[16]
  v19 = v1[17]
  v20 = v1[18]
  v21 = v1[19]
  v22 = v1[20]
  v23 = v1[21]
  v24 = v1[22]
  v25 = v1[23]
  v26 = v1[24]
  v27 = v1[25]
  v28 = v1[26]
  v29 = v1[27]
  v30 = v1[28]
  v31 = v1[29]
  v32 = v1[30]
  v33 = v1[31]
  v34 = v1[32]
  v35 = v1[33]
  v36 = v1[34]
  v37 = v1[35]
  v38 = v1[36]
  v39 = v1[37]
  v40 = v1[38]
  v41 = v1[39]
  v42 = v1[40]
  v43 = v1[41]
  v44 = v1[42]
  v45 = v1[43]
  v46 = v1[44]
  v47 = v1[45]
  v48 = v1[46]
  v49 = v1[47]
  v50 = v1[48]
  v51 = v1[49]
  v52 = v1[50]
  v53 = v1[51]
  v54 = v1[52]
  v55 = v1[53]
  v56 = v1[54]
  v57 = v1[55]
  v58 = v1[56]
  v59 = v1[57]
  v60 = v1[58]
  v61 = v1[59]
  v62 = v1[60]
  v63 = v1[61]
  v64 = v1[62]
  v65 = v1[63]
  v66 = v1[64]
  v67 = v1[65]
  v68 = v1[66]
  v69 = v1[67]
  v70 = v1[68]
  v71 = v1[69]
  v72 = v1[70]
  v73 = v1[71]
  v74 = v1[72]
  v75 = v1[73]
  v76 = v1[74]
  v77 = v1[75]
  v78 = v1[76]
  v79 = v1[77]
  v80 = v1[78]
  v81 = v1[79]
  v82 = v1[80]
  v83 = v1[81]
  v84 = v1[82]
  v85 = v1[83]
  v86 = v1[84]
  v87 = v1[85]
  v88 = v1[86]
  v89 = v1[87]
  v90 = v1[88]
  v91 = v1[89]
  v92 = v1[90]
  v93 = v1[91]
  v94 = v1[92]
  v95 = v1[93]
  v96 = v1[94]
  v97 = v1[95]
  v98 = v1[96]
  v99 = v1[97]
  v100 = v1[98]
  v101 = v1[99]
  v102 = v1[100]
  v103 = v1[101]
  v104 = v1[102]
  v105 = v1[103]
  v106 = v1[104]
  v107 = v1[105]
  v108 = v1[106]
  v109 = v1[107]
  v110 = v1[108]
  v111 = v1[109]
  v112 = v1[110]
  v113 = v1[111]
  v114 = v1[112]
  v115 = v1[113]
  v116 = v1[114]
  v117 = v1[115]
  v118 = v1[116]
  v119 = v1[117]
  v120 = v1[118]
  v121 = v1[119]
  v122 = v1[120]
  v123 = v1[121]
  v124 = v1[122]
  v125 = v1[123]
  v126 = v1[124]
  v127 = v1[125]
  v128 = v1[126]
  v129 = v1[127]
  v130 = v1[128]
  v131 = v1[129]
  v132 = v1[130]
  v133 = v1[131]
  v134 = v1[132]
  v135 = v1[133]
  v136 = v1[134]
  v137 = v1[135]
  v138 = v1[136]
  v139 = v1[137]
  v140 = v1[138]
  v141 = v1[139]
  v142 = v1[140]
  v143 = v1[141]
  v144 = v1[142]
  v145 = v1[143]
  v146 = v1[144]
  v147 = v1[145]
  v148 = v1[146]
  v149 = v1[147]
  v150 = v1[148]
  v151 = v1[149]
  v152 = v1[150]
  v153 = v1[151]
  v154 = v1[152]
  v155 = v1[153]
  v156 = v1[154]
  v157 = v1[155]
  v158 = v1[156]
  v159 = v1[157]
  v160 = v1[158]
  v161 = v1[159]
  v162 = v1[160]
  v163 = v1[161]
  v164 = [v93]
  v165 = forward_const_eval_0(v164)
  v166 = v165[0]
  ttnn.deallocate(v93, False)
  v167 = [v107]
  v168 = forward_const_eval_1(v167)
  v169 = v168[0]
  ttnn.deallocate(v107, False)
  v170 = [v94]
  v171 = forward_const_eval_2(v170)
  v172 = v171[0]
  ttnn.deallocate(v94, False)
  v173 = [v163]
  v174 = forward_const_eval_3(v173)
  v175 = v174[0]
  ttnn.deallocate(v163, False)
  v176 = [v108]
  v177 = forward_const_eval_4(v176)
  v178 = v177[0]
  ttnn.deallocate(v108, False)
  v179 = [v102]
  v180 = forward_const_eval_5(v179)
  v181 = v180[0]
  ttnn.deallocate(v102, False)
  v182 = [v95]
  v183 = forward_const_eval_6(v182)
  v184 = v183[0]
  ttnn.deallocate(v95, False)
  v185 = [v101]
  v186 = forward_const_eval_7(v185)
  v187 = v186[0]
  ttnn.deallocate(v101, False)
  v188 = [v96]
  v189 = forward_const_eval_8(v188)
  v190 = v189[0]
  ttnn.deallocate(v96, False)
  v191 = my_get_device.DeviceGetter.get_device()
  v192 = ttnn.permute(v2, [0, 2, 3, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), pad_value=0)
  ttnn.deallocate(v2, False)
  v193 = ttnn.reshape(v192, [1, 1, 401408, 3], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v192, False)
  v194 = ttnn.to_layout(v193, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v193, False)
  v195 = ttnn.conv2d(input_tensor=v194, weight_tensor=v109, device=v191, in_channels=3, out_channels=64, batch_size=8, input_height=224, input_width=224, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3, 3, 3], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v194, False)
  ttnn.deallocate(v109, False)
  v196 = ttnn.multiply(v195, v3, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v195, False)
  ttnn.deallocate(v3, False)
  v197 = ttnn.add(v196, v4, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v196, False)
  ttnn.deallocate(v4, False)
  v198 = ttnn.relu(v197, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v197, False)
  v199 = ttnn.to_layout(v198, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v198, False)
  v200 = ttnn.max_pool2d(v199, 8, 112, 112, 64, [3, 3], [2, 2], [1, 1], [1, 1], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None), applied_shard_scheme=None, ceil_mode=False, in_place_halo=False)
  ttnn.deallocate(v199, False)
  v201 = ttnn.conv2d(input_tensor=v200, weight_tensor=v110, device=v191, in_channels=64, out_channels=64, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v110, False)
  v202 = ttnn.multiply(v201, v5, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v201, False)
  ttnn.deallocate(v5, False)
  v203 = ttnn.add(v202, v6, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v202, False)
  ttnn.deallocate(v6, False)
  v204 = ttnn.relu(v203, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v203, False)
  v205 = ttnn.to_layout(v204, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v204, False)
  v206 = ttnn.conv2d(input_tensor=v205, weight_tensor=v111, device=v191, in_channels=64, out_channels=64, batch_size=8, input_height=56, input_width=56, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v205, False)
  ttnn.deallocate(v111, False)
  v207 = ttnn.multiply(v206, v7, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v206, False)
  ttnn.deallocate(v7, False)
  v208 = ttnn.add(v207, v8, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v207, False)
  ttnn.deallocate(v8, False)
  v209 = ttnn.relu(v208, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v208, False)
  v210 = ttnn.to_layout(v209, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v209, False)
  v211 = ttnn.conv2d(input_tensor=v210, weight_tensor=v112, device=v191, in_channels=64, out_channels=256, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v210, False)
  ttnn.deallocate(v112, False)
  v212 = ttnn.multiply(v211, v9, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v211, False)
  ttnn.deallocate(v9, False)
  v213 = ttnn.add(v212, v10, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v212, False)
  ttnn.deallocate(v10, False)
  v214 = ttnn.conv2d(input_tensor=v200, weight_tensor=v113, device=v191, in_channels=64, out_channels=256, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v200, False)
  ttnn.deallocate(v113, False)
  v215 = ttnn.multiply(v214, v11, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v214, False)
  ttnn.deallocate(v11, False)
  v216 = ttnn.add(v215, v12, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v215, False)
  ttnn.deallocate(v12, False)
  v217 = ttnn.add(v213, v216, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v216, False)
  ttnn.deallocate(v213, False)
  v218 = ttnn.relu(v217, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v217, False)
  v219 = ttnn.to_layout(v218, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v220 = ttnn.conv2d(input_tensor=v219, weight_tensor=v114, device=v191, in_channels=256, out_channels=64, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v219, False)
  ttnn.deallocate(v114, False)
  v221 = ttnn.multiply(v220, v13, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v220, False)
  ttnn.deallocate(v13, False)
  v222 = ttnn.add(v221, v14, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v221, False)
  ttnn.deallocate(v14, False)
  v223 = ttnn.relu(v222, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v222, False)
  v224 = ttnn.to_layout(v223, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v223, False)
  v225 = ttnn.conv2d(input_tensor=v224, weight_tensor=v115, device=v191, in_channels=64, out_channels=64, batch_size=8, input_height=56, input_width=56, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v224, False)
  ttnn.deallocate(v115, False)
  v226 = ttnn.multiply(v225, v15, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v225, False)
  ttnn.deallocate(v15, False)
  v227 = ttnn.add(v226, v16, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v226, False)
  ttnn.deallocate(v16, False)
  v228 = ttnn.relu(v227, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v227, False)
  v229 = ttnn.to_layout(v228, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v228, False)
  v230 = ttnn.conv2d(input_tensor=v229, weight_tensor=v116, device=v191, in_channels=64, out_channels=256, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v229, False)
  ttnn.deallocate(v116, False)
  v231 = ttnn.multiply(v230, v17, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v230, False)
  ttnn.deallocate(v17, False)
  v232 = ttnn.add(v231, v18, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v231, False)
  ttnn.deallocate(v18, False)
  v233 = ttnn.add(v232, v218, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v232, False)
  ttnn.deallocate(v218, False)
  v234 = ttnn.relu(v233, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v233, False)
  v235 = ttnn.to_layout(v234, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v236 = ttnn.conv2d(input_tensor=v235, weight_tensor=v117, device=v191, in_channels=256, out_channels=64, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v235, False)
  ttnn.deallocate(v117, False)
  v237 = ttnn.multiply(v236, v19, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v236, False)
  ttnn.deallocate(v19, False)
  v238 = ttnn.add(v237, v20, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v237, False)
  ttnn.deallocate(v20, False)
  v239 = ttnn.relu(v238, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v238, False)
  v240 = ttnn.to_layout(v239, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v239, False)
  v241 = ttnn.conv2d(input_tensor=v240, weight_tensor=v118, device=v191, in_channels=64, out_channels=64, batch_size=8, input_height=56, input_width=56, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v240, False)
  ttnn.deallocate(v118, False)
  v242 = ttnn.multiply(v241, v21, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v241, False)
  ttnn.deallocate(v21, False)
  v243 = ttnn.add(v242, v22, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v242, False)
  ttnn.deallocate(v22, False)
  v244 = ttnn.relu(v243, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v243, False)
  v245 = ttnn.to_layout(v244, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v244, False)
  v246 = ttnn.conv2d(input_tensor=v245, weight_tensor=v119, device=v191, in_channels=64, out_channels=256, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v245, False)
  ttnn.deallocate(v119, False)
  v247 = ttnn.multiply(v246, v23, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v246, False)
  ttnn.deallocate(v23, False)
  v248 = ttnn.add(v247, v24, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v247, False)
  ttnn.deallocate(v24, False)
  v249 = ttnn.add(v248, v234, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v248, False)
  ttnn.deallocate(v234, False)
  v250 = ttnn.relu(v249, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v249, False)
  v251 = ttnn.to_layout(v250, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v252 = ttnn.conv2d(input_tensor=v251, weight_tensor=v120, device=v191, in_channels=256, out_channels=128, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v251, False)
  ttnn.deallocate(v120, False)
  v253 = ttnn.multiply(v252, v25, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v252, False)
  ttnn.deallocate(v25, False)
  v254 = ttnn.add(v253, v26, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v253, False)
  ttnn.deallocate(v26, False)
  v255 = ttnn.relu(v254, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v254, False)
  v256 = ttnn.to_layout(v255, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v255, False)
  v257 = ttnn.conv2d(input_tensor=v256, weight_tensor=v121, device=v191, in_channels=128, out_channels=128, batch_size=8, input_height=56, input_width=56, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v256, False)
  ttnn.deallocate(v121, False)
  v258 = ttnn.multiply(v257, v27, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v257, False)
  ttnn.deallocate(v27, False)
  v259 = ttnn.add(v258, v28, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v258, False)
  ttnn.deallocate(v28, False)
  v260 = ttnn.relu(v259, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v259, False)
  v261 = ttnn.to_layout(v260, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v260, False)
  v262 = ttnn.conv2d(input_tensor=v261, weight_tensor=v122, device=v191, in_channels=128, out_channels=512, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v261, False)
  ttnn.deallocate(v122, False)
  v263 = ttnn.multiply(v262, v29, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v262, False)
  ttnn.deallocate(v29, False)
  v264 = ttnn.add(v263, v30, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v263, False)
  ttnn.deallocate(v30, False)
  v265 = ttnn.to_layout(v250, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v250, False)
  v266 = ttnn.conv2d(input_tensor=v265, weight_tensor=v123, device=v191, in_channels=256, out_channels=512, batch_size=8, input_height=56, input_width=56, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v265, False)
  ttnn.deallocate(v123, False)
  v267 = ttnn.multiply(v266, v31, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v266, False)
  ttnn.deallocate(v31, False)
  v268 = ttnn.add(v267, v32, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v267, False)
  ttnn.deallocate(v32, False)
  v269 = ttnn.add(v264, v268, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v268, False)
  ttnn.deallocate(v264, False)
  v270 = ttnn.relu(v269, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v269, False)
  v271 = ttnn.to_layout(v270, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v272 = ttnn.conv2d(input_tensor=v271, weight_tensor=v124, device=v191, in_channels=512, out_channels=128, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v271, False)
  ttnn.deallocate(v124, False)
  v273 = ttnn.multiply(v272, v33, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v272, False)
  ttnn.deallocate(v33, False)
  v274 = ttnn.add(v273, v34, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v273, False)
  ttnn.deallocate(v34, False)
  v275 = ttnn.relu(v274, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v274, False)
  v276 = ttnn.to_layout(v275, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v275, False)
  v277 = ttnn.conv2d(input_tensor=v276, weight_tensor=v125, device=v191, in_channels=128, out_channels=128, batch_size=8, input_height=28, input_width=28, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v276, False)
  ttnn.deallocate(v125, False)
  v278 = ttnn.multiply(v277, v35, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v277, False)
  ttnn.deallocate(v35, False)
  v279 = ttnn.add(v278, v36, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v278, False)
  ttnn.deallocate(v36, False)
  v280 = ttnn.relu(v279, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v279, False)
  v281 = ttnn.to_layout(v280, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v280, False)
  v282 = ttnn.conv2d(input_tensor=v281, weight_tensor=v126, device=v191, in_channels=128, out_channels=512, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v281, False)
  ttnn.deallocate(v126, False)
  v283 = ttnn.multiply(v282, v37, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v282, False)
  ttnn.deallocate(v37, False)
  v284 = ttnn.add(v283, v38, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v283, False)
  ttnn.deallocate(v38, False)
  v285 = ttnn.add(v284, v270, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v284, False)
  ttnn.deallocate(v270, False)
  v286 = ttnn.relu(v285, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v285, False)
  v287 = ttnn.to_layout(v286, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v288 = ttnn.conv2d(input_tensor=v287, weight_tensor=v127, device=v191, in_channels=512, out_channels=128, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v287, False)
  ttnn.deallocate(v127, False)
  v289 = ttnn.multiply(v288, v39, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v288, False)
  ttnn.deallocate(v39, False)
  v290 = ttnn.add(v289, v40, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v289, False)
  ttnn.deallocate(v40, False)
  v291 = ttnn.relu(v290, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v290, False)
  v292 = ttnn.to_layout(v291, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v291, False)
  v293 = ttnn.conv2d(input_tensor=v292, weight_tensor=v128, device=v191, in_channels=128, out_channels=128, batch_size=8, input_height=28, input_width=28, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v292, False)
  ttnn.deallocate(v128, False)
  v294 = ttnn.multiply(v293, v41, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v293, False)
  ttnn.deallocate(v41, False)
  v295 = ttnn.add(v294, v42, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v294, False)
  ttnn.deallocate(v42, False)
  v296 = ttnn.relu(v295, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v295, False)
  v297 = ttnn.to_layout(v296, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v296, False)
  v298 = ttnn.conv2d(input_tensor=v297, weight_tensor=v129, device=v191, in_channels=128, out_channels=512, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v297, False)
  ttnn.deallocate(v129, False)
  v299 = ttnn.multiply(v298, v43, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v298, False)
  ttnn.deallocate(v43, False)
  v300 = ttnn.add(v299, v44, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v299, False)
  ttnn.deallocate(v44, False)
  v301 = ttnn.add(v300, v286, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v300, False)
  ttnn.deallocate(v286, False)
  v302 = ttnn.relu(v301, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v301, False)
  v303 = ttnn.to_layout(v302, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v304 = ttnn.conv2d(input_tensor=v303, weight_tensor=v130, device=v191, in_channels=512, out_channels=128, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v303, False)
  ttnn.deallocate(v130, False)
  v305 = ttnn.multiply(v304, v45, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v304, False)
  ttnn.deallocate(v45, False)
  v306 = ttnn.add(v305, v46, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v305, False)
  ttnn.deallocate(v46, False)
  v307 = ttnn.relu(v306, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v306, False)
  v308 = ttnn.to_layout(v307, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v307, False)
  v309 = ttnn.conv2d(input_tensor=v308, weight_tensor=v131, device=v191, in_channels=128, out_channels=128, batch_size=8, input_height=28, input_width=28, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v308, False)
  ttnn.deallocate(v131, False)
  v310 = ttnn.multiply(v309, v47, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v309, False)
  ttnn.deallocate(v47, False)
  v311 = ttnn.add(v310, v48, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v310, False)
  ttnn.deallocate(v48, False)
  v312 = ttnn.relu(v311, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v311, False)
  v313 = ttnn.to_layout(v312, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v312, False)
  v314 = ttnn.conv2d(input_tensor=v313, weight_tensor=v132, device=v191, in_channels=128, out_channels=512, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v313, False)
  ttnn.deallocate(v132, False)
  v315 = ttnn.multiply(v314, v49, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v314, False)
  ttnn.deallocate(v49, False)
  v316 = ttnn.add(v315, v50, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v315, False)
  ttnn.deallocate(v50, False)
  v317 = ttnn.add(v316, v302, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v316, False)
  ttnn.deallocate(v302, False)
  v318 = ttnn.relu(v317, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v317, False)
  v319 = ttnn.to_layout(v318, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v320 = ttnn.conv2d(input_tensor=v319, weight_tensor=v133, device=v191, in_channels=512, out_channels=256, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v319, False)
  ttnn.deallocate(v133, False)
  v321 = ttnn.multiply(v320, v51, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v320, False)
  ttnn.deallocate(v51, False)
  v322 = ttnn.add(v321, v52, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v321, False)
  ttnn.deallocate(v52, False)
  v323 = ttnn.relu(v322, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v322, False)
  v324 = ttnn.to_layout(v323, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v323, False)
  v325 = ttnn.conv2d(input_tensor=v324, weight_tensor=v134, device=v191, in_channels=256, out_channels=256, batch_size=8, input_height=28, input_width=28, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v324, False)
  ttnn.deallocate(v134, False)
  v326 = ttnn.multiply(v325, v53, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v325, False)
  ttnn.deallocate(v53, False)
  v327 = ttnn.add(v326, v54, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v326, False)
  ttnn.deallocate(v54, False)
  v328 = ttnn.relu(v327, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v327, False)
  v329 = ttnn.to_layout(v328, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v328, False)
  v330 = ttnn.conv2d(input_tensor=v329, weight_tensor=v135, device=v191, in_channels=256, out_channels=1024, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v329, False)
  ttnn.deallocate(v135, False)
  v331 = ttnn.multiply(v330, v55, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v330, False)
  ttnn.deallocate(v55, False)
  v332 = ttnn.add(v331, v56, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v331, False)
  ttnn.deallocate(v56, False)
  v333 = ttnn.to_layout(v318, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v318, False)
  v334 = ttnn.conv2d(input_tensor=v333, weight_tensor=v136, device=v191, in_channels=512, out_channels=1024, batch_size=8, input_height=28, input_width=28, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v333, False)
  ttnn.deallocate(v136, False)
  v335 = ttnn.multiply(v334, v57, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v334, False)
  ttnn.deallocate(v57, False)
  v336 = ttnn.add(v335, v58, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v335, False)
  ttnn.deallocate(v58, False)
  v337 = ttnn.add(v332, v336, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v336, False)
  ttnn.deallocate(v332, False)
  v338 = ttnn.relu(v337, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v337, False)
  v339 = ttnn.to_layout(v338, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v340 = ttnn.conv2d(input_tensor=v339, weight_tensor=v137, device=v191, in_channels=1024, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v339, False)
  ttnn.deallocate(v137, False)
  v341 = ttnn.multiply(v340, v59, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v340, False)
  ttnn.deallocate(v59, False)
  v342 = ttnn.add(v341, v60, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v341, False)
  ttnn.deallocate(v60, False)
  v343 = ttnn.relu(v342, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v342, False)
  v344 = ttnn.to_layout(v343, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v343, False)
  v345 = ttnn.conv2d(input_tensor=v344, weight_tensor=v138, device=v191, in_channels=256, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v344, False)
  ttnn.deallocate(v138, False)
  v346 = ttnn.multiply(v345, v61, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v345, False)
  ttnn.deallocate(v61, False)
  v347 = ttnn.add(v346, v62, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v346, False)
  ttnn.deallocate(v62, False)
  v348 = ttnn.relu(v347, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v347, False)
  v349 = ttnn.to_layout(v348, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v348, False)
  v350 = ttnn.conv2d(input_tensor=v349, weight_tensor=v139, device=v191, in_channels=256, out_channels=1024, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v349, False)
  ttnn.deallocate(v139, False)
  v351 = ttnn.multiply(v350, v63, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v350, False)
  ttnn.deallocate(v63, False)
  v352 = ttnn.add(v351, v64, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v351, False)
  ttnn.deallocate(v64, False)
  v353 = ttnn.add(v352, v338, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v352, False)
  ttnn.deallocate(v338, False)
  v354 = ttnn.relu(v353, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v353, False)
  v355 = ttnn.to_layout(v354, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v356 = ttnn.conv2d(input_tensor=v355, weight_tensor=v140, device=v191, in_channels=1024, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v355, False)
  ttnn.deallocate(v140, False)
  v357 = ttnn.multiply(v356, v65, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v356, False)
  ttnn.deallocate(v65, False)
  v358 = ttnn.add(v357, v66, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v357, False)
  ttnn.deallocate(v66, False)
  v359 = ttnn.relu(v358, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v358, False)
  v360 = ttnn.to_layout(v359, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v359, False)
  v361 = ttnn.conv2d(input_tensor=v360, weight_tensor=v141, device=v191, in_channels=256, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v360, False)
  ttnn.deallocate(v141, False)
  v362 = ttnn.multiply(v361, v67, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v361, False)
  ttnn.deallocate(v67, False)
  v363 = ttnn.add(v362, v68, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v362, False)
  ttnn.deallocate(v68, False)
  v364 = ttnn.relu(v363, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v363, False)
  v365 = ttnn.to_layout(v364, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v364, False)
  v366 = ttnn.conv2d(input_tensor=v365, weight_tensor=v142, device=v191, in_channels=256, out_channels=1024, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v365, False)
  ttnn.deallocate(v142, False)
  v367 = ttnn.multiply(v366, v69, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v366, False)
  ttnn.deallocate(v69, False)
  v368 = ttnn.add(v367, v70, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v367, False)
  ttnn.deallocate(v70, False)
  v369 = ttnn.add(v368, v354, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v368, False)
  ttnn.deallocate(v354, False)
  v370 = ttnn.relu(v369, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v369, False)
  v371 = ttnn.to_layout(v370, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v372 = ttnn.conv2d(input_tensor=v371, weight_tensor=v143, device=v191, in_channels=1024, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v371, False)
  ttnn.deallocate(v143, False)
  v373 = ttnn.multiply(v372, v71, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v372, False)
  ttnn.deallocate(v71, False)
  v374 = ttnn.add(v373, v72, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v373, False)
  ttnn.deallocate(v72, False)
  v375 = ttnn.relu(v374, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v374, False)
  v376 = ttnn.to_layout(v375, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v375, False)
  v377 = ttnn.conv2d(input_tensor=v376, weight_tensor=v144, device=v191, in_channels=256, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v376, False)
  ttnn.deallocate(v144, False)
  v378 = ttnn.multiply(v377, v73, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v377, False)
  ttnn.deallocate(v73, False)
  v379 = ttnn.add(v378, v74, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v378, False)
  ttnn.deallocate(v74, False)
  v380 = ttnn.relu(v379, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v379, False)
  v381 = ttnn.to_layout(v380, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v380, False)
  v382 = ttnn.conv2d(input_tensor=v381, weight_tensor=v145, device=v191, in_channels=256, out_channels=1024, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v381, False)
  ttnn.deallocate(v145, False)
  v383 = ttnn.multiply(v382, v75, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v382, False)
  ttnn.deallocate(v75, False)
  v384 = ttnn.add(v383, v76, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v383, False)
  ttnn.deallocate(v76, False)
  v385 = ttnn.add(v384, v370, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v384, False)
  ttnn.deallocate(v370, False)
  v386 = ttnn.relu(v385, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v385, False)
  v387 = ttnn.to_layout(v386, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v388 = ttnn.conv2d(input_tensor=v387, weight_tensor=v146, device=v191, in_channels=1024, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v387, False)
  ttnn.deallocate(v146, False)
  v389 = ttnn.multiply(v388, v77, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v388, False)
  ttnn.deallocate(v77, False)
  v390 = ttnn.add(v389, v78, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v389, False)
  ttnn.deallocate(v78, False)
  v391 = ttnn.relu(v390, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v390, False)
  v392 = ttnn.to_layout(v391, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v391, False)
  v393 = ttnn.conv2d(input_tensor=v392, weight_tensor=v147, device=v191, in_channels=256, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v392, False)
  ttnn.deallocate(v147, False)
  v394 = ttnn.multiply(v393, v79, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v393, False)
  ttnn.deallocate(v79, False)
  v395 = ttnn.add(v394, v80, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v394, False)
  ttnn.deallocate(v80, False)
  v396 = ttnn.relu(v395, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v395, False)
  v397 = ttnn.to_layout(v396, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v396, False)
  v398 = ttnn.conv2d(input_tensor=v397, weight_tensor=v148, device=v191, in_channels=256, out_channels=1024, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v397, False)
  ttnn.deallocate(v148, False)
  v399 = ttnn.multiply(v398, v81, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v398, False)
  ttnn.deallocate(v81, False)
  v400 = ttnn.add(v399, v82, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v399, False)
  ttnn.deallocate(v82, False)
  v401 = ttnn.add(v400, v386, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v400, False)
  ttnn.deallocate(v386, False)
  v402 = ttnn.relu(v401, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v401, False)
  v403 = ttnn.to_layout(v402, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v404 = ttnn.conv2d(input_tensor=v403, weight_tensor=v149, device=v191, in_channels=1024, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v403, False)
  ttnn.deallocate(v149, False)
  v405 = ttnn.multiply(v404, v83, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v404, False)
  ttnn.deallocate(v83, False)
  v406 = ttnn.add(v405, v84, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v405, False)
  ttnn.deallocate(v84, False)
  v407 = ttnn.relu(v406, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v406, False)
  v408 = ttnn.to_layout(v407, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v407, False)
  v409 = ttnn.conv2d(input_tensor=v408, weight_tensor=v150, device=v191, in_channels=256, out_channels=256, batch_size=8, input_height=14, input_width=14, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v408, False)
  ttnn.deallocate(v150, False)
  v410 = ttnn.multiply(v409, v85, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v409, False)
  ttnn.deallocate(v85, False)
  v411 = ttnn.add(v410, v86, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v410, False)
  ttnn.deallocate(v86, False)
  v412 = ttnn.relu(v411, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v411, False)
  v413 = ttnn.to_layout(v412, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v412, False)
  v414 = ttnn.conv2d(input_tensor=v413, weight_tensor=v151, device=v191, in_channels=256, out_channels=1024, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v413, False)
  ttnn.deallocate(v151, False)
  v415 = ttnn.multiply(v414, v87, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v414, False)
  ttnn.deallocate(v87, False)
  v416 = ttnn.add(v415, v88, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v415, False)
  ttnn.deallocate(v88, False)
  v417 = ttnn.add(v416, v402, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v416, False)
  ttnn.deallocate(v402, False)
  v418 = ttnn.relu(v417, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v417, False)
  v419 = ttnn.to_layout(v418, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v420 = ttnn.conv2d(input_tensor=v419, weight_tensor=v152, device=v191, in_channels=1024, out_channels=512, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v419, False)
  ttnn.deallocate(v152, False)
  v421 = ttnn.multiply(v420, v89, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v420, False)
  ttnn.deallocate(v89, False)
  v422 = ttnn.add(v421, v90, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v421, False)
  ttnn.deallocate(v90, False)
  v423 = ttnn.relu(v422, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v422, False)
  v424 = ttnn.to_layout(v423, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v423, False)
  v425 = ttnn.conv2d(input_tensor=v424, weight_tensor=v153, device=v191, in_channels=512, out_channels=512, batch_size=8, input_height=14, input_width=14, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v424, False)
  ttnn.deallocate(v153, False)
  v426 = ttnn.multiply(v425, v91, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v425, False)
  ttnn.deallocate(v91, False)
  v427 = ttnn.add(v426, v92, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v426, False)
  ttnn.deallocate(v92, False)
  v428 = ttnn.relu(v427, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v427, False)
  v429 = ttnn.to_layout(v428, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v428, False)
  v430 = ttnn.conv2d(input_tensor=v429, weight_tensor=v154, device=v191, in_channels=512, out_channels=2048, batch_size=8, input_height=7, input_width=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v429, False)
  ttnn.deallocate(v154, False)
  v431 = ttnn.multiply(v430, v166, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v430, False)
  ttnn.deallocate(v166, False)
  v432 = ttnn.add(v431, v172, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v431, False)
  ttnn.deallocate(v172, False)
  v433 = ttnn.to_layout(v418, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v418, False)
  v434 = ttnn.conv2d(input_tensor=v433, weight_tensor=v155, device=v191, in_channels=1024, out_channels=2048, batch_size=8, input_height=14, input_width=14, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v433, False)
  ttnn.deallocate(v155, False)
  v435 = ttnn.multiply(v434, v184, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v434, False)
  ttnn.deallocate(v184, False)
  v436 = ttnn.add(v435, v190, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v435, False)
  ttnn.deallocate(v190, False)
  v437 = ttnn.add(v432, v436, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v436, False)
  ttnn.deallocate(v432, False)
  v438 = ttnn.relu(v437, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v437, False)
  v439 = ttnn.to_layout(v438, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v440 = ttnn.conv2d(input_tensor=v439, weight_tensor=v156, device=v191, in_channels=2048, out_channels=512, batch_size=8, input_height=7, input_width=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v439, False)
  ttnn.deallocate(v156, False)
  v441 = ttnn.multiply(v440, v97, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v440, False)
  ttnn.deallocate(v97, False)
  v442 = ttnn.add(v441, v98, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v441, False)
  ttnn.deallocate(v98, False)
  v443 = ttnn.relu(v442, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v442, False)
  v444 = ttnn.to_layout(v443, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v443, False)
  v445 = ttnn.conv2d(input_tensor=v444, weight_tensor=v157, device=v191, in_channels=512, out_channels=512, batch_size=8, input_height=7, input_width=7, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v444, False)
  ttnn.deallocate(v157, False)
  v446 = ttnn.multiply(v445, v99, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v445, False)
  ttnn.deallocate(v99, False)
  v447 = ttnn.add(v446, v100, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v446, False)
  ttnn.deallocate(v100, False)
  v448 = ttnn.relu(v447, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v447, False)
  v449 = ttnn.to_layout(v448, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v448, False)
  v450 = ttnn.conv2d(input_tensor=v449, weight_tensor=v158, device=v191, in_channels=512, out_channels=2048, batch_size=8, input_height=7, input_width=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v449, False)
  ttnn.deallocate(v158, False)
  v451 = ttnn.multiply(v450, v187, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v450, False)
  ttnn.deallocate(v187, False)
  v452 = ttnn.add(v451, v181, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v451, False)
  ttnn.deallocate(v181, False)
  v453 = ttnn.add(v452, v438, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v452, False)
  ttnn.deallocate(v438, False)
  v454 = ttnn.relu(v453, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v453, False)
  v455 = ttnn.to_layout(v454, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v456 = ttnn.conv2d(input_tensor=v455, weight_tensor=v159, device=v191, in_channels=2048, out_channels=512, batch_size=8, input_height=7, input_width=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v455, False)
  ttnn.deallocate(v159, False)
  v457 = ttnn.multiply(v456, v103, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v456, False)
  ttnn.deallocate(v103, False)
  v458 = ttnn.add(v457, v104, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v457, False)
  ttnn.deallocate(v104, False)
  v459 = ttnn.relu(v458, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v458, False)
  v460 = ttnn.to_layout(v459, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v459, False)
  v461 = ttnn.conv2d(input_tensor=v460, weight_tensor=v160, device=v191, in_channels=512, out_channels=512, batch_size=8, input_height=7, input_width=7, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v460, False)
  ttnn.deallocate(v160, False)
  v462 = ttnn.multiply(v461, v105, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v461, False)
  ttnn.deallocate(v105, False)
  v463 = ttnn.add(v462, v106, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v462, False)
  ttnn.deallocate(v106, False)
  v464 = ttnn.relu(v463, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v463, False)
  v465 = ttnn.to_layout(v464, ttnn.Layout.ROW_MAJOR, None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v464, False)
  v466 = ttnn.conv2d(input_tensor=v465, weight_tensor=v161, device=v191, in_channels=512, out_channels=2048, batch_size=8, input_height=7, input_width=7, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, bias_tensor=None, conv_config=None, compute_config=None, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v465, False)
  ttnn.deallocate(v161, False)
  v467 = ttnn.multiply(v466, v169, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v466, False)
  ttnn.deallocate(v169, False)
  v468 = ttnn.add(v467, v178, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v467, False)
  ttnn.deallocate(v178, False)
  v469 = ttnn.add(v468, v454, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v468, False)
  ttnn.deallocate(v454, False)
  v470 = ttnn.relu(v469, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v469, False)
  v471 = ttnn.reshape(v470, [8, 1, 49, 2048], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v470, False)
  v472 = ttnn.mean(v471, [-2], True, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v471, False)
  v473 = ttnn.reshape(v472, [8, 2048], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v472, False)
  v474 = ttnn.matmul(v473, v162, transpose_a=False, transpose_b=False, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v473, False)
  ttnn.deallocate(v162, False)
  v475 = ttnn.add(v474, v175, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  ttnn.deallocate(v474, False)
  ttnn.deallocate(v175, False)
  v476 = [v475]
  return v476

def create_inputs_for_forward(): 
  v1 = my_get_device.DeviceGetter.get_device()
  v2 = ttnn.ones(shape=ttnn.Shape([8, 3, 224, 224]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v3 = ttnn.to_device(v2, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v4 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v5 = ttnn.to_device(v4, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v6 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v7 = ttnn.to_device(v6, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v8 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v9 = ttnn.to_device(v8, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v10 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v11 = ttnn.to_device(v10, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v12 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v13 = ttnn.to_device(v12, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v14 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v15 = ttnn.to_device(v14, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v16 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v17 = ttnn.to_device(v16, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v18 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v19 = ttnn.to_device(v18, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v20 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v21 = ttnn.to_device(v20, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v22 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v23 = ttnn.to_device(v22, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v24 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v25 = ttnn.to_device(v24, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v26 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v27 = ttnn.to_device(v26, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v28 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v29 = ttnn.to_device(v28, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v30 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v31 = ttnn.to_device(v30, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v32 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v33 = ttnn.to_device(v32, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v34 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v35 = ttnn.to_device(v34, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v36 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v37 = ttnn.to_device(v36, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v38 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v39 = ttnn.to_device(v38, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v40 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v41 = ttnn.to_device(v40, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v42 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 64]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v43 = ttnn.to_device(v42, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v44 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v45 = ttnn.to_device(v44, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v46 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v47 = ttnn.to_device(v46, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v48 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v49 = ttnn.to_device(v48, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v50 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v51 = ttnn.to_device(v50, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v52 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v53 = ttnn.to_device(v52, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v54 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v55 = ttnn.to_device(v54, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v56 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v57 = ttnn.to_device(v56, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v58 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v59 = ttnn.to_device(v58, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v60 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v61 = ttnn.to_device(v60, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v62 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v63 = ttnn.to_device(v62, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v64 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v65 = ttnn.to_device(v64, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v66 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v67 = ttnn.to_device(v66, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v68 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v69 = ttnn.to_device(v68, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v70 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v71 = ttnn.to_device(v70, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v72 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v73 = ttnn.to_device(v72, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v74 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v75 = ttnn.to_device(v74, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v76 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v77 = ttnn.to_device(v76, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v78 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v79 = ttnn.to_device(v78, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v80 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v81 = ttnn.to_device(v80, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v82 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v83 = ttnn.to_device(v82, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v84 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v85 = ttnn.to_device(v84, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v86 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v87 = ttnn.to_device(v86, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v88 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v89 = ttnn.to_device(v88, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v90 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v91 = ttnn.to_device(v90, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v92 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v93 = ttnn.to_device(v92, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v94 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 128]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v95 = ttnn.to_device(v94, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v96 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v97 = ttnn.to_device(v96, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v98 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v99 = ttnn.to_device(v98, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v100 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v101 = ttnn.to_device(v100, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v102 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v103 = ttnn.to_device(v102, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v104 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v105 = ttnn.to_device(v104, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v106 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v107 = ttnn.to_device(v106, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v108 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v109 = ttnn.to_device(v108, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v110 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v111 = ttnn.to_device(v110, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v112 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v113 = ttnn.to_device(v112, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v114 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v115 = ttnn.to_device(v114, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v116 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v117 = ttnn.to_device(v116, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v118 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v119 = ttnn.to_device(v118, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v120 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v121 = ttnn.to_device(v120, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v122 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v123 = ttnn.to_device(v122, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v124 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v125 = ttnn.to_device(v124, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v126 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v127 = ttnn.to_device(v126, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v128 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v129 = ttnn.to_device(v128, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v130 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v131 = ttnn.to_device(v130, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v132 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v133 = ttnn.to_device(v132, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v134 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v135 = ttnn.to_device(v134, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v136 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v137 = ttnn.to_device(v136, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v138 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v139 = ttnn.to_device(v138, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v140 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v141 = ttnn.to_device(v140, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v142 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v143 = ttnn.to_device(v142, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v144 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v145 = ttnn.to_device(v144, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v146 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v147 = ttnn.to_device(v146, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v148 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v149 = ttnn.to_device(v148, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v150 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v151 = ttnn.to_device(v150, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v152 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v153 = ttnn.to_device(v152, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v154 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v155 = ttnn.to_device(v154, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v156 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v157 = ttnn.to_device(v156, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v158 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v159 = ttnn.to_device(v158, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v160 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v161 = ttnn.to_device(v160, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v162 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v163 = ttnn.to_device(v162, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v164 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v165 = ttnn.to_device(v164, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v166 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v167 = ttnn.to_device(v166, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v168 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v169 = ttnn.to_device(v168, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v170 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 256]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v171 = ttnn.to_device(v170, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v172 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v173 = ttnn.to_device(v172, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v174 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 1024]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v175 = ttnn.to_device(v174, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v176 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v177 = ttnn.to_device(v176, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v178 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v179 = ttnn.to_device(v178, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v180 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v181 = ttnn.to_device(v180, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v182 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v183 = ttnn.to_device(v182, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v184 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v185 = ttnn.to_device(v184, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v186 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v187 = ttnn.to_device(v186, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v188 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v189 = ttnn.to_device(v188, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v190 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v191 = ttnn.to_device(v190, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v192 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v193 = ttnn.to_device(v192, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v194 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v195 = ttnn.to_device(v194, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v196 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v197 = ttnn.to_device(v196, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v198 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v199 = ttnn.to_device(v198, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v200 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v201 = ttnn.to_device(v200, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v202 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v203 = ttnn.to_device(v202, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v204 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v205 = ttnn.to_device(v204, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v206 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v207 = ttnn.to_device(v206, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v208 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v209 = ttnn.to_device(v208, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v210 = ttnn.ones(shape=ttnn.Shape([1, 1, 1, 512]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v211 = ttnn.to_device(v210, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v212 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v213 = ttnn.to_device(v212, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v214 = ttnn.ones(shape=ttnn.Shape([1, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v215 = ttnn.to_device(v214, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v216 = ttnn.ones(shape=ttnn.Shape([64, 3, 7, 7]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v217 = ttnn.ones(shape=ttnn.Shape([64, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v218 = ttnn.ones(shape=ttnn.Shape([64, 64, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v219 = ttnn.ones(shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v220 = ttnn.ones(shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v221 = ttnn.ones(shape=ttnn.Shape([64, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v222 = ttnn.ones(shape=ttnn.Shape([64, 64, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v223 = ttnn.ones(shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v224 = ttnn.ones(shape=ttnn.Shape([64, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v225 = ttnn.ones(shape=ttnn.Shape([64, 64, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v226 = ttnn.ones(shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v227 = ttnn.ones(shape=ttnn.Shape([128, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v228 = ttnn.ones(shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v229 = ttnn.ones(shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v230 = ttnn.ones(shape=ttnn.Shape([512, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v231 = ttnn.ones(shape=ttnn.Shape([128, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v232 = ttnn.ones(shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v233 = ttnn.ones(shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v234 = ttnn.ones(shape=ttnn.Shape([128, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v235 = ttnn.ones(shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v236 = ttnn.ones(shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v237 = ttnn.ones(shape=ttnn.Shape([128, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v238 = ttnn.ones(shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v239 = ttnn.ones(shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v240 = ttnn.ones(shape=ttnn.Shape([256, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v241 = ttnn.ones(shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v242 = ttnn.ones(shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v243 = ttnn.ones(shape=ttnn.Shape([1024, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v244 = ttnn.ones(shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v245 = ttnn.ones(shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v246 = ttnn.ones(shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v247 = ttnn.ones(shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v248 = ttnn.ones(shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v249 = ttnn.ones(shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v250 = ttnn.ones(shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v251 = ttnn.ones(shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v252 = ttnn.ones(shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v253 = ttnn.ones(shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v254 = ttnn.ones(shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v255 = ttnn.ones(shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v256 = ttnn.ones(shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v257 = ttnn.ones(shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v258 = ttnn.ones(shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v259 = ttnn.ones(shape=ttnn.Shape([512, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v260 = ttnn.ones(shape=ttnn.Shape([512, 512, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v261 = ttnn.ones(shape=ttnn.Shape([2048, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v262 = ttnn.ones(shape=ttnn.Shape([2048, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v263 = ttnn.ones(shape=ttnn.Shape([512, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v264 = ttnn.ones(shape=ttnn.Shape([512, 512, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v265 = ttnn.ones(shape=ttnn.Shape([2048, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v266 = ttnn.ones(shape=ttnn.Shape([512, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v267 = ttnn.ones(shape=ttnn.Shape([512, 512, 3, 3]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v268 = ttnn.ones(shape=ttnn.Shape([2048, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.ROW_MAJOR, device=None)
  v269 = ttnn.ones(shape=ttnn.Shape([2048, 1000]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v270 = ttnn.to_device(v269, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v271 = ttnn.ones(shape=ttnn.Shape([1000]), dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE, device=None)
  v272 = ttnn.to_device(v271, device=v1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
  v273 = [v3, v5, v7, v9, v11, v13, v15, v17, v19, v21, v23, v25, v27, v29, v31, v33, v35, v37, v39, v41, v43, v45, v47, v49, v51, v53, v55, v57, v59, v61, v63, v65, v67, v69, v71, v73, v75, v77, v79, v81, v83, v85, v87, v89, v91, v93, v95, v97, v99, v101, v103, v105, v107, v109, v111, v113, v115, v117, v119, v121, v123, v125, v127, v129, v131, v133, v135, v137, v139, v141, v143, v145, v147, v149, v151, v153, v155, v157, v159, v161, v163, v165, v167, v169, v171, v173, v175, v177, v179, v181, v183, v185, v187, v189, v191, v193, v195, v197, v199, v201, v203, v205, v207, v209, v211, v213, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226, v227, v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241, v242, v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255, v256, v257, v258, v259, v260, v261, v262, v263, v264, v265, v266, v267, v268, v270, v272]
  return v273


def measure(inputs, loop_count):

  batch_size = inputs[0].shape[0]

  start = time.time()
  for _ in range(loop_count):
    outputs = forward(inputs)
    for tensor in outputs:
      ttnn.from_device(tensor, True)
  end = time.time()

  verbose(inputs, outputs, loop_count, batch_size, start, end)


def verbose(inputs, outputs, loop_count, batch_size, start, end):

  samples = loop_count * batch_size
  exec_time = end - start

  print(50 * "=")
  print(f"First Warmup Run: ")
  print(50 * "=")
  print(f"Type of inputs: {type(inputs)}")
  print(f"Number of inputs: {len(inputs)}")
  print(f"Type of outputs: {type(outputs)}")
  print(f"Number of outputs: {len(outputs)}")
  print(f"Number of samples: {samples}")
  print(f"Batch size: {batch_size}")
  print(f"Loop count: {loop_count}")
  print(f"Execution time: {exec_time}")
  print(f"Samples per second: {samples / exec_time}")
  print(50 * "=")


def main():
  
  v1 = create_inputs_for_forward()

  # First Warmup Run
  measure(v1, loop_count=1)

  # Next 4 Warmup Runs
  measure(v1, loop_count=4)

  # Next LOOP-COUNT Runs
  measure(v1, loop_count=16)

  v3 = 0
  return v3

if __name__ == '__main__':
  main()


