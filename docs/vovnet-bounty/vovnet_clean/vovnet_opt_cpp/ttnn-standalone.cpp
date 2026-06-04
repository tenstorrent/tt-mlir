#include "ttnn-precompiled.hpp"
::std::vector<::ttnn::Tensor>
main_const_eval_0(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ::ttnn::Tensor v4 = v1[2];
  ::ttnn::Tensor v5 = v1[3];
  ::ttnn::Tensor v6 = v1[4];
  ::ttnn::Tensor v7 = v1[5];
  ::ttnn::Tensor v8 = v1[6];
  ::ttnn::Tensor v9 = v1[7];
  ::ttnn::Tensor v10 = v1[8];
  ::ttnn::Tensor v11 = v1[9];
  ::ttnn::Tensor v12 = v1[10];
  ::ttnn::Tensor v13 = v1[11];
  ::ttnn::Tensor v14 = v1[12];
  ::ttnn::Tensor v15 = v1[13];
  ::ttnn::Tensor v16 = v1[14];
  ::ttnn::Tensor v17 = v1[15];
  ::ttnn::Tensor v18 = v1[16];
  ::ttnn::Tensor v19 = v1[17];
  ::ttnn::Tensor v20 = v1[18];
  ::ttnn::Tensor v21 = v1[19];
  ::ttnn::Tensor v22 = v1[20];
  ::ttnn::Tensor v23 = v1[21];
  ::ttnn::Tensor v24 = v1[22];
  ::ttnn::Tensor v25 = v1[23];
  ::ttnn::Tensor v26 = v1[24];
  ::ttnn::Tensor v27 = v1[25];
  ::ttnn::Tensor v28 = v1[26];
  ::ttnn::Tensor v29 = v1[27];
  ::ttnn::Tensor v30 = v1[28];
  ::ttnn::Tensor v31 = v1[29];
  ::ttnn::Tensor v32 = v1[30];
  ::ttnn::Tensor v33 = v1[31];
  ::ttnn::Tensor v34 = v1[32];
  ::ttnn::Tensor v35 = v1[33];
  ::ttnn::Tensor v36 = v1[34];
  ::ttnn::Tensor v37 = v1[35];
  ::ttnn::Tensor v38 = v1[36];
  ::ttnn::Tensor v39 = v1[37];
  ::ttnn::Tensor v40 = v1[38];
  ::ttnn::Tensor v41 = v1[39];
  ::ttnn::Tensor v42 = v1[40];
  ::ttnn::Tensor v43 = v1[41];
  ::ttnn::Tensor v44 = v1[42];
  ::ttnn::Tensor v45 = v1[43];
  ::ttnn::Tensor v46 = v1[44];
  ::ttnn::Tensor v47 = v1[45];
  ::ttnn::Tensor v48 = v1[46];
  ::ttnn::Tensor v49 = v1[47];
  ::ttnn::Tensor v50 = v1[48];
  ::ttnn::Tensor v51 = v1[49];
  ::ttnn::Tensor v52 = v1[50];
  ::ttnn::Tensor v53 = v1[51];
  ::ttnn::Tensor v54 = v1[52];
  ::ttnn::Tensor v55 = v1[53];
  ::ttnn::Tensor v56 = v1[54];
  ::ttnn::Tensor v57 = v1[55];
  ::ttnn::Tensor v58 = v1[56];
  ::ttnn::Tensor v59 = v1[57];
  ::ttnn::Tensor v60 = v1[58];
  ::ttnn::Tensor v61 = v1[59];
  ::ttnn::Tensor v62 = v1[60];
  ::ttnn::Tensor v63 = v1[61];
  ::ttnn::Tensor v64 = v1[62];
  ::ttnn::Tensor v65 = v1[63];
  ::ttnn::Tensor v66 = v1[64];
  ::ttnn::Tensor v67 = v1[65];
  ::ttnn::Tensor v68 = v1[66];
  ::ttnn::Tensor v69 = v1[67];
  ::ttnn::Tensor v70 = v1[68];
  ::ttnn::Tensor v71 = v1[69];
  ::ttnn::Tensor v72 = v1[70];
  ::ttnn::Tensor v73 = v1[71];
  ::ttnn::Tensor v74 = v1[72];
  ::ttnn::Tensor v75 = v1[73];
  ::ttnn::Tensor v76 = v1[74];
  ::ttnn::Tensor v77 = v1[75];
  ::ttnn::Tensor v78 = v1[76];
  ::ttnn::Tensor v79 = v1[77];
  ::ttnn::Tensor v80 = v1[78];
  ::ttnn::Tensor v81 = v1[79];
  ::ttnn::Tensor v82 = v1[80];
  ::ttnn::Tensor v83 = v1[81];
  ::ttnn::Tensor v84 = v1[82];
  ::ttnn::Tensor v85 = v1[83];
  ::ttnn::Tensor v86 = v1[84];
  ::ttnn::Tensor v87 = v1[85];
  ::ttnn::Tensor v88 = v1[86];
  ::ttnn::Tensor v89 = v1[87];
  ::ttnn::Tensor v90 = v1[88];
  ::ttnn::Tensor v91 = v1[89];
  ::ttnn::Tensor v92 = v1[90];
  ::ttnn::Tensor v93 = v1[91];
  ::ttnn::Tensor v94 = v1[92];
  ::ttnn::Tensor v95 = v1[93];
  ::ttnn::Tensor v96 = v1[94];
  ::ttnn::Tensor v97 = v1[95];
  ::ttnn::Tensor v98 = v1[96];
  ::ttnn::Tensor v99 = v1[97];
  ::ttnn::Tensor v100 = v1[98];
  ::ttnn::Tensor v101 = v1[99];
  ::ttnn::Tensor v102 = v1[100];
  ::ttnn::Tensor v103 = v1[101];
  ::ttnn::Tensor v104 = v1[102];
  ::ttnn::Tensor v105 = v1[103];
  ::ttnn::Tensor v106 = v1[104];
  ::ttnn::Tensor v107 = v1[105];
  ::ttnn::Tensor v108 = v1[106];
  ::ttnn::Tensor v109 = v1[107];
  ::ttnn::Tensor v110 = v1[108];
  ::ttnn::Tensor v111 = v1[109];
  ::ttnn::Tensor v112 = v1[110];
  ::ttnn::Tensor v113 = v1[111];
  ::ttnn::Tensor v114 = v1[112];
  ::ttnn::Tensor v115 = v1[113];
  ::ttnn::Tensor v116 = v1[114];
  ttnn::distributed::MeshDevice *v117 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v118 = ttnn::typecast(
      v112, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v119 = ttnn::typecast(
      v115, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v120 = ttnn::typecast(
      v116, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v121 = ttnn::typecast(
      v113, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v122 = ttnn::typecast(
      v114, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v123 = ttnn::typecast(
      v107, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v124 = ttnn::typecast(
      v110, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v125 = ttnn::typecast(
      v111, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v126 = ttnn::typecast(
      v108, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v127 = ttnn::typecast(
      v109, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v128 = ttnn::typecast(
      v102, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v129 = ttnn::typecast(
      v105, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v130 = ttnn::typecast(
      v106, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v131 = ttnn::typecast(
      v103, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v132 = ttnn::typecast(
      v104, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v133 = ttnn::typecast(
      v97, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v134 = ttnn::typecast(
      v100, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v135 = ttnn::typecast(
      v101, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v136 = ttnn::typecast(
      v98, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v137 = ttnn::typecast(
      v99, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v138 = ttnn::typecast(
      v92, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v139 = ttnn::typecast(
      v95, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v140 = ttnn::typecast(
      v96, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v141 = ttnn::typecast(
      v93, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v142 = ttnn::typecast(
      v94, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v143 = ttnn::typecast(
      v87, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v144 = ttnn::typecast(
      v90, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v145 = ttnn::typecast(
      v91, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v146 = ttnn::typecast(
      v88, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v147 = ttnn::typecast(
      v89, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v148 = ttnn::typecast(
      v82, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v149 = ttnn::typecast(
      v85, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v150 = ttnn::typecast(
      v86, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v151 = ttnn::typecast(
      v83, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v152 = ttnn::typecast(
      v84, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v153 = ttnn::typecast(
      v77, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v154 = ttnn::typecast(
      v80, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v155 = ttnn::typecast(
      v81, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v156 = ttnn::typecast(
      v78, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v157 = ttnn::typecast(
      v79, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v158 = ttnn::typecast(
      v72, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v159 = ttnn::typecast(
      v75, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v160 = ttnn::typecast(
      v76, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v161 = ttnn::typecast(
      v73, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v162 = ttnn::typecast(
      v74, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v163 = ttnn::typecast(
      v67, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v164 = ttnn::typecast(
      v70, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v165 = ttnn::typecast(
      v71, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v166 = ttnn::typecast(
      v68, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v167 = ttnn::typecast(
      v69, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v168 = ttnn::typecast(
      v62, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v169 = ttnn::typecast(
      v65, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v170 = ttnn::typecast(
      v66, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v171 = ttnn::typecast(
      v63, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v172 = ttnn::typecast(
      v64, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v173 = ttnn::typecast(
      v57, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v174 = ttnn::typecast(
      v60, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v175 = ttnn::typecast(
      v61, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v176 = ttnn::typecast(
      v58, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v177 = ttnn::typecast(
      v59, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v178 = ttnn::typecast(
      v52, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v179 = ttnn::typecast(
      v55, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v180 = ttnn::typecast(
      v56, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v181 = ttnn::typecast(
      v53, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v182 = ttnn::typecast(
      v54, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v183 = ttnn::typecast(
      v47, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v184 = ttnn::typecast(
      v50, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v185 = ttnn::typecast(
      v51, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v186 = ttnn::typecast(
      v48, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v187 = ttnn::typecast(
      v49, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v188 = ttnn::typecast(
      v42, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v189 = ttnn::typecast(
      v45, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v190 = ttnn::typecast(
      v46, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v191 = ttnn::typecast(
      v43, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v192 = ttnn::typecast(
      v44, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v193 = ttnn::typecast(
      v37, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v194 = ttnn::typecast(
      v40, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v195 = ttnn::typecast(
      v41, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v196 = ttnn::typecast(
      v38, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v197 = ttnn::typecast(
      v39, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v198 = ttnn::typecast(
      v32, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v199 = ttnn::typecast(
      v35, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v200 = ttnn::typecast(
      v36, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v201 = ttnn::typecast(
      v33, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v202 = ttnn::typecast(
      v34, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v203 = ttnn::typecast(
      v27, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v204 = ttnn::typecast(
      v30, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v205 = ttnn::typecast(
      v31, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v206 = ttnn::typecast(
      v28, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v207 = ttnn::typecast(
      v29, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v208 = ttnn::typecast(
      v22, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v209 = ttnn::typecast(
      v25, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v210 = ttnn::typecast(
      v26, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v211 = ttnn::typecast(
      v23, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v212 = ttnn::typecast(
      v24, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v213 = ttnn::typecast(
      v17, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v214 = ttnn::typecast(
      v20, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v215 = ttnn::typecast(
      v21, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v216 = ttnn::typecast(
      v18, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v217 = ttnn::typecast(
      v19, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v218 = ttnn::typecast(
      v12, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v219 = ttnn::typecast(
      v15, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v220 = ttnn::typecast(
      v16, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v221 = ttnn::typecast(
      v13, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v222 = ttnn::typecast(
      v14, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v223 = ttnn::typecast(
      v7, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v224 = ttnn::typecast(
      v10, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v225 = ttnn::typecast(
      v11, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v226 = ttnn::typecast(
      v8, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v227 = ttnn::typecast(
      v9, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v228 = ttnn::typecast(
      v2, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v229 = ttnn::typecast(
      v5, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v230 = ttnn::typecast(
      v6, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v231 = ttnn::typecast(
      v3, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v232 = ttnn::typecast(
      v4, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v233;
  ::ttnn::Tensor v234;
  ::ttnn::Tensor v235;
  ::ttnn::Tensor v236;
  ::ttnn::Tensor v237;
  ::ttnn::Tensor v238;
  ::ttnn::Tensor v239;
  ::ttnn::Tensor v240;
  ::ttnn::Tensor v241;
  ::ttnn::Tensor v242;
  ::ttnn::Tensor v243;
  ::ttnn::Tensor v244;
  ::ttnn::Tensor v245;
  ::ttnn::Tensor v246;
  ::ttnn::Tensor v247;
  ::ttnn::Tensor v248;
  ::ttnn::Tensor v249;
  ::ttnn::Tensor v250;
  ::ttnn::Tensor v251;
  ::ttnn::Tensor v252;
  ::ttnn::Tensor v253;
  ::ttnn::Tensor v254;
  ::ttnn::Tensor v255;
  ::ttnn::Tensor v256;
  ::ttnn::Tensor v257;
  ::ttnn::Tensor v258;
  ::ttnn::Tensor v259;
  ::ttnn::Tensor v260;
  ::ttnn::Tensor v261;
  ::ttnn::Tensor v262;
  ::ttnn::Tensor v263;
  ::ttnn::Tensor v264;
  ::ttnn::Tensor v265;
  ::ttnn::Tensor v266;
  ::ttnn::Tensor v267;
  ::ttnn::Tensor v268;
  ::ttnn::Tensor v269;
  ::ttnn::Tensor v270;
  ::ttnn::Tensor v271;
  ::ttnn::Tensor v272;
  ::ttnn::Tensor v273;
  ::ttnn::Tensor v274;
  ::ttnn::Tensor v275;
  ::ttnn::Tensor v276;
  ::ttnn::Tensor v277;
  ::ttnn::Tensor v278;
  std::tie(v233, v234, v235, v236, v237, v238, v239, v240, v241, v242, v243,
           v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254,
           v255, v256, v257, v258, v259, v260, v261, v262, v263, v264, v265,
           v266, v267, v268, v269, v270, v271, v272, v273, v274, v275, v276,
           v277, v278) =
      cpu_hoisted_const_eval_b2f462d7(
          v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128,
          v129, v130, v131, v132, v133, v134, v135, v136, v137, v138, v139,
          v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150,
          v151, v152, v153, v154, v155, v156, v157, v158, v159, v160, v161,
          v162, v163, v164, v165, v166, v167, v168, v169, v170, v171, v172,
          v173, v174, v175, v176, v177, v178, v179, v180, v181, v182, v183,
          v184, v185, v186, v187, v188, v189, v190, v191, v192, v193, v194,
          v195, v196, v197, v198, v199, v200, v201, v202, v203, v204, v205,
          v206, v207, v208, v209, v210, v211, v212, v213, v214, v215, v216,
          v217, v218, v219, v220, v221, v222, v223, v224, v225, v226, v227,
          v228, v229, v230, v231, v232);
  ttnn::deallocate(v232, false);
  ttnn::deallocate(v231, false);
  ttnn::deallocate(v230, false);
  ttnn::deallocate(v229, false);
  ttnn::deallocate(v228, false);
  ttnn::deallocate(v227, false);
  ttnn::deallocate(v226, false);
  ttnn::deallocate(v225, false);
  ttnn::deallocate(v224, false);
  ttnn::deallocate(v223, false);
  ttnn::deallocate(v222, false);
  ttnn::deallocate(v221, false);
  ttnn::deallocate(v220, false);
  ttnn::deallocate(v219, false);
  ttnn::deallocate(v218, false);
  ttnn::deallocate(v217, false);
  ttnn::deallocate(v216, false);
  ttnn::deallocate(v215, false);
  ttnn::deallocate(v214, false);
  ttnn::deallocate(v213, false);
  ttnn::deallocate(v212, false);
  ttnn::deallocate(v211, false);
  ttnn::deallocate(v210, false);
  ttnn::deallocate(v209, false);
  ttnn::deallocate(v208, false);
  ttnn::deallocate(v207, false);
  ttnn::deallocate(v206, false);
  ttnn::deallocate(v205, false);
  ttnn::deallocate(v204, false);
  ttnn::deallocate(v203, false);
  ttnn::deallocate(v202, false);
  ttnn::deallocate(v201, false);
  ttnn::deallocate(v200, false);
  ttnn::deallocate(v199, false);
  ttnn::deallocate(v198, false);
  ttnn::deallocate(v197, false);
  ttnn::deallocate(v196, false);
  ttnn::deallocate(v195, false);
  ttnn::deallocate(v194, false);
  ttnn::deallocate(v193, false);
  ttnn::deallocate(v192, false);
  ttnn::deallocate(v191, false);
  ttnn::deallocate(v190, false);
  ttnn::deallocate(v189, false);
  ttnn::deallocate(v188, false);
  ttnn::deallocate(v187, false);
  ttnn::deallocate(v186, false);
  ttnn::deallocate(v185, false);
  ttnn::deallocate(v184, false);
  ttnn::deallocate(v183, false);
  ttnn::deallocate(v182, false);
  ttnn::deallocate(v181, false);
  ttnn::deallocate(v180, false);
  ttnn::deallocate(v179, false);
  ttnn::deallocate(v178, false);
  ttnn::deallocate(v177, false);
  ttnn::deallocate(v176, false);
  ttnn::deallocate(v175, false);
  ttnn::deallocate(v174, false);
  ttnn::deallocate(v173, false);
  ttnn::deallocate(v172, false);
  ttnn::deallocate(v171, false);
  ttnn::deallocate(v170, false);
  ttnn::deallocate(v169, false);
  ttnn::deallocate(v168, false);
  ttnn::deallocate(v167, false);
  ttnn::deallocate(v166, false);
  ttnn::deallocate(v165, false);
  ttnn::deallocate(v164, false);
  ttnn::deallocate(v163, false);
  ttnn::deallocate(v162, false);
  ttnn::deallocate(v161, false);
  ttnn::deallocate(v160, false);
  ttnn::deallocate(v159, false);
  ttnn::deallocate(v158, false);
  ttnn::deallocate(v157, false);
  ttnn::deallocate(v156, false);
  ttnn::deallocate(v155, false);
  ttnn::deallocate(v154, false);
  ttnn::deallocate(v153, false);
  ttnn::deallocate(v152, false);
  ttnn::deallocate(v151, false);
  ttnn::deallocate(v150, false);
  ttnn::deallocate(v149, false);
  ttnn::deallocate(v148, false);
  ttnn::deallocate(v147, false);
  ttnn::deallocate(v146, false);
  ttnn::deallocate(v145, false);
  ttnn::deallocate(v144, false);
  ttnn::deallocate(v143, false);
  ttnn::deallocate(v142, false);
  ttnn::deallocate(v141, false);
  ttnn::deallocate(v140, false);
  ttnn::deallocate(v139, false);
  ttnn::deallocate(v138, false);
  ttnn::deallocate(v137, false);
  ttnn::deallocate(v136, false);
  ttnn::deallocate(v135, false);
  ttnn::deallocate(v134, false);
  ttnn::deallocate(v133, false);
  ttnn::deallocate(v132, false);
  ttnn::deallocate(v131, false);
  ttnn::deallocate(v130, false);
  ttnn::deallocate(v129, false);
  ttnn::deallocate(v128, false);
  ttnn::deallocate(v127, false);
  ttnn::deallocate(v126, false);
  ttnn::deallocate(v125, false);
  ttnn::deallocate(v124, false);
  ttnn::deallocate(v123, false);
  ttnn::deallocate(v122, false);
  ttnn::deallocate(v121, false);
  ttnn::deallocate(v120, false);
  ttnn::deallocate(v119, false);
  ttnn::deallocate(v118, false);
  ::ttnn::Tensor v279 = ttnn::typecast(
      v233, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v233, false);
  ::ttnn::Tensor v280 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v279,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt},
      ::ttnn::Layout::TILE, "OIHW", 3, 64, 1, 224, 224,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 32,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v279, false);
  ::ttnn::Tensor v281 = ttnn::typecast(
      v234, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v234, false);
  ::ttnn::Tensor v282 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v281,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt},
      ::ttnn::Layout::TILE, 3, 64, 1, 224, 224, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, v117, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 32,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v281, false);
  ::ttnn::Tensor v283 = ttnn::typecast(
      v235, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v235, false);
  ::ttnn::Tensor v284 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v283,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 6}}}},
                            ::std::array<uint32_t, 2>{224, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 64, 64, 1, 112, 112,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v283, false);
  ::ttnn::Tensor v285 = ttnn::typecast(
      v236, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v236, false);
  ::ttnn::Tensor v286 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v285,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 6}}}},
                            ::std::array<uint32_t, 2>{224, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 64, 64, 1, 112, 112,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v285, false);
  ::ttnn::Tensor v287 = ttnn::typecast(
      v237, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v237, false);
  ::ttnn::Tensor v288 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v287,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 64, 64, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v287, false);
  ::ttnn::Tensor v289 = ttnn::typecast(
      v238, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v238, false);
  ::ttnn::Tensor v290 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v289,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 64, 64, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v117, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v289, false);
  ::ttnn::Tensor v291 = ttnn::typecast(
      v239, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v239, false);
  ::ttnn::Tensor v292 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v291,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 64, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v291, false);
  ::ttnn::Tensor v293 = ttnn::typecast(
      v240, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v240, false);
  ::ttnn::Tensor v294 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v293,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 64, 128, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v117, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v293, false);
  ::ttnn::Tensor v295 = ttnn::typecast(
      v241, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v241, false);
  ::ttnn::Tensor v296 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v295,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v295, false);
  ::ttnn::Tensor v297 = ttnn::typecast(
      v242, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v242, false);
  ::ttnn::Tensor v298 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v297,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v297, false);
  ::ttnn::Tensor v299 = ttnn::typecast(
      v243, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v243, false);
  ::ttnn::Tensor v300 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v299,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v299, false);
  ::ttnn::Tensor v301 = ttnn::typecast(
      v244, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v244, false);
  ::ttnn::Tensor v302 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v301,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v301, false);
  ::ttnn::Tensor v303 = ttnn::typecast(
      v245, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v245, false);
  ::ttnn::Tensor v304 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v303,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v303, false);
  ::ttnn::Tensor v305 = ttnn::typecast(
      v246, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v246, false);
  ::ttnn::Tensor v306 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v305,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v305, false);
  ::ttnn::Tensor v307 = ttnn::typecast(
      v247, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v247, false);
  ::ttnn::Tensor v308 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v307,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 448},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 448, 256, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v307, false);
  ::ttnn::Tensor v309 = ttnn::typecast(
      v248, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v248, false);
  ::ttnn::Tensor v310 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v309,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 448},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 448, 256, 1, 56, 56,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v309, false);
  ::ttnn::Tensor v311 = ttnn::typecast(
      v249, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v249, false);
  ::ttnn::Tensor v312 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v311,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{16, 256},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::ROW_MAJOR, "OIHW", 256, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v311, false);
  ::ttnn::Tensor v313 = ttnn::typecast(
      v250, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v250, false);
  ::ttnn::Tensor v314 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v313,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{16, 256},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::ROW_MAJOR, 256, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v313, false);
  ::ttnn::Tensor v315 = ttnn::typecast(
      v251, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v251, false);
  ::ttnn::Tensor v316 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v315,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v315, false);
  ::ttnn::Tensor v317 = ttnn::typecast(
      v252, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v252, false);
  ::ttnn::Tensor v318 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v317,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v317, false);
  ::ttnn::Tensor v319 = ttnn::typecast(
      v253, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v253, false);
  ::ttnn::Tensor v320 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v319,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v319, false);
  ::ttnn::Tensor v321 = ttnn::typecast(
      v254, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v254, false);
  ::ttnn::Tensor v322 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v321,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v321, false);
  ::ttnn::Tensor v323 = ttnn::typecast(
      v255, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v255, false);
  ::ttnn::Tensor v324 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v323,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v323, false);
  ::ttnn::Tensor v325 = ttnn::typecast(
      v256, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v256, false);
  ::ttnn::Tensor v326 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v325,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v325, false);
  ::ttnn::Tensor v327 = ttnn::typecast(
      v257, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v257, false);
  ::ttnn::Tensor v328 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v327,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 736},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 736, 512, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 64,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v327, false);
  ::ttnn::Tensor v329 = ttnn::typecast(
      v258, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v258, false);
  ::ttnn::Tensor v330 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v329,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 736},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 736, 512, 1, 28, 28,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 64,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v329, false);
  ::ttnn::Tensor v331 = ttnn::typecast(
      v259, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v259, false);
  ::ttnn::Tensor v332 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v331,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::Layout::ROW_MAJOR, "OIHW", 512, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .config_tensors_in_dram = true,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v331, false);
  ::ttnn::Tensor v333 = ttnn::typecast(
      v260, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v260, false);
  ::ttnn::Tensor v334 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v333,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::Layout::ROW_MAJOR, 512, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .config_tensors_in_dram = true,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v333, false);
  ::ttnn::Tensor v335 = ttnn::typecast(
      v261, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v261, false);
  ::ttnn::Tensor v336 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v335,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v335, false);
  ::ttnn::Tensor v337 = ttnn::typecast(
      v262, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v262, false);
  ::ttnn::Tensor v338 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v337,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v337, false);
  ::ttnn::Tensor v339 = ttnn::typecast(
      v263, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v263, false);
  ::ttnn::Tensor v340 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v339,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v339, false);
  ::ttnn::Tensor v341 = ttnn::typecast(
      v264, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v264, false);
  ::ttnn::Tensor v342 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v341,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v341, false);
  ::ttnn::Tensor v343 = ttnn::typecast(
      v265, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v265, false);
  ::ttnn::Tensor v344 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v343,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v343, false);
  ::ttnn::Tensor v345 = ttnn::typecast(
      v266, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v266, false);
  ::ttnn::Tensor v346 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v345,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v345, false);
  ::ttnn::Tensor v347 = ttnn::typecast(
      v267, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v267, false);
  ::ttnn::Tensor v348 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v347,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 0}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 1},
                                                  ::ttnn::CoreCoord{3, 1}}}},
                            ::std::array<uint32_t, 2>{224, 96},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 1088, 768, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v347, false);
  ::ttnn::Tensor v349 = ttnn::typecast(
      v268, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v268, false);
  ::ttnn::Tensor v350 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v349,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 0}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 1},
                                                  ::ttnn::CoreCoord{3, 1}}}},
                            ::std::array<uint32_t, 2>{224, 96},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 1088, 768, 1, 14, 14,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v349, false);
  ::ttnn::Tensor v351 = ttnn::typecast(
      v269, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v269, false);
  ::ttnn::Tensor v352 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v351,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{49, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::ROW_MAJOR, "OIHW", 768, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v351, false);
  ::ttnn::Tensor v353 = ttnn::typecast(
      v270, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v270, false);
  ::ttnn::Tensor v354 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v353,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{49, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::ROW_MAJOR, 768, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v353, false);
  ::ttnn::Tensor v355 = ttnn::typecast(
      v271, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v271, false);
  ::ttnn::Tensor v356 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v355,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 224, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v355, false);
  ::ttnn::Tensor v357 = ttnn::typecast(
      v272, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v272, false);
  ::ttnn::Tensor v358 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v357,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v117, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v357, false);
  ::ttnn::Tensor v359 = ttnn::typecast(
      v273, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v273, false);
  ::ttnn::Tensor v360 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v359,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 224, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v359, false);
  ::ttnn::Tensor v361 = ttnn::typecast(
      v274, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v274, false);
  ::ttnn::Tensor v362 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v361,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v117, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v361, false);
  ::ttnn::Tensor v363 = ttnn::typecast(
      v275, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v275, false);
  ::ttnn::Tensor v364 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v363,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 224, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v363, false);
  ::ttnn::Tensor v365 = ttnn::typecast(
      v276, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v276, false);
  ::ttnn::Tensor v366 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v365,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v117, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v365, false);
  ::ttnn::Tensor v367 = ttnn::typecast(
      v277, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v277, false);
  ::ttnn::Tensor v368 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v367,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 1}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 2},
                                                  ::ttnn::CoreCoord{6, 2}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 1440, 1024, 1, 7, 7,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v367, false);
  ::ttnn::Tensor v369 = ttnn::typecast(
      v278, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v278, false);
  ::ttnn::Tensor v370 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v369,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 1}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 2},
                                                  ::ttnn::CoreCoord{6, 2}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 1440, 1024, 1, 7, 7,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v117, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v369, false);
  ::std::vector<::ttnn::Tensor> v371 = util_create_vec(
      v280, v282, v284, v286, v288, v290, v292, v294, v296, v298, v300, v302,
      v304, v306, v308, v310, v312, v314, v316, v318, v320, v322, v324, v326,
      v328, v330, v332, v334, v336, v338, v340, v342, v344, v346, v348, v350,
      v352, v354, v356, v358, v360, v362, v364, v366, v368, v370);
  return v371;
}
::std::vector<::ttnn::Tensor>
main_const_eval_1(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::typecast(
      v2, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = cpu_hoisted_const_eval_b3a3bf24(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::to_layout(
      v5, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_device(
      v6, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}
::std::vector<::ttnn::Tensor>
main_const_eval_2(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 6}}}},
                            ::std::array<uint32_t, 2>{224, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 64, 64, 1, 112, 112,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 64, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_3(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 224, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 224, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_4(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 256, 256, 1, 1, 1,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_5(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 128, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_6(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 1024, 1024, 1, 1, 1,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_7(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 512, 512, 1, 1, 1,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_8(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::typecast(
      v2, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = cpu_hoisted_const_eval_24665087(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::typecast(
      v5, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v6,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 512, 512, 1, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}
::std::vector<::ttnn::Tensor>
main_const_eval_9(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 160, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_10(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 224, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 224, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_11(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 128, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_12(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 192, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_13(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{256, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 64, 64, 1, 112, 112,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 64, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_14(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 128, 128, 1, 56, 56,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 128, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_15(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::typecast(
      v2, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = cpu_hoisted_const_eval_8b0e4be0(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::typecast(
      v5, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v6,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 1024, 1024, 1, 1, 1,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1,
      v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}
::std::vector<::ttnn::Tensor>
main_const_eval_16(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 160, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_17(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::typecast(
      v2, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = cpu_hoisted_const_eval_382fc5dd(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::to_layout(
      v5, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_device(
      v6, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}
::std::vector<::ttnn::Tensor>
main_const_eval_18(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 192, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 64,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_19(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 192, 192, 1, 14, 14,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 192, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_20(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 768, 768, 1, 1, 1,
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1},
      true, 1, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_21(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::typecast(
      v2, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = cpu_hoisted_const_eval_5c372694(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::typecast(
      v5, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v6,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 768, 768, 1, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}
::std::vector<::ttnn::Tensor>
main_const_eval_22(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::typecast(
      v2, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = cpu_hoisted_const_eval_8b649ce3(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::typecast(
      v5, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::operations::conv::conv2d::prepare_conv_bias(
      v6,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, 256, 256, 1, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT16,
      ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt);
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}
::std::vector<::ttnn::Tensor>
main_const_eval_23(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 224, 224, 1, 7, 7,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 224, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
::std::vector<::ttnn::Tensor>
main_const_eval_24(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(
      v2,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Layout::TILE, "OIHW", 160, 160, 1, 28, 28,
      ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1},
      false, 160, v3, ::ttnn::DataType::BFLOAT16, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_0;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_1;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_2;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_3;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_4;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_5;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_6;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_7;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_8;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_9;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_10;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_11;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_12;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_13;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_14;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_15;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_16;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_17;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_18;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_19;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_20;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_21;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_22;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_23;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_24;
::std::vector<::ttnn::Tensor> _main(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ::ttnn::Tensor v4 = v1[2];
  ::ttnn::Tensor v5 = v1[3];
  ::ttnn::Tensor v6 = v1[4];
  ::ttnn::Tensor v7 = v1[5];
  ::ttnn::Tensor v8 = v1[6];
  ::ttnn::Tensor v9 = v1[7];
  ::ttnn::Tensor v10 = v1[8];
  ::ttnn::Tensor v11 = v1[9];
  ::ttnn::Tensor v12 = v1[10];
  ::ttnn::Tensor v13 = v1[11];
  ::ttnn::Tensor v14 = v1[12];
  ::ttnn::Tensor v15 = v1[13];
  ::ttnn::Tensor v16 = v1[14];
  ::ttnn::Tensor v17 = v1[15];
  ::ttnn::Tensor v18 = v1[16];
  ::ttnn::Tensor v19 = v1[17];
  ::ttnn::Tensor v20 = v1[18];
  ::ttnn::Tensor v21 = v1[19];
  ::ttnn::Tensor v22 = v1[20];
  ::ttnn::Tensor v23 = v1[21];
  ::ttnn::Tensor v24 = v1[22];
  ::ttnn::Tensor v25 = v1[23];
  ::ttnn::Tensor v26 = v1[24];
  ::ttnn::Tensor v27 = v1[25];
  ::ttnn::Tensor v28 = v1[26];
  ::ttnn::Tensor v29 = v1[27];
  ::ttnn::Tensor v30 = v1[28];
  ::ttnn::Tensor v31 = v1[29];
  ::ttnn::Tensor v32 = v1[30];
  ::ttnn::Tensor v33 = v1[31];
  ::ttnn::Tensor v34 = v1[32];
  ::ttnn::Tensor v35 = v1[33];
  ::ttnn::Tensor v36 = v1[34];
  ::ttnn::Tensor v37 = v1[35];
  ::ttnn::Tensor v38 = v1[36];
  ::ttnn::Tensor v39 = v1[37];
  ::ttnn::Tensor v40 = v1[38];
  ::ttnn::Tensor v41 = v1[39];
  ::ttnn::Tensor v42 = v1[40];
  ::ttnn::Tensor v43 = v1[41];
  ::ttnn::Tensor v44 = v1[42];
  ::ttnn::Tensor v45 = v1[43];
  ::ttnn::Tensor v46 = v1[44];
  ::ttnn::Tensor v47 = v1[45];
  ::ttnn::Tensor v48 = v1[46];
  ::ttnn::Tensor v49 = v1[47];
  ::ttnn::Tensor v50 = v1[48];
  ::ttnn::Tensor v51 = v1[49];
  ::ttnn::Tensor v52 = v1[50];
  ::ttnn::Tensor v53 = v1[51];
  ::ttnn::Tensor v54 = v1[52];
  ::ttnn::Tensor v55 = v1[53];
  ::ttnn::Tensor v56 = v1[54];
  ::ttnn::Tensor v57 = v1[55];
  ::ttnn::Tensor v58 = v1[56];
  ::ttnn::Tensor v59 = v1[57];
  ::ttnn::Tensor v60 = v1[58];
  ::ttnn::Tensor v61 = v1[59];
  ::ttnn::Tensor v62 = v1[60];
  ::ttnn::Tensor v63 = v1[61];
  ::ttnn::Tensor v64 = v1[62];
  ::ttnn::Tensor v65 = v1[63];
  ::ttnn::Tensor v66 = v1[64];
  ::ttnn::Tensor v67 = v1[65];
  ::ttnn::Tensor v68 = v1[66];
  ::ttnn::Tensor v69 = v1[67];
  ::ttnn::Tensor v70 = v1[68];
  ::ttnn::Tensor v71 = v1[69];
  ::ttnn::Tensor v72 = v1[70];
  ::ttnn::Tensor v73 = v1[71];
  ::ttnn::Tensor v74 = v1[72];
  ::ttnn::Tensor v75 = v1[73];
  ::ttnn::Tensor v76 = v1[74];
  ::ttnn::Tensor v77 = v1[75];
  ::ttnn::Tensor v78 = v1[76];
  ::ttnn::Tensor v79 = v1[77];
  ::ttnn::Tensor v80 = v1[78];
  ::ttnn::Tensor v81 = v1[79];
  ::ttnn::Tensor v82 = v1[80];
  ::ttnn::Tensor v83 = v1[81];
  ::ttnn::Tensor v84 = v1[82];
  ::ttnn::Tensor v85 = v1[83];
  ::ttnn::Tensor v86 = v1[84];
  ::ttnn::Tensor v87 = v1[85];
  ::ttnn::Tensor v88 = v1[86];
  ::ttnn::Tensor v89 = v1[87];
  ::ttnn::Tensor v90 = v1[88];
  ::ttnn::Tensor v91 = v1[89];
  ::ttnn::Tensor v92 = v1[90];
  ::ttnn::Tensor v93 = v1[91];
  ::ttnn::Tensor v94 = v1[92];
  ::ttnn::Tensor v95 = v1[93];
  ::ttnn::Tensor v96 = v1[94];
  ::ttnn::Tensor v97 = v1[95];
  ::ttnn::Tensor v98 = v1[96];
  ::ttnn::Tensor v99 = v1[97];
  ::ttnn::Tensor v100 = v1[98];
  ::ttnn::Tensor v101 = v1[99];
  ::ttnn::Tensor v102 = v1[100];
  ::ttnn::Tensor v103 = v1[101];
  ::ttnn::Tensor v104 = v1[102];
  ::ttnn::Tensor v105 = v1[103];
  ::ttnn::Tensor v106 = v1[104];
  ::ttnn::Tensor v107 = v1[105];
  ::ttnn::Tensor v108 = v1[106];
  ::ttnn::Tensor v109 = v1[107];
  ::ttnn::Tensor v110 = v1[108];
  ::ttnn::Tensor v111 = v1[109];
  ::ttnn::Tensor v112 = v1[110];
  ::ttnn::Tensor v113 = v1[111];
  ::ttnn::Tensor v114 = v1[112];
  ::ttnn::Tensor v115 = v1[113];
  ::ttnn::Tensor v116 = v1[114];
  ::ttnn::Tensor v117 = v1[115];
  ::ttnn::Tensor v118 = v1[116];
  ::ttnn::Tensor v119 = v1[117];
  ::ttnn::Tensor v120 = v1[118];
  ::ttnn::Tensor v121 = v1[119];
  ::ttnn::Tensor v122 = v1[120];
  ::ttnn::Tensor v123 = v1[121];
  ::ttnn::Tensor v124 = v1[122];
  ::ttnn::Tensor v125 = v1[123];
  ::ttnn::Tensor v126 = v1[124];
  ::ttnn::Tensor v127 = v1[125];
  ::ttnn::Tensor v128 = v1[126];
  ::ttnn::Tensor v129 = v1[127];
  ::ttnn::Tensor v130 = v1[128];
  ::ttnn::Tensor v131 = v1[129];
  ::ttnn::Tensor v132 = v1[130];
  ::ttnn::Tensor v133 = v1[131];
  ::ttnn::Tensor v134 = v1[132];
  ::ttnn::Tensor v135 = v1[133];
  ::ttnn::Tensor v136 = v1[134];
  ::ttnn::Tensor v137 = v1[135];
  ::ttnn::Tensor v138 = v1[136];
  ::ttnn::Tensor v139 = v1[137];
  ::ttnn::Tensor v140 = v1[138];
  ::ttnn::Tensor v141 = v1[139];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v142 = &main_const_eval_0;
  ::std::vector<::ttnn::Tensor> v143 = util_create_vec(
      v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v17, v18, v19, v20, v21,
      v23, v24, v25, v26, v27, v29, v30, v31, v32, v33, v36, v37, v38, v39, v40,
      v41, v42, v43, v44, v45, v47, v48, v49, v50, v51, v53, v54, v55, v56, v57,
      v59, v60, v61, v62, v63, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75,
      v77, v78, v79, v80, v81, v83, v84, v85, v86, v87, v89, v90, v91, v92, v93,
      v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v107, v108, v109,
      v110, v111, v113, v114, v115, v116, v117, v119, v120, v121, v122, v123,
      v124, v125, v126, v127, v128, v130, v131, v132, v133, v134, v136, v137,
      v138, v139, v140);
  ::std::vector<::ttnn::Tensor> *v144 = &g_cached_result_main_const_eval_0;
  ttnn::constEvalFuncWrapper(v142, v143, v144);
  ::std::vector<::ttnn::Tensor> v145 = g_cached_result_main_const_eval_0;
  ::ttnn::Tensor v146 = v145[0];
  ::ttnn::Tensor v147 = v145[1];
  ::ttnn::Tensor v148 = v145[2];
  ::ttnn::Tensor v149 = v145[3];
  ::ttnn::Tensor v150 = v145[4];
  ::ttnn::Tensor v151 = v145[5];
  ::ttnn::Tensor v152 = v145[6];
  ::ttnn::Tensor v153 = v145[7];
  ::ttnn::Tensor v154 = v145[8];
  ::ttnn::Tensor v155 = v145[9];
  ::ttnn::Tensor v156 = v145[10];
  ::ttnn::Tensor v157 = v145[11];
  ::ttnn::Tensor v158 = v145[12];
  ::ttnn::Tensor v159 = v145[13];
  ::ttnn::Tensor v160 = v145[14];
  ::ttnn::Tensor v161 = v145[15];
  ::ttnn::Tensor v162 = v145[16];
  ::ttnn::Tensor v163 = v145[17];
  ::ttnn::Tensor v164 = v145[18];
  ::ttnn::Tensor v165 = v145[19];
  ::ttnn::Tensor v166 = v145[20];
  ::ttnn::Tensor v167 = v145[21];
  ::ttnn::Tensor v168 = v145[22];
  ::ttnn::Tensor v169 = v145[23];
  ::ttnn::Tensor v170 = v145[24];
  ::ttnn::Tensor v171 = v145[25];
  ::ttnn::Tensor v172 = v145[26];
  ::ttnn::Tensor v173 = v145[27];
  ::ttnn::Tensor v174 = v145[28];
  ::ttnn::Tensor v175 = v145[29];
  ::ttnn::Tensor v176 = v145[30];
  ::ttnn::Tensor v177 = v145[31];
  ::ttnn::Tensor v178 = v145[32];
  ::ttnn::Tensor v179 = v145[33];
  ::ttnn::Tensor v180 = v145[34];
  ::ttnn::Tensor v181 = v145[35];
  ::ttnn::Tensor v182 = v145[36];
  ::ttnn::Tensor v183 = v145[37];
  ::ttnn::Tensor v184 = v145[38];
  ::ttnn::Tensor v185 = v145[39];
  ::ttnn::Tensor v186 = v145[40];
  ::ttnn::Tensor v187 = v145[41];
  ::ttnn::Tensor v188 = v145[42];
  ::ttnn::Tensor v189 = v145[43];
  ::ttnn::Tensor v190 = v145[44];
  ::ttnn::Tensor v191 = v145[45];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v192 = &main_const_eval_1;
  ::std::vector<::ttnn::Tensor> v193 = util_create_vec(v3);
  ::std::vector<::ttnn::Tensor> *v194 = &g_cached_result_main_const_eval_1;
  ttnn::constEvalFuncWrapper(v192, v193, v194);
  ::std::vector<::ttnn::Tensor> v195 = g_cached_result_main_const_eval_1;
  ::ttnn::Tensor v196 = v195[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v197 = &main_const_eval_2;
  ::std::vector<::ttnn::Tensor> v198 = util_create_vec(v129);
  ::std::vector<::ttnn::Tensor> *v199 = &g_cached_result_main_const_eval_2;
  ttnn::constEvalFuncWrapper(v197, v198, v199);
  ::std::vector<::ttnn::Tensor> v200 = g_cached_result_main_const_eval_2;
  ::ttnn::Tensor v201 = v200[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v202 = &main_const_eval_3;
  ::std::vector<::ttnn::Tensor> v203 = util_create_vec(v22);
  ::std::vector<::ttnn::Tensor> *v204 = &g_cached_result_main_const_eval_3;
  ttnn::constEvalFuncWrapper(v202, v203, v204);
  ::std::vector<::ttnn::Tensor> v205 = g_cached_result_main_const_eval_3;
  ::ttnn::Tensor v206 = v205[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v207 = &main_const_eval_4;
  ::std::vector<::ttnn::Tensor> v208 = util_create_vec(v95);
  ::std::vector<::ttnn::Tensor> *v209 = &g_cached_result_main_const_eval_4;
  ttnn::constEvalFuncWrapper(v207, v208, v209);
  ::std::vector<::ttnn::Tensor> v210 = g_cached_result_main_const_eval_4;
  ::ttnn::Tensor v211 = v210[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v212 = &main_const_eval_5;
  ::std::vector<::ttnn::Tensor> v213 = util_create_vec(v112);
  ::std::vector<::ttnn::Tensor> *v214 = &g_cached_result_main_const_eval_5;
  ttnn::constEvalFuncWrapper(v212, v213, v214);
  ::std::vector<::ttnn::Tensor> v215 = g_cached_result_main_const_eval_5;
  ::ttnn::Tensor v216 = v215[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v217 = &main_const_eval_6;
  ::std::vector<::ttnn::Tensor> v218 = util_create_vec(v5);
  ::std::vector<::ttnn::Tensor> *v219 = &g_cached_result_main_const_eval_6;
  ttnn::constEvalFuncWrapper(v217, v218, v219);
  ::std::vector<::ttnn::Tensor> v220 = g_cached_result_main_const_eval_6;
  ::ttnn::Tensor v221 = v220[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v222 = &main_const_eval_7;
  ::std::vector<::ttnn::Tensor> v223 = util_create_vec(v65);
  ::std::vector<::ttnn::Tensor> *v224 = &g_cached_result_main_const_eval_7;
  ttnn::constEvalFuncWrapper(v222, v223, v224);
  ::std::vector<::ttnn::Tensor> v225 = g_cached_result_main_const_eval_7;
  ::ttnn::Tensor v226 = v225[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v227 = &main_const_eval_8;
  ::std::vector<::ttnn::Tensor> v228 = util_create_vec(v64);
  ::std::vector<::ttnn::Tensor> *v229 = &g_cached_result_main_const_eval_8;
  ttnn::constEvalFuncWrapper(v227, v228, v229);
  ::std::vector<::ttnn::Tensor> v230 = g_cached_result_main_const_eval_8;
  ::ttnn::Tensor v231 = v230[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v232 = &main_const_eval_9;
  ::std::vector<::ttnn::Tensor> v233 = util_create_vec(v88);
  ::std::vector<::ttnn::Tensor> *v234 = &g_cached_result_main_const_eval_9;
  ttnn::constEvalFuncWrapper(v232, v233, v234);
  ::std::vector<::ttnn::Tensor> v235 = g_cached_result_main_const_eval_9;
  ::ttnn::Tensor v236 = v235[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v237 = &main_const_eval_10;
  ::std::vector<::ttnn::Tensor> v238 = util_create_vec(v28);
  ::std::vector<::ttnn::Tensor> *v239 = &g_cached_result_main_const_eval_10;
  ttnn::constEvalFuncWrapper(v237, v238, v239);
  ::std::vector<::ttnn::Tensor> v240 = g_cached_result_main_const_eval_10;
  ::ttnn::Tensor v241 = v240[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v242 = &main_const_eval_11;
  ::std::vector<::ttnn::Tensor> v243 = util_create_vec(v106);
  ::std::vector<::ttnn::Tensor> *v244 = &g_cached_result_main_const_eval_11;
  ttnn::constEvalFuncWrapper(v242, v243, v244);
  ::std::vector<::ttnn::Tensor> v245 = g_cached_result_main_const_eval_11;
  ::ttnn::Tensor v246 = v245[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v247 = &main_const_eval_12;
  ::std::vector<::ttnn::Tensor> v248 = util_create_vec(v46);
  ::std::vector<::ttnn::Tensor> *v249 = &g_cached_result_main_const_eval_12;
  ttnn::constEvalFuncWrapper(v247, v248, v249);
  ::std::vector<::ttnn::Tensor> v250 = g_cached_result_main_const_eval_12;
  ::ttnn::Tensor v251 = v250[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v252 = &main_const_eval_13;
  ::std::vector<::ttnn::Tensor> v253 = util_create_vec(v135);
  ::std::vector<::ttnn::Tensor> *v254 = &g_cached_result_main_const_eval_13;
  ttnn::constEvalFuncWrapper(v252, v253, v254);
  ::std::vector<::ttnn::Tensor> v255 = g_cached_result_main_const_eval_13;
  ::ttnn::Tensor v256 = v255[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v257 = &main_const_eval_14;
  ::std::vector<::ttnn::Tensor> v258 = util_create_vec(v118);
  ::std::vector<::ttnn::Tensor> *v259 = &g_cached_result_main_const_eval_14;
  ttnn::constEvalFuncWrapper(v257, v258, v259);
  ::std::vector<::ttnn::Tensor> v260 = g_cached_result_main_const_eval_14;
  ::ttnn::Tensor v261 = v260[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v262 = &main_const_eval_15;
  ::std::vector<::ttnn::Tensor> v263 = util_create_vec(v4);
  ::std::vector<::ttnn::Tensor> *v264 = &g_cached_result_main_const_eval_15;
  ttnn::constEvalFuncWrapper(v262, v263, v264);
  ::std::vector<::ttnn::Tensor> v265 = g_cached_result_main_const_eval_15;
  ::ttnn::Tensor v266 = v265[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v267 = &main_const_eval_16;
  ::std::vector<::ttnn::Tensor> v268 = util_create_vec(v82);
  ::std::vector<::ttnn::Tensor> *v269 = &g_cached_result_main_const_eval_16;
  ttnn::constEvalFuncWrapper(v267, v268, v269);
  ::std::vector<::ttnn::Tensor> v270 = g_cached_result_main_const_eval_16;
  ::ttnn::Tensor v271 = v270[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v272 = &main_const_eval_17;
  ::std::vector<::ttnn::Tensor> v273 = util_create_vec(v2);
  ::std::vector<::ttnn::Tensor> *v274 = &g_cached_result_main_const_eval_17;
  ttnn::constEvalFuncWrapper(v272, v273, v274);
  ::std::vector<::ttnn::Tensor> v275 = g_cached_result_main_const_eval_17;
  ::ttnn::Tensor v276 = v275[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v277 = &main_const_eval_18;
  ::std::vector<::ttnn::Tensor> v278 = util_create_vec(v58);
  ::std::vector<::ttnn::Tensor> *v279 = &g_cached_result_main_const_eval_18;
  ttnn::constEvalFuncWrapper(v277, v278, v279);
  ::std::vector<::ttnn::Tensor> v280 = g_cached_result_main_const_eval_18;
  ::ttnn::Tensor v281 = v280[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v282 = &main_const_eval_19;
  ::std::vector<::ttnn::Tensor> v283 = util_create_vec(v52);
  ::std::vector<::ttnn::Tensor> *v284 = &g_cached_result_main_const_eval_19;
  ttnn::constEvalFuncWrapper(v282, v283, v284);
  ::std::vector<::ttnn::Tensor> v285 = g_cached_result_main_const_eval_19;
  ::ttnn::Tensor v286 = v285[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v287 = &main_const_eval_20;
  ::std::vector<::ttnn::Tensor> v288 = util_create_vec(v35);
  ::std::vector<::ttnn::Tensor> *v289 = &g_cached_result_main_const_eval_20;
  ttnn::constEvalFuncWrapper(v287, v288, v289);
  ::std::vector<::ttnn::Tensor> v290 = g_cached_result_main_const_eval_20;
  ::ttnn::Tensor v291 = v290[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v292 = &main_const_eval_21;
  ::std::vector<::ttnn::Tensor> v293 = util_create_vec(v34);
  ::std::vector<::ttnn::Tensor> *v294 = &g_cached_result_main_const_eval_21;
  ttnn::constEvalFuncWrapper(v292, v293, v294);
  ::std::vector<::ttnn::Tensor> v295 = g_cached_result_main_const_eval_21;
  ::ttnn::Tensor v296 = v295[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v297 = &main_const_eval_22;
  ::std::vector<::ttnn::Tensor> v298 = util_create_vec(v94);
  ::std::vector<::ttnn::Tensor> *v299 = &g_cached_result_main_const_eval_22;
  ttnn::constEvalFuncWrapper(v297, v298, v299);
  ::std::vector<::ttnn::Tensor> v300 = g_cached_result_main_const_eval_22;
  ::ttnn::Tensor v301 = v300[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v302 = &main_const_eval_23;
  ::std::vector<::ttnn::Tensor> v303 = util_create_vec(v16);
  ::std::vector<::ttnn::Tensor> *v304 = &g_cached_result_main_const_eval_23;
  ttnn::constEvalFuncWrapper(v302, v303, v304);
  ::std::vector<::ttnn::Tensor> v305 = g_cached_result_main_const_eval_23;
  ::ttnn::Tensor v306 = v305[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v307 = &main_const_eval_24;
  ::std::vector<::ttnn::Tensor> v308 = util_create_vec(v76);
  ::std::vector<::ttnn::Tensor> *v309 = &g_cached_result_main_const_eval_24;
  ttnn::constEvalFuncWrapper(v307, v308, v309);
  ::std::vector<::ttnn::Tensor> v310 = g_cached_result_main_const_eval_24;
  ::ttnn::Tensor v311 = v310[0];
  ttnn::distributed::MeshDevice *v312 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v313 = ttnn::to_layout(
      v141, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v141, false);
  ::ttnn::Tensor v314 = ttnn::permute(
      v313, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v313, false);
  ::ttnn::Tensor v315 = ttnn::reshape(
      v314, ::std::vector<int32_t>{1, 1, 50176, 3},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt});
  ttnn::deallocate(v314, false);
  ::ttnn::Tensor v316 = ::std::get<0>(ttnn::conv2d(
      v315, v146, v312, 3, 64, 1, 224, 224, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v147,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 32,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{256, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v317 = ::std::get<0>(ttnn::conv2d(
      v316, v256, v312, 64, 64, 1, 112, 112, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 64, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 6}}}},
                            ::std::array<uint32_t, 2>{224, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v318 = ::std::get<0>(ttnn::conv2d(
      v317, v148, v312, 64, 64, 1, 112, 112, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v149,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 6}}}},
                            ::std::array<uint32_t, 2>{224, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v319 = ::std::get<0>(ttnn::conv2d(
      v318, v201, v312, 64, 64, 1, 112, 112, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 64, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v320 = ::std::get<0>(ttnn::conv2d(
      v319, v150, v312, 64, 64, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v151,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v321 = ::std::get<0>(ttnn::conv2d(
      v320, v152, v312, 64, 128, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v153,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v322 = ::std::get<0>(ttnn::conv2d(
      v321, v261, v312, 128, 128, 1, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 128, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v323 = ::std::get<0>(ttnn::conv2d(
      v322, v154, v312, 128, 128, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v155,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v324 = ::std::get<0>(ttnn::conv2d(
      v323, v216, v312, 128, 128, 1, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 128, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v325 = ::std::get<0>(ttnn::conv2d(
      v324, v156, v312, 128, 128, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v157,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v326 = ::std::get<0>(ttnn::conv2d(
      v325, v246, v312, 128, 128, 1, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 128, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v327 = ::std::get<0>(ttnn::conv2d(
      v326, v158, v312, 128, 128, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v159,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 128},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::std::vector<::ttnn::Tensor> v328 = util_create_vec(v320, v323, v325, v327);
  ::ttnn::Tensor v329 = ttnn::concat(
      v328, 3,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 448},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v327, false);
  ttnn::deallocate(v325, false);
  ttnn::deallocate(v323, false);
  ttnn::deallocate(v320, false);
  ::ttnn::Tensor v330 = ::std::get<0>(ttnn::conv2d(
      v329, v160, v312, 448, 256, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v161,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 256},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v331 =
      ttnn::mean(v330, ::ttsl::SmallVector<int32_t>{2}, true,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::L1, ::std::nullopt},
                 ::std::nullopt);
  ::ttnn::Tensor v332 = ttnn::to_memory_config(
      v331,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v331, false);
  ::ttnn::Tensor v333 = ::std::get<0>(ttnn::conv2d(
      v332, v211, v312, 256, 256, 1, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v301,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 0}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v334 = ttnn::typecast(
      v333, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 0}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v333, false);
  ::ttnn::Tensor v335 = ttnn::hardsigmoid(
      v334,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 0}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v334, false);
  ::ttnn::Tensor v336 = ttnn::typecast(
      v335, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 0}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v335, false);
  ::ttnn::Tensor v337 = ttnn::multiply(
      v330, v336, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{64, 256},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v336, false);
  ttnn::deallocate(v330, false);
  ::std::vector<::ttnn::Tensor> v338 = ttnn::max_pool2d(
      v337, 1, 56, 56, 256, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 1, 0, 1},
      ::std::array<uint32_t, 2>{1, 1}, false,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 5}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6},
                                                  ::ttnn::CoreCoord{0, 6}}}},
                            ::std::array<uint32_t, 2>{16, 256},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v339 = v338[0];
  ttnn::deallocate(v337, false);
  ::ttnn::Tensor v340 = ::std::get<0>(ttnn::conv2d(
      v339, v162, v312, 256, 160, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v163,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v341 = ::std::get<0>(ttnn::conv2d(
      v340, v236, v312, 160, 160, 1, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 160, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v342 = ::std::get<0>(ttnn::conv2d(
      v341, v164, v312, 160, 160, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v165,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v343 = ::std::get<0>(ttnn::conv2d(
      v342, v271, v312, 160, 160, 1, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 160, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v344 = ::std::get<0>(ttnn::conv2d(
      v343, v166, v312, 160, 160, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v167,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v345 = ::std::get<0>(ttnn::conv2d(
      v344, v311, v312, 160, 160, 1, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 160, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v346 = ::std::get<0>(ttnn::conv2d(
      v345, v168, v312, 160, 160, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v169,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 160},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v347 = ttnn::to_memory_config(
      v339, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v339, false);
  ::ttnn::Tensor v348 = ttnn::to_memory_config(
      v342, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v342, false);
  ::ttnn::Tensor v349 = ttnn::to_memory_config(
      v344, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v344, false);
  ::ttnn::Tensor v350 = ttnn::to_memory_config(
      v346, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v346, false);
  ::std::vector<::ttnn::Tensor> v351 = util_create_vec(v347, v348, v349, v350);
  ::ttnn::Tensor v352 = ttnn::concat(
      v351, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v350, false);
  ttnn::deallocate(v349, false);
  ttnn::deallocate(v348, false);
  ttnn::deallocate(v347, false);
  ::ttnn::Tensor v353 = ttnn::to_layout(
      v352, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v352, false);
  ::ttnn::Tensor v354 = ttnn::to_memory_config(
      v353,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 736},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v353, false);
  ::ttnn::Tensor v355 = ::std::get<0>(ttnn::conv2d(
      v354, v170, v312, 736, 512, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v171,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 64,
          .shard_layout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 512},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v356 =
      ttnn::mean(v355, ::ttsl::SmallVector<int32_t>{2}, true,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::L1, ::std::nullopt},
                 ::std::nullopt);
  ::ttnn::Tensor v357 = ttnn::to_memory_config(
      v356,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v356, false);
  ::ttnn::Tensor v358 = ::std::get<0>(ttnn::conv2d(
      v357, v226, v312, 512, 512, 1, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v231,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 1}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v359 = ttnn::typecast(
      v358, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 1}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v358, false);
  ::ttnn::Tensor v360 = ttnn::hardsigmoid(
      v359,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 1}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v359, false);
  ::ttnn::Tensor v361 = ttnn::typecast(
      v360, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 1}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v360, false);
  ::ttnn::Tensor v362 = ttnn::multiply(
      v355, v361, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{32, 512},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v361, false);
  ttnn::deallocate(v355, false);
  ::std::vector<::ttnn::Tensor> v363 = ttnn::max_pool2d(
      v362, 1, 28, 28, 512, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 1, 0, 1},
      ::std::array<uint32_t, 2>{1, 1}, false,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 3},
                                                  ::ttnn::CoreCoord{0, 3}}}},
                            ::std::array<uint32_t, 2>{8, 512},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v364 = v363[0];
  ttnn::deallocate(v362, false);
  ::ttnn::Tensor v365 = ttnn::to_memory_config(
      v364, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v366 = ::std::get<0>(ttnn::conv2d(
      v365, v172, v312, 512, 192, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v173,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .config_tensors_in_dram = true,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ttnn::deallocate(v365, false);
  ::ttnn::Tensor v367 = ttnn::to_memory_config(
      v366,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v366, false);
  ::ttnn::Tensor v368 = ::std::get<0>(ttnn::conv2d(
      v367, v281, v312, 192, 192, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 192, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 64,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v369 = ::std::get<0>(ttnn::conv2d(
      v368, v174, v312, 192, 192, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v175,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v370 = ::std::get<0>(ttnn::conv2d(
      v369, v286, v312, 192, 192, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 192, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v371 = ::std::get<0>(ttnn::conv2d(
      v370, v176, v312, 192, 192, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v177,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v372 = ::std::get<0>(ttnn::conv2d(
      v371, v251, v312, 192, 192, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 192, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v373 = ::std::get<0>(ttnn::conv2d(
      v372, v178, v312, 192, 192, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v179,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{5, 6}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v374 = ttnn::to_memory_config(
      v364, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v364, false);
  ::ttnn::Tensor v375 = ttnn::to_memory_config(
      v369, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v369, false);
  ::ttnn::Tensor v376 = ttnn::to_memory_config(
      v371, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v371, false);
  ::ttnn::Tensor v377 = ttnn::to_memory_config(
      v373, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v373, false);
  ::std::vector<::ttnn::Tensor> v378 = util_create_vec(v374, v375, v376, v377);
  ::ttnn::Tensor v379 = ttnn::concat(
      v378, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v377, false);
  ttnn::deallocate(v376, false);
  ttnn::deallocate(v375, false);
  ttnn::deallocate(v374, false);
  ::ttnn::Tensor v380 = ttnn::to_layout(
      v379, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v379, false);
  ::ttnn::Tensor v381 = ttnn::to_memory_config(
      v380,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 0}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 1},
                                                  ::ttnn::CoreCoord{3, 1}}}},
                            ::std::array<uint32_t, 2>{224, 96},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v380, false);
  ::ttnn::Tensor v382 = ::std::get<0>(ttnn::conv2d(
      v381, v180, v312, 1088, 768, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v181,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{224, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v383 =
      ttnn::mean(v382, ::ttsl::SmallVector<int32_t>{2}, true,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::L1, ::std::nullopt},
                 ::std::nullopt);
  ::ttnn::Tensor v384 = ttnn::to_memory_config(
      v383,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v383, false);
  ::ttnn::Tensor v385 = ::std::get<0>(ttnn::conv2d(
      v384, v291, v312, 768, 768, 1, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v296,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v386 = ttnn::typecast(
      v385, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v385, false);
  ::ttnn::Tensor v387 = ttnn::hardsigmoid(
      v386,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v386, false);
  ::ttnn::Tensor v388 = ttnn::typecast(
      v387, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v387, false);
  ::ttnn::Tensor v389 = ttnn::multiply(
      v382, v388, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{224, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v388, false);
  ttnn::deallocate(v382, false);
  ::std::vector<::ttnn::Tensor> v390 = ttnn::max_pool2d(
      v389, 1, 14, 14, 768, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 1, 0, 1},
      ::std::array<uint32_t, 2>{1, 1}, false,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 2}}}},
                            ::std::array<uint32_t, 2>{49, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v391 = v390[0];
  ttnn::deallocate(v389, false);
  ::ttnn::Tensor v392 = ::std::get<0>(ttnn::conv2d(
      v391, v182, v312, 768, 224, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v183,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = false,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v393 = ::std::get<0>(ttnn::conv2d(
      v392, v241, v312, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 224, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v394 = ::std::get<0>(ttnn::conv2d(
      v393, v184, v312, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v185,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v395 = ::std::get<0>(ttnn::conv2d(
      v394, v206, v312, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 224, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v396 = ::std::get<0>(ttnn::conv2d(
      v395, v186, v312, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v187,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v397 = ::std::get<0>(ttnn::conv2d(
      v396, v306, v312, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 224, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = false,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v398 = ::std::get<0>(ttnn::conv2d(
      v397, v188, v312, 224, 224, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v189,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{6, 0}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v399 = ttnn::to_memory_config(
      v391, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v391, false);
  ::ttnn::Tensor v400 = ttnn::to_memory_config(
      v394, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v394, false);
  ::ttnn::Tensor v401 = ttnn::to_memory_config(
      v396, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v396, false);
  ::ttnn::Tensor v402 = ttnn::to_memory_config(
      v398, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v398, false);
  ::std::vector<::ttnn::Tensor> v403 = util_create_vec(v399, v400, v401, v402);
  ::ttnn::Tensor v404 = ttnn::concat(
      v403, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v402, false);
  ttnn::deallocate(v401, false);
  ttnn::deallocate(v400, false);
  ttnn::deallocate(v399, false);
  ::ttnn::Tensor v405 = ttnn::to_layout(
      v404, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v404, false);
  ::ttnn::Tensor v406 = ttnn::to_memory_config(
      v405,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 1}},
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 2},
                                                  ::ttnn::CoreCoord{6, 2}}}},
                            ::std::array<uint32_t, 2>{64, 64},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v405, false);
  ::ttnn::Tensor v407 = ::std::get<0>(ttnn::conv2d(
      v406, v190, v312, 1440, 1024, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v191,
      ::ttnn::Conv2dConfig{
          .weights_dtype = ::ttnn::DataType::BFLOAT16,
          .activation = ::ttnn::operations::unary::UnaryWithParam(
              ::ttnn::operations::unary::UnaryOpType::RELU),
          .deallocate_activation = true,
          .config_tensors_in_dram = true,
          .act_block_h_override = 0,
          .shard_layout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
          .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{64, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v408 = ttnn::to_memory_config(
      v407, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::L1, ::std::nullopt});
  ::ttnn::Tensor v409 = ttnn::reshape(
      v408, ::std::vector<int32_t>{1, 7, 7, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt});
  ttnn::deallocate(v408, false);
  ::ttnn::Tensor v410 = ttnn::permute(
      v409, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v409, false);
  ::ttnn::Tensor v411 =
      ttnn::mean(v407, ::ttsl::SmallVector<int32_t>{2}, true,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::L1, ::std::nullopt},
                 ::std::nullopt);
  ttnn::deallocate(v407, false);
  ::ttnn::Tensor v412 = ttnn::to_memory_config(
      v411,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v411, false);
  ::ttnn::Tensor v413 = ::std::get<0>(ttnn::conv2d(
      v412, v221, v312, 1024, 1024, 1, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v266,
      ::ttnn::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16,
                           .deallocate_activation = true,
                           .config_tensors_in_dram = true,
                           .act_block_h_override = 0,
                           .shard_layout =
                               ::ttnn::TensorMemoryLayout::WIDTH_SHARDED,
                           .enable_kernel_stride_folding = false},
      ::std::nullopt,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::Conv2dSliceConfig{.slice_type =
                                    ttnn::Conv2dSliceConfig::SliceType::L1_FULL,
                                .num_slices = 0}));
  ::ttnn::Tensor v414 = ttnn::typecast(
      v413, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v413, false);
  ::ttnn::Tensor v415 = ttnn::hardsigmoid(
      v414,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v414, false);
  ::ttnn::Tensor v416 = ttnn::typecast(
      v415, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v415, false);
  ::ttnn::Tensor v417 = ttnn::to_memory_config(
      v416, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::L1, ::std::nullopt});
  ttnn::deallocate(v416, false);
  ::ttnn::Tensor v418 = ttnn::reshape(
      v417, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt});
  ttnn::deallocate(v417, false);
  ::ttnn::Tensor v419 = ttnn::multiply(
      v410, v418, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 7}}}},
                            ::std::array<uint32_t, 2>{512, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v418, false);
  ttnn::deallocate(v410, false);
  ::ttnn::Tensor v420 = ttnn::typecast(
      v419, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 7}}}},
                            ::std::array<uint32_t, 2>{512, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v419, false);
  ::ttnn::Tensor v421 = ttnn::to_memory_config(
      v420,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{0, 7}}}},
                            ::std::array<uint32_t, 2>{4096, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v420, false);
  ::ttnn::Tensor v422 =
      ttnn::mean(v421, ::ttsl::SmallVector<int32_t>{2, 3}, true,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::L1, ::std::nullopt},
                 ::std::nullopt);
  ttnn::deallocate(v421, false);
  ::ttnn::Tensor v423 = ttnn::reshape(
      v422, ::std::vector<int32_t>{1, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::L1, ::std::nullopt});
  ttnn::deallocate(v422, false);
  ::ttnn::Tensor v424 = ttnn::linear(
      v423, v196, v276, false, false,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}},
      ::ttnn::DataType::FLOAT32, ::std::nullopt, ::std::nullopt,
      ::std::nullopt);
  ttnn::deallocate(v423, false);
  ::ttnn::Tensor v425 = ttnn::typecast(
      v424, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{
          ::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1,
          ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                                ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0},
                                                  ::ttnn::CoreCoord{7, 3}}}},
                            ::std::array<uint32_t, 2>{32, 32},
                            ::ttnn::types::ShardOrientation::ROW_MAJOR}});
  ttnn::deallocate(v424, false);
  ::ttnn::Tensor v426 = ttnn::to_memory_config(
      v425, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v425, false);
  ::std::vector<::ttnn::Tensor> v427 = util_create_vec(v426);
  return v427;
}
std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor,
           ::ttnn::Tensor, ::ttnn::Tensor>
cpu_hoisted_const_eval_b2f462d7() {}
::ttnn::Tensor cpu_hoisted_const_eval_24665087() {}
::ttnn::Tensor cpu_hoisted_const_eval_382fc5dd() {}
::ttnn::Tensor cpu_hoisted_const_eval_8b0e4be0() {}
::ttnn::Tensor cpu_hoisted_const_eval_b3a3bf24() {}
::ttnn::Tensor cpu_hoisted_const_eval_8b649ce3() {}
::ttnn::Tensor cpu_hoisted_const_eval_5c372694() {}
::std::vector<::ttnn::Tensor> create_inputs_for__main() {
  ttnn::distributed::MeshDevice *v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(
      ::ttnn::Shape({1000}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::ones(
      ::ttnn::Shape({1000, 1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(
      ::ttnn::Shape({1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::ones(
      ::ttnn::Shape({1024, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(
      ::ttnn::Shape({1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::ones(
      ::ttnn::Shape({1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::ones(
      ::ttnn::Shape({1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::ones(
      ::ttnn::Shape({1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::ones(
      ::ttnn::Shape({1024, 1440, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v17 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v18 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v19 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v20 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v21 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v22 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v23 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v24 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v25 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v26 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v27 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v28 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v29 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v30 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v31 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v32 = ttnn::ones(
      ::ttnn::Shape({224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v33 = ttnn::ones(
      ::ttnn::Shape({224, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v34 = ttnn::ones(
      ::ttnn::Shape({768}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v35 = ttnn::ones(
      ::ttnn::Shape({768, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v36 = ttnn::ones(
      ::ttnn::Shape({768}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v37 = ttnn::ones(
      ::ttnn::Shape({768}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v38 = ttnn::ones(
      ::ttnn::Shape({768}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v39 = ttnn::ones(
      ::ttnn::Shape({768}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v40 = ttnn::ones(
      ::ttnn::Shape({768, 1088, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v41 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v42 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v43 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v44 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v45 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v46 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v47 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v48 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v49 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v50 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v51 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v52 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v53 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v54 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v55 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v56 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v57 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v58 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v59 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v60 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v61 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v62 = ttnn::ones(
      ::ttnn::Shape({192}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v63 = ttnn::ones(
      ::ttnn::Shape({192, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v64 = ttnn::ones(
      ::ttnn::Shape({512}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v65 = ttnn::ones(
      ::ttnn::Shape({512, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v66 = ttnn::ones(
      ::ttnn::Shape({512}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v67 = ttnn::ones(
      ::ttnn::Shape({512}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v68 = ttnn::ones(
      ::ttnn::Shape({512}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v69 = ttnn::ones(
      ::ttnn::Shape({512}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v70 = ttnn::ones(
      ::ttnn::Shape({512, 736, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v71 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v72 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v73 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v74 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v75 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v76 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v77 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v78 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v79 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v80 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v81 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v82 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v83 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v84 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v85 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v86 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v87 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v88 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v89 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v90 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v91 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v92 = ttnn::ones(
      ::ttnn::Shape({160}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v93 = ttnn::ones(
      ::ttnn::Shape({160, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v94 = ttnn::ones(
      ::ttnn::Shape({256}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v95 = ttnn::ones(
      ::ttnn::Shape({256, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v96 = ttnn::ones(
      ::ttnn::Shape({256}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v97 = ttnn::ones(
      ::ttnn::Shape({256}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v98 = ttnn::ones(
      ::ttnn::Shape({256}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v99 = ttnn::ones(
      ::ttnn::Shape({256}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v100 = ttnn::ones(
      ::ttnn::Shape({256, 448, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v101 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v102 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v103 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v104 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v105 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v106 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v107 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v108 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v109 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v110 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v111 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v112 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v113 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v114 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v115 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v116 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v117 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v118 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v119 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v120 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v121 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v122 = ttnn::ones(
      ::ttnn::Shape({128}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v123 = ttnn::ones(
      ::ttnn::Shape({128, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v124 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v125 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v126 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v127 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v128 = ttnn::ones(
      ::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v129 = ttnn::ones(
      ::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v130 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v131 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v132 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v133 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v134 = ttnn::ones(
      ::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v135 = ttnn::ones(
      ::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v136 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v137 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v138 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v139 = ttnn::ones(
      ::ttnn::Shape({64}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v140 = ttnn::ones(
      ::ttnn::Shape({64, 3, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>>
      v141 = *v1;
  ::ttnn::Tensor v142 = ttnn::ones(
      ::ttnn::Shape({1, 3, 224, 224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, v141,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v143 = util_create_vec(
      v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17,
      v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32,
      v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47,
      v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62,
      v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77,
      v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92,
      v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105,
      v106, v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117,
      v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128, v129,
      v130, v131, v132, v133, v134, v135, v136, v137, v138, v139, v140, v142);
  return v143;
}
// ===================== N150 BENCHMARK HARNESS =====================
// Replaces the default single-shot main(). Measures eager and metal-trace
// throughput for VoVNet (batch=1, bf16) on a single Wormhole N150.
//
// Env vars:
//   BENCH_WARMUP        warmup iterations          (default 10)
//   BENCH_ITERS         timed iterations           (default 50)
//   BENCH_MODE          eager | trace | both       (default both)
//   TRACE_REGION_SIZE   DRAM trace region bytes    (default 128MB, see hpp)
//
// NOTE: weights here are ones() (codegen without exported tensors) — fine for
// throughput, NOT for accuracy. For accuracy regenerate with exported tensors.
#include <chrono>
#include <cstdlib>
#include <string>

static int bench_env_int(const char *name, int def) {
  if (const char *e = std::getenv(name))
    return std::atoi(e);
  return def;
}

int32_t main() {
  ttnn::MeshDevice *device = ttnn::DeviceGetter::getInstance();

  const int warmup = bench_env_int("BENCH_WARMUP", 10);
  const int iters = bench_env_int("BENCH_ITERS", 50);
  const std::string mode =
      std::getenv("BENCH_MODE") ? std::getenv("BENCH_MODE") : "both";

  // Inputs: weights (ones) + input image. Const-eval caches the weight
  // transforms internally, so repeated _main() calls reuse cached device-side
  // weights and only re-run the actual compute.
  ::std::vector<::ttnn::Tensor> v1 = create_inputs_for__main();

  // Warmup: triggers program compile + const-eval caching.
  for (int i = 0; i < warmup; ++i) {
    auto out = _main(v1);
    (void)out;
  }

  if (mode == "eager" || mode == "both") {
    // Drain before timing.
    {
      auto o = _main(v1);
      (void)ttnn::from_device(o[0]);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    ::std::vector<::ttnn::Tensor> out;
    for (int i = 0; i < iters; ++i) {
      out = _main(v1);
    }
    (void)ttnn::from_device(out[0]); // block until queue drains
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::cout << "[eager] " << ms << " ms/iter, " << (1000.0 / ms)
              << " FPS (b=1)\n";
  }

  if (mode == "trace" || mode == "both") {
    const ttnn::QueueId cq = ttnn::QueueId(0);
    // Capture the compute graph once (const-eval already cached above).
    auto tid = ttnn::begin_trace_capture(device, cq);
    auto traced = _main(v1);
    (void)traced;
    ttnn::end_trace_capture(device, tid, cq);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      ttnn::execute_trace(device, tid, cq, /*blocking=*/false);
    }
    ttnn::execute_trace(device, tid, cq, /*blocking=*/true); // drain
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() /
                (iters + 1);
    std::cout << "[trace] " << ms << " ms/iter, " << (1000.0 / ms)
              << " FPS (b=1)\n";

    ttnn::release_trace(device, tid);
  }

  return 0;
}
