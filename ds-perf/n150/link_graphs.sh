#!/usr/bin/env bash
# Re-lay the downloaded CI dumps into the layout run_model_fleet.sh expects:
#   graphs/<variant>/<graphname>/ttnn_runtime_<graphname>_..._g1_*.mlir
# before = DS off (nods), after = DS on (ds).
set -u
rm -rf graphs; mkdir -p graphs
declare -A MAP=( [before]=nods [after]=ds [patched]=guard )
for src in before after patched; do
  v=${MAP[$src]}
  find raw/$src -name 'ttnn_runtime_*_g1_*.mlir' | while read -r f; do
    base=$(basename "$f")
    name=${base#ttnn_runtime_}
    name=${name%%_bs[0-9]*}            # qwen_2_5_3b_bs32_isl128_run..._g1_... -> qwen_2_5_3b
    case "$name" in *_g1_*) continue;; esac   # no _bs marker (vllm/CNN dumps): skip
    mkdir -p "graphs/$v/$name"
    ln -sf "$(realpath "$f")" "graphs/$v/$name/$base"
  done
done
for v in nods ds guard; do echo "$v: $(ls graphs/$v 2>/dev/null | wc -l) models"; done
echo "in both:"; comm -12 <(ls graphs/nods|sort) <(ls graphs/ds|sort) | tr '\n' ' '; echo
