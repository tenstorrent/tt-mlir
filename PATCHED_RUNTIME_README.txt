*** tt-mlir's RUNTIME LIBRARY IS PATCHED -- 2026-11-02 ***
(separate from, and compatible with, tt-metal-src/PATCHED_BUILD_README.txt)

Patch: quasar_models/quasar_redirect_mul_sub_div_sigmoid.patch
       runtime/lib/ttnn/operations/eltwise/binary/binary.cpp
         Multiply, Subtract, Divide now redirect to
         ttnn::operations::experimental::quasar::binary::{multiply,subtract,divide}
         under utils::isQuasar(), exactly as Add already did.
       runtime/lib/ttnn/operations/eltwise/unary/unary.cpp
         the Quasar Relu rewrite is lifted into a helper
         runQuasarUnaryAsBinaryActivation() and reused for Sigmoid
         (x -> quasar::binary::add(x, 0.0f, lhs_activations={SIGMOID})).

Deployed artifact -- EXACTLY ONE FILE:
   third_party/tt-mlir/build/install/lib/libTTMLIRRuntime.so
       pre-patch md5  8c2a14d41d7cef7dad4833df1e4a8d64   (5241456 B)
       patched   md5  bd96ff34b1bb2d990f7a6d8f0e94c9a4   (5245760 B)
   Nothing else was written. tt-metal-src was NOT touched by this build.

Measured 2026-11-02 on ttsim (quasar-release-candidate, ttsim_rc0916):
   YOLOv8n, ORIGINAL plan_yolov8n_best.json : 87/87 of 261, pcc 0.9999924302101135
       -> IDENTICAL to the pre-patch baseline, to all 16 digits.
   ResNet-50, plan_par_ttsim.json           : 120/120 of 122, pcc 0.9998232126235962
       -> IDENTICAL to the recorded baseline, to all 16 digits.
   So this patch is INERT on every plan the campaign currently uses.
   It only matters if a plan ROUTES Mul/Sub/Div/Sigmoid to the device; the
   stock plans pin those to cpu. With a plan that does route them:
   +57 Mul and +57 Sigmoid segments execute on Quasar (87 -> 201).

TO RESTORE THE PRE-PATCH RUNTIME, one command:
   cp /proj_sw/user_dev/ctr-lelanchelian/lab_17_forge_onxx/quasar_models/REDIRECT_BACKUP_20261102/libTTMLIRRuntime.so.PRISTINE_PREPATCH \
      /proj_sw/user_dev/ctr-lelanchelian/lab_17_forge_onxx/tt-forge-onnx/third_party/tt-mlir/build/install/lib/libTTMLIRRuntime.so
   (The tt-mlir SOURCE tree still has the patch applied; `git -C
    third_party/tt-mlir checkout runtime/lib/ttnn/operations/eltwise/` reverts it.)

Full write-up: quasar_models/REDIRECT_BUILD_RESULT.txt
Backups      : quasar_models/REDIRECT_BACKUP_20261102/

================================================================================
*** SUPERSEDED -- 2026-11-12 -- EVERYTHING ABOVE IS NOW COMMITTED ***
================================================================================
All the layers described above are COMMITTED, not working-tree edits:

  tt-metal  branch quasar-resnet50-consolidated-20261112
            134b07c9791  inherited pre-2026-11-02 state (fill.h, permute, fill_pad,
                         tilize device factories, umd pointer) -- NOT part of this work
            bd017536f20  the six consolidated Quasar layers
  tt-mlir   branch quasar-resnet50-consolidated-20261112
            34bc1734b1   inherited pre-2026-10-21 runtime state
            7260e8bfe3   the eltwise redirects + unary-as-activation

ONE library set now carries ALL of them (md5 _ttnncpp.so bfab1e48,
libTTMLIRRuntime.so dfb4aa41), archived at
    quasar_models/resnet50_complete_20261112/lib/
and verified on ResNet-50 122/122, PDL, YOLOv8n and Llama-3.2-1B.

DO NOT RUN THE REVERT SCRIPTS REFERENCED ABOVE. They restore source files that are
now committed and libraries that are two generations behind; running one leaves the
tree inconsistent with HEAD. To drop a layer, revert its hunks from the commit.

Read quasar_models/RESNET50_COMPLETE.md instead of this file.
