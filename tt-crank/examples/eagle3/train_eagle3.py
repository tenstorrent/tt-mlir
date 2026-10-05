# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""EAGLE-3 draft training on a single tt chip with NeMo AutoModel's recipe.

Runs `nemo_automodel`'s `TrainEagle3Recipe` (target aux-state capture, draft
model, TTT-unrolled loss, LR schedule, W&B logging, checkpointing) with the
frozen target and the draft placed on the tt device. The target forward and the
trainer module's forward/backward run through `torch.compile(backend="tt")`
per the config's `tt_compile` block; the TTIR/TTNN IR of every compiled graph
is dumped under `$TT_CRANK_ARTIFACTS_DIR` and uploaded to the W&B run.

Configs use AutoModel's YAML schema; dotted CLI overrides work as in AutoModel:

    python tt-crank/examples/eagle3/train_eagle3.py \\
        --config tt-crank/examples/eagle3/llama3_1_8b_eagle3.yaml \\
        --recipe_args.train_split "train[:1000]"

Requires `nemo_automodel` (and `wandb` for logging) in the tt-mlir venv.
"""

import logging
import pathlib

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.filesystem as torch_dcp_filesystem
import tt_crank.torch  # noqa: F401  — registers the "tt" device and backend
import wandb
from nemo_automodel.components.checkpoint._backports import filesystem as dcp_filesystem
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.models.llama.rope_utils import LlamaRotaryEmbedding
from nemo_automodel.components.speculative.eagle import HFEagle3TargetModel
from nemo_automodel.components.utils.compile_utils import (
    build_compile_config,
    compile_module_inplace,
)
from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe
from transformers import AutoModelForCausalLM
from tt_crank.torch._artifacts import collect_artifacts, dump_artifacts

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = pathlib.Path(__file__).with_name("llama3_1_8b_eagle3.yaml")


class _SerialSizeSortedLoader(dcp_filesystem._SerialCpuLoader):
    """The checkpoint writers (model: AutoModel's backport, optimizer: torch's)
    overlap device->host copies on a device stream whenever a custom device is
    available; tt has no stream API. Stands in for that loader: copies serially,
    in the same size-sorted order the backport lays out its safetensors header in."""

    def __init__(self, resolve_fun, **_):
        super().__init__(resolve_fun)

    def start_loading(self):
        self.items.sort(key=lambda item: item[0])


dcp_filesystem._OverlappingCpuLoader = _SerialSizeSortedLoader
torch_dcp_filesystem._OverlappingCpuLoader = _SerialSizeSortedLoader


class TTTrainEagle3Recipe(TrainEagle3Recipe):
    """`TrainEagle3Recipe` on one tt chip.

    `setup()` picks CUDA/bf16 when available and falls back to CPU/fp32
    otherwise; pin tt/bf16 instead (the setters swallow its assignments).
    """

    device = property(lambda self: torch.device("tt"), lambda self, _: None)
    compute_dtype = property(lambda self: torch.bfloat16, lambda self, _: None)

    def _setup_colocated_target(self, recipe_cfg, target_path, draft_base_config):
        # `NeMoAutoModelForCausalLM` requires CUDA; load the frozen target with
        # plain HF instead and otherwise follow the upstream single-device path.
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path,
            dtype=self.compute_dtype,
            attn_implementation=recipe_cfg.get("target_attn_implementation", "sdpa"),
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
        )
        # safetensors loads parameters at 8-byte-aligned host addresses, and a
        # host->tt copy from such a buffer never completes on device. Clones
        # get torch's regular (64-byte-aligned) allocations.
        with torch.no_grad():
            for param in self.target_model.parameters():
                param.data = param.data.clone()
        self.target_model.to(self.device)
        self.target_model.requires_grad_(False)
        self.target_wrapper = HFEagle3TargetModel(
            self.target_model, aux_layer_ids=recipe_cfg.get("aux_layer_ids", None)
        )

    def setup(self):
        super().setup()
        # `setup()` opens a world-size-1 gloo group, which it only needs for the
        # CPU draft-vocab count reduction. gloo rejects tt tensors, and the
        # train loop's metric reductions and checkpoint barriers are identities
        # on one chip that skip themselves when no group is initialized.
        dist.destroy_process_group()

        compile_cfg = build_compile_config(self.cfg.get("tt_compile", None))
        compile_module_inplace(self.target_model, compile_cfg)
        compile_module_inplace(self.trainer_module, compile_cfg)
        # AutoModel turns on scalar capture for FlashAttention, which traces
        # `.item()` into data-dependent shapes; tt lowers static shapes only, so
        # let `.item()` break the graph and run eagerly.
        torch._dynamo.config.capture_scalar_outputs = False
        # The draft's rotary embedding sizes its cos/sin cache from
        # `position_ids.max().item()`. Under packing that value changes nearly
        # every batch, and static-shape dynamo recompiles the code after the
        # `.item()` for each new value; run these few ops eagerly instead.
        for module in self.draft_model.modules():
            if isinstance(module, LlamaRotaryEmbedding):
                module.forward = torch._dynamo.disable(module.forward)

    def _wandb_log(self, data, step):
        # With an explicit `step`, `run.log` defaults to `commit=False` and holds
        # the row until a later step is logged, so every point would reach W&B
        # one logging interval late.
        if self.wandb_run is not None:
            self.wandb_run.log(data, step=step, commit=True)

    def _train_epochs(self, *args, **kwargs):
        # Graphs compile lazily in the first steps. The collection is dumped
        # (and uploaded) here rather than after the loop, because the recipe
        # finishes the W&B run in its teardown.
        with collect_artifacts("eagle3_train"):
            try:
                return super()._train_epochs(*args, **kwargs)
            finally:
                self._log_ir(dump_artifacts())

    def _log_ir(self, ir_dir):
        if ir_dir is None:
            return
        logger.info("TTIR/TTNN IR of the compiled graphs: %s", ir_dir)
        if self.wandb_run is not None:
            artifact = wandb.Artifact(f"ir-{self.wandb_run.id}", type="ttnn-ir")
            artifact.add_dir(str(ir_dir))
            self.wandb_run.log_artifact(artifact)


def main() -> None:
    cfg = parse_args_and_load_config(str(_DEFAULT_CONFIG))
    recipe = TTTrainEagle3Recipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
