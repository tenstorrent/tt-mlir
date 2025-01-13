# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer


def load_model(model_path, **kwargs):
    # Default config values
    config = LlamaConfig.from_pretrained(model_path)

    # Use defaults or values from kwargs
    config.return_dict = kwargs.get("return_dict", False)
    config.use_cache = kwargs.get("use_cache", False)
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)

    # Load the model
    framework_model = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto", config=config
    )
    framework_model.eval()

    # Using AutoTokenizer for default tokenizers for both openllama and llama 3.2
    use_fast = kwargs.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)

    return framework_model, tokenizer


model, tokenizer = load_model("openlm-research/open_llama_3b", return_dict=True)
model_decoder = model.get_decoder()
