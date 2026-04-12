
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def pick_torch_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if torch.cuda.is_available():
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def infer_lora_targets(model_name_or_path: str) -> list[str]:
    lower = model_name_or_path.lower()
    if "pythia" in lower or "gpt-neox" in lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if "llama" in lower or "mistral" in lower or "qwen" in lower or "gemma" in lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "v_proj", "k_proj", "o_proj", "query_key_value"]


def maybe_set_hf_cache() -> None:
    # AutoDL 官方建议把 HF 缓存转到数据盘，避免系统盘被占满。
    # https://www.autodl.com/docs/huggingface/
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))


def is_local_peft_adapter(model_name_or_path: str | os.PathLike[str]) -> bool:
    path = Path(model_name_or_path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def load_causal_lm(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    is_trainable: bool = False,
):
    from transformers import AutoModelForCausalLM

    if is_local_peft_adapter(model_name_or_path):
        try:
            from peft import AutoPeftModelForCausalLM

            return AutoPeftModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                is_trainable=is_trainable,
            )
        except Exception:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch_dtype,
            )
            return PeftModel.from_pretrained(
                base_model,
                model_name_or_path,
                is_trainable=is_trainable,
            )

    return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)


def load_sequence_classification_model(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype,
    num_labels: int = 1,
    is_trainable: bool = False,
):
    from transformers import AutoModelForSequenceClassification

    if is_local_peft_adapter(model_name_or_path):
        try:
            from peft import AutoPeftModelForSequenceClassification

            return AutoPeftModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                is_trainable=is_trainable,
            )
        except Exception:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
            )
            return PeftModel.from_pretrained(
                base_model,
                model_name_or_path,
                is_trainable=is_trainable,
            )

    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        torch_dtype=torch_dtype,
    )
