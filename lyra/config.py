"""Configuration and exact architectural parameter accounting."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

@dataclass
class LyraConfig:
    model_name: str = "Lyra"
    version: str = "0.1.0"
    vocab_size: int = 16384
    context_length: int = 1024
    hidden_size: int = 1024
    num_layers: int = 20
    num_attention_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 3072
    rope_theta: float = 10000.0
    rope_scaling: Any = None
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    def __post_init__(self):
        assert self.hidden_size == self.num_attention_heads * self.head_dim
        assert self.num_attention_heads % self.num_kv_heads == 0
        assert self.head_dim % 2 == 0
        assert self.context_length > 0 and self.vocab_size > 0
        assert self.intermediate_size > 0 and self.num_layers > 0
        assert 0.0 <= self.dropout < 1.0
        if self.rope_scaling is not None:
            assert self.rope_scaling.get("type") == "linear" and self.rope_scaling.get("factor", 0) > 0

    @classmethod
    def from_json(cls, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        # Colab configs may include a separate `training` section; all model
        # keys stay strict so misspelled architecture fields still fail fast.
        raw.pop("training", None)
        return cls(**raw)

    def to_dict(self):
        return asdict(self)

    def parameter_count(self) -> int:
        h, d, m, v, n = self.hidden_size, self.head_dim, self.intermediate_size, self.vocab_size, self.num_layers
        # Q/O projections plus grouped K/V projections; SwiGLU has gate/up/down.
        attention = h*h + 2*self.num_kv_heads*d*h + h*h
        mlp = 3*h*m
        block_norms = 2*h
        final_norm = h
        embedding = v*h
        lm_head = 0 if self.tie_embeddings else v*h
        return embedding + n*(attention + mlp + block_norms) + final_norm + lm_head
