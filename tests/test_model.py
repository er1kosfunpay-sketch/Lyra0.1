import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel

def test_parameter_count_exact():
 c=LyraConfig.from_json('configs/lyra_0_1.json'); m=LyraModel(c)
 assert m.parameter_count()==c.parameter_count()
 assert 1_200_000_000 <= m.parameter_count() <= 1_800_000_000

def test_model_forward_backward():
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c); ids=torch.randint(0,c.vocab_size,(2,32)); o=m(ids,ids)
 assert o['logits'].shape==(2,32,c.vocab_size); o['loss'].backward()
 assert m.embed.weight.grad is not None

def test_attention_gqa_shapes():
 from lyra.model import CausalSelfAttention
 c=LyraConfig.from_json('configs/debug.json'); y=CausalSelfAttention(c)(torch.randn(2,12,c.hidden_size)); assert y.shape==(2,12,c.hidden_size)
