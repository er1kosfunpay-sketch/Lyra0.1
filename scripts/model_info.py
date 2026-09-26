import argparse,json
from lyra.config import LyraConfig
from lyra.model import LyraModel
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/lyra_0_1.json'); a=p.parse_args(); c=LyraConfig.from_json(a.config); m=LyraModel(c)
print(json.dumps({'config':c.to_dict(),'parameters':m.parameter_count(),'parameter_estimate':c.parameter_count(),'embedding_parameters':c.vocab_size*c.hidden_size,'weights_fp32_bytes':m.parameter_count()*4,'weights_bf16_bytes':m.parameter_count()*2},indent=2))
