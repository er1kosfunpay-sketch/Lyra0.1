"""Export original trained Lyra weights; optional dtype conversion is non-destructive."""
import argparse,json
from pathlib import Path
import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--tokenizer',required=True); p.add_argument('--out',default='exports/lyra'); p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='bfloat16'); a=p.parse_args()
c=LyraConfig.from_json(a.config); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); m=LyraModel(c); m.load_state_dict(ck['model']); dtype=getattr(torch,a.dtype); m.to(dtype=dtype)
out=Path(a.out); out.mkdir(parents=True,exist_ok=True); torch.save(m.state_dict(),out/'model.pt'); (out/'config.json').write_text(json.dumps(c.to_dict(),indent=2),encoding='utf8'); LyraTokenizer.from_file(a.tokenizer).save(out/'tokenizer.json'); (out/'generation_config.json').write_text(json.dumps({'preset':'lyra_chat','temperature':0.8,'top_p':0.9,'top_k':50,'repetition_penalty':1.08,'max_new_tokens':256,'eos_token':'<END>'},indent=2),encoding='utf8'); (out/'README.md').write_text('# Lyra 0.1 export\n\nWeights are from the specified local Lyra training checkpoint. The model was randomly initialized before training. See the repository report for training/evaluation provenance.\n',encoding='utf8')
print(f'Exported {m.parameter_count():,} Lyra parameters to {out}')
