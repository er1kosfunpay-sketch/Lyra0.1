"""Interactive local chat keeps recent context and prints assistant replies."""
import argparse
from pathlib import Path
import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
from lyra.generation import GENERATION_DEFAULTS,fit_history_to_context

SYSTEM_PROMPT='You are Lyra 0.1. Be conversational, concise when appropriate, and do not invent facts you do not know.'

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--model',default='exports/lyra'); p.add_argument('--max-new-tokens',type=int,default=256); a=p.parse_args(argv); root=Path(a.model)
 c=LyraConfig.from_json(root/'config.json'); tok=LyraTokenizer.from_file(root/'tokenizer.json'); dev='cuda' if torch.cuda.is_available() else 'cpu'; m=LyraModel(c).to(dev); m.load_state_dict(torch.load(root/'model.pt',map_location=dev,weights_only=True)); m.eval(); d=GENERATION_DEFAULTS
 hist=[{'role':'system','content':SYSTEM_PROMPT}]
 print('Lyra 0.1 — type /exit to quit')
 while True:
  try: user=input('you> ').strip()
  except (EOFError,KeyboardInterrupt): print(); break
  if user=='/exit':break
  if not user: continue
  hist.append({'role':'user','content':user})
  hist,ids=fit_history_to_context(hist,tok,c.context_length,reserve=a.max_new_tokens+8)
  x=torch.tensor([ids],device=dev)
  with torch.inference_mode():
   y=m.generate(x,max_new_tokens=a.max_new_tokens,temperature=d['temperature'],top_p=d['top_p'],top_k=d['top_k'],repetition_penalty=d['repetition_penalty'],eos_token_id=tok.id(d['eos_token']))
  gen=y[0,len(ids):].tolist(); end=tok.id(d['eos_token'])
  if end in gen: gen=gen[:gen.index(end)]
  answer=tok.decode(gen)
  print(f'lyra> {answer}',flush=True); hist.append({'role':'assistant','content':answer})

if __name__=='__main__':main()
