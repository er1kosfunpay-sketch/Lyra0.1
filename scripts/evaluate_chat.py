"""Run a text-only chat benchmark and save model outputs for human review."""
import argparse,json,math
from pathlib import Path
import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
p=argparse.ArgumentParser(); p.add_argument('--model',default='exports/lyra'); p.add_argument('--benchmark',default='lyra/evaluation/chat_benchmark_0_1.jsonl'); p.add_argument('--out',default='logs/benchmark_outputs.jsonl'); p.add_argument('--max-new-tokens',type=int,default=128); a=p.parse_args()
r=Path(a.model); c=LyraConfig.from_json(r/'config.json'); tok=LyraTokenizer.from_file(r/'tokenizer.json'); dev='cuda' if torch.cuda.is_available() else 'cpu'; m=LyraModel(c).to(dev); m.load_state_dict(torch.load(r/'model.pt',map_location=dev,weights_only=True)); m.eval(); seen=total=repeat=0; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
with open(out,'w',encoding='utf8') as fo, open(a.benchmark,encoding='utf8') as fi:
 for line in fi:
  case=json.loads(line); ids=[]
  for msg in case['messages']:
   ids.append(tok.id('<USER>' if msg['role']=='user' else '<ASSISTANT>')); ids+=tok.encode(msg['content'])
  x=torch.tensor([ids],device=dev)
  y=m.generate(x,max_new_tokens=a.max_new_tokens,temperature=.8,top_p=.9,top_k=50,repetition_penalty=1.08,eos_token_id=tok.id('<END>'))
  answer=tok.decode(y[0,len(ids):].tolist()); toks=tok.encode(answer); repeated=(len(toks)!=len(set(toks)))
  row={**case,'generated':answer,'generated_tokens':len(toks),'token_repeat':repeated}; fo.write(json.dumps(row,ensure_ascii=False)+'\n'); total+=1; repeat+=repeated
print(json.dumps({'cases':total,'token_repetition_case_rate':repeat/total if total else None,'human_quality_scores':'NOT TESTED','outputs':str(out)},ensure_ascii=False))
