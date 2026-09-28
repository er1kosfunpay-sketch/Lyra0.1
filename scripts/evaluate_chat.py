"""Run a text-only chat benchmark and save model outputs for human review."""
import argparse,json
from pathlib import Path
import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
from lyra.generation import GENERATION_DEFAULTS,format_chat

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--model',default='exports/lyra'); p.add_argument('--benchmark',default='lyra/evaluation/chat_benchmark_0_1.jsonl'); p.add_argument('--out',default='logs/benchmark_outputs.jsonl'); p.add_argument('--max-new-tokens',type=int,default=128); a=p.parse_args(argv)
 r=Path(a.model); c=LyraConfig.from_json(r/'config.json'); tok=LyraTokenizer.from_file(r/'tokenizer.json'); dev='cuda' if torch.cuda.is_available() else 'cpu'; m=LyraModel(c).to(dev); m.load_state_dict(torch.load(r/'model.pt',map_location=dev,weights_only=True)); m.eval()
 d=GENERATION_DEFAULTS; end_id=tok.id(d['eos_token'])
 role_ids={tok.id(t) for t in ('<USER>','<ASSISTANT>','<SYSTEM>','<TOOL>')}
 seen=total=repeat=empty=role_leak=no_eos=0; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
 with open(out,'w',encoding='utf8') as fo, open(a.benchmark,encoding='utf8') as fi:
  for line in fi:
   case=json.loads(line)
   # Identical prompt construction to training: role+body+END per turn plus
   # a trailing <ASSISTANT> trigger (see lyra.generation.format_chat).
   ids=format_chat(case['messages'],tok,add_assistant_trigger=True)
   if len(ids)>=c.context_length: ids=ids[-(c.context_length-1):]
   x=torch.tensor([ids],device=dev)
   y=m.generate(x,max_new_tokens=a.max_new_tokens,temperature=d['temperature'],top_p=d['top_p'],top_k=d['top_k'],repetition_penalty=d['repetition_penalty'],eos_token_id=end_id)
   gen=y[0,len(ids):].tolist(); stopped=bool(gen and gen[-1]==end_id)
   if end_id in gen: gen=gen[:gen.index(end_id)]
   answer=tok.decode(gen); toks=tok.encode(answer)
   repeated=(len(toks)>8 and len(set(toks))/len(toks)<0.25)
   row={**case,'generated':answer,'generated_tokens':len(toks),'token_repeat':repeated,'empty':not answer.strip(),'role_leak':bool(role_ids.intersection(gen)),'stopped_on_eos':stopped}
   fo.write(json.dumps(row,ensure_ascii=False)+'\n'); total+=1; repeat+=repeated; empty+=not answer.strip(); role_leak+=row['role_leak']; no_eos+=not stopped
 print(json.dumps({'cases':total,'token_repetition_case_rate':repeat/total if total else None,'empty_rate':empty/total if total else None,'role_leak_rate':role_leak/total if total else None,'no_eos_rate':no_eos/total if total else None,'human_quality_scores':'NOT TESTED','outputs':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
