"""Interactive local chat keeps recent context and prints decoded tokens incrementally."""
import argparse,torch
from pathlib import Path
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
p=argparse.ArgumentParser(); p.add_argument('--model',default='exports/lyra'); p.add_argument('--max-new-tokens',type=int,default=256); a=p.parse_args(); root=Path(a.model)
c=LyraConfig.from_json(root/'config.json'); tok=LyraTokenizer.from_file(root/'tokenizer.json'); dev='cuda' if torch.cuda.is_available() else 'cpu'; m=LyraModel(c).to(dev); m.load_state_dict(torch.load(root/'model.pt',map_location=dev,weights_only=True)); m.eval(); end=tok.id('<END>'); hist=[{'role':'system','content':'You are Lyra 0.1. Be conversational, concise when appropriate, and do not invent facts you do not know.'}]
print('Lyra 0.1 — type /exit to quit')
while True:
 try: user=input('you> ').strip()
 except (EOFError,KeyboardInterrupt): print(); break
 if user=='/exit':break
 hist.append({'role':'user','content':user});
 def flatten():
  out=[]
  for msg in hist:
   out.append(tok.id(f"<{msg['role'].upper()}>")); out.extend(tok.encode(msg['content']))
  out.append(tok.id('<ASSISTANT>')); return out
 ids=flatten()
 while len(ids)>=c.context_length and len(hist)>2:
  del hist[1:3]; ids=flatten()
 if len(ids)>=c.context_length: ids=ids[-(c.context_length-1):]
 x=torch.tensor([ids],device=dev); answer_ids=[]; printed=''
 print('lyra> ',end='',flush=True)
 for _ in range(a.max_new_tokens):
  nxt=m.generate(x,max_new_tokens=1,temperature=.8,top_p=.9,top_k=50,repetition_penalty=1.08,eos_token_id=end)[:,-1:]
  if nxt.item()==end:break
  x=torch.cat((x,nxt),1); answer_ids.append(nxt.item()); text=tok.decode(answer_ids)
  if text.startswith(printed): print(text[len(printed):],end='',flush=True); printed=text
 print(); hist.append({'role':'assistant','content':tok.decode(answer_ids)})
