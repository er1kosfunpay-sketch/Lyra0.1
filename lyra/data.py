"""Streaming JSONL conversation packing, optionally masking non-assistant targets."""
import json,random
import torch
from torch.utils.data import IterableDataset
class PackedTextDataset(IterableDataset):
    def __init__(self, files, tokenizer, seq_len, seed=17, assistant_only=False): self.files=list(files); self.tokenizer=tokenizer; self.seq_len=seq_len; self.seed=seed; self.epoch=0; self.assistant_only=assistant_only
    def set_epoch(self,epoch): self.epoch=epoch
    def _conversation(self,row):
        ids=[]; labels=[]
        for m in row.get('messages',[]):
            role=m.get('role','user').upper()
            if role not in ('USER','ASSISTANT','SYSTEM'):continue
            r=[self.tokenizer.id(f'<{role}>')]
            body=self.tokenizer.encode(m.get('content',''))
            end=[self.tokenizer.id('<END>')]
            segment=r+body+end; ids.extend(segment)
            if self.assistant_only and role!='ASSISTANT':labels.extend([-100]*len(segment))
            else:labels.extend(segment)
        return ids,labels
    def __iter__(self):
        paths=self.files.copy(); random.Random(self.seed+self.epoch).shuffle(paths); buf=[]; targets=[]
        for path in paths:
            with open(path,encoding='utf-8') as f:
                for line in f:
                    try: row=json.loads(line)
                    except json.JSONDecodeError: continue
                    if 'messages' in row: ids,labs=self._conversation(row)
                    else:
                        ids=self.tokenizer.encode(row.get('text',line),add_eos=True); labs=ids.copy()
                    buf.extend(ids); targets.extend(labs)
                    while len(buf)>=self.seq_len+1:
                        chunk=buf[:self.seq_len+1]; lab=targets[:self.seq_len+1]
                        del buf[:self.seq_len]; del targets[:self.seq_len]
                        # Avoid optimizer steps on windows containing only
                        # masked user/system targets during assistant-only SFT.
                        if self.assistant_only and not any(x!=-100 for x in lab[1:]): continue
                        yield {'input_ids':torch.tensor(chunk[:-1],dtype=torch.long),'labels':torch.tensor(lab[:-1],dtype=torch.long)}
