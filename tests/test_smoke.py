"""End-to-end smoke tests: tokenizer -> dataset -> model -> train step -> checkpoint -> fresh-process reload -> generation.

Run with:  python tests/run_all.py   (or: pytest tests/test_smoke.py -q)
CPU-friendly: the model-level checks use configs/debug.json.
"""
import json,subprocess,sys
import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
from lyra.data import PackedTextDataset
from lyra.generation import format_chat,GENERATION_DEFAULTS
from lyra.checkpoint import save_checkpoint,load_checkpoint

def _tiny_tokenizer(tmp_path):
 corpus=tmp_path/'corpus.txt'
 corpus.write_text(
  ('Привет! Как дела? Hello world. User speaks, assistant answers.\n'
   'line one\nline two https://example.com 12345 test, test; punctuation!?\n'
   'Смешанный mixed текст with English words внутри предложения.\n')*20,encoding='utf8')
 return LyraTokenizer.train([corpus],vocab_size=512)

def test_tokenizer_bilingual_roundtrip(tmp_path):
 tok=_tiny_tokenizer(tmp_path)
 for s in ['Привет, как дела?','Hello, world!','Смешанный mixed текст ok?','12345 https://example.com','line one\nline two','a,b;c:d!e?f','   spaced   out   ']:
  assert tok.decode(tok.encode(s))==s, s
 for special in ("<PAD>","<UNK>","<BOS>","<EOS>","<SYSTEM>","<USER>","<ASSISTANT>","<TOOL>","<END>"):
  tok.id(special)  # must exist, must not raise
 assert tok.vocab_size <= 512

def test_chat_format_matches_training_stream(tmp_path):
 tok=_tiny_tokenizer(tmp_path)
 messages=[{'role':'system','content':'Be kind.'},{'role':'user','content':'Привет!'},{'role':'assistant','content':'Привет! Чем помочь?'}]
 prompt=format_chat(messages,tok,add_assistant_trigger=True)
 assert prompt[-1]==tok.id('<ASSISTANT>')
 ds=PackedTextDataset([],tok,seq_len=64)
 train_ids,_=ds._conversation({'messages':messages})
 # Training stream for the same turns (no trailing trigger) must be a prefix of the inference prompt.
 assert prompt[:len(train_ids)]==train_ids
 # Every turn is ROLE body END; bodies never smuggle a second role boundary.
 assert train_ids.count(tok.id('<END>'))==3
 assert train_ids.count(tok.id('<USER>'))==1 and train_ids.count(tok.id('<ASSISTANT>'))==1

def test_model_forward_and_loss(tmp_path):
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c)
 ids=torch.randint(0,c.vocab_size,(2,32)); out=m(ids,ids)
 assert out['logits'].shape==(2,32,c.vocab_size)
 assert torch.isfinite(out['loss']).item()

def test_minimal_training_step_updates_weights():
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c); opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
 before=m.embed.weight.detach().clone()
 ids=torch.randint(0,c.vocab_size,(2,32)); opt.zero_grad(set_to_none=True); m(ids,ids)['loss'].backward()
 torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
 assert not torch.equal(before,m.embed.weight)

def test_generation_shape_and_eos_stop():
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c)
 ids=torch.tensor([[1,2,3]])
 out=m.generate(ids,max_new_tokens=5,temperature=0,top_k=0,eos_token_id=-1)
 assert out.dim()==2 and out.shape[0]==1 and 4<=out.shape[1]<=8
 assert out[0,:3].tolist()==[1,2,3]

def test_checkpoint_save_load_and_fingerprint(tmp_path):
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c); opt=torch.optim.AdamW(m.parameters())
 ids=torch.randint(0,c.vocab_size,(1,16)); m(ids,ids)['loss'].backward(); opt.step()
 path=tmp_path/'latest.pt'
 save_checkpoint(path,m,opt,c,step=3,tokens_seen=48,tokenizer_fingerprint='tok',dataset_version='ds')
 assert not path.with_suffix('.pt.tmp').exists()
 m2=LyraModel(c); opt2=torch.optim.AdamW(m2.parameters())
 step,tokens,best,epoch=load_checkpoint(path,m2,opt2,c,'tok',dataset_version='ds')
 assert (step,tokens,best,epoch)==(3,48,None,0)
 for k,v in m.state_dict().items(): assert torch.equal(v,m2.state_dict()[k])
 try: load_checkpoint(path,m2,opt2,c,'other',dataset_version='ds')
 except ValueError as e: assert 'tokenizer_fingerprint' in str(e)
 else: raise AssertionError('incompatible tokenizer was accepted')

def test_fresh_process_reload_and_generate(tmp_path):
 """Checkpoint + tokenizer must work in a NEW process (Kaggle resume reality)."""
 import hashlib
 tok=_tiny_tokenizer(tmp_path); tok_path=tmp_path/'tok.json'; tok.save(tok_path)
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c); opt=torch.optim.AdamW(m.parameters())
 ckpt=tmp_path/'latest.pt'
 save_checkpoint(ckpt,m,opt,c,step=1,tokens_seen=16,tokenizer_fingerprint=hashlib.sha256(tok_path.read_bytes()).hexdigest(),dataset_version='ds')
 probe=tmp_path/'probe.py'
 probe.write_text(
  'import sys,torch\n'
  'from lyra.config import LyraConfig\n'
  'from lyra.model import LyraModel\n'
  'from lyra.tokenizer import LyraTokenizer\n'
  'from lyra.checkpoint import load_checkpoint\n'
  f"c=LyraConfig.from_json('configs/debug.json'); tok=LyraTokenizer.from_file(r'{tok_path}'); m=LyraModel(c); opt=torch.optim.AdamW(m.parameters())\n"
  f"step,tokens,best,epoch=load_checkpoint(r'{ckpt}',m,opt,c,__import__('hashlib').sha256(open(r'{tok_path}','rb').read()).hexdigest(),dataset_version='ds')\n"
  'm.eval(); x=torch.tensor([[tok.id(\"<USER>\")]]); y=m.generate(x,max_new_tokens=3,temperature=0,top_k=0)\n'
  'assert y.shape==(1,4),(y.shape,); print(f\"FRESH_RELOAD_OK step={step} out={y.shape}\")\n',encoding='utf8')
 r=subprocess.run([sys.executable,str(probe)],capture_output=True,text=True,timeout=300)
 assert r.returncode==0,(r.stdout,r.stderr); assert 'FRESH_RELOAD_OK' in r.stdout

def test_packed_sft_batch_minimal_step(tmp_path):
 tok=_tiny_tokenizer(tmp_path)
 f=tmp_path/'train.jsonl'
 rows=[{'messages':[{'role':'user','content':'Привет, как дела?'},{'role':'assistant','content':'Хорошо! А у тебя?'}]},
       {'messages':[{'role':'user','content':'What is your name?'},{'role':'assistant','content':'I am Lyra.'}]}]
 f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
 c=LyraConfig.from_json('configs/debug.json')
 ds=PackedTextDataset([f],tok,seq_len=8,assistant_only=True)
 batch=[next(iter(ds)),next(iter(ds))]
 ids=torch.stack([b['input_ids'] for b in batch]); labs=torch.stack([b['labels'] for b in batch])
 m=LyraModel(c); out=m(ids,labs)
 assert torch.isfinite(out['loss']).item()
 assert (labs==-100).any() and (labs!=-100).any()
 assert GENERATION_DEFAULTS['eos_token']=='<END>'
