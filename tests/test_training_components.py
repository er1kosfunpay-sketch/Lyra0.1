import json,torch
from lyra.checkpoint import save_checkpoint,load_checkpoint
from lyra.config import LyraConfig
from lyra.data import PackedTextDataset
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer

def test_generation_token_loop():
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c); ids=torch.tensor([[1,2,3]]); out=m.generate(ids,max_new_tokens=3,temperature=0,top_k=0)
 assert out.shape==(1,6)

def test_checkpoint_resume_and_compatibility(tmp_path):
 c=LyraConfig.from_json('configs/debug.json'); m=LyraModel(c); opt=torch.optim.AdamW(m.parameters()); ids=torch.randint(0,c.vocab_size,(1,16)); m(ids,ids)['loss'].backward(); opt.step()
 path=tmp_path/'latest.pt'; save_checkpoint(path,m,opt,c,step=3,tokens_seen=48,tokenizer_fingerprint='tok',dataset_version='ds')
 m2=LyraModel(c); opt2=torch.optim.AdamW(m2.parameters()); step,tokens,best=load_checkpoint(path,m2,opt2,c,'tok',dataset_version='ds')
 assert (step,tokens,best)==(3,48,None)
 for k,v in m.state_dict().items(): assert torch.equal(v,m2.state_dict()[k])
 try: load_checkpoint(path,m2,opt2,c,'other',dataset_version='ds')
 except ValueError as e: assert 'tokenizer_fingerprint' in str(e)
 else: raise AssertionError('incompatible tokenizer was accepted')

def test_sft_data_masks_user_targets(tmp_path):
 f=tmp_path/'train.jsonl'; f.write_text(json.dumps({'messages':[{'role':'user','content':'Please tell me a story.'},{'role':'assistant','content':'A small fox walked home.'}]})+'\n',encoding='utf8')
 tok=LyraTokenizer.train([f],vocab_size=512); ds=PackedTextDataset([f],tok,seq_len=8,assistant_only=True); item=next(iter(ds))
 assert item['input_ids'].shape==(8,); assert (item['labels']==-100).any(); assert (item['labels']!=-100).any()
