"""Stage 1/2 conversation LM training with AMP, accumulation, schedules, and resume."""
import argparse,glob,hashlib,json,random,subprocess,sys
from pathlib import Path

# Add project root to sys.path so `import lyra` works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
from lyra.data import PackedTextDataset
from lyra.checkpoint import save_checkpoint,load_checkpoint,fingerprint

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/debug.json'); p.add_argument('--tokenizer'); p.add_argument('--data'); p.add_argument('--validation'); p.add_argument('--stage',choices=['pretrain','sft']); p.add_argument('--steps',type=int,help='target total optimizer steps for this stage; resume runs until this target'); p.add_argument('--batch-size',type=int); p.add_argument('--grad-accum',type=int); p.add_argument('--lr',type=float); p.add_argument('--warmup-steps',type=int); p.add_argument('--save-every',type=int); p.add_argument('--archive-every',type=int); p.add_argument('--keep-last-checkpoints',type=int,default=5); p.add_argument('--eval-every',type=int); p.add_argument('--resume'); p.add_argument('--reset-stage',action='store_true',help='load weights but restart stage step/scheduler/data cursor; permits a new dataset while still validating model/tokenizer config'); p.add_argument('--allow-dataset-change',action='store_true',help='resume optimizer/scheduler/step while accepting a new dataset fingerprint'); p.add_argument('--out'); p.add_argument('--seed',type=int); a=p.parse_args()
 raw_cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig')); tr=raw_cfg.get('training',{})
 defaults={'tokenizer':'artifacts/tokenizer/tokenizer.json','data':'data/processed/oasst_ru/train.jsonl','validation':'data/processed/oasst_ru/validation.jsonl','stage':'pretrain','steps':1000,'batch_size':1,'grad_accum':8,'lr':3e-4,'warmup_steps':100,'save_every':100,'archive_every':0,'eval_every':100,'out':'checkpoints','seed':17}
 for key,value in defaults.items():
  if getattr(a,key) is None:setattr(a,key,tr.get(key,value))
 if tr.get('require_cuda',False) and not torch.cuda.is_available(): raise RuntimeError('This config requires CUDA. Local CPU runs are disabled for the Colab training profile.')
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
 device='cuda' if torch.cuda.is_available() else 'cpu'; cfg=LyraConfig.from_json(a.config); tok=LyraTokenizer.from_file(a.tokenizer); tfp=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
 if tok.vocab_size!=cfg.vocab_size: raise ValueError(f'Tokenizer vocabulary ({tok.vocab_size}) does not match model config ({cfg.vocab_size}).')
 files=glob.glob(a.data,recursive=True)
 if not files: raise FileNotFoundError(f'No training files matched {a.data!r}')
 if a.reset_stage and not a.resume: raise ValueError('--reset-stage requires --resume with a real checkpoint')
 if a.keep_last_checkpoints < 0: raise ValueError('--keep-last-checkpoints cannot be negative')
 stats_path=Path(files[0]).parent/'dataset_stats.json'
 if stats_path.exists(): dataset_version=fingerprint(json.loads(stats_path.read_text(encoding='utf-8-sig')))
 else: dataset_version=fingerprint([(Path(f).name,Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in sorted(files)])
 val_files=glob.glob(a.validation,recursive=True) if a.validation else []
 dataset_info={'train_files':[{'path':f,'bytes':Path(f).stat().st_size} for f in sorted(files)],'validation_files':[{'path':f,'bytes':Path(f).stat().st_size} for f in sorted(val_files)],'stats':json.loads(stats_path.read_text(encoding='utf-8-sig')) if stats_path.exists() else {}}
 ds=PackedTextDataset(files,tok,cfg.context_length,seed=a.seed,assistant_only=(a.stage=='sft')); dl=DataLoader(ds,batch_size=a.batch_size,num_workers=0)
 model=LyraModel(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.1)
 def lr_scale(s):
  if s<a.warmup_steps:return max(1e-3,(s+1)/max(1,a.warmup_steps))
  progress=min(1.0,(s-a.warmup_steps)/max(1,a.steps-a.warmup_steps)); return 0.5*(1+np.cos(np.pi*progress))
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_scale)
 amp_dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
 try:
  scaler=torch.amp.GradScaler('cuda',enabled=(device=='cuda' and amp_dtype==torch.float16))
 except (AttributeError,TypeError):
  scaler=torch.cuda.amp.GradScaler(enabled=(device=='cuda' and amp_dtype==torch.float16))
 step=tokens=0; best_val=float('inf'); ckpt_epoch=0
 if a.resume is None and (Path(a.out)/'latest.pt').exists(): a.resume='latest'
 if a.resume and a.resume!='latest':
  step,tokens,loaded_best,ckpt_epoch=load_checkpoint(a.resume,model,opt,cfg,tfp,scaler,scheduler=None if a.reset_stage else sched,dataset_version=None if (a.reset_stage or a.allow_dataset_change) else dataset_version)
  if a.reset_stage: step=0
  elif loaded_best is not None: best_val=loaded_best
 elif a.resume=='latest':
  latest=Path(a.out)/'latest.pt'
  if not latest.exists(): raise FileNotFoundError(f'--resume latest requested, but no checkpoint exists at {latest}')
  step,tokens,loaded_best,ckpt_epoch=load_checkpoint(latest,model,opt,cfg,tfp,scaler,scheduler=None if a.reset_stage else sched,dataset_version=None if (a.reset_stage or a.allow_dataset_change) else dataset_version)
  if a.reset_stage: step=0
  elif loaded_best is not None: best_val=loaded_best
 if a.reset_stage:
  # The optimizer state is retained, but the new phase starts at its own LR schedule.
  for group,base_lr in zip(opt.param_groups,sched.base_lrs): group['lr']=base_lr*lr_scale(0)
 # Recreate the deterministic stream and skip already-consumed microbatches on resume.
 epoch=0; ds.set_epoch(epoch); it=iter(dl)
 for _ in range(step*a.grad_accum):
  try: next(it)
  except StopIteration:
   epoch+=1; ds.set_epoch(epoch); it=iter(dl); next(it)
 if (a.resume and not a.reset_stage) and epoch!=ckpt_epoch:
  print(json.dumps({'warning':'resumed data epoch differs from checkpoint epoch','replayed_epoch':epoch,'checkpoint_epoch':ckpt_epoch}),flush=True)
 val_dataset=lambda: PackedTextDataset(val_files,tok,cfg.context_length,seed=a.seed,assistant_only=(a.stage=='sft'))
 val_it=iter(DataLoader(val_dataset(),batch_size=a.batch_size)) if val_files else None
 def batch_next(iterator,is_val=False):
  nonlocal epoch,it
  try:return next(iterator)
  except StopIteration:
   if is_val:return None
   epoch+=1; ds.set_epoch(epoch); it=iter(dl); return next(it)
 model.train(); opt.zero_grad(set_to_none=True); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 log_path=out/'training.jsonl'
 def emit(record):
  line=json.dumps(record,ensure_ascii=False); print(line,flush=True)
  with log_path.open('a',encoding='utf-8') as log: log.write(line+'\n'); log.flush()
 def save(path):
  save_checkpoint(path,model,opt,cfg,step,tokens,tfp,dataset_version=dataset_version,scaler=scaler,scheduler=sched,epoch=epoch,best_validation_loss=best_val,stage=a.stage,dataset_info=dataset_info)
 def prune_archives():
  archives=sorted(out.glob('checkpoint_step_*.pt'),key=lambda f:f.stat().st_mtime_ns)
  for old in archives[:-a.keep_last_checkpoints] if a.keep_last_checkpoints else archives:
   old.unlink(missing_ok=True)
 while step<a.steps:
  total_loss=0.0
  for _ in range(a.grad_accum):
   batch=batch_next(it); ids=batch['input_ids'].to(device); labels=batch['labels'].to(device)
   with torch.autocast(device_type=device,dtype=amp_dtype,enabled=(device=='cuda')): loss=model(ids,labels)['loss']/a.grad_accum
   scaler.scale(loss).backward(); total_loss+=float(loss.detach()); tokens+=ids.numel()
  scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step+=1
  record={'step':step,'stage':a.stage,'loss':total_loss,'tokens_seen':tokens,'lr':sched.get_last_lr()[0],'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None}
  if device=='cuda' and step%100==0:
   record.update(gpu=torch.cuda.get_device_name(0),gpu_memory_allocated_gib=round(torch.cuda.memory_allocated(0)/1024**3,2),gpu_memory_reserved_gib=round(torch.cuda.memory_reserved(0)/1024**3,2))
   try:
    util=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits','--id=0'],text=True,stderr=subprocess.DEVNULL,timeout=2).strip().splitlines()[0]
    record['gpu_utilization_percent']=int(util)
   except (OSError,subprocess.SubprocessError,ValueError,IndexError): pass
  if step%10==0: emit(record)
  if val_it is not None and step%a.eval_every==0:
   model.eval(); vals=[]
   with torch.no_grad():
    for _ in range(10):
     vb=batch_next(val_it,True)
     if vb is None: val_it=iter(DataLoader(val_dataset(),batch_size=a.batch_size)); vb=next(val_it)
     with torch.autocast(device_type=device,dtype=amp_dtype,enabled=(device=='cuda')): vals.append(float(model(vb['input_ids'].to(device),vb['labels'].to(device))['loss']))
   val_loss=sum(vals)/len(vals); emit({'step':step,'stage':a.stage,'validation_loss':val_loss,'tokens_seen':tokens})
   if val_loss<best_val:
    best_val=val_loss; save(out/'best.pt')
   model.train()
  if step%a.save_every==0:
   if a.archive_every and step%a.archive_every==0:
    save(out/f'checkpoint_step_{step:08d}.pt'); prune_archives()
   save(out/'latest.pt')
 # latest.pt is the resumable final state; avoid a second multi-gigabyte copy.
 if step%a.save_every: save(out/'latest.pt')
if __name__=='__main__':main()
