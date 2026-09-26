"""Portable, compatibility-checked checkpoints with optimizer and RNG state."""
import hashlib,json,random,subprocess
from pathlib import Path
import numpy as np
import torch

def fingerprint(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: return "uncommitted"
def save_checkpoint(path,model,optimizer,config,step,tokens_seen,tokenizer_fingerprint,dataset_version="unspecified",scaler=None,scheduler=None,epoch=0,best_validation_loss=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,"scaler":scaler.state_dict() if scaler else None,"config":config.to_dict(),"step":step,"epoch":epoch,"tokens_seen":tokens_seen,"best_validation_loss":best_validation_loss,"metadata":{"model_name":config.model_name,"model_version":config.version,"git_commit":git_commit(),"config_hash":fingerprint(config.to_dict()),"tokenizer_fingerprint":tokenizer_fingerprint,"dataset_version":dataset_version},"rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    tmp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,tmp); tmp.replace(path)
def load_checkpoint(path,model,optimizer,config,tokenizer_fingerprint,scaler=None,scheduler=None,strict=True,dataset_version=None):
    ckpt=torch.load(path,map_location="cpu",weights_only=False); m=ckpt["metadata"]
    expected={"model_name":config.model_name,"model_version":config.version,"config_hash":fingerprint(config.to_dict()),"tokenizer_fingerprint":tokenizer_fingerprint}
    if dataset_version is not None: expected["dataset_version"]=dataset_version
    mismatches=[f"{k}: checkpoint={m.get(k)!r}, current={v!r}" for k,v in expected.items() if m.get(k)!=v]
    if mismatches and strict: raise ValueError("Checkpoint is incompatible:\n"+"\n".join(mismatches))
    model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    random.setstate(ckpt["rng"]["python"]); np.random.set_state(ckpt["rng"]["numpy"]); torch.set_rng_state(ckpt["rng"]["torch"])
    if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None: torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    return ckpt["step"],ckpt["tokens_seen"],ckpt.get("best_validation_loss")
