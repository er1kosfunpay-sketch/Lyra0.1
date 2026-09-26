"""Minimal local inference API."""
import os
from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
app=FastAPI(title='Lyra 0.1 API')
_model=_tokenizer=_config=None
class GenerateRequest(BaseModel): prompt:str; max_new_tokens:int=Field(default=128,ge=1,le=2048); temperature:float=Field(default=0.8,ge=0)
class ChatRequest(BaseModel): messages:list[dict[str,str]]; max_new_tokens:int=128; temperature:float=0.8
def load():
 global _model,_tokenizer,_config
 root=Path(os.getenv('LYRA_MODEL_DIR','exports/lyra'))
 try:
  _config=LyraConfig.from_json(root/'config.json'); _tokenizer=LyraTokenizer.from_file(root/'tokenizer.json'); device='cuda' if torch.cuda.is_available() else 'cpu'; _model=LyraModel(_config).to(device); _model.load_state_dict(torch.load(root/'model.pt',map_location=device,weights_only=True)); _model.eval()
 except FileNotFoundError: _model=None
@app.on_event('startup')
def startup(): load()
@app.get('/health')
def health(): return {'status':'ok' if _model else 'model_not_loaded'}
@app.get('/info')
def info():
 if not _config: raise HTTPException(503,'Model is not loaded')
 return {'name':_config.model_name,'version':_config.version,'parameters':_model.parameter_count(),'context_length':_config.context_length,'vocab_size':_config.vocab_size,'dtype':_config.dtype}
def generate(prompt,max_new_tokens,temperature):
 if not _model: raise HTTPException(503,'Exported model is not available; train and export Lyra first.')
 ids=torch.tensor([_tokenizer.encode(prompt)],device=next(_model.parameters()).device); out=_model.generate(ids,max_new_tokens=max_new_tokens,temperature=temperature,eos_token_id=_tokenizer.id('<END>'))
 return _tokenizer.decode(out[0,len(ids[0]):].tolist())
@app.post('/generate')
def generate_route(req:GenerateRequest): return {'model':_config.model_name if _config else 'Lyra','text':generate(req.prompt,req.max_new_tokens,req.temperature)}
@app.post('/chat')
def chat(req:ChatRequest):
 prompt=''.join(f"<{m['role'].upper()}>{m['content']}" for m in req.messages)+'<ASSISTANT>'
 return {'model':_config.model_name if _config else 'Lyra','text':generate(prompt,req.max_new_tokens,req.temperature)}
