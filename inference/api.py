"""Minimal local inference API."""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer
from lyra.generation import GENERATION_DEFAULTS,format_chat

_model=_tokenizer=_config=None

class GenerateRequest(BaseModel): prompt:str; max_new_tokens:int=Field(default=128,ge=1,le=2048); temperature:float=Field(default=0.8,ge=0)
class ChatRequest(BaseModel): messages:list[dict[str,str]]; max_new_tokens:int=Field(default=128,ge=1,le=2048); temperature:float=Field(default=0.8,ge=0)

def load():
 global _model,_tokenizer,_config
 root=Path(os.getenv('LYRA_MODEL_DIR','exports/lyra'))
 try:
  _config=LyraConfig.from_json(root/'config.json'); _tokenizer=LyraTokenizer.from_file(root/'tokenizer.json'); device='cuda' if torch.cuda.is_available() else 'cpu'; _model=LyraModel(_config).to(device); _model.load_state_dict(torch.load(root/'model.pt',map_location=device,weights_only=True)); _model.eval()
 except FileNotFoundError: _model=None

@asynccontextmanager
async def lifespan(app):
 load()
 yield

app=FastAPI(title='Lyra 0.1 API',lifespan=lifespan)

@app.get('/health')
def health(): return {'status':'ok' if _model else 'model_not_loaded'}
@app.get('/info')
def info():
 if not _config: raise HTTPException(503,'Model is not loaded')
 return {'name':_config.model_name,'version':_config.version,'parameters':_model.parameter_count(),'context_length':_config.context_length,'vocab_size':_config.vocab_size,'dtype':_config.dtype}
def generate_ids(ids,max_new_tokens,temperature):
 d=GENERATION_DEFAULTS; ctx=_config.context_length
 if len(ids)>=ctx: ids=ids[-(ctx-1):]
 x=torch.tensor([ids],device=next(_model.parameters()).device)
 with torch.inference_mode():
  out=_model.generate(x,max_new_tokens=max_new_tokens,temperature=temperature,top_k=d['top_k'],top_p=d['top_p'],repetition_penalty=d['repetition_penalty'],eos_token_id=_tokenizer.id(d['eos_token']))
 gen=out[0,len(ids):].tolist(); end=_tokenizer.id(d['eos_token'])
 if end in gen: gen=gen[:gen.index(end)]
 return gen
def generate(prompt,max_new_tokens,temperature):
 if not _model: raise HTTPException(503,'Exported model is not available; train and export Lyra first.')
 ids=_tokenizer.encode(prompt); return _tokenizer.decode(generate_ids(ids,max_new_tokens,temperature))
@app.post('/generate')
def generate_route(req:GenerateRequest): return {'model':_config.model_name if _config else 'Lyra','text':generate(req.prompt,req.max_new_tokens,req.temperature)}
@app.post('/chat')
def chat(req:ChatRequest):
 if not _model: raise HTTPException(503,'Exported model is not available; train and export Lyra first.')
 try: ids=format_chat(req.messages,_tokenizer,add_assistant_trigger=True)
 except ValueError as e: raise HTTPException(400,str(e))
 return {'model':_config.model_name if _config else 'Lyra','text':_tokenizer.decode(generate_ids(ids,req.max_new_tokens,req.temperature))}
