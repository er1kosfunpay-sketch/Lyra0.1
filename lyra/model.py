"""Small decoder-only Transformer built from random initialization."""
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from .config import LyraConfig

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight

def apply_rope(q, k, theta, scaling=None):
    # q [B,H,T,D], k [B,KV,T,D]. Compute phases in fp32 for stability.
    d = q.shape[-1]; t = q.shape[-2]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d))
    factor = scaling.get("factor",1.0) if scaling else 1.0
    phase = torch.outer(torch.arange(t, device=q.device, dtype=torch.float32)/factor, inv)
    cos, sin = phase.cos()[None,None], phase.sin()[None,None]
    def rotate(x):
        a, b = x.float().reshape(*x.shape[:-1], d//2, 2).unbind(-1)
        out = torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2)
        return out.to(x.dtype)
    return rotate(q), rotate(k)

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.h=c.num_attention_heads; self.kv=c.num_kv_heads; self.d=c.head_dim
        self.q_proj=nn.Linear(c.hidden_size,self.h*self.d,bias=False)
        self.k_proj=nn.Linear(c.hidden_size,self.kv*self.d,bias=False)
        self.v_proj=nn.Linear(c.hidden_size,self.kv*self.d,bias=False)
        self.o_proj=nn.Linear(self.h*self.d,c.hidden_size,bias=False); self.theta=c.rope_theta; self.scaling=c.rope_scaling; self.dropout=c.dropout
    def forward(self,x):
        b,t,_=x.shape
        q=self.q_proj(x).view(b,t,self.h,self.d).transpose(1,2)
        k=self.k_proj(x).view(b,t,self.kv,self.d).transpose(1,2)
        v=self.v_proj(x).view(b,t,self.kv,self.d).transpose(1,2)
        q,k=apply_rope(q,k,self.theta,self.scaling)
        # PyTorch SDPA selects memory-efficient/Flash kernels when supported.
        if self.h != self.kv:
            try:
                y=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True,dropout_p=self.dropout if self.training else 0.0)
            except TypeError:  # Older PyTorch: preserve behavior with explicit KV expansion.
                repeat=self.h//self.kv; k=k.repeat_interleave(repeat,dim=1); v=v.repeat_interleave(repeat,dim=1)
                y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.dropout if self.training else 0.0)
        else: y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,t,-1))

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__(); self.gate=nn.Linear(c.hidden_size,c.intermediate_size,bias=False); self.up=nn.Linear(c.hidden_size,c.intermediate_size,bias=False); self.down=nn.Linear(c.intermediate_size,c.hidden_size,bias=False); self.dropout=c.dropout
    def forward(self,x): return F.dropout(self.down(F.silu(self.gate(x))*self.up(x)),p=self.dropout,training=self.training)

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.attn_norm=RMSNorm(c.hidden_size,c.norm_eps); self.attn=CausalSelfAttention(c); self.mlp_norm=RMSNorm(c.hidden_size,c.norm_eps); self.mlp=SwiGLU(c)
    def forward(self,x):
        x=x+self.attn(self.attn_norm(x)); return x+self.mlp(self.mlp_norm(x))

class LyraModel(nn.Module):
    def __init__(self, config: LyraConfig):
        super().__init__(); self.config=config
        self.embed=nn.Embedding(config.vocab_size,config.hidden_size)
        self.layers=nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm=RMSNorm(config.hidden_size,config.norm_eps)
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        if config.tie_embeddings: self.lm_head.weight=self.embed.weight
        self.gradient_checkpointing=config.gradient_checkpointing
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,mean=0.0,std=0.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,mean=0.0,std=0.02)
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
    def forward(self,input_ids,labels=None):
        x=self.embed(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and x.requires_grad:
                x=torch.utils.checkpoint.checkpoint(layer,x,use_reentrant=False)
            else: x=layer(x)
        logits=self.lm_head(self.norm(x))
        loss=None
        if labels is not None:
            targets=labels[:,1:].reshape(-1)
            token_losses=F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),targets,ignore_index=-100,reduction='none')
            loss=token_losses.sum()/(targets.ne(-100).sum().clamp_min(1))
        return {"logits":logits,"loss":loss}
    @torch.inference_mode()
    def generate(self,input_ids,max_new_tokens=128,temperature=0.8,top_k=50,top_p=0.9,repetition_penalty=1.08,eos_token_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx=input_ids[:,-self.config.context_length:]
            logits=self(idx)["logits"][:,-1].float()
            if temperature <= 0: next_id=logits.argmax(-1,keepdim=True)
            else:
                logits/=max(temperature,1e-6)
                if repetition_penalty != 1.0:
                    # Batch-safe: penalize each row by its own seen tokens only,
                    # vectorized over unique ids instead of a Python-level set loop.
                    for i in range(input_ids.size(0)):
                        uniq=torch.unique(input_ids[i])
                        vals=logits[i,uniq]
                        logits[i,uniq]=torch.where(vals > 0, vals/repetition_penalty, vals*repetition_penalty)
                if top_k: vals,_=torch.topk(logits,min(top_k,logits.size(-1))); logits[logits<vals[:,-1,None]]=-float('inf')
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, -1)
                    remove = sorted_probs.cumsum(-1) - sorted_probs >= top_p
                    sorted_logits[remove] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
                next_id=torch.multinomial(torch.softmax(logits,dim=-1),1)
            input_ids=torch.cat((input_ids,next_id),dim=1)
            if eos_token_id is not None and bool((next_id==eos_token_id).all()): break
        return input_ids

