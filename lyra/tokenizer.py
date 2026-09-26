"""Train a standalone byte-level BPE tokenizer from a text corpus."""
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

SPECIAL_TOKENS=["<PAD>","<UNK>","<BOS>","<EOS>","<SYSTEM>","<USER>","<ASSISTANT>","<TOOL>","<END>"]
class LyraTokenizer:
    def __init__(self, tokenizer): self.backend=tokenizer
    @classmethod
    def train(cls, files, vocab_size=32768):
        tok=Tokenizer(models.BPE(unk_token="<UNK>")); tok.normalizer=normalizers.NFC()
        tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder=decoders.ByteLevel()
        tok.train([str(p) for p in files],trainers.BpeTrainer(vocab_size=vocab_size,special_tokens=SPECIAL_TOKENS,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),min_frequency=2))
        return cls(tok)
    @classmethod
    def from_file(cls,path): return cls(Tokenizer.from_file(str(path)))
    def save(self,path): self.backend.save(str(path))
    def encode(self,text,add_bos=False,add_eos=False):
        ids=self.backend.encode(text).ids
        if add_bos: ids=[self.id("<BOS>")]+ids
        if add_eos: ids += [self.id("<EOS>")]
        return ids
    def decode(self,ids,skip_special_tokens=True): return self.backend.decode(list(ids),skip_special_tokens=skip_special_tokens)
    def tokenize(self,text): return self.backend.encode(text).tokens
    def detokenize(self,tokens): return self.decode([self.id(t) for t in tokens],skip_special_tokens=False)
    def batch_encode(self,texts,**kwargs): return [self.encode(t,**kwargs) for t in texts]
    def id(self,token):
        value=self.backend.token_to_id(token)
        if value is None: raise KeyError(token)
        return value
    @property
    def vocab_size(self): return self.backend.get_vocab_size()
