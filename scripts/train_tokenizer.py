"""Train byte-BPE on train-split conversation text only (streaming extraction)."""
import argparse,glob,json,tempfile,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyra.tokenizer import LyraTokenizer

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True,help='JSONL glob or UTF-8 text files (use the TRAIN split only)'); p.add_argument('--vocab-size',type=int,default=16384); p.add_argument('--out',default='artifacts/tokenizer/tokenizer.json'); a=p.parse_args(argv)
 files=glob.glob(a.data,recursive=True)
 if not files: raise FileNotFoundError(a.data)
 if not any('train' in Path(f).name for f in files):
  print('WARNING: tokenizer input does not look like a train split; training on validation/test text leaks eval data into the vocabulary.')
 with tempfile.TemporaryDirectory() as td:
  corpus=Path(td)/'corpus.txt'
  with corpus.open('w',encoding='utf8') as out:
   for src in files:
    with open(src,encoding='utf-8-sig',errors='replace') as f:
     for line in f:
      try: row=json.loads(line)
      except json.JSONDecodeError: out.write(line if line.endswith('\n') else line+'\n'); continue
      for m in row.get('messages',[]):
       # Keep original newlines: ByteLevel BPE must see real line breaks,
       # otherwise inference on multiline input meets byte sequences the
       # tokenizer rarely observed during training.
       if m.get('content'):out.write(m['content']+'\n')
  tok=LyraTokenizer.train([corpus],a.vocab_size); Path(a.out).parent.mkdir(parents=True,exist_ok=True); tok.save(a.out)
  print(f'vocab_size={tok.vocab_size}; target={a.vocab_size}; file={a.out}')
  samples=['Привет, как дела?','Здравствуйте! Рад тебя видеть.','Я создаю собственную языковую модель.','Hello.','How are you?','I am building my own language model.','game:GetService("ReplicatedStorage")','line one\nline two','你好 🌍']
  for s in samples:
   ids=tok.encode(s); words=max(1,len(s.split())); print({'text':s,'tokens':len(ids),'tokens_per_word':round(len(ids)/words,3),'tokens_per_character':round(len(ids)/max(1,len(s)),3),'roundtrip':tok.decode(ids)==s})

if __name__=='__main__':main()
