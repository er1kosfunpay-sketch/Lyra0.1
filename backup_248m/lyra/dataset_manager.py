"""Versioned public Hugging Face source manifest and on-disk cache index."""
from pathlib import Path
import hashlib,json,time
SOURCES={
 "OpenAssistant/oasst1":{"revision":"main","license":"apache-2.0","languages":["ru","en"],"use":"Primary human-written, human-rated branches. Filter ru/en; branch select using +1/-1 ratings.","url":"https://huggingface.co/datasets/OpenAssistant/oasst1"},
 "kukunechka/russian-everyday-dialogues":{"revision":"main","license":"cc-by-4.0","languages":["ru"],"use":"Small native-authored everyday dialogue seed; retain attribution.","url":"https://huggingface.co/datasets/kukunechka/russian-everyday-dialogues"},
 "roskoN/dailydialog":{"revision":"main","license":"cc-by-nc-sa-4.0","languages":["en"],"use":"Optional non-commercial only; human-written daily dialogues; preserve license and attribution.","url":"https://huggingface.co/datasets/roskoN/dailydialog"},
 "HuggingFaceH4/ultrachat_200k":{"revision":"main","license":"mit","languages":["en"],"use":"Capped SFT-only supplement; synthetic ChatGPT-generated instructional multi-turn text, filter long prompts.","url":"https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"},
 "SiberiaSoft/SiberianPersonaChat-2":{"revision":"main","license":"mit","languages":["ru"],"use":"Default Russian supplement parsed to multi-turn dialogue; reject QA, persona scaffolding, low-turn, repeated-message and repeated-prompt rows.","url":"https://huggingface.co/datasets/SiberiaSoft/SiberianPersonaChat-2"}}
def file_hash(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def register(repo,path,license):
 p=Path(path); idx=p.parent/'dataset_cache.json'; doc=json.loads(idx.read_text()) if idx.exists() else {}
 doc[repo]={'path':str(p),'sha256':file_hash(p),'bytes':p.stat().st_size,'license':license,'cached_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}; idx.write_text(json.dumps(doc,indent=2),encoding='utf8')
