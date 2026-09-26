"""License-aware Hugging Face ingestion and quality-controlled conversation build.

Dependencies: python -m pip install -e ".[data]" (datasets, huggingface_hub)
"""
import argparse,hashlib,json,random
from collections import Counter
from pathlib import Path
from lyra.conversations import clean_text,detect_lang,valid_messages,duplicate_key,stats
SOURCES={
 "oasst":("OpenAssistant/oasst1","apache-2.0","human annotated; prefer consensus-rated assistant children"),
 "ru_everyday":("kukunechka/russian-everyday-dialogues","cc-by-4.0","20 native-authored everyday multi-turn dialogue seeds; attribute author"),
 "daily":("roskoN/dailydialog","cc-by-nc-sa-4.0","non-commercial share-alike; opt-in, English, translated rows not present"),
 "ultra":("HuggingFaceH4/ultrachat_200k","mit","synthetic ChatGPT dialogues; use capped, filtered SFT-only subset"),
 "siberian":("SiberiaSoft/SiberianPersonaChat-2","mit","Russian persona chat, synthetic/template-like, inspect and cap heavily")}
REVISION={}
def pin(repo):
 from huggingface_hub import HfApi
 if repo not in REVISION: REVISION[repo]=HfApi().dataset_info(repo).sha
 return REVISION[repo]
def normalize(ms):
 out=[]
 for m in ms:
  role=m.get('role','').lower(); role={'prompter':'user','human':'user','assistant':'assistant','user':'user','system':'system'}.get(role)
  if not role: return None
  text=clean_text(m.get('content',m.get('text','')))
  if not text:return None
  out.append({'role':role,'content':text})
 return out

def rows(source,limit):
 if source=='oasst':
  import gzip
  repo='OpenAssistant/oasst1'; tree_file=Path('data/cache/oasst/all.trees.jsonl.gz')
  if not tree_file.exists():
   from huggingface_hub import hf_hub_download
   revision=pin(repo); tree_file=Path(hf_hub_download(repo_id=repo,filename='2023-04-12_oasst_all.trees.jsonl.gz',repo_type='dataset',revision=revision,cache_dir='data/cache'))
  else: REVISION[repo]='fdf72ae0827c1cda404aff25b6603abec9e3399b'
  def quality(node):
   votes=node.get('emojis') or {}; labels=node.get('labels') or {}
   helpful=labels.get('helpfulness',{}).get('value',0); rated=labels.get('quality',{}).get('value',0)
   return (int(votes.get('+1',0))-int(votes.get('-1',0)),float(helpful)+float(rated),int(node.get('review_count') or 0))
  emitted=0
  with gzip.open(tree_file,'rt',encoding='utf-8') as f:
   for line in f:
    tree=json.loads(line); root=tree.get('prompt') or {}
    if tree.get('tree_state')!='ready_for_export' or root.get('lang') not in ('ru','en') or root.get('deleted') or root.get('synthetic'):continue
    chain=[root]; node=root
    while node.get('replies') and len(chain)<100:
     children=[r for r in node['replies'] if not r.get('deleted') and not r.get('synthetic') and r.get('lang')==root['lang']]
     if not children:break
     node=max(children,key=quality); chain.append(node)
    ms=normalize([{'role':r.get('role',''),'content':r.get('text','')} for r in chain])
    if ms and len(ms)>=2 and all(ms[i]['role']!=ms[i-1]['role'] for i in range(1,len(ms))):
     yield {'messages':ms,'lang':root['lang'],'source':source}; emitted+=1
     if emitted>=limit:break
 elif source=='ru_everyday':
  repo='kukunechka/russian-everyday-dialogues'; local=Path('data/curated/russian_everyday_dialogues.jsonl')
  if local.exists():
   with local.open(encoding='utf-8-sig') as f:
    for i,line in enumerate(f):
     if i>=limit:break
     row=json.loads(line); yield row
   REVISION[repo]='3d9c43ca85a50e7a32fa7b05e27c0637a710e988'; return
  revision=pin(repo)
  from huggingface_hub import hf_hub_download
  path=hf_hub_download(repo_id=repo,filename='russian_everyday_dialogues.jsonl',repo_type='dataset',revision=revision,cache_dir='data/cache')
  emitted=0
  with open(path,encoding='utf-8-sig') as f:
   for line in f:
    r=json.loads(line); ms=normalize([{'role':'user','content':r.get('user','')},{'role':'assistant','content':r.get('assistant','')}])
    if ms:
     yield {'messages':ms,'lang':'ru','source':source}; emitted+=1
     if emitted>=limit:break
 elif source=='daily':
  from datasets import load_dataset
  ds=load_dataset('roskoN/dailydialog',split='train',revision=pin('roskoN/dailydialog'),cache_dir='data/cache')
  # Keep full dialogues, not every cumulative prefix.
  seen=set(); emitted=0
  for r in ds:
   utterances=r.get('dialog') or r.get('dialogue')
   if not utterances or len(utterances)<4:continue
   key=tuple(utterances)
   if key in seen:continue
   seen.add(key)
   ms=normalize([{'role':'user' if i%2==0 else 'assistant','content':x} for i,x in enumerate(utterances)])
   if ms:
    yield {'messages':ms,'lang':'en','source':source}; emitted+=1
    if emitted>=limit:break
 elif source=='ultra':
  from datasets import load_dataset
  ds=load_dataset('HuggingFaceH4/ultrachat_200k',split='train_sft',streaming=True,revision=pin('HuggingFaceH4/ultrachat_200k'),cache_dir='data/cache').shuffle(seed=2026,buffer_size=10000)
  emitted=0
  for r in ds:
   ms=normalize(r.get('messages',[]))
   # Strip system scaffolding, cap meandering / generated task prompts.
   ms=[m for m in (ms or []) if m['role']!='system']
   if not ms or len(ms)>20:continue
   text=' '.join(m['content'] for m in ms)
   if len(text)>9000 or any(x in text.lower() for x in ('write a 1000 word','create a comprehensive','as an ai language model')):continue
   if valid_messages(ms):
    yield {'messages':ms,'lang':detect_lang(text),'source':source}; emitted+=1
    if emitted>=limit:break
 elif source=='siberian':
  from datasets import load_dataset
  ds=load_dataset('SiberiaSoft/SiberianPersonaChat-2',split='train',streaming=True,revision=pin('SiberiaSoft/SiberianPersonaChat-2'),cache_dir='data/cache'); emitted=0
  for r in ds:
   raw=r.get('input','')+'\n'+r.get('output','')
   # Existing dialog embedded in input field; retain only role-marked, coherent turns.
   ms=[]
   for line in raw.splitlines():
    if ':' not in line:continue
    who,content=line.split(':',1); who=who.strip().lower()
    role='user' if who in ('ты','user') else 'assistant' if who in ('я','assistant') else None
    if role:ms.append({'role':role,'content':clean_text(content)})
   if valid_messages(ms) and detect_lang(' '.join(m['content'] for m in ms))=='ru':
    yield {'messages':ms,'lang':'ru','source':source}; emitted+=1
    if emitted>=limit:break

def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',default='data/processed'); p.add_argument('--sources',default='oasst,ru_everyday,ultra'); p.add_argument('--max-per-source',type=int,default=50000); p.add_argument('--ru-share',type=float,default=.5); p.add_argument('--include-daily-nc',action='store_true'); p.add_argument('--include-siberian',action='store_true'); p.add_argument('--seed',type=int,default=17); a=p.parse_args()
 names=[x.strip() for x in a.sources.split(',') if x.strip()]
 if a.include_daily_nc and 'daily' not in names:names.append('daily')
 if a.include_siberian and 'siberian' not in names:names.append('siberian')
 unknown=set(names)-set(SOURCES)
 if unknown:raise ValueError(f'Unknown data source(s): {sorted(unknown)}')
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); dedup=set(); accepted=[]; raw_count=filtered=dupes=0
 for name in names:
  for row in rows(name,a.max_per_source):
   raw_count+=1; text=' '.join(m['content'] for m in row['messages']); lang=detect_lang(text)
   if row['lang']=='ru' and lang=='mixed': lang='ru'
   if lang not in ('ru','en') or not valid_messages(row['messages']): filtered+=1; continue
   row['lang']=lang; key=duplicate_key(row['messages'])
   if key in dedup:dupes+=1; continue
   dedup.add(key); accepted.append(row)
 curated=Path('data/curated/identity_and_honesty.jsonl')
 if curated.exists():
  for line in curated.read_text(encoding='utf-8-sig').splitlines():
   raw_count+=1; row=json.loads(line); ms=normalize(row.get('messages',[]))
   if not ms or not valid_messages(ms): filtered+=1; continue
   row['messages']=ms; row['lang']=row.get('lang',detect_lang(' '.join(m['content'] for m in ms)))
   key=duplicate_key(ms)
   if key in dedup:dupes+=1; continue
   dedup.add(key); accepted.append(row)
 accepted_before_balance=len(accepted); source_counts_before_balance=dict(Counter(x['source'] for x in accepted))
 language_counts_before_balance=dict(Counter(x['lang'] for x in accepted))
 if not 0<a.ru_share<1:raise ValueError('--ru-share must be between 0 and 1')
 rng=random.Random(a.seed); by_lang={lang:[x for x in accepted if x['lang']==lang] for lang in ('ru','en')}
 # Balance without replacement: do not compensate for scarce Russian by repeating it.
 n=min(int(len(by_lang['ru'])/a.ru_share),int(len(by_lang['en'])/(1-a.ru_share)))
 if n:
  ru=rng.sample(by_lang['ru'],int(n*a.ru_share)); en=rng.sample(by_lang['en'],n-len(ru)); accepted=ru+en
 else: accepted=[]
 rng.shuffle(accepted)
 # 90/5/5 split after de-dup; keep original test/validation upstream if available as future enhancement.
 n=len(accepted); ntest=max(1,int(n*.05)); nval=max(1,int(n*.05)); splits={'train':accepted[:n-ntest-nval],'validation':accepted[n-ntest-nval:n-ntest],'test':accepted[n-ntest:]}
 for split,items in splits.items():
  with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
   for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 meta={'dataset_version':'lyra-conversation-0.1','source_names':names,'source_revisions':{k:REVISION.get(SOURCES[k][0]) for k in names},'licenses':{k:SOURCES[k][1] for k in names},'seed':a.seed,'max_per_source':a.max_per_source,'source_conversation_candidates_seen':raw_count,'accepted_before_balance':accepted_before_balance,'filtered_after_extraction':filtered,'duplicates_after_normalization':dupes,'quality_filter_rate_of_extracted_candidates':filtered/raw_count if raw_count else None,'duplicate_rate_of_extracted_candidates':dupes/raw_count if raw_count else None,'language_counts_before_balance':language_counts_before_balance,'source_conversations_before_balance':source_counts_before_balance,'accepted_after_balance':len(accepted),'ru_share_target':a.ru_share,'source_conversations_after_balance':dict(Counter(x['source'] for x in accepted)),'splits':{k:stats(v) for k,v in splits.items()},'note':'Counts/rates describe extracted candidates after source-specific extraction, not the full upstream files.'}
 (out/'dataset_stats.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
