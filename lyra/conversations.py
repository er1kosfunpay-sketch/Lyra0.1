"""Conversation validation, text hygiene and duplicate fingerprints."""
import hashlib,re,unicodedata
from collections import Counter
BAD = re.compile(r"(?:https?://\S+|\b(?:buy now|subscribe|click here)\b)",re.I)
SHORT_OK = {"hi","hello","hey","yes","no","ok","okay","lol","haha","привет","да","нет","ок","ага","спасибо","пока"}
def clean_text(s):
    s=unicodedata.normalize("NFC",str(s or "")); s="".join(c for c in s if c in "\n\t" or unicodedata.category(c)[0]!="C")
    alpha=sum(c.isalpha() for c in s)
    if alpha and (s.count("Р")+s.count("С"))/alpha>0.12:
        try:
            repaired=s.encode("cp1251").decode("utf-8")
            if sum(c.isalpha() for c in repaired)>alpha*0.7:s=repaired
        except (UnicodeEncodeError,UnicodeDecodeError):
            pass
    return re.sub(r"[ \t]+"," ",s).strip()
def detect_lang(text):
    letters=[c for c in text.lower() if c.isalpha()]
    if not letters:return "unknown"
    cyr=sum("а"<=c<="я" or c in "ёъыэ" for c in letters)/len(letters)
    return "ru" if cyr>0.25 else "en" if cyr<0.05 else "mixed"
def valid_messages(messages,min_turns=2,max_chars=24000):
    if not isinstance(messages,list) or len(messages)<min_turns:return False
    total=0; prev=None
    for i,m in enumerate(messages):
        role=m.get("role"); text=clean_text(m.get("content"))
        if role not in {"user","assistant","system"} or (i and role==prev and role!="system"):return False
        if not text or len(text)>12000 or BAD.search(text):return False
        if len(text)<3 and text.lower() not in SHORT_OK:return False
        if len(set(text))<3 and len(text)>8:return False
        total+=len(text); prev=role
    return total<=max_chars and any(m["role"]=="assistant" for m in messages) and any(m["role"]=="user" for m in messages)
def normalized_hash(messages):
    norm=" ".join(re.sub(r"\W+"," ",m["content"].lower()).split() for m in messages)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()
def duplicate_key(messages):
    # Exact and coarse near-duplicate fingerprint. Kept intentionally language agnostic.
    words=" ".join(m["content"].lower() for m in messages)
    words=re.sub(r"[^\w\s]"," ",words,flags=re.UNICODE)
    return hashlib.sha256(" ".join(words.split()).encode()).hexdigest()
def stats(rows):
    turns=[]; responses=[]; users=[]; langs=Counter(); nmsg=0
    for row in rows:
        ms=row["messages"]; nmsg+=len(ms); turns.append(sum(m["role"] in {"user","assistant"} for m in ms))
        langs[row.get("lang","unknown")]+=1
        responses.extend(len(m["content"]) for m in ms if m["role"]=="assistant")
        users.extend(len(m["content"]) for m in ms if m["role"]=="user")
    def med(v):
        if not v:return 0
        s=sorted(v); return s[len(s)//2]
    def percentiles(v):
        if not v:return {"p25":0,"p50":0,"p75":0,"p90":0}
        s=sorted(v); return {f"p{p}":s[min(len(s)-1,int((p/100)*(len(s)-1)))] for p in (25,50,75,90)}
    return {"conversations":len(rows),"messages":nmsg,"average_turns":sum(turns)/len(turns) if turns else 0,"median_turns":med(turns),"max_turns":max(turns,default=0),"language_conversations":dict(langs),"assistant_response_chars_mean":sum(responses)/len(responses) if responses else 0,"assistant_response_chars_percentiles":percentiles(responses),"user_message_chars_mean":sum(users)/len(users) if users else 0,"user_message_chars_percentiles":percentiles(users)}
