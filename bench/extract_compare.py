#!/usr/bin/env python3
"""Fact-extraction comparison for the Hindsight retain path: nano gemma-E2B vs AGX `cheap` (8B).
Pre-registered: metric = identifier recall (ids/numbers in the source that appear in the extracted
facts), JSON validity, fact count, wall latency. Same prompt, same docs, temperature 0.1. n = 5 docs."""
import json, os, re, sys, time, glob, urllib.request
KEY = [l.split('=',1)[1].strip() for l in open(os.path.expanduser('~/litellm-compose/litellm.env')) if l.startswith('LITELLM_MASTER_KEY=')][0]
ONLY=os.environ.get('ARMS')
NANO=os.environ.get('NANO_HOST') or sys.exit('set NANO_HOST=address of the nano')
# (url, model, key, style): style 'sys' = system prompt + doc; 'nuextract' = NuExtract '# Template:' format.
# 2026-09-12 second pass: bench arms on nano :8081 (ephemeral llama-server, gemma stopped), grammar-enforced JSON.
ARMS = {
  'nano-gemma-e2b': (f'http://{NANO}:8080/v1/chat/completions', 'gemma-e2b', None, 'sys'),
  'agx-cheap-8b':   ('http://127.0.0.1:5555/gateway/openai/v1/chat/completions', 'cheap', KEY, 'sys'),
  'nano-nuextract-2b': (f'http://{NANO}:8081/v1/chat/completions', 'bench', None, 'nuextract'),
  'nano-lfm2-extract': (f'http://{NANO}:8081/v1/chat/completions', 'bench', None, 'sys'),
  'nano-qwen35-2b':  (f'http://{NANO}:8081/v1/chat/completions', 'bench', None, 'sys'),
}
GRAMMAR = os.environ.get('GRAMMAR','1') == '1'
FACT = {"type":"object","properties":{"fact":{"type":"string"},"entities":{"type":"array","items":{"type":"string"}}},"required":["fact","entities"]}
SCHEMA_ARR = {"type":"array","items":FACT}
SCHEMA_OBJ = {"type":"object","properties":{"facts":SCHEMA_ARR},"required":["facts"]}
NU_TEMPLATE = {"facts":[{"fact":"verbatim-string","entities":["string"]}]}
SYS = ("Extract the atomic, durable facts from the session closeout below as a JSON array of objects "
       "{\"fact\": string, \"entities\": [string]}. Keep identifiers, numbers, file paths and names verbatim. "
       "Be concise: one fact per object, no commentary, output JSON only.")
docs = sorted(glob.glob(os.path.expanduser('~/.hermes/claudex-integration/session_closeouts/*.md')), key=os.path.getmtime)[-5:]
if os.environ.get('DOCS_FILE'): docs=[l.strip() for l in open(os.environ['DOCS_FILE']) if l.strip()]  # pinned doc set across separately-run arms
OUTDIR=os.environ.get('OUTDIR', os.path.expanduser('~/jetson-llm-benchoff/results/raw/nano-extract-20260911'))
idre = re.compile(r'(?<![\w/])(?:#\d{2,}|[0-9a-f]{12}|[0-9a-f]{7,10}(?![\w])|\d{3,}(?:\.\d+)?)(?![\w/])')
out = {}
if ONLY: ARMS={k:v for k,v in ARMS.items() if k in ONLY.split(',')}
for arm,(url,model,key,style) in ARMS.items():
    rows=[]
    for d in docs:
        text=open(d).read()[:6000]
        ids=set(m.group(0) for m in idre.finditer(text))
        if style=='nuextract':
            msgs=[{"role":"user","content":"# Template:\n"+json.dumps(NU_TEMPLATE,indent=4)+"\n"+text}]; schema=SCHEMA_OBJ
        else:
            msgs=[{"role":"system","content":SYS},{"role":"user","content":text}]; schema=SCHEMA_ARR
        body={"model":model,"messages":msgs,"temperature":0.1,"max_tokens":int(os.environ.get("MAX_TOKENS","1500")),"chat_template_kwargs":{"enable_thinking":False}}
        if GRAMMAR: body["response_format"]={"type":"json_schema","json_schema":{"name":"facts","schema":schema}}
        req=urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json", **({"Authorization":"Bearer "+key} if key else {})})
        t0=time.time()
        try:
            r=json.load(urllib.request.urlopen(req, timeout=600)); dt=time.time()-t0
            content=r['choices'][0]['message'].get('content') or ''
            usage=r.get('usage',{})
        except Exception as e:
            rows.append({"doc":os.path.basename(d),"error":str(e)[:120]}); continue
        m=re.search(r'\[.*\]', content, re.S)
        try: facts=json.loads(m.group(0)) if m else None
        except Exception: facts=None
        joined=json.dumps(facts) if facts else content
        hit=sum(1 for i in ids if i in joined)
        rows.append({"doc":os.path.basename(d)[:8],"secs":round(dt,1),"json_ok":facts is not None,"n_facts":len(facts) if isinstance(facts,list) else 0,
                     "ids_in_src":len(ids),"ids_recalled":hit,"recall":round(hit/len(ids),2) if ids else None,"out_tokens":usage.get('completion_tokens')})
    out[arm]=rows
json.dump(out, open(os.path.join(OUTDIR,'results-'+(ONLY or 'all').replace(',','_')+'.json'),'w'), indent=1)
for arm,rows in out.items():
    print("==",arm)
    for r in rows: print(" ",r)
    ok=[r for r in rows if 'recall' in r and r['recall'] is not None]
    if ok: print("  MEAN recall %.2f | json_ok %d/%d | mean secs %.1f | mean facts %.1f" % (sum(r['recall'] for r in ok)/len(ok), sum(r['json_ok'] for r in ok), len(ok), sum(r['secs'] for r in ok)/len(ok), sum(r['n_facts'] for r in ok)/len(ok)))
