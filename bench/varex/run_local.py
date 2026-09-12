#!/usr/bin/env python3
"""VAREX plain-text (text_flow) inference against a local llama-server, from the local parquet shards.

Sampling: fixed-seed stratified sample of N doc_ids from the manifest (same set for every model).
Prompting: the README's system prompt + response_format json_object (paper-faithful), or
  --style nuextract (NuExtract native '# Template:' format, JSON Schema converted to a typed template),
  or --style schema (llama.cpp json_schema constrained decoding on the doc's own schema).
Output: <out>/<model>/text_flow/<doc_id>.pred.json (+ .meta.json with timing/usage), scoreable by evaluation/score.py.
"""
import argparse, glob, json, os, random, time, urllib.request
import pyarrow.parquet as pq

SYS = ("Extract structured data from this document.\n"
       "Return a JSON object matching this schema:\n\n{schema}\n\n"
       "Return null for fields you cannot find.\nReturn ONLY valid JSON.\n"
       "Return an instance of the JSON with extracted values, not the schema itself.")

def resolve(schema, root):
    if isinstance(schema, dict) and '$ref' in schema:
        ref = schema['$ref']
        node = root
        for part in ref.lstrip('#/').split('/'):
            node = node[part]
        return resolve(node, root)
    return schema

def to_nu_template(schema, root=None):
    """JSON Schema -> NuExtract typed template (strings -> 'verbatim-string', numbers -> 'number', ...)."""
    root = root or schema
    s = resolve(schema, root)
    t = s.get('type')
    if t == 'object' or 'properties' in s:
        return {k: to_nu_template(v, root) for k, v in s.get('properties', {}).items()}
    if t == 'array':
        return [to_nu_template(s.get('items', {'type': 'string'}), root)]
    if t in ('integer', 'number'):
        return 'number'
    if t == 'boolean':
        return ['true', 'false']
    if 'enum' in s:
        return [str(e) for e in s['enum']]
    return 'verbatim-string'

def load_docs(data_dir):
    docs = {}
    for f in sorted(glob.glob(os.path.join(data_dir, 'benchmark-*.parquet'))):
        t = pq.read_table(f, columns=['doc_id', 'split', 'schema', 'text_flow'])
        for r in t.to_pylist():
            docs[str(r['doc_id'])] = r
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='hfdata')
    ap.add_argument('--manifest', default='evaluation/manifest.json')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=20260912)
    ap.add_argument('--base-url', default='http://127.0.0.1:8012/v1')
    ap.add_argument('--model', required=True, help='served model name (also the output dir name)')
    ap.add_argument('--out', default='results')
    ap.add_argument('--style', choices=['json_object', 'nuextract', 'schema'], default='json_object')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--ids-file', default=None, help='write/read the sampled doc ids (shared across models)')
    a = ap.parse_args()

    docs = load_docs(a.data_dir)
    manifest = json.load(open(a.manifest))
    ids = [str(i) for i in (manifest if isinstance(manifest, list) else (manifest.get('active_doc_ids') or manifest.get('doc_ids')))]
    ids = [i for i in ids if i in docs]
    if a.ids_file and os.path.exists(a.ids_file):
        sample = [l.strip() for l in open(a.ids_file) if l.strip()]
    else:
        rnd = random.Random(a.seed)
        by_split = {}
        for i in ids:
            by_split.setdefault(docs[i]['split'], []).append(i)
        sample = []
        for sp, lst in sorted(by_split.items()):
            k = round(a.n * len(lst) / len(ids))
            sample += rnd.sample(sorted(lst), k)
        if a.ids_file:
            open(a.ids_file, 'w').write('\n'.join(sample) + '\n')
    out_dir = os.path.join(a.out, a.model, 'text_flow')
    os.makedirs(out_dir, exist_ok=True)
    print(f'{a.model} style={a.style} n={len(sample)} -> {out_dir}', flush=True)

    for n, doc_id in enumerate(sample, 1):
        pred_path = os.path.join(out_dir, f'{doc_id}.pred.json')
        if os.path.exists(pred_path):
            continue
        d = docs[doc_id]
        schema = json.loads(d['schema'])
        if a.style == 'nuextract':
            msgs = [{'role': 'user', 'content': '# Template:\n' + json.dumps(to_nu_template(schema), indent=4) + '\n' + d['text_flow']}]
        else:
            msgs = [{'role': 'system', 'content': SYS.format(schema=json.dumps(schema, indent=2))},
                    {'role': 'user', 'content': d['text_flow']}]
        body = {'model': a.model, 'messages': msgs, 'temperature': 0, 'max_tokens': a.max_tokens,
                'chat_template_kwargs': {'enable_thinking': False}}
        if a.style == 'schema':
            body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'doc', 'schema': schema}}
        else:
            body['response_format'] = {'type': 'json_object'}
        req = urllib.request.Request(a.base_url + '/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.time()
        meta = {'doc_id': doc_id, 'split': d['split'], 'style': a.style}
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900))
            content = r['choices'][0]['message'].get('content') or ''
            meta.update(secs=round(time.time() - t0, 1), usage=r.get('usage'), finish=r['choices'][0].get('finish_reason'))
            try:
                pred = json.loads(content)
            except Exception:
                s, e = content.find('{'), content.rfind('}')
                try:
                    pred = json.loads(content[s:e + 1]) if s >= 0 else {}
                except Exception:
                    pred = {}
                    meta['parse_error'] = True
            if a.style == 'nuextract' and isinstance(pred, dict) and set(pred) == {'facts'}:
                pred = pred['facts']
        except Exception as ex:
            meta.update(secs=round(time.time() - t0, 1), error=str(ex)[:200])
            pred = {}
        json.dump(pred, open(pred_path, 'w'), indent=1)
        json.dump(meta, open(os.path.join(out_dir, f'{doc_id}.meta.json'), 'w'))
        if n % 10 == 0 or 'error' in meta:
            print(f'  {n}/{len(sample)} {doc_id} {meta.get("secs")}s {"ERR " + meta["error"] if "error" in meta else ""}', flush=True)
    print('DONE', a.model, flush=True)

if __name__ == '__main__':
    main()
