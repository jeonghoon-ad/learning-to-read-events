# Builds the section 3 tables from the per-window rows/summary files in data/. CP bounds are the summaries' own
# (computed on the GPU boxes by code/*/cot_stats.py). Standard library only. Run: python3 analysis_tables.py
import json, collections, statistics as st, os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
B=os.path.join(D,'box_big_full'); S=os.path.join(D,'box_small')
def rows(p): return [json.loads(l) for l in open(p)]
def summ(p): return json.load(open(p))
out={}
cells={}
for p in (S+'/out', B+'/out', B+'/out2', B+'/out3', B+'/out4', B+'/out5', B+'/out6'):
    cells.update(summ(p+'/summary.json')['cells'])
print('| cell | exact | acc | 95% CP two-sided | one-sided upper | format fail |'); print('|---|---|---|---|---|---|')
for k,v in cells.items():
    b=v['bounds']; print(f"| {k} | {v['correct']}/{v['total']} | {v['accuracy']:.3f} | {b['lower_two_sided']:.4f}–{b['upper_two_sided']:.4f} | {b['upper_one_sided']:.4f} | {v['format_fail']} |")
print()
allrows=[]
for p in (S+'/out', B+'/out', B+'/out2', B+'/out3', B+'/out4', B+'/out5', B+'/out6'): allrows+=rows(p+'/rows.jsonl')
d=collections.defaultdict(list)
for r in allrows: d[(r['host'],r['arm'],r['n'])].append(r)
print('| cell | windows | gen tokens/window mean | median | max | tokens/event mean | median | budget hits | think closed | final-line exact | modal first wrong step |'); print('|---|---|---|---|---|---|---|---|---|---|---|')
for k,v in d.items():
    g=[r['gen_tokens'] for r in v]; n=k[2]
    hit=sum(bool(r.get('hit_budget', r['gen_tokens']>=8192 if k[1]=='B-think' else False)) for r in v)
    closed=sum('</think>' in r['generation'] for r in v) if k[1]=='B-think' else '-'
    fw=collections.Counter(r.get('first_wrong_step') for r in v if not r['exact']).most_common(1)
    print(f"| {k[0]}:{k[1]}:{n} | {len(v)} | {sum(g)/len(g):.1f} | {st.median(g):.1f} | {max(g)} | {sum(g)/len(g)/n:.1f} | {st.median(g)/n:.1f} | {hit} | {closed} | {sum(bool(r.get('final_line_exact')) for r in v)} | {fw[0] if fw else '-'} |")
print()
cost=[]
for p in (S+'/out', B+'/out', B+'/out2', B+'/out3', B+'/out4'):
    for c in summ(p+'/summary.json')['cost']: c['box']='small(5090)' if p.startswith(S) else 'big'; cost.append(c)
c2=collections.defaultdict(list)
for c in cost: c2[(c['box'],c['host'],c['arm'],c['n'])].append(c)
def med(v,f):
    x=[a[f] for a in v if a.get(f) is not None]; return st.median(x) if x else None
print('| box | cell | cost windows | prompt tok | gen tok | prefill s | decode s | s/event (prefill+decode) | peak MB above weights | cache MB |'); print('|---|---|---|---|---|---|---|---|---|---|')
for k,v in sorted(c2.items(), key=lambda x:(x[0][0],x[0][1],x[0][2],x[0][3])):
    pf=med(v,'prefill_seconds'); dc=med(v,'decode_seconds') or 0; sr=med(v,'selector_register_seconds') or 0
    print(f"| {k[0]} | {k[1]}:{k[2]}:{k[3]} | {len(v)} | {med(v,'prompt_tokens'):.0f} | {med(v,'generated_tokens'):.0f} | {pf:.4f} | {dc+sr:.4f} | {(pf+dc+sr)/k[3]:.5f} | {med(v,'peak_cuda_bytes_above_weights')/1e6:.1f} | {med(v,'cache_bytes')/1e6:.1f} |")
print()
for h,p in (('H1',S),('H2',S),('H3',B)):
    s=summ(p+'/out/summary.json'); c=s['copy'][h]; print('copy',h,c['correct'],c['total']); print('null',h,{a:b for a,b in s['null'].get(h,{}).items() if a not in('rows','bounds')})
r5=rows(B+'/out5/rows.jsonl')[0]; print('n128 batch wall s',round(r5['batch_wall_s'],1),'batch_n',r5['batch_n'],'peak alloc GB',round(r5['batch_peak_alloc_bytes']/1e9,2))
