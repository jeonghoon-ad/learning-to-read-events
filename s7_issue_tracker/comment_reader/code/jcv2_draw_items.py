"""Draw and freeze the preregistered sample, after the prior is frozen."""
import gzip, hashlib, json, random, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'jcv2/out'; SPOOL=Path('/private/tmp/semistructured_main_20261113_spool')
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from select_jcomment import core
from aliasing import alias_feed
FIELDS=('status','resolution')
def project_data(project):
    with gzip.open(SPOOL/f'{project}.jsonl.gz','rt') as f: issues=[json.loads(line) for line in f]
    vocab={f:set() for f in core.FIELDS}
    for issue in issues:
        for field in core.FIELDS:vocab[field].add(core.text_value(issue['fields'].get(field)))
        for ch in issue['changes']:
            if ch.get('field') in vocab:vocab[ch['field']].update((core.text_value(ch.get('fromString')),core.text_value(ch.get('toString'))))
    return issues,{f:sorted(v) for f,v in vocab.items()}
def make_candidates(issues):
    groups={'implicit':[],'explicit':[]}
    for iss in issues:
        if not iss.get('created'):continue
        comments=[c for c in iss['comments'] if c.get('created')]
        for ch in iss['changes']:
            if ch.get('field') not in FIELDS or not ch.get('created') or not ch.get('author'):continue
            pair=[c for c in comments if c.get('author') and core.norm(c['author'])==core.norm(ch['author']) and abs((core.dt(c['created'])-core.dt(ch['created'])).total_seconds())<=3600]
            if not pair:continue
            t=max([core.dt(ch['created'])]+[core.dt(c['created']) for c in pair])
            group='explicit' if core.supported(ch,comments,t) else 'implicit'
            groups[group].append((iss,ch,t))
    return groups
def materialize(project,entry,group,vocab):
    iss,ch,t=entry; field=ch['field']; key=iss['key']; ini=core.initial(iss)
    e={'kind':'created','time':iss['created'],'issue':key,'author':iss.get('creator'),'values':ini,'type':iss.get('type'),'summary':iss.get('summary')}
    comments=[dict(c,kind='comment',time=c['created'],issue=key) for c in iss['comments'] if c.get('created') and core.dt(c['created'])<=t]
    comments.sort(key=lambda x:core.dt(x['time']))
    events=[e]+comments
    latest=sorted((c for c in iss['changes'] if c.get('field')==field and c.get('created') and core.dt(c['created'])<=t),key=lambda c:core.dt(c['created']))
    gold=core.text_value(latest[-1]['toString']) if latest else core.text_value(ini[field])
    events,_,vocab,_=alias_feed(events,[],vocab)
    lines=[core.render_event(e) for e in events]
    return {'project':project,'issue':key,'field':field,'t':t.isoformat(),'lines':lines,'events':events,'vocab':vocab,'gold':gold,'group':group}
def draw():
    split=json.loads((ROOT/'split.json').read_text());excluded=set(split['selected_test_projects'])|set(split['train'])|set(split['val'])|{'AMQCPP','AIRAVATA'}
    projects=sorted(p.name[:-9] for p in SPOOL.glob('*.jsonl.gz') if p.name[:-9] not in excluded)
    assert not (set(projects)&excluded)
    rng=random.Random(20261114);rng.shuffle(projects);rows=[];counts=Counter();longs=Counter();considered=Counter()
    for project in projects:
        issues,vocab=project_data(project);groups=make_candidates(issues)
        for group,cap,target in [('implicit',3,500),('explicit',1,100)]:
            rng.shuffle(groups[group]);accepted=0
            for entry in groups[group]:
                if counts[group]>=target or accepted>=cap:break
                row=materialize(project,entry,group,vocab);considered[group]+=1
                if len('\n'.join(row['lines']))>20000:longs[group]+=1;continue
                row['id']=f'jcv2-{len(rows)+1:04d}';rows.append(row);accepted+=1;counts[group]+=1
        print(project,dict(counts),flush=True)
        if counts['implicit']==500 and counts['explicit']==100:break
    if dict(counts)!={'implicit':500,'explicit':100}:raise RuntimeError(f'insufficient draw {counts}')
    return rows,{'projects_in_pool':len(projects),'excluded':sorted(excluded),'selected_counts':dict(counts),'long_context_excluded_among_examined':dict(longs),'examined_candidates':dict(considered),'note':'Counts cover candidates examined in randomized draw before each cap was met.'}
def main():
    prior=OUT/'jcv2_prior_table.json'
    if not prior.is_file():raise RuntimeError('prior must be frozen first')
    paths=[OUT/'jcv2_items.jsonl',OUT/'jcv2_lead_label_subset.json',OUT/'jcv2_haiku_subset.json',OUT/'jcv2_draw_report.json']
    if any(p.exists() for p in paths):raise FileExistsError('refusing overwrite')
    rows,report=draw()
    with paths[0].open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    r=random.Random(20261115);lead=r.sample([x['id'] for x in rows if x['group']=='implicit'],50)+r.sample([x['id'] for x in rows if x['group']=='explicit'],50);r.shuffle(lead)
    haiku=random.Random(20261116).sample([x['id'] for x in rows],200)
    for path,obj in zip(paths[1:],[{'seed':20261115,'ids':lead},{'seed':20261116,'ids':haiku},report]):
        with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    for path in paths:print(path.name,hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':main()
