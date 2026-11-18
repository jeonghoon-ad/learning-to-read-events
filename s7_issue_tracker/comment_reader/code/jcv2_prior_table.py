"""Freeze train-project-only workflow priors before drawing any test items."""
import gzip, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from select_jcomment import core
SPOOL=Path('/private/tmp/semistructured_main_20261113_spool')
OUT=ROOT/'jcv2/out/jcv2_prior_table.json'
KEYWORDS=['fixed','fix','committed','commit','merged','merge','applied','resolved','done','pushed','closing','closed','duplicate',"won't fix",'wont fix','invalid','not a bug','reopen']
def first_keyword(body):
    s=str(body or '').casefold()
    hits=[(s.find(k),i,k) for i,k in enumerate(KEYWORDS) if k in s]
    return min(hits)[2] if hits else None
def main():
    train=json.loads((ROOT/'split.json').read_text())['train']
    k=defaultdict(Counter); z=defaultdict(Counter); pairs=0
    for project in train:
        path=SPOOL/f'{project}.jsonl.gz'
        if not path.is_file():raise RuntimeError(f'missing training project {project}')
        with gzip.open(path,'rt') as f:
            for line in f:
                issue=json.loads(line); initial=core.initial(issue)
                comments=[c for c in issue['comments'] if c.get('created') and c.get('author')]
                for ch in issue['changes']:
                    field=ch.get('field')
                    if field not in ('status','resolution') or not ch.get('created') or not ch.get('author'):continue
                    for c in comments:
                        if core.norm(c['author'])!=core.norm(ch['author']) or abs((core.dt(c['created'])-core.dt(ch['created'])).total_seconds())>3600:continue
                        pairs+=1; created=core.text_value(initial[field]); value=core.text_value(ch['toString']); word=first_keyword(c.get('body'))
                        z[(field,created)][value]+=1
                        if word:k[(field,created,word)][value]+=1
    def rows(table):
        return [{'key':list(key),'value':sorted(counts.items(),key=lambda x:(-x[1],x[0]))[0][0], 'counts':dict(sorted(counts.items()))} for key,counts in sorted(table.items())]
    data={'keywords':KEYWORDS,'training_projects':train,'training_pairs':pairs,'prior_k':rows(k),'prior_0':rows(z),'tie_break':'lexicographically smallest value','keyword_rule':'earliest occurrence in comment; keyword-list order breaks ties'}
    with OUT.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    print('prior_table sha256',hashlib.sha256(OUT.read_bytes()).hexdigest(),'pairs',pairs,'K keys',len(k),'0 keys',len(z),flush=True)
if __name__=='__main__':main()
