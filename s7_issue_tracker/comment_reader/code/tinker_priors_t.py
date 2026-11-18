"""Refit the frozen v2 prior recipe on the reader's non-dev projects."""
import argparse, gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'jcv2'), str(ROOT)]
sys.dont_write_bytecode=True
from select_jcomment import core
from prior_table import KEYWORDS, first_keyword
from prior_arm import build, predict
from draw_items import SPOOL

def fit(projects):
    k=defaultdict(Counter); z=defaultdict(Counter); pairs=0
    for project in projects:
        with gzip.open(SPOOL/f'{project}.jsonl.gz','rt') as f:
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
    return {'keywords':KEYWORDS,'training_projects':projects,'training_pairs':pairs,'prior_k':rows(k),'prior_0':rows(z),'tie_break':'lexicographically smallest value','keyword_rule':'earliest occurrence in comment; keyword-list order breaks ties'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build-report',type=Path);args=parser.parse_args()
    here=Path(__file__).resolve().parent
    table_path=here/'out/prior_t_table.json'; pred_path=here/'out/prior_t_predictions.jsonl'
    if table_path.exists() or pred_path.exists():raise FileExistsError('refusing to overwrite')
    report=json.loads((args.build_report or here/'data/build_report.json').read_text())
    table=fit(report['train_projects']); tables=build(table)
    items=[json.loads(l) for l in (ROOT/'jcv2/out/jcv2_items.jsonl').open()]
    preds=[predict(row,tables) for row in items]
    table_path.parent.mkdir(parents=True,exist_ok=True)
    with table_path.open('x') as f:json.dump(table,f,ensure_ascii=False,indent=2);f.write('\n')
    with pred_path.open('x') as f:
        for row in preds:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('training projects',len(report['train_projects']),'pairs',table['training_pairs'],'predictions',len(preds))
if __name__=='__main__':main()
