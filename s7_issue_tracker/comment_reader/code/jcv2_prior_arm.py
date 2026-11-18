"""Apply the 22:44 amendment to the frozen train-only frequency table."""
import json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT/'jcv2'));sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from prior_table import first_keyword
OUT=ROOT/'jcv2/out'

def build(table):
    k={tuple(r['key']):r['value'] for r in table['prior_k']}
    z={tuple(r['key']):r['value'] for r in table['prior_0']}
    fk=defaultdict(Counter); field=defaultdict(Counter)
    for r in table['prior_k']:
        f,_,word=r['key'];fk[(f,word)].update(r['counts'])
    for r in table['prior_0']:
        f,_=r['key'];field[f].update(r['counts'])
    mode=lambda c:sorted(c.items(),key=lambda x:(-x[1],x[0]))[0][0]
    return k,z,{x:mode(c) for x,c in fk.items()},{x:mode(c) for x,c in field.items()}

def predict(r,tables):
    k,z,fk,fields=tables;field=r['field'];initial=r['events'][0]['values'][field];word=None
    for e in reversed(r['events'][1:]):
        word=first_keyword(e.get('body'))
        if word:break
    if not word:pk,kl=initial,'no_keyword_created'
    elif (field,initial,word) in k:pk,kl=k[(field,initial,word)],'field_created_keyword'
    elif (field,word) in fk:pk,kl=fk[(field,word)],'field_keyword'
    elif field in fields:pk,kl=fields[field],'field'
    else:pk,kl=initial,'created_fallback'
    if (field,initial) in z:p0,zl=z[(field,initial)],'field_created'
    elif field in fields:p0,zl=fields[field],'field'
    else:p0,zl=initial,'created_fallback'
    return {'id':r['id'],'prior_k':pk,'prior_0':p0,'keyword':word,'prior_k_level':kl,'prior_0_level':zl}

def main():
    tables=build(json.loads((OUT/'jcv2_prior_table.json').read_text()));counts={'prior_k':Counter(),'prior_0':Counter()}
    path=OUT/'jcv2_prior_predictions_backoff_2244.jsonl'
    with path.open('x') as f:
        for line in (OUT/'jcv2_items.jsonl').open():
            row=predict(json.loads(line),tables);counts['prior_k'][row['prior_k_level']]+=1;counts['prior_0'][row['prior_0_level']]+=1
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'amendment':'22:44','items':sum(counts['prior_k'].values()),'levels':{k:dict(v) for k,v in counts.items()}}
    with (OUT/'jcv2_prior_backoff_2244_report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
