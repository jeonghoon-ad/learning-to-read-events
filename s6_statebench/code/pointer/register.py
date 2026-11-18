"""String-keyed execution; only IC and SS receive benchmark modular arithmetic."""
from sb_labels import formatter,parse_choice
from sb_register import CONST,COPY,INC,NOOP
from pointer.labels import normal

MODULAR={'integer-code-modular','spreadsheet-cells-modular'}

def execute(reads,fmt,num_values):
    state={}; ranks={}
    for r in reads:
        op=r['op'];tgt=r.get('tgt');src=r.get('src');val=r.get('val')
        for name in (tgt,src):
            if name is not None and name not in ranks: ranks[name]=len(ranks)
        if op==NOOP or tgt is None: continue
        if op==CONST: state[tgt]=val
        elif op==COPY: state[tgt]=state.get(src)
        elif op==INC:
            if fmt not in MODULAR or num_values<=0: state[tgt]=None;continue
            try: state[tgt]=str((int(state[tgt])+int(val))%num_values)
            except (ValueError,TypeError,KeyError): state[tgt]=None
        else: raise ValueError('unknown op')
    return state,ranks

def choice_strings(choice,fmt):
    _,variables,values,_=formatter(fmt)
    return {normal(variables[k]):normal(values[v]) for k,v in parse_choice(choice,fmt).items()}

def choose(state,choices,fmt):
    options=[choice_strings(c,fmt) for c in choices]
    if len(options)!=5: raise ValueError('five choices required')
    keys=set(options[0])
    if not keys or any(set(o)!=keys for o in options):raise ValueError('choice keys differ')
    distances=[sum(state.get(k)!=v for k,v in o.items()) for o in options]
    minimum=min(distances);winners=[i for i,d in enumerate(distances) if d==minimum]
    return {'choice':winners[0],'distance':minimum,'tie_count':int(len(winners)>1)}

def score(reads,row):
    state,ranks=execute(reads,row['formatter'],int(row['num_values']))
    names=row['surface_variables']; vals=row['surface_values']; expected=row['final_state']
    if len(names)<len(expected):raise ValueError('missing expected variables')
    exact=all(state.get(normal(names[i]))==normal(str(vals[v])) for i,v in enumerate(expected))
    choice=choose(state,row['choices'],row['formatter'])
    return {'register_exact':exact,'five_choice':choice['choice']==row['correct_choice_index'],
            'five_choice_ties_wrong':choice['choice']==row['correct_choice_index'] and not choice['tie_count'],
            'ties':choice['tie_count'],'choice':choice['choice'],'ranks':ranks}
