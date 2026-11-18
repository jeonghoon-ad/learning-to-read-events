"""The benchmark's two-pattern parser and comment synonym baseline."""
import re
from select_jcomment import core
CREATED = re.compile(r'^\[\d{4}-\d\d-\d\d \d\d:\d\d\] \S+ created (\S+) \(type .*?, priority (.*?), status (.*?), resolution (.*?), assignee (\S+?)\): ')
CHANGED = re.compile(r'^\[\d{4}-\d\d-\d\d \d\d:\d\d\] \S+ changed (status|resolution|priority|assignee) of (\S+) from (.*) to (.*)$')
COMMENT = re.compile(r'^\[\d{4}-\d\d-\d\d \d\d:\d\d\] \S+ commented on (\S+): ', re.S)

def parser_state(lines):
    st={}
    for n,line in enumerate(lines,1):
        m=CREATED.match(line)
        if m:
            for f,v in zip(('priority','status','resolution','assignee'),m.groups()[1:]): st[(m.group(1),f)]=(v,'exact',n)
            continue
        m=CHANGED.match(line)
        if m and (m.group(2),m.group(1)) in st: st[(m.group(2),m.group(1))]=(m.group(4),'exact',n)
    return st

def comment_state(events,vocab):
    st={};line_no=1
    for e in events:
        if e['kind']=='change':continue
        line=core.render_event(e);m=CREATED.match(line)
        if m:
            for f,v in zip(('priority','status','resolution','assignee'),m.groups()[1:]):st[(m.group(1),f)]=(v,'exact',line_no)
        elif e['kind']=='comment':
            m=COMMENT.match(line)
            if m:
                key=m.group(1);body=line[m.end():].casefold()
                for field in ('status','resolution'):
                    values=[v for v in vocab[field] if any(p.casefold() in body for p in [v,*core.SYNONYMS.get(field,{}).get(v,[])])]
                    if len(values)==1 and (key,field) in st:st[(key,field)]=(values[0],'exact',line_no)
                    elif len(values)>1 and (key,field) in st:st[(key,field)]=('unknown','unknown',line_no)
        line_no+=len(line.split('\n'))
    return st
