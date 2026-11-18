# Phase 0 exploratory evaluation.
"""Offline-capable S5 CoT evaluation. See --help and workload JSON."""
import math
import argparse, hashlib, json, os, random, re, signal, sys, tarfile, threading, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import ref_s5 as s5
from cot_core import prefixes, prompt, score
from cot_stats import bounds

PERMS = [''.join(str(d + 1) for d in p) for p in s5.ALL_PERMS]
MODEL_PINS = {'H1': ('Qwen/Qwen3.5-0.8B-Base','dc7cdfe2'), 'H2': ('Qwen/Qwen3.5-0.8B','2fc06364'), 'H3': ('Qwen/Qwen3.8-27B','37561816')}
STOP = threading.Event()

def fixed_row(seed, n=43690):
    return torch.randint(0,120,(n,),generator=torch.Generator().manual_seed(seed)).tolist()

def events(seed, n):
    return [PERMS[x] for x in fixed_row(seed)[:n]]

def controls():
    for seed in range(86000,86299):
        row=fixed_row(seed); x=[PERMS[i] for i in row]
        if prefixes(x) != [PERMS[i] for i in s5.prefix_products([j+1 for j in row])[1:]]:
            raise ValueError('composer mismatch seed %s' % seed)
        ev=x[:8]; states=prefixes(ev)
        oracle='\n'.join(f'{i}: {e} > {v}' for i,(e,v) in enumerate(zip(ev,states),1))+'\nfinal: '+states[-1]
        shifted=next(p for p in PERMS if p!=states[-1])
        if not score('A',ev,states[-1])['exact'] or not score('B',ev,oracle)['exact'] or score('A',ev,shifted)['exact'] or score('B',ev,oracle.replace('final: '+states[-1], 'final: '+shifted))['exact']:
            raise ValueError('scorer control mismatch')

def demos(): return [events(99000,1),events(99001,2),events(99002,3)]

def chat_messages(arm, ev):
    # Fix 17:02 (L-118 one fix): chat hosts get the same instruction and demos as alternating user/assistant turns,
    # because one packed user message produced format failures on H2 (explanations in A, missing 't:' in B).
    base='B' if arm=='B-think' else arm
    full=prompt(base, ev, demos())
    intro, rest = full.split('\n\n', 1)[0], None
    msgs=[]
    for i,row in enumerate(demos()):
        st=prefixes(row); q='Events: '+''.join(e+',' for e in row)
        a=st[-1] if base=='A' else ''.join(f'{j}: {e} > {v}\n' for j,(e,v) in enumerate(zip(row,st),1))+'final: '+st[-1]
        msgs.append({'role':'user','content':(intro+'\n\n' if i==0 else '')+q}); msgs.append({'role':'assistant','content':a})
    msgs.append({'role':'user','content':'Events: '+''.join(e+',' for e in ev)})
    return msgs

def render(tokenizer, host, arm, ev):
    if host == 'H1':
        return prompt('B' if arm=='B-think' else arm, ev, demos())
    return tokenizer.apply_chat_template(chat_messages(arm, ev),tokenize=False,add_generation_prompt=True,enable_thinking=(arm=='B-think'))

def final_line_exact(arm, answer, truth):
    # Diagnostic only (not the primary metric): the last stated final state equals the truth, ignoring step-line format.
    found=re.findall(r'final\s*:\s*([1-5]{5})', answer, re.IGNORECASE) if arm!='A' else re.findall(r'(?<![0-9])([1-5]{5})(?![0-9])', answer)
    return bool(found) and found[-1]==truth

def copy_prompt(tokenizer, host, ev):
    raw='Write the last event as five digits.\n'
    for i in range(3):
        example=events(99000+i,i+1)
        raw+='Events: '+''.join(x+',' for x in example)+'\n'+example[-1]+'\n\n'
    raw+='Events: '+''.join(x+',' for x in ev)+'\n'
    if host!='H1': raw=tokenizer.apply_chat_template([{'role':'user','content':raw}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return raw

def answer_finished(text, mode):
    if mode in ('A', 'copy'):
        return bool(re.fullmatch(r'\s*[1-5]{5}',text) or '\n' in text)
    if mode == 'B-think':
        if '</think>' not in text: return False
        text=text.rsplit('</think>',1)[-1]
    return bool(re.search(r'(?:^|\n)\s*final\s*:\s*[1-5]{5}$',text,re.IGNORECASE))

TAIL=64
def tail_finished(tok,ids,mode,closed):
    # Review F3: decode only the last TAIL tokens; for B-think remember whether </think> has appeared.
    if mode in ('A','copy'): return answer_finished(tok.decode(ids,skip_special_tokens=True),mode)
    if mode=='B-think' and not closed.get('v'):
        if '</think>' in tok.decode(ids[-TAIL:],skip_special_tokens=False): closed['v']=True
        else: return False
        text=tok.decode(ids[-TAIL:],skip_special_tokens=False).rsplit('</think>',1)[-1]
        return bool(re.search(r'(?:^|\n)\s*final\s*:\s*[1-5]{5}$',text,re.IGNORECASE))
    return answer_finished(tok.decode(ids[-TAIL:],skip_special_tokens=True),'B')

def b_cap(tok,ev):
    # Review F1: cap from the token count of the correct transcript, not 16n+32.
    states=prefixes(ev)
    oracle='\n'.join(f'{i}: {e} > {v}' for i,(e,v) in enumerate(zip(ev,states),1))+'\nfinal: '+states[-1]
    return math.ceil(1.25*len(tok(oracle,add_special_tokens=False)['input_ids']))+64

def max_new_for(tok,arm,evs):
    if arm=='B-think': return 8192
    if arm=='B': return max(b_cap(tok,e) for e in evs)
    return 12

class AnswerStop(StoppingCriteria):
    def __init__(self,tok,width,mode): self.tok,self.width,self.mode,self.at,self.closed=tok,width,mode,{},{}
    def __call__(self,input_ids,scores,**kwargs):
        decisions=[]
        for i,row in enumerate(input_ids):
            if i not in self.at and tail_finished(self.tok,row[self.width:].tolist(),self.mode,self.closed.setdefault(i,{})):
                self.at[i]=len(row)-self.width
            decisions.append(i in self.at)
        return torch.tensor(decisions,device=input_ids.device,dtype=torch.bool)

def generate(model, tok, texts, max_new, device, mode):
    batch=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to(device)
    width=batch['input_ids'].shape[1]; criterion=AnswerStop(tok,width,mode)
    with torch.inference_mode():
        out=model.generate(**batch,do_sample=False,max_new_tokens=max_new,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,use_cache=True,stopping_criteria=StoppingCriteriaList([criterion]))
    generated=out.sequences[:,width:]
    eos=set([tok.eos_token_id] if isinstance(tok.eos_token_id,int) else (tok.eos_token_id or []))
    counts=[]
    for i,row in enumerate(generated.tolist()):
        if i in criterion.at:
            counts.append(criterion.at[i]); continue
        count=0
        for x in row:
            count+=1
            if x in eos: break
        counts.append(count)
    result=[]
    for i in range(len(texts)):
        ids=generated[i,:counts[i]]
        result.append((tok.decode(ids,skip_special_tokens=True),tok.decode(ids,skip_special_tokens=False),int(batch['attention_mask'][i].sum()),counts[i]))
    return result

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)

def cache_bytes(obj,seen=None):
    if seen is None: seen=set()
    if id(obj) in seen: return 0
    seen.add(id(obj))
    if isinstance(obj,torch.Tensor): return obj.numel()*obj.element_size()
    if isinstance(obj,dict): return sum(cache_bytes(v,seen) for v in obj.values())
    if isinstance(obj,(list,tuple)): return sum(cache_bytes(v,seen) for v in obj)
    if hasattr(obj,'__dict__'): return sum(cache_bytes(v,seen) for v in vars(obj).values())
    return 0

def cost(model,tok,text,max_new,device,mode):
    encoded=tok(text,return_tensors='pt',add_special_tokens=False).to(device)
    if device.type=='cuda':
        synchronize(device); torch.cuda.reset_peak_memory_stats(device); baseline=torch.cuda.memory_allocated(device)
    else: baseline=None
    with torch.inference_mode():
        synchronize(device); start=time.perf_counter(); output=model(**encoded,use_cache=True,logits_to_keep=1); synchronize(device)
        prefill=time.perf_counter()-start; cache=output.past_key_values
        token=output.logits[:,-1:].argmax(-1); generated=[int(token.item())]; gen=1; synchronize(device); start=time.perf_counter()
        closed={}
        while gen < max_new and int(token.item()) != tok.eos_token_id and not tail_finished(tok,generated,mode,closed):
            output=model(input_ids=token,past_key_values=cache,use_cache=True)
            cache=output.past_key_values; token=output.logits[:,-1:].argmax(-1); gen+=1; generated.append(int(token.item()))
        synchronize(device); decode=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated(device)-baseline if device.type=='cuda' else None
    return {'prompt_tokens':int(encoded['input_ids'].shape[1]),'generated_tokens':gen,'prefill_seconds':prefill,'decode_seconds':decode,'peak_cuda_bytes_above_weights':peak,'cache_bytes':cache_bytes(cache)}

def esl_cost(model,row,device):
    # Same token IDs and three identity events as trigger_register.encode.
    ids=[]
    for action in [0,0,0]+row:
        ids.extend(16+d for d in s5.ALL_PERMS[action]); ids.append(248052)
    ids=torch.tensor([ids],dtype=torch.long,device=device)
    backbone=model.model
    old=backbone.layers; old_norm=backbone.norm
    baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
    torch.manual_seed(0)
    head=torch.nn.Sequential(torch.nn.Linear(6144,512),torch.nn.GELU(),torch.nn.Linear(512,120)).to(device).float().eval()
    captured=[]
    hook=old[9].register_forward_hook(lambda _m,_a,out: captured.append(out[0] if isinstance(out,tuple) else out))
    backbone.layers=torch.nn.ModuleList(list(old)[:10]); backbone.norm=torch.nn.Identity()
    if device.type=='cuda': synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            synchronize(device); start=time.perf_counter(); out=backbone(input_ids=ids,use_cache=True); synchronize(device); prefill=time.perf_counter()-start
            features=captured[0][:,18:,:].reshape(1,len(row),6144).float()
            synchronize(device); start=time.perf_counter(); logits=head(features); actions=logits.argmax(-1)[0].tolist(); state=0
            for action in actions: state=int(s5.MUL[action,state])
            synchronize(device); selector=time.perf_counter()-start
        return {'n':len(row),'boot_events':3,'prompt_tokens':int(ids.shape[1]),'generated_tokens':0,'prefill_seconds':prefill,'selector_register_seconds':selector,'cache_bytes':cache_bytes(out.past_key_values),'peak_cuda_bytes_above_weights':torch.cuda.max_memory_allocated(device)-baseline if baseline is not None else None}
    finally: hook.remove(); backbone.layers=old; backbone.norm=old_norm

def aggregate(cells):
    groups={}
    for key, value in cells.items():
        host,arm,ns=key.split(':'); n=int(ns)
        groups.setdefault((host,arm),[]).append((n,value))
    result={}
    for (host,arm), records in groups.items():
        records.sort(key=lambda x:x[0])
        fall=next((n for n,v in records if v['bounds']['upper_one_sided']<.99003),None)
        chance=next((n for n,v in records if v['bounds']['upper_one_sided']<.05),None)
        matched=[n for n,v in records if arm=='B' and v['accuracy']>=.95 and v['bounds']['upper_two_sided']>=.99003]
        result[host+':'+arm]={'n_fall':fall,'n_chance':chance,'matched_accuracy_lengths':matched if matched else 'not reached'}
    return result

def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)

def write_rows(path,rows):
    tmp=path.with_suffix('.jsonl.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def archive(out):
    target=out/'ARCHIVE.tgz'; temp=out/'ARCHIVE.tgz.tmp'
    with tarfile.open(temp,'w:gz') as tar:
        for p in out.iterdir():
            if p.is_file() and p.name not in ('ARCHIVE.tgz','ARCHIVE.tgz.tmp','ARCHIVE.tgz.sha256','DONE'): tar.add(p,arcname=p.name)
    os.replace(temp,target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest(); (out/'ARCHIVE.tgz.sha256').write_text(digest+'  ARCHIVE.tgz\n')

def heartbeat(out):
    while not STOP.wait(30): (out/'heartbeat').write_text(str(time.time())+'\n')

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/'heartbeat').write_text(str(time.time())+'\n')
    threading.Thread(target=heartbeat,args=(out,),daemon=True).start()
    def term(_sig,_frame):
        atomic(out/'STATUS.json',{'stage':'interrupted'}); archive(out); (out/'PARTIAL').write_text('SIGTERM\n'); os._exit(143)
    signal.signal(signal.SIGTERM,term)
    workload_bytes=Path(args.workload).read_bytes(); workload_hash=hashlib.sha256(workload_bytes).hexdigest()
    manifest=out/'workload_sha256'
    if manifest.exists() and manifest.read_text().strip()!=workload_hash: raise ValueError('resume workload differs')
    manifest.write_text(workload_hash+'\n')
    config=json.loads(workload_bytes); device=torch.device(args.device)
    if device.type not in ('cpu','cuda'): raise ValueError('only CPU or CUDA')
    torch.set_num_threads(args.threads)
    atomic(out/'STATUS.json',{'stage':'controls'})
    try: controls()
    except Exception as exc:
        atomic(out/'STATUS.json',{'stage':'control_failure','error':str(exc)}); return 20
    atomic(out/'controls.json',{'composer':'PASS all 299 full windows','scorer':'PASS oracle and shifted'})
    rows_path=out/'rows.jsonl'; rows=[]
    if rows_path.exists():
        for line in rows_path.read_text().splitlines(): rows.append(json.loads(line))
    prior=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
    summaries=prior.get('cells',{}); costrows=prior.get('cost',[]); copies=prior.get('copy',{}); nulls=prior.get('null',{}); examples=json.loads((out/'examples.json').read_text()) if (out/'examples.json').exists() else {}
    for host in config['hosts']:
        name=host['host']; revision=MODEL_PINS[name][1]
        model_manifest=Path(__file__).resolve().parent/'model_paths.json'
        local_paths=json.loads(model_manifest.read_text()) if model_manifest.exists() else {}
        modelpath=host.get('path',local_paths.get(name,MODEL_PINS[name][0]))
        if Path(modelpath).is_dir() and revision not in str(Path(modelpath).resolve()): raise ValueError('local model revision mismatch')
        tok=AutoTokenizer.from_pretrained(modelpath,revision=revision,local_files_only=args.offline,padding_side='left')
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(modelpath,revision=revision,local_files_only=args.offline,dtype=torch.float32 if device.type=='cpu' else torch.bfloat16).to(device).eval()
        for p in model.parameters(): p.requires_grad_(False)
        atomic(out/'STATUS.json',{'stage':'copy_control','host':name})
        copy_evs=[events(seed,8) for seed in range(99100,99120)]; correct=0; copy_rows=[]
        for start in range(0,20,args.batch_size):
            atomic(out/'STATUS.json',{'stage':'copy_control','host':name,'completed':start,'total':20})
            evs=copy_evs[start:start+args.batch_size]; prompts=[copy_prompt(tok,name,e) for e in evs]
            for ev,(answer,raw,nt,ng) in zip(evs,generate(model,tok,prompts,12,device,'copy')):
                ok=answer.strip()==ev[-1]; correct+=ok; copy_rows.append({'seed':99100+len(copy_rows),'answer':answer,'generation':raw,'truth':ev[-1],'exact':ok,'prompt_tokens':nt,'generated_tokens':ng})
            atomic(out/'copy_control_partial.json',{'host':name,'completed':len(copy_rows),'rows':copy_rows})
            print('copy',name,len(copy_rows),'/20',flush=True)
        copies[name]={'correct':correct,'total':20,'bounds':bounds(correct,20),'format_not_read':correct<18,'rows':copy_rows}
        histories={}; best_a=None
        for arm in host['arms']:
            chance=None; confirmation=False
            for n in host['lengths']:
                if arm=='B-think' and (name!='H2' or n>16): continue
                if name=='H3' and n not in (8,32,128,512): raise ValueError('invalid H3 length')
                if chance is not None:
                    if name=='H3' or confirmation or n!=chance*2 or n>256: break
                    confirmation=True
                count=int(host.get('windows', 299 if n<=256 else 100)); seeds=list(range(86000,86000+count))
                existing=[r for r in rows if (r['host'],r['arm'],r['n'])==(name,arm,n)]
                complete={r['seed'] for r in existing}
                if len(complete)!=len(existing) or not complete.issubset(seeds): raise ValueError('invalid resume rows')
                pending=[s for s in seeds if s not in complete]
                atomic(out/'STATUS.json',{'stage':'accuracy','host':name,'arm':arm,'n':n,'pending':len(pending)})
                for start in range(0,len(pending),args.batch_size):
                    batch=pending[start:start+args.batch_size]; evs=[events(s,n) for s in batch]; prompts=[render(tok,name,arm,e) for e in evs]
                    results=generate(model,tok,prompts,max_new_for(tok,arm,evs),device,arm)
                    for seed,ev,text,(output,raw,nt,ng) in zip(batch,evs,prompts,results):
                        answer=output.rsplit('</think>',1)[-1] if arm=='B-think' else output
                        scored=score('B' if arm=='B-think' else arm,ev,answer)
                        row={'seed':seed,'n':n,'arm':arm,'host':name,**scored,'tokens':{'prompt':nt,'generated':ng},'generation':raw,'generation_scored':output,'prompt_tokens':nt,'gen_tokens':ng,'prefill_s':None,'decode_s':None,'peak_mem_bytes':None,'cache_bytes':None,'format_not_read':copies[name]['format_not_read'],'final_line_exact':final_line_exact(arm,answer,scored['truth'])}
                        with rows_path.open('a') as f: f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
                        rows.append(row); examples.setdefault(arm,{'prompt':text,'generation':raw})
                selected=[r for r in rows if (r['host'],r['arm'],r['n'])==(name,arm,n)]
                k=sum(r['exact'] for r in selected); cp=bounds(k,count); summaries[f'{name}:{arm}:{n}']={'correct':k,'total':count,'bounds':cp,'format_fail':sum(not r['format_ok'] for r in selected),'accuracy':k/count}
                if arm=='A' and k/count>=.5: best_a=n
                if cp['upper_one_sided']<.05 and chance is None: chance=n
                for seed in seeds[:min(5,count)]:
                    if any((c.get('host'),c.get('arm'),c.get('n'),c.get('seed'))==(name,arm,n,seed) for c in costrows): continue
                    text=render(tok,name,arm,events(seed,n)); metrics=cost(model,tok,text,max_new_for(tok,arm,[events(seed,n)]),device,arm)
                    costrows.append({'host':name,'arm':arm,'n':n,'seed':seed,**metrics,'per_event':{k:v/n if v is not None else None for k,v in metrics.items()}})
                    target=next(r for r in rows if (r['host'],r['arm'],r['n'],r['seed'])==(name,arm,n,seed))
                    target.update({'prefill_s':metrics['prefill_seconds'],'decode_s':metrics['decode_seconds'],'peak_mem_bytes':metrics['peak_cuda_bytes_above_weights'],'cache_bytes':metrics['cache_bytes'],'cost_prompt_tokens':metrics['prompt_tokens'],'cost_gen_tokens':metrics['generated_tokens']})
                    write_rows(rows_path,rows)
                atomic(out/'summary.json',{'cells':summaries,'collapse':aggregate(summaries),'copy':copies,'cost':costrows,'null':nulls})
                atomic(out/'examples.json',examples)
                print('cell',name,arm,n,k,'/',count,flush=True)
        if best_a is None: nulls[name]={'status':'not informative (A at chance)'}
        elif args.skip_null:
            nulls[name]={'status':'skipped for CPU dry run','eligible_n':best_a}
        else:
            total=299; k=0; null_same=0
            for start in range(0,total,args.batch_size):
                ids=range(86000+start,min(86000+start+args.batch_size,86300-1)); evs=[events(s,best_a) for s in ids]; shuffled=[]
                for seed,ev in zip(ids,evs):
                    v=ev[:]; random.Random(seed).shuffle(v); shuffled.append(v)
                outputs=generate(model,tok,[render(tok,name,'A',v) for v in shuffled],12,device,'A')
                for seed,orig,v,o in zip(ids,evs,shuffled,outputs):
                    ex=score('A',orig,o[0])['exact']; k+=ex; same=prefixes(v)[-1]==prefixes(orig)[-1]; null_same+=same
                    with (out/'null_rows.jsonl').open('a') as f: f.write(json.dumps({'host':name,'n':best_a,'seed':seed,'shuffled':v,'generation':o[1],'exact':bool(ex),'shuffled_state_equals_true':bool(same)})+'\n')
            nulls[name]={'n':best_a,'correct':k,'total':total,'bounds':bounds(k,total),'shuffled_state_equals_true':null_same}
        if name=='H1':
            for n in host.get('esl_lengths',[]):
                if not any(c.get('host')==name and c.get('arm')=='E' and c.get('n')==n for c in costrows):
                    costrows.append({'host':name,'arm':'E',**esl_cost(model,fixed_row(86000)[:n],device)})
        del model
    atomic(out/'examples.json',examples); atomic(out/'summary.json',{'cells':summaries,'collapse':aggregate(summaries),'copy':copies,'cost':costrows,'null':nulls})
    atomic(out/'STATUS.json',{'stage':'done'}); STOP.set(); archive(out); (out/'DONE').write_text('ok\n'); return 0

def main():
    p=argparse.ArgumentParser(); p.add_argument('--workload',required=True); p.add_argument('--out',default='out'); p.add_argument('--device',default='cuda'); p.add_argument('--offline',action='store_true'); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--threads',type=int,default=6); p.add_argument('--skip-null',action='store_true'); args=p.parse_args()
    try: return run(args)
    except Exception as exc:
        out=Path(args.out); out.mkdir(parents=True,exist_ok=True); atomic(out/'STATUS.json',{'stage':'error','type':type(exc).__name__,'error':str(exc)}); raise
if __name__=='__main__': sys.exit(main())
