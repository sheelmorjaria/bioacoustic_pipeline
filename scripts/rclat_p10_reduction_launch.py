"""Bounded, prospectively specified adaptive engineering reduction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['search','baseline','confirm'],required=True);a=p.parse_args()
out=Path('outputs/rclat_p10_reduction_v1');plan=json.loads((out/'plan.json').read_text())
env=os.environ.copy();env['ONEDNN_PRIMITIVE_CACHE_CAPACITY']='8192';env['ONEDNN_VERBOSE']='profile_create,filter=conv'
def run(name,ids,device):
 candidate=dict(recipe_ids=ids,device=device,plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest())
 with (out/f'{name}_candidate.json').open('x') as f:json.dump(candidate,f,indent=2)
 with (out/f'{name}_xpu{device}_execution.log').open('x') as log:
  r=subprocess.run(['timeout','--signal=INT','--kill-after=10s','930s',sys.executable,'scripts/rclat_p10_reduction.py','--condition',name,'--device',str(device)],env=env,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{name}:exit{r.returncode};no retries')
 d=json.loads((out/f'{name}_xpu{device}_result.json').read_text())
 if not d['completed']:raise RuntimeError(f'{name}:incomplete;abstain')
 phase='initial';miss=0
 for line in (out/f'{name}_xpu{device}_execution.log').open():
  if line.startswith('ATTRIBUTION_VALIDATION_BEGIN:'):phase=line.strip().split(':')[-1]
  parts=line.split(',')
  if phase=='2' and len(parts)>6 and parts[:3]==['onednn_verbose','v1','primitive'] and parts[3]=='create:cache_miss' and parts[5]=='convolution':miss+=1
 snaps={v['phase']:v for v in d['snapshots']};delta=(snaps['validation_2_end']['native_in_use']-snaps['validation_2_begin']['native_in_use'])/2**20
 row=dict(name=name,device=device,recipe_count=len(ids),convolution_misses=miss,native_delta_MiB=delta,positive=miss>0 and delta>.1)
 (out/f'{name}_decision.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True);return row
if a.mode=='baseline':
 run('baseline_other',plan['source_recipe_ids'],1)
elif a.mode=='confirm':
 state=json.loads((out/'search_result.json').read_text());run('confirm_best',state['best_recipe_ids'],1)
else:
 history=[];memo={};best=plan['source_recipe_ids'];n=2
 def evaluate(ids):
  key=tuple(ids)
  if key in memo:return memo[key]
  if len(history)>=plan['max_search_tests']:return None
  row=run(f'test{len(history)+1:02}',ids,0);history.append(row);memo[key]=row
  (out/'search_progress.json').write_text(json.dumps(dict(history=history,current_best_recipe_ids=best),indent=2)+'\n')
  return row
 baseline=evaluate(best)
 if not baseline['positive']:raise RuntimeError('Baseline negative;abstain')
 while len(history)<plan['max_search_tests'] and len(best)>1:
  partitions=[best[(i*len(best))//n:((i+1)*len(best))//n] for i in range(n)]
  candidates=partitions if n==2 else [[r for r in best if r not in set(part)] for part in partitions]
  changed=False
  for ids in candidates:
   row=evaluate(ids)
   if row is None:break
   if row['positive']:best=ids;n=2;changed=True;break
  if changed:continue
  if n>=len(best):break
  n=min(len(best),n*2)
 (out/'search_result.json').write_text(json.dumps(dict(best_recipe_ids=best,best_recipe_count=len(best),history=history,tests=len(history),minimality_proven=False,scientific_restart=False),indent=2)+'\n')
