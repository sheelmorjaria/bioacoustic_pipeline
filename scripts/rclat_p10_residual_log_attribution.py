import json
from pathlib import Path
from collections import defaultdict,Counter
src=Path('outputs/rclat_p10_cache_fullworkload_engineering_v1');out=Path('outputs/rclat_p10_residual_attribution_v1')
results={}
for arm in ['mel','combined']:
 seen=set(); counts=defaultdict(Counter); epoch=0;phase='initial'
 for line in (src/f'{arm}_execution.log').open():
  if line.startswith('ENGINEERING_TRAIN_BEGIN:'):epoch=int(line.split(':')[1]);phase='train'
  elif line.startswith('ENGINEERING_VALIDATION_FORWARD_BEGIN:'):phase='validation'
  if not line.startswith('onednn_verbose,'):continue
  parts=line.strip().split(',')
  if len(parts)<9 or parts[2]!='primitive' or not parts[3].startswith('create:cache_'):continue
  kind=parts[5];hit=parts[3].split(':')[1];key=tuple(parts[4:-1]);c=counts[(epoch,phase,kind)]
  c[hit]+=1
  if hit=='cache_miss':c['previously_logged_descriptor_miss' if key in seen else 'first_logged_descriptor_miss']+=1
  seen.add(key)
 resources={}
 for line in (src/f'{arm}_steps.jsonl').open():
  r=json.loads(line)
  if r.get('event') in ['validation_begin','validation_end','checkpoint_saved']:
   resources[(r['epoch'],r['event'])]=r
 rows=[]
 for e in range(1,13):
  a=resources[e,'validation_begin'];b=resources[e,'validation_end'];c=resources[e,'checkpoint_saved']
  rows.append(dict(epoch=e,validation_native_MiB=(b['glibc_in_use']-a['glibc_in_use'])/2**20,checkpoint_native_MiB=(c['glibc_in_use']-b['glibc_in_use'])/2**20,counts={phase:{kind:dict(v) for (ep,ph,kind),v in counts.items() if ep==e and ph==phase} for phase in ['train','validation']}))
 results[arm]=rows
 print(arm)
 for r in rows:
  print(r['epoch'],round(r['validation_native_MiB'],2),json.dumps(r['counts']))
(out/'phase_attribution.json').write_text(json.dumps(dict(source=str(src),descriptor_caveat='Logged descriptor excludes runtime cache-key fields not printed by oneDNN; repeated logged descriptors do not prove identical internal keys or eviction.',arms=results),indent=2)+'\n')
