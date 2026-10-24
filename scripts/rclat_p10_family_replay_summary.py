"""Summarize completed bounded engineering runs; no model inference."""
import json
from collections import Counter,defaultdict
from pathlib import Path
p=Path('outputs/rclat_p10_family_replay_v1');results={}
for f in sorted(p.glob('*_result.json')):
 name=f.name.removesuffix('_result.json');d=json.loads(f.read_text());seen=set();counts=defaultdict(Counter);phase='initial'
 for line in (p/f'{name}_execution.log').open():
  if line.startswith('ATTRIBUTION_TRAIN_BEGIN:'):phase='training_'+line.strip().split(':')[-1]
  if line.startswith('ATTRIBUTION_VALIDATION_BEGIN:'):phase='validation_'+line.strip().split(':')[-1]
  a=line.strip().split(',')
  if len(a)<9 or a[:3]!=['onednn_verbose','v1','primitive'] or a[3] not in ['create:cache_hit','create:cache_miss']:continue
  key=tuple(a[4:-1]);c=counts[(phase,a[5])];c[a[3].split(':')[1]]+=1
  if a[3]=='create:cache_miss':c['seen_logged_descriptor_miss' if key in seen else 'first_logged_descriptor_miss']+=1
  seen.add(key)
 snaps={r['phase']:r for r in d['snapshots']};passes=[]
 for i in range(1,4):
  a=snaps.get(f'validation_{i}_begin');b=snaps.get(f'validation_{i}_end')
  if a and b:passes.append(dict(pass_number=i,native_delta_MiB=(b['native_in_use']-a['native_in_use'])/2**20,RSS_delta_MiB=(int(b['VmRSS'].split()[0])-int(a['VmRSS'].split()[0]))/1024,primitive_counts={kind:dict(v) for (ph,kind),v in counts.items() if ph==f'validation_{i}'}))
 results[name]=dict(completed=d['completed'],error=d.get('error'),updates=d['updates'],seconds=d['seconds'],passes=passes)
(p/'summary.json').write_text(json.dumps(dict(runs=results,internal_cache_keys_known=False,scientific_scores=False,scientific_restart=False),indent=2)+'\n')
for name,r in results.items():print(name,r['completed'],[(v['pass_number'],round(v['native_delta_MiB'],3),v['primitive_counts'].get('convolution',{}).get('cache_miss',0)) for v in r['passes']])
