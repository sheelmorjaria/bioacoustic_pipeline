"""Bounded reduced-workload cache-policy comparison with parity prerequisites."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--device',type=int,required=True);a=p.parse_args()
out=Path('outputs/rclat_p10_reduced_cache_intervention_v1')
for policy in ['capacity8192','capacity16384']:
 env=os.environ.copy();env['ONEDNN_PRIMITIVE_CACHE_CAPACITY']=policy.removeprefix('capacity');env['ONEDNN_VERBOSE']='profile_create,filter=conv'
 for action in ['parity','run']:
  command=[sys.executable,'scripts/rclat_p10_reduced_cache_parity.py','parity','--policy',policy,'--device',str(a.device)] if action=='parity' else [sys.executable,'scripts/rclat_p10_reduced_cache_intervention.py','--condition',policy,'--device',str(a.device)]
  with (out/f'{policy}_xpu{a.device}_{action}_execution.log').open('x') as f:
   r=subprocess.run(['timeout','--signal=INT','--kill-after=10s','930s',*command],env=env,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{policy}/{action} failed; no retry')
  path=out/f'{policy}_xpu{a.device}_parity.json' if action=='parity' else out/f'{policy}_xpu{a.device}_result.json'
  result=json.loads(path.read_text())
  if not result.get('passed',result.get('completed',False)):raise RuntimeError(f'{policy}/{action} gate failed; stop')
  print(policy+' '+action+' complete',flush=True)
