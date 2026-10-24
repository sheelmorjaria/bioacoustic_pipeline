"""Run each frozen engineering condition once, sequentially on one device."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--device',type=int,required=True);a=p.parse_args()
out=Path('outputs/rclat_p10_matrix_composition_v1')
env=os.environ.copy();env['ONEDNN_PRIMITIVE_CACHE_CAPACITY']='8192';env['ONEDNN_VERBOSE']='profile_create,filter=conv'
for condition in ['prefix128','seed17','seed29','seed43','prefix512']:
 name=f'{condition}_xpu{a.device}'
 with (out/f'{name}_execution.log').open('x') as log:
  run=subprocess.run(['timeout','--signal=INT','--kill-after=10s','930s',sys.executable,'scripts/rclat_p10_matrix_composition.py','--condition',condition,'--device',str(a.device)],env=env,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode!=0:raise RuntimeError(f'{name}: exit {run.returncode}; no retry')
 result=json.loads((out/f'{name}_result.json').read_text())
 if not result['completed']:raise RuntimeError(f'{name} incomplete; no retry')
 print(name+' complete',flush=True)
