"""Offline retained native stack summaries, preserving incomplete coverage."""
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.append('/tmp/rclat_memray')
import memray
p=Path('outputs/rclat_p10_alternation_native_v1');out={}
for result in sorted(p.glob('*_result.json')):
 name=result.name.removesuffix('_result.json');d=json.loads(result.read_text());capture=p/f'{name}_second_validation.bin'
 if not capture.exists():continue
 reader=memray.FileReader(capture);records=sorted(reader.get_leaked_allocation_records(),key=lambda r:r.size,reverse=True)
 total=sum(r.size for r in records);categories=defaultdict(int);inclusive=defaultdict(int);top=[]
 for i,r in enumerate(records):
  frames=r.native_stack_trace();funcs=[v[0] for v in frames]
  tags=[]
  if any('gemm' in f.lower() or 'matmul' in f.lower() or 'addmm' in f.lower() for f in funcs):tags.append('GEMM_matmul_path')
  if any('conv' in f.lower() for f in funcs):tags.append('convolution_path')
  if any('dnnl' in f.lower() or 'onednn' in f.lower() for f in funcs):tags.append('oneDNN')
  if any('getKernel' in f or 'create_kernel' in f or 'jit' in f.lower() for f in funcs):tags.append('JIT_or_kernel_creation')
  if any('urProgram' in f or 'ur_program' in f for f in funcs):tags.append('UR_program')
  if any('Decompress' in f for f in funcs):tags.append('decompression')
  if any('XPUCachingAllocator' in f for f in funcs):tags.append('XPU_allocator')
  for tag in tags:inclusive[tag]+=r.size
  categories['+'.join(tags) or 'other_unresolved']+=r.size
  if i<20:top.append(dict(bytes=r.size,allocations=r.n_allocations,native_frames=[list(f) for f in frames[:35]],python_frames=[list(f) for f in r.stack_trace()[:8]]))
 row=dict(capture=str(capture),capture_batches=d['capture_batches'],capture_complete=d['capture_complete'],process_completed=d['completed'],retained_bytes=total,retained_allocations=sum(r.n_allocations for r in records),stack_groups=len(records),inclusive_path_bytes=dict(inclusive),path_combinations=dict(categories),top=top)
 out[name]=row
 print(name,'batches',d['capture_batches'],'complete',d['capture_complete'],'retained_MiB',round(total/2**20,3),'inclusive', {k:round(v/max(total,1)*100,2) for k,v in inclusive.items()})
(p/'native_stack_summary.json').write_text(json.dumps(dict(captures=out,inclusive_categories_overlap=True,retained_at_capture_end_not_leak_proof=True),indent=2)+'\n')
