"""Bounded engineering-only repeated validation, with device crossover."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time
import torch
from torch.nn.utils.rnn import pad_sequence
import rclat_p10_cache_fullworkload as base
from rclat_p10_train import epoch_order, sequence
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text
from bioacoustic_pipeline.representation.rclat_numerics import check_gradients, check_parameters_and_optimizer
from rclat_p10_memory_attribution import Mallinfo2
from rclat_p10_c1_shape_replay import telemetry

OUT=Path('outputs/rclat_p10_validation_repeat_v1')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',type=int,required=True);ap.add_argument('--condition',choices=['v_only','t_v'],required=True);args=ap.parse_args()
    plan=json.loads((OUT/'plan.json').read_text())
    if base.protocol.sha(__file__)!=plan['runner_sha256']:raise ValueError('Runner changed')
    _,cohort=base.check()
    name=f'{args.condition}_xpu{args.device}'
    with (OUT/f'{name}_STARTED.json').open('x') as f:json.dump(dict(plan_sha256=base.protocol.sha(OUT/'plan.json')),f)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(0);torch.xpu.manual_seed_all(0)
    device=f'xpu:{args.device}';torch.xpu.set_device(device)
    model=base.build(64).to(device)
    scaler=json.loads((base.predecessor.original.OUT/'mel_scaler.json').read_text())
    libc=ctypes.CDLL(None);libc.mallinfo2.restype=Mallinfo2
    validation=sorted(cohort['selected_validation'],key=lambda r:r['uid'])
    training=epoch_order(cohort['selected_training'],1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    started=time.monotonic();snapshots=[];updates=0
    progress=(OUT/f'{name}_progress.jsonl').open('x',buffering=1)
    def snapshot(phase):
        torch.xpu.synchronize(device);m=libc.mallinfo2()
        r=dict(phase=phase,seconds=time.monotonic()-started,native_in_use=m.uordblks,**telemetry(device));snapshots.append(r);progress.write(json.dumps(r)+'\n');print('ATTRIBUTION_PHASE '+json.dumps(r),flush=True)
    def batch(rows):
        if time.monotonic()-started>plan['seconds_bound']:raise TimeoutError('Prospective diagnostic time bound')
        arrays=[frozen_scale(*sequence(r['uid'],'mel'),scaler) for r in rows]
        return pad_sequence([torch.from_numpy(x) for x,m in arrays],batch_first=True).to(device),pad_sequence([torch.from_numpy(m) for x,m in arrays],batch_first=True).to(device),torch.tensor([len(x) for x,m in arrays])
    result=dict(condition=args.condition,device=device,scientific_scores=False,scientific_restart=False)
    try:
        snapshot('initial')
        if args.condition=='t_v':
            print('ATTRIBUTION_TRAIN_BEGIN',flush=True);model.train()
            for pos in range(0,len(training),4):
                rows=training[pos:pos+4];optimizer.zero_grad(set_to_none=True)
                inputs=batch(rows);logits,lengths=model(*inputs)
                if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Training logits')
                loss=ctc_objective(logits,lengths,[encode_text(r['transcript']) for r in rows]);loss.backward();check_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();check_parameters_and_optimizer(model,optimizer)
                del inputs,logits,lengths,loss;updates+=1
            optimizer.zero_grad(set_to_none=True);snapshot('training_end')
        model.eval()
        for cycle in range(1,4):
            print(f'ATTRIBUTION_VALIDATION_BEGIN:{cycle}',flush=True);snapshot(f'validation_{cycle}_begin')
            with torch.no_grad():
                for pos in range(0,len(validation),4):
                    inputs=batch(validation[pos:pos+4]);logits,lengths=model(*inputs)
                    if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Validation logits')
                    del inputs,logits,lengths
            snapshot(f'validation_{cycle}_end')
        result['completed']=True
    except Exception as exc:result.update(completed=False,error=repr(exc))
    result.update(updates=updates,snapshots=snapshots,seconds=time.monotonic()-started)
    (OUT/f'{name}_result.json').write_text(json.dumps(result,indent=2)+'\n');progress.close()
    print('ATTRIBUTION_RESULT '+json.dumps({k:v for k,v in result.items() if k!='snapshots'}),flush=True)

if __name__=='__main__':main()
