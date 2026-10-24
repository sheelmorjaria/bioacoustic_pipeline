#!/usr/bin/env python3
"""Bounded MACDE stage-1 pilot: prepare -> preregister -> one protected run.

No unit/social labels. Existing bat corpus is previously explored, so this is a
new split pilot, not untouched confirmatory evidence. BirdPark is descriptive
zero-shot transfer only (two recordings). Stage 2 remains gated.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from bioacoustic_pipeline.embeddings.macde import (
    MACDE, MaskedStandardizer, eligible_mask, forecast_objective, CONTEXT_FRAMES, HORIZONS,
)
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations, waveform_detail_reference, BAND_EDGES
from bioacoustic_pipeline.diagnostics.acoustic_derivatives import supported_features, FEATURE_SCHEMA_VERSION
from bioacoustic_pipeline.pipeline.release_contract import (
    ScientificRunContract, MaskProvenance, source_manifest_from, raise_for_science,
)
from acoustic_dynamics_report import ALPHAS, ridge_fit, ridge_predict

CACHE=ROOT/'cache/macde_stage1.npz'
MANIFEST=ROOT/'outputs/macde_stage1_source_audit.json'
PREREG=ROOT/'outputs/macde_stage1_preregistration.json'
FRAMES=128
SEED=0
EPOCHS=20
BATCH=8
COUNTS={'fit':64,'validation':16,'heldout':16}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def text(value):
    return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'


def freeze(path,value):
    payload=text(value)
    if path.exists() and path.read_text()!=payload:
        raise RuntimeError(f'Refusing to overwrite differing artifact: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(payload)


def code_hashes():
    paths=[Path(__file__),ROOT/'src/bioacoustic_pipeline/embeddings/macde.py',
           ROOT/'src/bioacoustic_pipeline/diagnostics/macde_data.py',
           ROOT/'src/bioacoustic_pipeline/diagnostics/acoustic_derivatives.py',
           ROOT/'src/bioacoustic_pipeline/diagnostics/acoustic_fingerprint/physics.py',
           ROOT/'src/bioacoustic_pipeline/diagnostics/acoustic_fingerprint/_dsp.py',
           ROOT/'scripts/acoustic_dynamics_report.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def partition(group):
    value=int(hashlib.sha256(('macde-stage1-v1:'+group).encode()).hexdigest(),16)%100
    return 'heldout' if value>=80 else ('validation' if value>=60 else 'fit')


def prepare():
    if CACHE.exists() or MANIFEST.exists():
        raise RuntimeError('Prepared cohort already exists; no automatic replacement')
    raw=ROOT/'data/egyptian_fruit_bats'
    dates={}
    with (raw/'FileInfo.csv').open() as f:
        for row in csv.DictReader(f):
            dates[int(row['FileID'])]=row['Recording time'].split()[0]
    by_name={}
    with (raw/'Annotations_orig.csv').open() as f:
        for i,row in enumerate(csv.DictReader(f)):
            by_name[f'{i}.wav']=dates[int(row['FileID'])]
    # Index/date only: no emitter, recipient, context, unit or call-type label read.
    paths=sorted((raw/'audio').glob('*.wav'),key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())
    items,observations=[],[]
    counts=Counter()
    group_counts=Counter()
    used_hashes=set()
    exclusions=Counter()
    for path in paths:
        if all(counts[p]>=n for p,n in COUNTS.items()):
            break
        if path.name not in by_name:
            exclusions['missing_source_date']+=1
            continue
        group=by_name[path.name]
        split=partition(group)
        if counts[split]>=COUNTS[split] or group_counts[group]>=2:
            continue
        info=sf.info(path)
        required=round(info.samplerate*.04)+(FRAMES-1)*round(info.samplerate*.01)
        if info.frames<required:
            exclusions['too_short_for_declared_frame_cap']+=1
            continue
        fingerprint=sha(path)
        if fingerprint in used_hashes:
            exclusions['duplicate_audio']+=1
            continue
        audio,sr=sf.read(path,frames=required,dtype='float32',always_2d=True)
        obs=frame_observations(audio.mean(1),sr,max_frames=FRAMES)
        positions=eligible_mask(torch.from_numpy(obs['time_valid'])[None]).sum().item()
        if positions<8:
            exclusions['insufficient_contiguous_support']+=1
            continue
        used_hashes.add(fingerprint)
        observations.append(obs)
        items.append({'name':path.name,'path':str(path.relative_to(ROOT)),'group':group,
                      'split':split,'sha256':fingerprint,'sample_rate':sr,'positions':int(positions),
                      'kind':'bat','frames':len(obs['time_valid'])})
        counts[split]+=1
        group_counts[group]+=1
        if len(items)%8==0:
            print(f'prepared bat sources: {len(items)} {dict(counts)}',flush=True)
    if dict(counts)!=COUNTS:
        raise RuntimeError(f'Insufficient eligible recordings for fixed pilot: {dict(counts)}')
    for path in sorted((ROOT/'data/birdpark_pairs/Data').glob('*.h5')):
        import h5py
        with h5py.File(path) as f:
            sr=float(f['recInfo'].attrs['sampleRate_Hz'])
            # BirdPark fs is fractional; use exact sample count/rate, not rounded seconds.
            required=round(sr*.04)+(FRAMES-1)*round(sr*.01)
            audio=f['daqSignals'][0,:required]
        obs=frame_observations(audio,sr,max_frames=FRAMES)
        observations.append(obs)
        items.append({'name':path.name,'path':str(path.relative_to(ROOT)),
                      'group':path.stem,'split':'transfer','sha256':sha(path),
                      'sample_rate':sr,'positions':int(eligible_mask(torch.from_numpy(obs['time_valid'])[None]).sum()),
                      'kind':'birdpark','frames':len(obs['time_valid'])})
    # Controlled detail fixtures: paired gains share a family split; no network fits on fixtures.
    for family in range(24):
        rng=np.random.default_rng(10000+family)
        sr=250000
        t=np.arange(round(sr*.04)+(FRAMES-1)*round(sr*.01))/sr
        carrier=rng.uniform(700,30000)
        rate=rng.uniform(3,18)
        depth=rng.uniform(.1,.8)
        ratio=rng.uniform(0,.6)
        phase=rng.uniform(0,2*np.pi)
        wave=(np.sin(2*np.pi*carrier*t+phase)+ratio*np.sin(4*np.pi*carrier*t))*(1+depth*np.sin(2*np.pi*rate*t))
        for gain in (.15,.30):
            audio=(gain*wave).astype(np.float32)
            obs=frame_observations(audio,sr,max_frames=FRAMES)
            observations.append(obs)
            items.append({'name':f'fixture_{family}_{gain}','group':f'fixture_{family}',
                          'split':'detail_fit' if family<16 else 'detail_eval','kind':'fixture',
                          'sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'sample_rate':sr,
                          'frames':len(obs['time_valid']),'reference':waveform_detail_reference(audio,sr).tolist()})
        if family%8==7:
            print(f'prepared independent detail families: {family+1}',flush=True)
    arrays={key:np.stack([o[key] for o in observations]) for key in
            ('spectral','spectral_valid','physical','physical_valid','source_valid','detected_activity','time_valid')}
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(CACHE,**arrays)
    freeze(MANIFEST,{'schema':1,'items':items,'exclusions':dict(exclusions),
                     'feature_schema_version':FEATURE_SCHEMA_VERSION,'feature_names':list(supported_features()),
                     'frequency_band_edges_hz':BAND_EDGES.tolist(),
                     'cache_sha256':sha(CACHE),'code_provenance':code_hashes(),
                     'metadata_sha256':{name:sha(raw/name) for name in ('FileInfo.csv','Annotations_orig.csv')},
                     'masks':{'source_validity':'all samples read from real file and finite; no introduced padding',
                              'detected_activity':'native RMS >=1e-3, matching frozen fingerprint absolute gate',
                              'feature_support':'all twelve physical values finite and quality supported; at least one supported spectral band',
                              'task_eligibility':'52 preceding/current frames and 20 future frames contiguous; no compressed gaps'},
                     'source_scope':'date-blocked, label-free deterministic subset of previously explored bat recordings; pilot only',
                     'transfer_scope':'first native frames of Mic1 in two BirdPark recordings; no behavioural labels or adaptation',
                     'models_fitted':False})
    print('Prepared:',str(MANIFEST),flush=True)


def load():
    manifest=json.loads(MANIFEST.read_text())
    if sha(CACHE)!=manifest['cache_sha256']:
        raise RuntimeError('Prepared cache changed')
    if manifest['code_provenance']!=code_hashes():
        raise RuntimeError('Preparation code changed; do not silently regenerate cohort')
    with np.load(CACHE,allow_pickle=False) as z:
        arrays={key:z[key] for key in z.files}
    return manifest,arrays


def protocol():
    manifest,arrays=load()
    counts=Counter()
    groups={}
    for i,item in enumerate(manifest['items']):
        if item['kind']=='bat':
            counts[item['split']]+=int(eligible_mask(torch.from_numpy(arrays['time_valid'][i])[None]).sum())
            groups.setdefault(item['split'],set()).add(item['group'])
    return {'schema':1,'study':'MACDE stage-1 bounded acoustic representation pilot',
            'registration':'local prospective specification; not external registry',
            'source_audit_sha256':sha(MANIFEST),'cache_sha256':sha(CACHE),'code_provenance':code_hashes(),
            'seed':SEED,'frames_per_file':FRAMES,'frame_window_ms':40,'frame_hop_ms':10,
            'frame_time':'each observation is known only at window end; no center padding or future samples',
            'actual_fractional_sr_note':'hop is rounded native 10ms sample count; window is exactly four hops so h=4 is nonoverlapping even at fractional sample rates',
            'reference_features':list(supported_features()),'latent_width':64,
            'change_lags_ms':[20,50,100,200],'horizons_ms':[40,80,200],
            'eligibility':'same full 52-frame history + 20-frame future contiguous positions for every representation and horizon',
            'eligible_positions':dict(counts),'groups':{p:len(g) for p,g in groups.items()},
            'representations':{'R0':'physical state and previous state; existing frozen Ridge implementation and alpha grid',
                               'R0_long':'all 52 physical states; stronger matched-support history control',
                               'R1':'64-D spectral-only causal encoder, frozen physical forecasting supervision',
                               'R2':'same 64-D causal encoder with spectral and physical streams',
                               'R3':'identical architecture to R2 plus multiscale change auxiliary loss, lambda=.1'},
            'encoder_training':{'optimizer':'Adam','lr':.001,'weight_decay':.0001,'max_epochs':EPOCHS,
                                'batch_size':BATCH,'gradient_clip':5,'threads':2,'device':'cpu',
                                'selection':'minimum validation 40ms physical MSE; fixed budget for each arm; save selected epoch',
                                'refit':'same seed, selected epochs on fit+validation, refit input scalers only on fit+validation',
                                'R3_gate':'evaluate R3 only if train-side validation physical forecast MSE beats R2; no heldout or detail labels used to decide'},
            'prediction_probes':{'family':'same multi-output Ridge for all arms', 'alpha_grid':list(ALPHAS),
                                 'selection':'select alpha on validation using fit-only encoders/scalers; refit encoder and Ridge on fit+validation after freezing alpha',
                                 'targets':'independent frozen twelve physical features at prospective horizons; targets are not learned latents',
                                 'normalization':'per-horizon target mean/std fitted on forecast fit targets for selection, then fit+validation for final scoring'},
            'primary':'mean normalized per-feature RMSE(R0) - RMSE(R2), 40ms, paired date-block bootstrap 2000 draws',
            'secondary':['R0_long versus R2 at 40ms','80ms and 200ms forecasting','R3 versus R2 if validation gate passes',
                         'independent full-waveform detail probe','zero-shot BirdPark transfer, descriptive only'],
            'detail_probe':{'targets':['full-waveform log1p spectral centroid','full-waveform log1p spectral bandwidth','Hilbert-envelope CV'],
                            'material':'24 independent waveform families x2 gains; first16 families probe fit, last8 heldout',
                            'fit':'fixed Ridge alpha=1; network frozen; all preprocessing probe-fit-only; same fixtures across arms',
                            'representation':'mean valid state after complete causal warmup; R0 mean physical features',
                            'retention_rule':'R2 aggregate normalized RMSE <=1.05 times R0 detail error; this is preservation, not proof of perceptual relevance'},
            'stage1_success':'R2 primary and stronger-history improvements have positive 95% paired date intervals AND independent detail retention passes',
            'stage2_gate':'remain blocked pending stage1 success and independent transfer evidence with >=3 recording groups; two BirdPark clips cannot establish transfer generalization',
            'scope_limits':['no unit discovery, boundaries, social labels or semantics','existing bat material previously explored; no untouched confirmation claim',
                            'native-rate absolute-frequency bands; unavailable frequency bins explicitly masked','recorded clips conditional on valid acoustic support, not full conversation coverage'],
            'stop_rule':'one pilot run after specification; no adaptive architecture/loss/hyperparameter expansion'}


def scaled(arrays,indices):
    output={}
    states={}
    for key in ('spectral','physical'):
        mask=arrays[key+'_valid'] & arrays['time_valid'][...,None]
        scaler=MaskedStandardizer().fit(arrays[key][indices],mask[indices])
        output[key],output[key+'_valid']=scaler.transform(arrays[key],mask)
        states[key]=scaler.state()
    output['valid']=arrays['time_valid']
    return {k:torch.from_numpy(v) for k,v in output.items()},states


def network(kind,data,train,val=None,epochs=None):
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=MACDE(use_physical=kind!='R1')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf')
    best_state=None
    best_epoch=0
    trace=[]
    for epoch in range(1,(epochs or EPOCHS)+1):
        model.train()
        order=rng.permutation(train)
        for start in range(0,len(order),BATCH):
            idx=order[start:start+BATCH]
            batch={k:v[idx] for k,v in data.items()}
            optimizer.zero_grad()
            result=model(batch['spectral'],batch['physical'],batch['valid'],batch['spectral_valid'],batch['physical_valid'])
            loss=forecast_objective(result,batch['physical'],batch['valid'],aux_weight=.1 if kind=='R3' else 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                batch={k:v[val] for k,v in data.items()}
                result=model(batch['spectral'],batch['physical'],batch['valid'],batch['spectral_valid'],batch['physical_valid'])
                value=float(forecast_objective(result,batch['physical'],batch['valid']))
            trace.append(value)
            if value<best:
                best,best_epoch,best_state=value,epoch,copy.deepcopy(model.state_dict())
    if val is not None:
        model.load_state_dict(best_state)
    return model,{'selected_epochs':best_epoch if val is not None else epochs,'validation_mse':best if val is not None else None,'trace':trace}


def encode(model,data):
    model.eval()
    output=[]
    with torch.no_grad():
        for start in range(0,len(data['valid']),BATCH):
            idx=slice(start,start+BATCH)
            out=model(data['spectral'][idx],data['physical'][idx],data['valid'][idx],data['spectral_valid'][idx],data['physical_valid'][idx])
            output.append(out['state'].numpy())
    return np.concatenate(output)


def designs(arrays,embeddings,items,indices,horizon):
    rows=[]
    targets=[]
    groups=[]
    for i in indices:
        keep=eligible_mask(torch.from_numpy(arrays['time_valid'][i])[None])[0].numpy()
        for t in np.flatnonzero(keep):
            rows.append((i,t))
            targets.append(arrays['physical'][i,t+horizon])
            groups.append(items[i]['group'])
    x={}
    p=arrays['physical']
    x['R0']=np.array([np.concatenate([p[i,t],p[i,t-1]]) for i,t in rows])
    x['R0_long']=np.array([p[i,t-CONTEXT_FRAMES+1:t+1].reshape(-1) for i,t in rows])
    for kind,z in embeddings.items():
        x[kind]=np.array([z[i,t] for i,t in rows])
    return x,np.asarray(targets),groups,rows


def choose_ridge(x,y,xval,yval):
    scale=StandardScaler().fit(x)
    mu,sd=y.mean(0),np.maximum(y.std(0),1e-6)
    best=(float('inf'),None)
    for alpha in ALPHAS:
        w,b=ridge_fit(scale.transform(x),(y-mu)/sd,alpha)
        residual=ridge_predict(w,b,scale.transform(xval))-(yval-mu)/sd
        loss=float(np.sqrt((residual**2).mean(0)).mean())
        if loss<best[0]:
            best=(loss,alpha)
    return {'validation_nrmse':best[0],'alpha':best[1]}


def fit_ridge(x,y,alpha):
    scale=StandardScaler().fit(x)
    mu,sd=y.mean(0),np.maximum(y.std(0),1e-6)
    w,b=ridge_fit(scale.transform(x),(y-mu)/sd,alpha)
    return scale,mu,sd,w,b


def residual(predictor,x,y):
    scale,mu,sd,w,b=predictor
    return ridge_predict(w,b,scale.transform(x))-(y-mu)/sd


def loss(resid):
    return float(np.sqrt((resid**2).mean(0)).mean())


def boot(a,b,groups):
    rng=np.random.default_rng(SEED)
    unique=sorted(set(groups))
    grouped=[np.where(np.asarray(groups)==g)[0] for g in unique]
    draws=[]
    for _ in range(2000):
        idx=np.concatenate([grouped[j] for j in rng.integers(0,len(unique),len(unique))])
        draws.append(loss(a[idx])-loss(b[idx]))
    return {'point':loss(a)-loss(b),'ci95':np.quantile(draws,[.025,.975]).tolist(),'clusters':len(unique)}


def run():
    manifest,arrays=load()
    plan=protocol()
    if not PREREG.exists() or PREREG.read_text()!=text(plan):
        raise RuntimeError('Missing or changed preregistration')
    marker=ROOT/'outputs/macde_stage1_RUN_STARTED.json'
    if marker.exists():
        raise RuntimeError('One-run stop rule: run already started')
    freeze(marker,{'preregistration_sha256':sha(PREREG)})
    items=manifest['items']
    partitions={p:np.array([i for i,item in enumerate(items) if item['split']==p],int)
                for p in ('fit','validation','heldout','transfer','detail_fit','detail_eval')}
    contract=ScientificRunContract(run_name='macde_stage1',feature_schema_version=FEATURE_SCHEMA_VERSION,
        masks={},source_manifest={},eval_split={})
    contract.masks={name:MaskProvenance(kind=name,produced_by=description,win_ms=40,hop_ms=10,amplitude_stage='native waveform')
                    for name,description in manifest['masks'].items()}
    bats=[item for item in items if item['kind']=='bat']
    contract.source_manifest=source_manifest_from([i['name'] for i in bats],['eval' if i['split']=='heldout' else i['split'] for i in bats],[i['group'] for i in bats])
    contract.eval_split={'provenance':'date-group hash; heldout never fits weights/scalers/alpha','unit':'recording date'}
    contract.fitted_preprocessing={'encoder_and_ridge_scalers':'fit for selection; fit+validation for final refit'}
    release=raise_for_science(contract)
    selection_data,_=scaled(arrays,partitions['fit'])
    selection_models,training={},{}
    for kind in ('R1','R2','R3'):
        print(f'{kind}: training-side representation fit',flush=True)
        model,info=network(kind,selection_data,partitions['fit'],partitions['validation'])
        selection_models[kind]=model
        training[kind]=info
    retain_r3=training['R3']['validation_mse']<training['R2']['validation_mse']
    retained=['R1','R2']+(['R3'] if retain_r3 else [])
    print('Auxiliary objective passes training-side gate:',retain_r3,flush=True)
    selection_z={k:encode(selection_models[k],selection_data) for k in retained}
    chosen={}
    for horizon in HORIZONS:
        xt,yt,_,_=designs(arrays,selection_z,items,partitions['fit'],horizon)
        xv,yv,_,_=designs(arrays,selection_z,items,partitions['validation'],horizon)
        chosen[str(horizon)]={k:choose_ridge(xt[k],yt,xv[k],yv) for k in xt}
    allfit=np.sort(np.concatenate([partitions['fit'],partitions['validation']]))
    final_data,scaler_states=scaled(arrays,allfit)
    models={}
    final_z={}
    checkpoint_paths={}
    for kind in retained:
        print(f'{kind}: final refit for {training[kind]["selected_epochs"]} epochs',flush=True)
        model,_=network(kind,final_data,allfit,epochs=training[kind]['selected_epochs'])
        models[kind]=model
        final_z[kind]=encode(model,final_data)
        path=ROOT/f'checkpoints/macde_stage1_{kind}.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.state_dict(),'scalers':scaler_states,'preregistration_sha256':sha(PREREG),
                    'width':64,'use_physical':kind!='R1','physical_feature_names':supported_features(),
                    'frequency_band_edges_hz':BAND_EDGES.tolist(),'stage2_ready':False},path)
        checkpoint_paths[kind]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),
                                'parameters':sum(p.numel() for p in model.parameters()),
                                'state_std_fit':final_z[kind][allfit][arrays['time_valid'][allfit]].std(0).tolist()}
    forecasts={}
    residual_artifact={}
    for horizon in HORIZONS:
        xf,yf,_,_=designs(arrays,final_z,items,allfit,horizon)
        xe,ye,groups,positions=designs(arrays,final_z,items,partitions['heldout'],horizon)
        fits={k:fit_ridge(xf[k],yf,chosen[str(horizon)][k]['alpha']) for k in xf}
        res={k:residual(fits[k],xe[k],ye) for k in fits}
        transfer={}
        if len(partitions['transfer']):
            xx,yy,tg,_=designs(arrays,final_z,items,partitions['transfer'],horizon)
            if len(yy):
                transfer={'positions':len(yy),'recording_groups':len(set(tg)),
                          'nrmse':{k:loss(residual(fits[k],xx[k],yy)) for k in fits},'inference':'descriptive, no adaptation; only two recordings'}
        forecasts[str(horizon)]={'nrmse':{k:loss(r) for k,r in res.items()},'positions':len(ye),
                                'primary_R0_minus_R2':boot(res['R0'],res['R2'],groups),
                                'strong_control_R0long_minus_R2':boot(res['R0_long'],res['R2'],groups),
                                'R2_minus_R3':boot(res['R2'],res['R3'],groups) if retain_r3 else None,
                                'transfer':transfer}
        residual_artifact.update({f'h{horizon}_{k}':v for k,v in res.items()})
        residual_artifact[f'h{horizon}_source_frame']=np.asarray(positions)
    detail={}
    detail_x={'R0':[]}
    for k in retained:
        detail_x[k]=[]
    fixture_indices=np.concatenate([partitions['detail_fit'],partitions['detail_eval']])
    for i in fixture_indices:
        keep=eligible_mask(torch.from_numpy(arrays['time_valid'][i])[None],history=CONTEXT_FRAMES,horizon=0)[0].numpy()
        detail_x['R0'].append(arrays['physical'][i,keep].mean(0))
        for k in retained:
            detail_x[k].append(final_z[k][i,keep].mean(0))
    n=len(partitions['detail_fit'])
    refs=np.array([items[i]['reference'] for i in fixture_indices])
    for key,values in detail_x.items():
        values=np.asarray(values)
        model=fit_ridge(values[:n],refs[:n],1.)
        detail[key]={'normalized_rmse':loss(residual(model,values[n:],refs[n:]))}
    detail_ok=detail['R2']['normalized_rmse']<=1.05*detail['R0']['normalized_rmse']
    primary=forecasts['4']
    success=primary['primary_R0_minus_R2']['ci95'][0]>0 and primary['strong_control_R0long_minus_R2']['ci95'][0]>0 and detail_ok
    residual_path=ROOT/'outputs/macde_stage1_residuals.npz'
    np.savez_compressed(residual_path,**residual_artifact)
    result={'study':plan['study'],'preregistration_sha256':sha(PREREG),'code_provenance':code_hashes(),
            'release_contract':release,'training_selection':training,'retained_multiscale_auxiliary':retain_r3,
            'ridge_selection':chosen,'forecasting':forecasts,'detail_probe':detail,
            'detail_retention_passed':detail_ok,'stage1_pilot_criteria_passed':success,
            'stage2_ready':False,'stage2_block':'independent transfer validation requires >=3 recording groups; no automatic CAVEM integration',
            'checkpoints':checkpoint_paths,'residual_artifact':{'path':str(residual_path.relative_to(ROOT)),'sha256':sha(residual_path)},
            'limits':plan['scope_limits']}
    freeze(ROOT/'outputs/macde_stage1_results.json',result)
    print(text({'forecast_40ms':primary,'detail_probe':detail,'stage1_pilot_criteria_passed':success,'stage2_ready':False}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--preregister',action='store_true')
    group.add_argument('--run',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if args.prepare:
            prepare()
        elif args.preregister:
            plan=protocol()
            freeze(PREREG,plan)
            print(text({'preregistration':str(PREREG),'sha256':sha(PREREG),'eligible_positions':plan['eligible_positions'],'groups':plan['groups']}))
        else:
            run()


if __name__=='__main__':
    main()
