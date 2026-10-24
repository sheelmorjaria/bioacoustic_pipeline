#!/usr/bin/env python3
"""Close frozen transfer and draft a separate, unexecuted multi-bird protocol."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    if path.exists():
        raise RuntimeError('Refusing to overwrite research record: '+str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    results_path = ROOT/'outputs/bfsongrepo_frozen_transfer_results.json'
    spec_path = ROOT/'outputs/bfsongrepo_frozen_transfer_preregistration.json'
    result = json.loads(results_path.read_text())
    spec = json.loads(spec_path.read_text())
    if result['preregistration_sha256'] != sha(spec_path):
        raise ValueError('Preregistration mismatch')
    for name, digest in spec['dependencies'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen dependency changed: '+name)
    if sha(ROOT/result['residual_artifact']['path']) != result['residual_artifact']['sha256']:
        raise ValueError('Residual provenance mismatch')
    unseen = ['gy6or6','bl26lb16','gr41rd51']
    effects = {b:{k:result['evaluations'][b][k] for k in
                 ['R0long_minus_R2','R0_minus_R2','shuffled_minus_ordered']} for b in unseen}
    if not all(effects[b]['R0_minus_R2']['point'] < 0 for b in unseen):
        raise ValueError('Closeout conclusion disagrees with source results')
    recovery_path = ROOT/'outputs/bfsongrepo_frozen_transfer_recovery_verification.json'
    recovery = json.loads(recovery_path.read_text())
    closeout = {
        'study':result['study'],'status':'completed_closed_mixed_result',
        'result_sha256':sha(results_path),'preregistration_sha256':sha(spec_path),
        'recovery_verification_sha256':sha(recovery_path),
        'original_neural_residuals_reproduced_exactly':all(v == 0 for k,v in recovery['max_absolute_residual_differences'].items() if k.endswith('__R2') or k.endswith('__R2_shuffled')),
        'results_modified':False,'further_tuning_permitted_in_this_branch':False,
        'unseen_bird_effects':effects,
        'conclusion':'Heterogeneous frozen end-to-end cross-individual forecasting. R2 improves over full-history Ridge on two unseen birds but has higher point error than short-history Ridge on all three. No consistent advantage over the complete baseline family.',
        'uncertainty_qualification':'Conditional recording bootstrap per bird/day. R0-R2 CI includes zero for bl26lb16; all-three point-error direction is not three separately significant failures.',
        'order_control':'Separate neural arms trained AND evaluated with their respective ordered/disrupted histories. Shuffling was not limited to inference. Ordered arm selected11epochs,disrupted6epochs;procedure-level contrast.',
        'mechanism':'Individual-specific dynamics is a working hypothesis,not an identified mechanism. Individual,recording domain,repertoire and other factors remain confounded.',
        'transfer_scope':'Frozen acoustic encoder plus existing forecast head. Does not test a frozen encoder with a newly fitted head.',
        'stage2_ready':False,'tier_b_certified':False,'multi_bird_training_performed':False,
    }
    manifest_path = ROOT/'outputs/bfsongrepo_frozen_transfer_population.json'
    manifest = json.loads(manifest_path.read_text())
    counts = Counter(i['recording_id'].split('/')[0] for i in manifest['items'])
    days = Counter('/'.join(i['recording_id'].split('/')[:2]) for i in manifest['items'])
    if sum(counts.values()) != 552 or len(counts) != 4 or len(days) != 6:
        raise ValueError('Population mismatch')
    birds = sorted(counts)
    draft = {
        'study':'BFSongRepo multi-bird acoustic forecasting successor',
        'status':'prospective_design_draft_not_preregistered_not_executable',
        'parent_closeout':'outputs/bfsongrepo_frozen_transfer_closeout.json',
        'population_reference':{'path':str(manifest_path.relative_to(ROOT)),'sha256':sha(manifest_path),
                                'recordings_by_bird':dict(sorted(counts.items())),
                                'recordings_by_bird_day':dict(sorted(days.items())),
                                'targets':manifest['targets']},
        'hypothesis':'Training on multiple identified individuals may improve forecasting for a held-out individual relative to training-selected Ridge baselines.',
        'separation':'New from-scratch models;not a repair of frozen transfer. No CAVEM integration,compression,adaptation,denoising,syllable labels or new objectives.',
        'outer_evaluation':'Four leave-one-bird-out folds;all dates and recordings of one bird held out together. Fit final models on remaining three birds only.',
        'proposed_outer_folds':[{'held_out_bird':b,'training_birds':[x for x in birds if x != b]} for b in birds],
        'eligibility':'Reuse frozen16 uniformly spaced eligible targets per recording,128-frame support,52-frame input,40ms12-physical-feature targets. No random frame splits or cross-recording histories.',
        'arms':['R0 physical short-history Ridge','R0_long physical52-frame Ridge','R2 combined MACDE','R2_shuffled matched independently trained disrupted-history MACDE'],
        'neural_policy':{'initialization':'From scratch,seed0,no prior checkpoint warm start',
                         'epochs':12,'batch_size':32,'optimizer':'Adam lr0.001 weight_decay0.0001 clip5',
                         'selection':'Fixed12epochs for both neural arms,based on pre-existing training budget;no early stopping or held-out-bird selection',
                         'sampling':'All training rows once per epoch in deterministic shuffled order. Unequal source recording counts are retained and disclosed;bird-balanced training is not an additional arm.',
                         'history_control':'Permute older50frames jointly with masks,anchor newest2;train and evaluate each arm under its declared policy.'},
        'ridge_comparator_selection':{
            'inner_unit':'Bird-disjoint leave-one-bird-out within the three outer-training birds:fit on two,validate on third,three inner folds.',
            'candidates':'R0 and R0_long crossed with the unchanged original alpha grid;grid values must be pinned in runnable preregistration.',
            'score':'Mean of the three validation-bird normalized RMSE values,each evaluated using physical-target feature scales estimated from its two fit birds only.',
            'tie_rule':'Exact ties prefer R0,then smaller alpha.',
            'refit':'Freeze selected family and alpha,fit on all three outer-training birds before opening held-out-bird results. Also fit/report both Ridge families with their own inner-selected alpha.',
            'prohibited':'Choosing the primary comparator using held-out-bird performance.'},
        'preprocessing':'Every fold estimates input scalers,target transforms and common physical-target reporting scale exclusively from its fit birds. Outer target bird never contributes fitted statistics.',
        'primary':'Four separately reported held-out-bird differences:nRMSE(training-selected Ridge)-nRMSE(R2). No pooled-frame primary endpoint.',
        'criterion':'Prospective stringent screening:all four bird-level point differences positive and all four conditional recording-bootstrap lower95%bounds positive. Even a pass is conditional acoustic evidence,not a species/population-level confidence statement or Stage2 unlock.',
        'secondary':'All-arm per-bird errors,comparisons with both Ridge families,and matched shuffled-minus-ordered contrast;do not replace the primary comparator after scoring.',
        'uncertainty':'Report four effects individually and optional unweighted descriptive mean. Recording bootstrap conditional on each bird,days and fixed fits;no bird-population confidence interval. Outer fits overlap;four folds are not four independent experiments.',
        'exposure':'All four birds already evaluated in previous research. Protocol can be prospective for a new training procedure but cannot make this corpus an untouched confirmation set.',
        'generalisation_limit':'Bird identity separation in one laboratory collection;independent sessions,populations and acquisition environments not certified.',
        'stop':'One preregistered four-fold run after implementation and leakage tests are frozen. No outcome-driven reseeding,extra epochs,fold replacement,objectives or architecture changes.',
        'required_before_execution':['Implement new runner without editing closed scripts',
                                     'Test outer and inner bird isolation,train-only preprocessing,target alignment and matched shuffle policies',
                                     'Pin alpha grid,code,dependencies,source/cache hashes and all fit/validation/evaluation manifests',
                                     'Freeze executable preregistration and fail-closed one-run marker before fitting'],
        'models_fitted':False,'evaluation_executed':False,'stage2_ready':False,'tier_b_certified':False,
    }
    closeout_path = ROOT/'outputs/bfsongrepo_frozen_transfer_closeout.json'
    draft_path = ROOT/'outputs/bfsongrepo_multibird_design_draft.json'
    if closeout_path.exists() or draft_path.exists():
        raise RuntimeError('Closeout or design draft already exists')
    write_new(closeout_path,closeout)
    write_new(draft_path,draft)
    print('Closed frozen transfer;separate multi-bird design draft saved;no fitting or evaluation.')


if __name__ == '__main__':
    main()
