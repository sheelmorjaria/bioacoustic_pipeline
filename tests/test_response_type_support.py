"""A label-support screen must not silently change the protected cohort."""
import copy

import pytest

from bioacoustic_pipeline.contextual.response_type_support import label_support


def row(i, label, partition, date):
    return {'event_id':str(i),'target':label,'partition':partition,'date':date}


def test_unseen_heldout_category_blocks_the_whole_comparison_without_filtering():
    rows=[row(1,'Phee','fit','a'),row(2,'Trill','fit','b'),
          row(3,'Phee','heldout','c'),row(4,'Chirp','heldout','d')]
    original=copy.deepcopy(rows)
    out=label_support(rows)
    assert not out['fit_allowed']
    assert out['heldout_labels_absent_from_fit']==['Chirp']
    assert out['partition_counts']=={'fit':2,'heldout':2}
    assert rows==original
    assert out['class_support']['Chirp']['heldout_count']==1


def test_unassigned_rows_do_not_supply_training_label_support():
    rows=[row(1,'Phee','fit','a'),row(2,'Chirp','unassigned',None),
          row(3,'Chirp','heldout','b')]
    out=label_support(rows)
    assert out['class_support']['Chirp']['fit_count']==0
    assert out['class_support']['Chirp']['unassigned_count']==1
    assert 'unseen_heldout_target_class' in out['blocked_by']


def test_screen_reports_class_cluster_support_and_sparse_classes():
    rows=[row(1,'Phee','fit','a'),row(2,'Trill','fit','a'),row(3,'Phee','fit','b'),
          row(4,'Trill','fit','c'),row(5,'Phee','heldout','d'),row(6,'Trill','heldout','d'),
          row(7,'Phee','heldout','e'),row(8,'Trill','heldout','f')]
    out=label_support(rows)
    assert not out['fit_allowed']
    assert out['class_support']['Phee']['fit_dates']==2
    assert out['class_support']['Trill']['heldout_dates']==2
    assert out['sparse_fit_classes_below_5']==['Phee','Trill']


def test_duplicate_ids_missing_targets_and_date_leakage_fail_closed():
    rows=[row(1,'Phee','fit','a'),row(1,'Trill','heldout','b')]
    with pytest.raises(ValueError,match='unique'):
        label_support(rows)
    rows=[row(1,'Phee','fit','a'),row(2,None,'heldout','b')]
    with pytest.raises(ValueError,match='target'):
        label_support(rows)
    rows=[row(1,'Phee','fit','a'),row(2,'Trill','heldout','a')]
    with pytest.raises(ValueError,match='date'):
        label_support(rows)


def test_supported_multiclass_screen_can_pass_without_recoding():
    rows=[]
    for label in ['Phee','Trill']:
        for j in range(5):
            rows.append(row(f'{label}-fit-{j}',label,'fit',f'fit-{j%3}'))
        for j in range(3):
            rows.append(row(f'{label}-held-{j}',label,'heldout',f'held-{j}'))
    assert label_support(rows)['fit_allowed']
