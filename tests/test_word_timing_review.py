import pytest
from bioacoustic_pipeline.ingestion.word_timing_review import validate_row, review_gate


def valid_row():
    return {'target_id':'t1','status':'supported','onset_sample':'100','offset_sample':'300',
            'onset_uncertainty_samples':'20','offset_uncertainty_samples':'30',
            'evidence':'Independent listening and waveform inspection'}


def test_unreviewed_fields_cannot_certify_timing():
    row = valid_row()
    row['status'] = ''
    with pytest.raises(ValueError,match='Unreviewed'):
        validate_row(row,1000)
    empty = {'target_id':'t1','status':''}
    assert not validate_row(empty,1000)['supported']


def test_bounds_uncertainty_and_evidence_required():
    for field,value in [('offset_sample','1100'),('onset_sample','350'),
                        ('onset_uncertainty_samples','-1'),('evidence','')]:
        row = valid_row()
        row[field] = value
        with pytest.raises(ValueError):
            validate_row(row,1000)
    assert validate_row(valid_row(),1000)['supported']


def test_independent_reviews_and_adjudication_are_separate_gates():
    a={'annotator_id':'human-a','independent_review':True,'model_boundaries_hidden':True}
    b={'annotator_id':'human-b','independent_review':True,'model_boundaries_hidden':True}
    row=valid_row()
    assert not review_gate([row],[row],a,b,{}, {'t1':1000})['ready']
    with pytest.raises(ValueError,match='distinct'):
        review_gate([row],[row],a,a,{}, {'t1':1000})
    adjudicated={'t1':{'status':'approved','onset_sample':'110','offset_sample':'290',
                       'onset_uncertainty_samples':'30','offset_uncertainty_samples':'30',
                       'evidence':'Joint review resolved interval','adjudicator_id':'human-a'}}
    assert review_gate([row],[row],a,b,adjudicated,{'t1':1000})['ready']
