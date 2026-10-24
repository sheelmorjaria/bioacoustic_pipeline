"""Prespecified sampling and independent evidence flags for human alignment."""


def sampled_windows(duration, count=24, width=20):
    if duration/count <= width:
        raise ValueError('Sampling windows require disjoint strata wider than each window')
    return [{'window_id': f'W{i+1:02d}', 'start_seconds': int((i+.5)*duration/count-width/2),
             'end_seconds': int((i+.5)*duration/count-width/2)+width} for i in range(count)]


def validate_review(review):
    flags = ('boundary_verified', 'catalogue_id_verified', 'session_independence_verified')
    if review.get('status') != 'reviewed' and any(review.get(k) for k in flags):
        raise ValueError('Unreviewed rows cannot certify evidence')
    if review.get('boundary_verified') and not review.get('evidence'):
        raise ValueError('Boundary requires explicit source evidence beyond acoustic novelty')
    if review.get('catalogue_id_verified') and not (review.get('catalogue_id') and review.get('evidence')):
        raise ValueError('Catalogue linkage requires its own identifier and evidence')
    if review.get('session_independence_verified') and not review.get('session_evidence'):
        raise ValueError('Session independence requires separate evidence')
