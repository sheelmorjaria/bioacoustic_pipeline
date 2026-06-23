"""Repertoire taxonomy package — classify a corpus's vocal repertoire from validated diagnostics.

Architecture v1.0 Step 3. See ``classifier`` for the type vocabulary + classifier.
"""

from bioacoustic_pipeline.diagnostics.repertoire_taxonomy.classifier import (
    RepertoireProfile,
    RepertoireType,
    classify_repertoire,
    route_by_continuous,
)

__all__ = [
    "RepertoireProfile",
    "RepertoireType",
    "classify_repertoire",
    "route_by_continuous",
]
