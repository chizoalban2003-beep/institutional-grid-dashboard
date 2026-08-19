"""MIMIC-IV Data Engine — relational EHR to (N, K×3) tensor."""
from data_engine.mimic_ingest import (
    MIMIC_TO_FEATURE,
    FEATURE_NAMES,
    FEATURE_INDEX,
    K,
    K3,
    MIMICStayAssembler,
    export_stay_tensors,
    run_pipeline,
)
from data_engine.synthetic_mimic import (
    generate_cohort,
    generate_stay,
    ITEMIDS,
)

__all__ = [
    "MIMIC_TO_FEATURE",
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "K",
    "K3",
    "MIMICStayAssembler",
    "export_stay_tensors",
    "run_pipeline",
    "generate_cohort",
    "generate_stay",
    "ITEMIDS",
]
