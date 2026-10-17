"""FAF-FCTPBench benchmark generation, validation, and reference-solving tools."""

__version__ = "0.0.0"

from .catalogue import CandidateSpecification, curate_benchmark_core, generate_candidate_catalogue
from .instance import FCTPInstance

__all__ = [
    "CandidateSpecification",
    "FCTPInstance",
    "curate_benchmark_core",
    "generate_candidate_catalogue",
]
