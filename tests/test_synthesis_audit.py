import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "synthesis_audit", ROOT / "scripts" / "synthesis_audit.py"
)
synthesis_audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(synthesis_audit)


def test_synthesis_audit_flags_omitted_unpromoted_candidate():
    dossier = {
        "run_meta": {"run_id": "run-test"},
        "clusters": [
            {
                "cluster_id": "cluster-1",
                "canonical_url": "https://github.com/example/project-x",
                "titles": ["Project-X-1B release"],
                "snippets": ["Project-X-1B is an open source editing model."],
                "topic_terms": ["project", "editing", "model"],
            }
        ],
        "extracted_pages": [],
        "candidate_audit": {
            "unpromoted_candidates": [
                {
                    "candidate_id": "candidate-1",
                    "name": "Project-X-1B",
                    "surface_forms": ["example/Project-X-1B"],
                    "scores": {"promotion_score": 0.72},
                    "why_notable": "High-fit retrieved candidate.",
                    "trace": {"cluster_ids": ["cluster-1"]},
                }
            ]
        },
    }
    answer = (
        "The final shortlist includes only another model with strong editing support."
    )
    audit = synthesis_audit.build_synthesis_audit(dossier, answer)

    assert audit["status"]["requires_revision"] is True
    assert audit["omitted_high_fit_candidates"][0]["name"] == "Project-X-1B"
    assert (
        "include_or_exclude_unpromoted_candidate:Project-X-1B"
        in audit["required_actions"]
    )
